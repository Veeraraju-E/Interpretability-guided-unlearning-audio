#!/usr/bin/env python3
"""
VoxCeleb1/2 targeted speaker unlearning pipeline following LibriSpeech framework.
- Updated to match librispeech.py structure and flow exactly
- Preserves VoxCeleb-specific functionality (manifest format, precomputed embeddings)
- Key alignments with LibriSpeech:
  * Same pipeline step numbering and order
  * Consistent parameter handling and defaults
  * Identical result structure and evaluation metrics
  * Same CLI argument structure where applicable
"""

from __future__ import annotations
import os, time, json, random
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import numpy as np
import torch

from common import (
    seed_all, human_time, extract_embeddings, fit_whitener, apply_whitener,
    invert_whitener, l2_normalize, TopKSAE, train_sae, sae_encode_batch,
    prune_and_normalize_decoder, feature_stats, select_features_from_stats, build_U_from_decoder,
    leace_projector_in_U, apply_subspace_leace, apply_sauce_gating,
    inject_noise_in_subspace, compute_scores, eer_from_scores, build_verification_trials, 
    train_classifier, evaluate_classifier, linear_probe_U, per_class_eers, avg_eers
)


# -------------------- Manifest --------------------
def read_voxceleb_manifest(manifest_path: str) -> List[Dict]:
    """Read VoxCeleb manifest file and convert to common format."""
    manifest = []
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            seg = {
                'path': parts[1],  # wav_path -> path for common.py compatibility
                'speaker': parts[4] if len(parts) > 4 else os.path.basename(os.path.dirname(parts[1])),
                'segment_id': parts[0],
                'start': float(parts[2]) if len(parts) > 2 and parts[2] != '' else 0.0,
                'end': float(parts[3]) if len(parts) > 3 and parts[3] != '' else None,
                'session_id': parts[5] if len(parts) > 5 else None
            }
            manifest.append(seg)
    return manifest


def build_manifest(manifest_path: str, embeddings_path: str = None, 
                  dev_frac_per_spk: float = 0.75, seed: int = 1337) -> List[Dict]:
    """Build manifest and create dev/test splits similar to LibriSpeech pipeline.
    
    Returns list of dict entries with keys: path, speaker, set, split
    (matching LibriSpeech format exactly)
    """
    
    if embeddings_path is not None:
        # Load precomputed embeddings
        data = np.load(embeddings_path, allow_pickle=True)
        labels = data['labels'].tolist() if isinstance(data['labels'], np.ndarray) else list(data['labels'])
        ids = data['segment_ids'].tolist() if isinstance(data['segment_ids'], np.ndarray) else list(data['segment_ids'])
        raw_manifest = [{'path': f'dummy_{i}', 'speaker': labels[i], 'segment_id': ids[i]} for i in range(len(labels))]
    else:
        # Read manifest
        raw_manifest = read_voxceleb_manifest(manifest_path)
    
    # Group utterances by speaker (same as LibriSpeech)
    by_speaker = {}
    for i, m in enumerate(raw_manifest):
        speaker = m['speaker']
        by_speaker.setdefault(speaker, []).append(i)
    
    # Create dev/test splits per speaker (same logic as LibriSpeech)
    rng = random.Random(seed)
    manifest = []
    
    for speaker, indices in by_speaker.items():
        rng.shuffle(indices)
        n_dev = max(1, int(len(indices) * dev_frac_per_spk))
        dev_indices = indices[:n_dev]
        test_indices = indices[n_dev:]
        
        for idx in dev_indices:
            entry = raw_manifest[idx].copy()
            entry['split'] = 'dev'
            # 'set' will be added later based on forget_spks
            manifest.append(entry)
        
        for idx in test_indices:
            entry = raw_manifest[idx].copy()
            entry['split'] = 'test'
            # 'set' will be added later based on forget_spks
            manifest.append(entry)
    
    if len(manifest) == 0:
        raise RuntimeError(f"No entries found in manifest")
    
    return manifest


# -------------------- Orchestration: full run --------------------

def run_pipeline(manifest_path: str,
                 save_dir: str = "./runs/exp",
                 device: str = "cuda",
                 seed: int = 1337,
                 retain_frac: float = 0.975,
                 dev_frac_per_spk: float = 0.75,
                 sae_hidden_mult: float = 4.0,
                 sae_topk: int = 128,
                 sae_epochs: int = 30,
                 build_U_top: int = 200,
                 beta: float = 0.0,
                 leace: bool = True,
                 leace_ridge: float = 1e-6,
                 noise_sigma: float = 0.02,
                 per_speaker_union: bool = False,
                 forget_spks: Optional[List[str]] = None,
                 encoder: str = "ecapa",
                 embeddings_path: str = None,
                 wav2vec2_model: str = "facebook/wav2vec2-base",
                 hubert_model: str = "facebook/hubert-base-ls960"):
    """
    Main pipeline orchestration. Matches librispeech.py structure exactly.
    """
    seed_all(seed)
    os.makedirs(save_dir, exist_ok=True)
    t0_all = time.time()

    # 1) manifest + splits
    print("[1/12] Building manifest and splits...")
    entries = build_manifest(manifest_path, embeddings_path, dev_frac_per_spk, seed)

    # group utterances by speaker (same as LibriSpeech)
    by = {}
    for e in entries:
        by.setdefault(e["speaker"], []).append(e["path"])

    spks = sorted(by.keys())

    # decide which speakers to forget (same logic as LibriSpeech)
    rng = random.Random(seed)
    if forget_spks is None:
        # use retain_frac to derive how many speakers to *retain*; the rest become forget
        n_total = len(spks)
        n_retain = max(1, int(round(retain_frac * n_total)))
        n_forget = max(1, n_total - n_retain)
        forget_spks = rng.sample(spks, k=n_forget)
    elif isinstance(forget_spks, str):
        forget_spks = [forget_spks]
    else:
        # ensure we have a list copy
        forget_spks = list(forget_spks)

    # Validate that all requested forget speakers exist
    for spk in forget_spks:
        if spk not in spks:
            raise ValueError(f"Requested forget speaker {spk} not found in dataset")

    print(f"  FORGETTING SPEAKERS ({len(forget_spks)}): {forget_spks}")

    retain_spks = [s for s in spks if s not in forget_spks]

    # Add 'set' labels to manifest entries (same as LibriSpeech)
    manifest = []
    for entry in entries:
        entry_copy = entry.copy()
        entry_copy['set'] = 'forget' if entry['speaker'] in forget_spks else 'retain'
        manifest.append(entry_copy)

    # create lists for each split+set (same as LibriSpeech)
    dev_manifest = [m for m in manifest if m['split'] == 'dev']
    test_manifest = [m for m in manifest if m['split'] == 'test']
    dev_retain = [m for m in dev_manifest if m['set'] == 'retain']
    dev_forget = [m for m in dev_manifest if m['set'] == 'forget']
    test_retain = [m for m in test_manifest if m['set'] == 'retain']
    test_forget = [m for m in test_manifest if m['set'] == 'forget']
    print(f"  retain_spks {len(retain_spks)} forget_spks {len(forget_spks)}")
    print(f"  dev_retain {len(dev_retain)} dev_forget {len(dev_forget)} | test_retain {len(test_retain)} test_forget {len(test_forget)}")

    # 2) Embedding extraction
    if embeddings_path is not None:
        print("[2/12] Loading precomputed embeddings...")
        data = np.load(embeddings_path, allow_pickle=True)
        E_all = data['embeddings'].astype(np.float32)
        # Align labels and paths with manifest order
        L_all = [m['speaker'] for m in manifest]
        P_all = [m['path'] for m in manifest]
        print(f"  Loaded {len(manifest)} samples, {len(set(L_all))} speakers")
        print(f"  Embeddings shape: {E_all.shape}")
    else:
        print("[2/12] Extracting embeddings...")
        all_manifest = dev_manifest + test_manifest
        E_all, L_all, P_all = extract_embeddings(all_manifest,
                                                 encoder=encoder,
                                                 device=device,
                                                 wav2vec2_model=wav2vec2_model,
                                                 hubert_model=hubert_model,
                                                 label_key='speaker')
    
    labels = L_all

    # Create set labels (same as LibriSpeech)
    sets = [f"{m['set']}:{m['split']}" for m in manifest]

    # masks (boolean arrays aligned with all_manifest / E_all / labels) - same as LibriSpeech
    M_ret_dev = np.array([s == 'retain:dev' for s in sets], dtype=bool)
    M_fgt_dev = np.array([s == 'forget:dev' for s in sets], dtype=bool)
    M_ret_test = np.array([s == 'retain:test' for s in sets], dtype=bool)
    M_fgt_test = np.array([s == 'forget:test' for s in sets], dtype=bool)

    # 3) Whitening on retain-dev (same as LibriSpeech)
    print("[3/12] Fitting whitener on Train-retain (retain-dev)...")
    E_ret_train = E_all[M_ret_dev]
    if E_ret_train.shape[0] < 2:
        raise RuntimeError("Not enough retain-dev samples to fit whitener")
    mu, W = fit_whitener(E_ret_train)
    Z_all = apply_whitener(E_all, mu, W)

    # 4) SAE training on dev (retain + forget) - same as LibriSpeech
    print("[4/12] Training SAE on Z_train (dev union: retain+forget)...")
    Z_train = Z_all[M_ret_dev | M_fgt_dev]
    perm = np.random.permutation(Z_train.shape[0])
    cut = int(0.8 * len(perm))
    Z_sae_train = Z_train[perm[:cut]]
    Z_sae_val = Z_train[perm[cut:]]
    sae = TopKSAE(input_dim=Z_all.shape[1], hidden_mult=sae_hidden_mult, k=sae_topk)
    sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=sae_epochs, batch_size=4096, lr=1e-3, device=device)

    # 5) SAE encodes for feature mining (train-retain & train-forget) - same as LibriSpeech
    print("[5/12] Encoding SAE activations for train-retain and train-forget...")
    Z_ret_dev = Z_all[M_ret_dev]
    Z_fgt_dev = Z_all[M_fgt_dev]
    A_ret = sae_encode_batch(sae, Z_ret_dev, device=device, batch=8192)
    A_fgt = sae_encode_batch(sae, Z_fgt_dev, device=device, batch=8192)

    # 6) feature stats & selection (supports per-speaker union) - same as LibriSpeech
    print("[6/12] Computing feature statistics and selecting features...")
    stats = feature_stats(A_fgt, A_ret)
    feat_idx_global = select_features_from_stats(stats, top_k=build_U_top, fdr=0.05)
    feat_idx = feat_idx_global
    if per_speaker_union:
        print("  computing per-forget-speaker feature unions... (not implemented, using global)")
        feat_idx = feat_idx_global

    # 7) prune & normalize decoder & build U - same as LibriSpeech
    print("[7/12] Building U subspace...")
    dec_w = sae.dec.weight.detach().cpu().numpy()   # [d, m]
    dec_pruned = prune_and_normalize_decoder(dec_w, prune_threshold=1e-6)  # returns [d, r]
    U = build_U_from_decoder(dec_w, feat_idx)
    print(f"  U shape: {U.shape}")

    # 8) compute LEACE in U if requested - same as LibriSpeech
    P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
    mu_r = None
    if leace:
        print("[8/12] Computing LEACE projector in U...")
        mask_dev = np.array([s.endswith(':dev') for s in sets], dtype=bool)
        # use indices to avoid repeated index() lookups (same fix as LibriSpeech)
        dev_indices = np.where(mask_dev)[0]
        Y_dev = np.array([1 if manifest[i]['speaker'] in forget_spks else 0 for i in dev_indices], dtype=int).reshape(-1, 1)
        # safety check: shapes must align
        assert Y_dev.shape[0] == Z_all[mask_dev].shape[0], "Y_dev rows must match dev rows of Z_all"
        try:
            P_r, mu_r = leace_projector_in_U(Z_all[mask_dev], U, Y_dev, ridge=leace_ridge)
        except Exception as e:
            print(f"  Warning: LEACE computation failed ({e}), falling back to identity projector")
            P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
            mu_r = np.zeros((1, U.shape[1]))
    else:
        print("[8/12] Skipping LEACE (identity)")

    # 9) Apply SAUCE gating (optional) then LEACE and optional noise - same as LibriSpeech
    print("[9/12] Applying SAUCE gating (beta) and LEACE edit...")
    Z_edit = Z_all.copy()
    if beta > 0.0:
        print("  applying SAUCE gating with beta", beta)
        Z_edit = apply_sauce_gating(sae, Z_all, feat_idx, beta=beta, device=device, batch=8192)
    if leace:
        Z_edit = apply_subspace_leace(Z_edit, U, P_r, mu_r)
    if noise_sigma and noise_sigma > 0:
        Z_edit = inject_noise_in_subspace(Z_edit, U, sigma=noise_sigma)

    # map back to embedding space and l2 normalize
    E_edit = invert_whitener(Z_edit, mu, W)
    E_all_norm, E_edit_norm = l2_normalize(E_all), l2_normalize(E_edit)

    # -------------------- 10) Per-class EERs - same as LibriSpeech --------------------
    print("[10/12] Per-class verification EERs (each class vs all others) ...")
    test_trials, test_labels = build_verification_trials(labels, forget_spks, M_ret_test | M_fgt_test)
    scores_test_before = compute_scores(E_all_norm, test_trials)
    scores_test_after = compute_scores(E_edit_norm, test_trials)
    eer_before = eer_from_scores(scores_test_before, test_labels)
    eer_after = eer_from_scores(scores_test_after, test_labels)
    results["EER_before_raw"] = eer_before
    results["EER_after_raw"] = eer_after
    print(f"  Global EER: before {eer_before:.4f}, after {eer_after:.4f}")
    all_spks = sorted(set(labels))
    spk2id = {d: i for i, d in enumerate(all_spks)}

    # Build test lists
    E_ret_test = E_all_norm[M_ret_test]
    L_ret_test = [spk2id[d] for d in np.array(labels)[M_ret_test]]
    E_fgt_test = E_all_norm[M_fgt_test]
    L_fgt_test = [spk2id[d] for d in np.array(labels)[M_fgt_test]]
    E_ret_test_edit = E_edit_norm[M_ret_test]
    E_fgt_test_edit = E_edit_norm[M_fgt_test]

    per_class_test = per_class_eers(E_all_norm, E_edit_norm, labels, mask=M_ret_test | M_fgt_test, rng_seed=seed)
    per_class_ret = {c: vals for c, vals in per_class_test.items() if c in retain_spks}
    per_class_fgt = {c: vals for c, vals in per_class_test.items() if c in forget_spks}

    retain_avg_before, retain_avg_after = avg_eers(per_class_ret)
    forget_avg_before, forget_avg_after = avg_eers(per_class_fgt)

    results = {}
    results["retain_EER_before_avg"] = retain_avg_before
    results["retain_EER_after_avg"] = retain_avg_after
    results["forget_EER_before_avg"] = forget_avg_before
    results["forget_EER_after_avg"] = forget_avg_after

    # -------------------- 11) Linear Probing Evaluation - same as LibriSpeech --------------------
    print("[11/12] Linear probing evaluation in U subspace (forget vs retain speakers)...")
    split_mask = {"dev": M_ret_dev | M_fgt_dev, "test": M_ret_test | M_fgt_test}
    acc_before, auc_before = linear_probe_U(Z_all, labels, U, split_mask, forget_classes=forget_spks)
    acc_after, auc_after = linear_probe_U(Z_edit, labels, U, split_mask, forget_classes=forget_spks)

    print(f'  Linear Probing accuracies and AUC, before: ({acc_before:.4f}, {auc_before:.4f}), after: ({acc_after:.4f}, {auc_after:.4f})')
    results["linear_probe_acc_before"] = acc_before
    results["linear_probe_acc_after"] = acc_after
    results["linear_probe_auc_before"] = auc_before
    results["linear_probe_auc_after"] = auc_after

    # Print detailed and summary outputs (same as LibriSpeech)
    print("\nPer-class EERs (sample):")
    for i, (c, (eb, ea)) in enumerate(sorted(per_class_test.items())):
        if c in forget_spks:
            print("  [FORGET SPEAKER]")
        print(f"  class {c}: EER_before={eb:.4f}, EER_after={ea:.4f}")

    print("\n--- Average EER summary ---")
    print(f"  Retain avg EER: before {retain_avg_before:.4f}, after {retain_avg_after:.4f}")
    print(f"  Forget avg EER: before {forget_avg_before:.4f}, after {forget_avg_after:.4f}")

    # Save summary JSON (same format as LibriSpeech)
    stats_path = os.path.join(save_dir, "verification_eer_summary.json")
    with open(stats_path, "w") as fh:
        json.dump({
            "per_class_count_test": len(per_class_test),
            "retain_avg": {"before": retain_avg_before, "after": retain_avg_after},
            "forget_avg": {"before": forget_avg_before, "after": forget_avg_after},
            "linear_probe": {
                "acc_before": acc_before, "acc_after": acc_after,
                "auc_before": auc_before, "auc_after": auc_after
            },
            "forget_spks": forget_spks,
        }, fh, indent=2)

    # -------------------- 12) Speaker classifier (ResNet) training - same as LibriSpeech --------------------
    print("[12/12] Speaker classification accuracy (ResNet) before/after edits...")
    all_spks = sorted(list(set(labels)))
    spk2id = {s: i for i, s in enumerate(all_spks)}

    X_cls_train = E_all_norm[M_ret_dev | M_fgt_dev]
    y_cls_train = [spk2id[s] for i, s in enumerate(labels) if (M_ret_dev | M_fgt_dev)[i]]
    X_cls_val = E_all_norm[M_ret_test | M_fgt_test]
    y_cls_val = [spk2id[s] for i, s in enumerate(labels) if (M_ret_test | M_fgt_test)[i]]

    clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(all_spks), device=device, epochs=20, batch_size=512, lr=1e-3)

    acc_ret_before = evaluate_classifier(clf, E_ret_test, L_ret_test, device=device)
    acc_fgt_before = evaluate_classifier(clf, E_fgt_test, L_fgt_test, device=device)
    acc_ret_after = evaluate_classifier(clf, E_ret_test_edit, L_ret_test, device=device)
    acc_fgt_after = evaluate_classifier(clf, E_fgt_test_edit, L_fgt_test, device=device)

    print(f"  Classifier Retain: before {acc_ret_before*100:.2f}% after {acc_ret_after*100:.2f}%")
    print(f"  Classifier Forget: before {acc_fgt_before*100:.2f}% after {acc_fgt_after*100:.2f}%")

    torch.save(sae.state_dict(), os.path.join(save_dir, "sae_state.pt"))
    print("Done. Total time:", human_time(time.time() - t0_all))


# -------------------- CLI --------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_path", default="../datasets/voxceleb2/manifest.csv", help="VoxCeleb manifest file")
    p.add_argument("--save_dir", default="./outputs", help="store artifacts")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--retain_frac", type=float, default=0.7)
    p.add_argument("--dev_frac_per_spk", type=float, default=0.7)
    p.add_argument("--sae_hidden_mult", type=float, default=10.0)
    p.add_argument("--sae_topk", type=int, default=128)
    p.add_argument("--sae_epochs", type=int, default=150)
    p.add_argument("--build_U_top", type=int, default=200)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--leace", action="store_true")
    p.add_argument("--leace_ridge", type=float, default=1e-6)
    p.add_argument("--noise_sigma", type=float, default=0.02)
    p.add_argument("--forget_spks", nargs='+', default=["id00017"], help="Speaker IDs to forget (e.g. 'id00017' 'id00018')")
    p.add_argument("--encoder", type=str, default="ecapa", help="Encoder to use - 'ecapa', 'wav2vec2', 'hubert'")
    p.add_argument("--embeddings_path", type=str, default=None, help="Load precomputed embeddings .npz")
    p.add_argument("--wav2vec2_model", type=str, default="facebook/wav2vec2-base")
    p.add_argument("--hubert_model", type=str, default="facebook/hubert-base-ls960")
    
    args = p.parse_args()

    run_pipeline(
        manifest_path=args.manifest_path, save_dir=args.save_dir,
        device=args.device, seed=args.seed,
        retain_frac=args.retain_frac, dev_frac_per_spk=args.dev_frac_per_spk,
        sae_hidden_mult=args.sae_hidden_mult, sae_topk=args.sae_topk, sae_epochs=args.sae_epochs,
        build_U_top=args.build_U_top, beta=args.beta, leace=args.leace, leace_ridge=args.leace_ridge,
        noise_sigma=args.noise_sigma, encoder=args.encoder, forget_spks=args.forget_spks,
        embeddings_path=args.embeddings_path, wav2vec2_model=args.wav2vec2_model, hubert_model=args.hubert_model
    )