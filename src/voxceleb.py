from __future__ import annotations
import os, time, json, random
from typing import List, Dict, Optional
import argparse
import numpy as np

from common import (
    seed_all, human_time, extract_embeddings, fit_whitener, apply_whitener,
    invert_whitener, l2_normalize, SAE, train_sae, sae_encode_batch,
    prune_and_normalize_decoder, feature_stats, select_features_from_stats, build_U_from_decoder,
    leace_projector_in_U, apply_subspace_leace, apply_sauce_gating, inject_noise_in_subspace, 
    compute_scores, eer_from_scores, build_verification_trials,
    train_classifier, evaluate_classifier, per_class_eers, avg_eers, safe_linear_probe,
    calculate_monosemanticity_score, compute_monosemanticity_statistics
)

def read_voxceleb_manifest(manifest_path: str) -> List[Dict]:
    manifest = []
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            seg = {
                'path': parts[1],
                'speaker': parts[4] if len(parts) > 4 else os.path.basename(os.path.dirname(parts[1])),
                'segment_id': parts[0],
                'start': float(parts[2]) if len(parts) > 2 and parts[2] != '' else 0.0,
                'end': float(parts[3]) if len(parts) > 3 and parts[3] != '' else None,
                'session_id': parts[5] if len(parts) > 5 else None
            }
            manifest.append(seg)
    return manifest

def build_manifest(manifest_path: str, embeddings_path: str = None, dev_frac_per_spk: float = 0.75, seed: int = 1337) -> List[Dict]:
    if embeddings_path is not None:
        data = np.load(embeddings_path, allow_pickle=True)
        labels = data['labels'].tolist() if 'labels' in data else (data['y'].tolist() if 'y' in data else None)
        ids = data['segment_ids'].tolist() if 'segment_ids' in data else (data['ids'].tolist() if 'ids' in data else None)
        if labels is None:
            raise RuntimeError("embeddings .npz must contain 'labels' or 'y' array when embeddings_path is used")
        raw_manifest = [{'path': f'dummy_{i}', 'speaker': labels[i], 'segment_id': ids[i] if ids is not None else None} for i in range(len(labels))]
    else:
        raw_manifest = read_voxceleb_manifest(manifest_path)

    by_speaker = {}
    for i, m in enumerate(raw_manifest):
        speaker = m['speaker']
        by_speaker.setdefault(speaker, []).append(i)

    rng = random.Random(seed)
    manifest = []
    for speaker, indices in by_speaker.items():
        rng.shuffle(indices)
        n_dev = max(1, int(len(indices) * dev_frac_per_spk))
        if len(indices) > 1 and n_dev >= len(indices): 
            n_dev = len(indices) - 1
        dev_indices = indices[:n_dev]
        test_indices = indices[n_dev:]
        for idx in dev_indices:
            entry = raw_manifest[idx].copy()
            entry['split'] = 'dev'
            manifest.append(entry)
        for idx in test_indices:
            entry = raw_manifest[idx].copy()
            entry['split'] = 'test'
            manifest.append(entry)
    if len(manifest) == 0:
        raise RuntimeError("No entries found in manifest")
    return manifest

def run_pipeline(
    manifest_path: str, save_dir: str = "./runs/exp", device: str = "cuda", seed: int = 1337, 
    retain_frac: float = 0.975, dev_frac_per_spk: float = 0.75, sae_multiplier: float = 4.0,
    sae_variant: str = "topk", sae_topk: int = 128, lambda_sparsity: float = 0.0, sae_epochs: int = 30, build_U_top: int = 200, beta: float = 0.0, 
    leace: bool = True, leace_ridge: float = 1e-6, noise_sigma: float = 0.02,
    per_speaker_union: bool = False, forget_spks: Optional[List[str]] = None,
    encoder: str = "ecapa", embeddings_path: str = None, 
    wav2vec2_model: str = "facebook/wav2vec2-base", 
    hubert_model: str = "facebook/hubert-base-ls960",
    conformer_model: str = "facebook/wav2vec2-conformer-rel-pos-large"):

    seed_all(seed)
    os.makedirs(save_dir, exist_ok=True)
    t0_all = time.time()

    # 1) Build manifest and splits
    print("[1/12] Building manifest and splits...")
    entries = build_manifest(manifest_path, embeddings_path, dev_frac_per_spk, seed)

    by = {}
    for e in entries:
        by.setdefault(e["speaker"], []).append(e)
    spks = sorted(by.keys())

    test_counts = {s: sum(1 for x in by[s] if x.get('split') == 'test') for s in spks}

    rng = random.Random(seed)
    if forget_spks is None:
        n_total = len(spks)
        n_retain = max(1, int(round(retain_frac * n_total)))
        n_forget = max(1, n_total - n_retain)
        eligible = [s for s in spks if test_counts.get(s, 0) > 0 and len(by[s]) > 1]
        if len(eligible) >= n_forget:
            forget_spks = rng.sample(eligible, k=n_forget)
        else:
            print(f"  Warning: not enough eligible speakers with test samples; sampling from all speakers (eligible {len(eligible)} < required {n_forget})")
            forget_spks = rng.sample(spks, k=n_forget)
    elif isinstance(forget_spks, str):
        forget_spks = [forget_spks]
    else:
        forget_spks = list(forget_spks)

    for spk in forget_spks:
        if spk not in spks:
            raise ValueError(f"Requested forget speaker {spk} not found in dataset")

    print(f"  FORGETTING SPEAKERS ({len(forget_spks)}): {forget_spks}")

    retain_spks = [s for s in spks if s not in forget_spks]

    manifest = []
    for entry in entries:
        entry_copy = entry.copy()
        entry_copy['set'] = 'forget' if entry['speaker'] in forget_spks else 'retain'
        manifest.append(entry_copy)

    dev_manifest = [m for m in manifest if m['split'] == 'dev']
    test_manifest = [m for m in manifest if m['split'] == 'test']
    dev_retain = [m for m in dev_manifest if m['set'] == 'retain']
    dev_forget = [m for m in dev_manifest if m['set'] == 'forget']
    test_retain = [m for m in test_manifest if m['set'] == 'retain']
    test_forget = [m for m in test_manifest if m['set'] == 'forget']
    
    print(f"  retain_spks {len(retain_spks)} forget_spks {len(forget_spks)}")
    print(f"  dev_retain {len(dev_retain)} dev_forget {len(dev_forget)} | test_retain {len(test_retain)} test_forget {len(test_forget)}")

    # 2) Embedding extraction or load
    if embeddings_path:
        print("[2/12] Loading precomputed embeddings and aligning to manifest order...")
        data = np.load(embeddings_path, allow_pickle=True)
        embs = data['embeddings']
        
        seg_ids = None
        if 'segment_ids' in data:
            seg_ids = list(map(str, data['segment_ids']))
        elif 'ids' in data:
            seg_ids = list(map(str, data['ids']))

        if seg_ids is not None:
            id_to_idx = {sid: i for i, sid in enumerate(seg_ids)}
            E_all = np.zeros((len(manifest), embs.shape[1]), dtype=np.float32)
            for i, entry in enumerate(manifest):
                sid = entry.get('segment_id', None)
                if sid is None:
                    raise RuntimeError("Embeddings provided but manifest entries lack 'segment_id' to align")
                sid_s = str(sid)
                if sid_s not in id_to_idx:
                    raise RuntimeError(f"segment_id {sid} not found in embeddings file keys")
                E_all[i] = embs[id_to_idx[sid_s]]
            L_all = [m['speaker'] for m in manifest]
            P_all = [m['path'] for m in manifest]
        else:
            print("  Warning: embeddings file lacks 'segment_ids' - assuming ordering already matches manifest")
            if embs.shape[0] != len(manifest):
                raise RuntimeError("Embeddings count does not match manifest length and no segment_ids provided for alignment")
            E_all = embs.astype(np.float32)
            L_all = [m['speaker'] for m in manifest]
            P_all = [m['path'] for m in manifest]
        print(f"  Loaded {E_all.shape[0]} embeddings, shape {E_all.shape}")
    else:
        print("[2/12] Extracting embeddings...")
        all_manifest = dev_manifest + test_manifest
        E_all, L_all, P_all = extract_embeddings(all_manifest, encoder=encoder, device=device, wav2vec2_model=wav2vec2_model, hubert_model=hubert_model, conformer_model=conformer_model, label_key='speaker')
        manifest = list(dev_manifest) + list(test_manifest)

    labels = L_all
    sets = [f"{m['set']}:{m['split']}" for m in manifest]
    
    # Create masks
    M_ret_dev = np.array([s == 'retain:dev' for s in sets], dtype=bool)
    M_fgt_dev = np.array([s == 'forget:dev' for s in sets], dtype=bool)
    M_ret_test = np.array([s == 'retain:test' for s in sets], dtype=bool)
    M_fgt_test = np.array([s == 'forget:test' for s in sets], dtype=bool)

    # 3) Whitening on retain-dev
    print("[3/12] Fitting whitener on Train-retain (retain-dev)...")
    E_ret_train = E_all[M_ret_dev]
    if E_ret_train.shape[0] < 2: 
        raise RuntimeError("Not enough retain-dev samples to fit whitener")
    mu, W = fit_whitener(E_ret_train)
    Z_all = apply_whitener(E_all, mu, W)

    # 4) SAE training on dev
    print("[4/12] Training SAE on Z_train (dev union: retain+forget)...")
    Z_train = Z_all[M_ret_dev | M_fgt_dev]
    perm = np.random.permutation(Z_train.shape[0])
    cut = int(0.8 * len(perm))
    Z_sae_train, Z_sae_val = Z_train[perm[:cut]], Z_train[perm[cut:]]
    sae = SAE(input_dim=Z_all.shape[1], hidden_mult=sae_multiplier, k=sae_topk, variant=sae_variant)
    sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=sae_epochs, batch_size=64, lr=1e-3, device=device, lambda_sparsity=lambda_sparsity)

    dec_w = sae.dec.weight.detach().cpu().numpy()

    # 5) SAE encodes for feature mining
    print("[5/12] Encoding SAE activations for train-retain and train-forget...")
    Z_ret_dev = Z_all[M_ret_dev]
    Z_fgt_dev = Z_all[M_fgt_dev]
    A_ret, A_fgt = sae_encode_batch(sae, Z_ret_dev, device=device, batch=8192), sae_encode_batch(sae, Z_fgt_dev, device=device, batch=8192)
    
    # Calculate Monosemanticity Score for original embeddings
    print("[5.1/12] Computing Monosemanticity Score for original embeddings...")
    E_all_norm = l2_normalize(E_all)
    ms_original = calculate_monosemanticity_score(E_all_norm, E_all_norm, top_m=20, min_activations=5)
    ms_original_stats = compute_monosemanticity_statistics(ms_original)
    print(f"  Original embeddings MS: mean={ms_original_stats['mean']:.4f}, std={ms_original_stats['std']:.4f}, "
          f"min={ms_original_stats['min']:.4f}, max={ms_original_stats['max']:.4f}")
    
    # Calculate Monosemanticity Score for SAE activations
    print("[5.2/12] Computing Monosemanticity Score for SAE activations...")
    A_all = sae_encode_batch(sae, Z_all, device=device, batch=8192)
    ms_sae = calculate_monosemanticity_score(A_all, E_all_norm, top_m=20, min_activations=5)
    ms_sae_stats = compute_monosemanticity_statistics(ms_sae)
    print(f"  SAE activations MS: mean={ms_sae_stats['mean']:.4f}, std={ms_sae_stats['std']:.4f}, "
          f"min={ms_sae_stats['min']:.4f}, max={ms_sae_stats['max']:.4f}")

    # 6) Feature stats & selection
    print("[6/12] Computing feature statistics and selecting features...")
    stats = feature_stats(A_fgt, A_ret)
    feat_idx_global = select_features_from_stats(stats, top_k=build_U_top, fdr=0.05)
    feat_idx = feat_idx_global
    if per_speaker_union:
        print("  computing per-forget-speaker feature unions... (not implemented, using global)")
        feat_idx = feat_idx_global

    # 7) Build U subspace
    print("[7/12] Building U subspace...")
    dec_w = sae.dec.weight.detach().cpu().numpy()
    dec_pruned = prune_and_normalize_decoder(dec_w, prune_threshold=1e-6)
    U = build_U_from_decoder(dec_w, feat_idx)
    print(f"  U shape: {U.shape}")

    # 8) Compute LEACE in U
    P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
    mu_r = None
    if leace:
        print("[8/12] Computing LEACE projector in U...")
        mask_dev = np.array([s.endswith(':dev') for s in sets], dtype=bool)
        dev_indices = np.where(mask_dev)[0]
        Y_dev = np.array([1 if manifest[i]['speaker'] in forget_spks else 0 for i in dev_indices], dtype=int).reshape(-1, 1)
        
        if Y_dev.shape[0] != Z_all[mask_dev].shape[0]: 
            raise RuntimeError("Mismatch between Y_dev and Z_all[mask_dev] lengths")
        try:
            P_r, mu_r = leace_projector_in_U(Z_all[mask_dev], U, Y_dev, ridge=leace_ridge)
        except Exception as e:
            print(f"  Warning: LEACE computation failed ({e}), falling back to identity projector")
            P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
            mu_r = np.zeros((1, U.shape[1]))

        ZU_dev = Z_all[mask_dev] @ U
    else:
        print("[8/12] Skipping LEACE (identity)")
        ZU_dev = Z_all[mask_dev] @ U if U.shape[1] > 0 else Z_all[mask_dev][:, :0]
        mask_dev = np.array([s.endswith(':dev') for s in sets], dtype=bool)
        dev_indices = np.where(mask_dev)[0]
        Y_dev = np.array([1 if manifest[i]['speaker'] in forget_spks else 0 for i in dev_indices], dtype=int).reshape(-1, 1)

    # 9) Apply SAUCE gating and LEACE edit
    print("[9/12] Applying SAUCE gating (beta) and LEACE edit...")
    Z_edit = Z_all.copy()
    if beta > 0.0:
        print(" applying SAUCE gating with beta", beta)
        Z_edit = apply_sauce_gating(sae, Z_all, feat_idx, beta=beta, device=device, batch=8192)
    if leace:
        Z_edit = apply_subspace_leace(Z_edit, U, P_r, mu_r)
    if noise_sigma and noise_sigma > 0:
        Z_edit = inject_noise_in_subspace(Z_edit, U, sigma=noise_sigma)

    E_edit = invert_whitener(Z_edit, mu, W)
    E_all_norm, E_edit_norm = l2_normalize(E_all), l2_normalize(E_edit)    
    
    # Calculate Monosemanticity Score for final edited embeddings
    print("[9.1/12] Computing Monosemanticity Score for final edited embeddings...")
    ms_final = calculate_monosemanticity_score(E_edit_norm, E_edit_norm, top_m=20, min_activations=5)
    ms_final_stats = compute_monosemanticity_statistics(ms_final)
    print(f"  Final edited embeddings MS: mean={ms_final_stats['mean']:.4f}, std={ms_final_stats['std']:.4f}, "
          f"min={ms_final_stats['min']:.4f}, max={ms_final_stats['max']:.4f}")

    # 10) Per-class EERs
    results = {}
    print("[10/12] Per-class verification EERs...")
    test_trials, test_labels = build_verification_trials(labels, forget_spks, M_ret_test | M_fgt_test)
    scores_test_before, scores_test_after = compute_scores(E_all_norm, test_trials), compute_scores(E_edit_norm, test_trials)
    eer_before, eer_after = eer_from_scores(scores_test_before, test_labels), eer_from_scores(scores_test_after, test_labels) 
    results["EER_before_raw"], results["EER_after_raw"] = eer_before, eer_after
    print(f"  Global EER: before {eer_before:.4f}, after {eer_after:.4f}")

    per_class_test = per_class_eers(E_all_norm, E_edit_norm, labels, mask=M_ret_test | M_fgt_test, rng_seed=seed)
    per_class_ret = {c: vals for c, vals in per_class_test.items() if c in retain_spks}
    per_class_fgt = {c: vals for c, vals in per_class_test.items() if c in forget_spks}

    retain_avg_before, retain_avg_after = avg_eers(per_class_ret)
    forget_avg_before, forget_avg_after = avg_eers(per_class_fgt)
    results.update({
        "retain_EER_before_avg": retain_avg_before,
        "retain_EER_after_avg": retain_avg_after,
        "forget_EER_before_avg": forget_avg_before,
        "forget_EER_after_avg": forget_avg_after,
    })

    # 11) Linear Probing (robust, using linear_probe_U)
    print("[11/12] Linear probing evaluation...")
    split_mask = {"dev": M_ret_dev | M_fgt_dev, "test": M_ret_test | M_fgt_test}
    acc_before, auc_before = safe_linear_probe(Z_all, labels, U, split_mask, forget_classes=forget_spks)
    acc_after, auc_after = safe_linear_probe(Z_edit, labels, U, split_mask, forget_classes=forget_spks)
    results.update({
        "linear_probe_acc_before": acc_before,
        "linear_probe_acc_after": acc_after,
        "linear_probe_auc_before": auc_before,
        "linear_probe_auc_after": auc_after,
    })
    print(f'  Linear Probing accuracies and AUC, before: ({acc_before:.4f}, {auc_before:.4f}), after: ({acc_after:.4f}, {auc_after:.4f})')

    # 12) Classifier (ResNet)
    print("[12/12] Speaker classification accuracy (ResNet) ...")
    all_spks = sorted(list(set(labels)))
    spk2id = {s: i for i, s in enumerate(all_spks)}

    X_cls_train = E_all_norm[M_ret_dev | M_fgt_dev]
    y_cls_train = [spk2id[s] for i, s in enumerate(labels) if (M_ret_dev | M_fgt_dev)[i]]
    X_cls_val = E_all_norm[M_ret_test | M_fgt_test]
    y_cls_val = [spk2id[s] for i, s in enumerate(labels) if (M_ret_test | M_fgt_test)[i]]

    clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(all_spks), device=device, epochs=20, batch_size=512, lr=1e-3)

    # Evaluate retain/forget sets before and after
    E_ret_test = E_all_norm[M_ret_test]
    L_ret_test = [spk2id[d] for d in np.array(labels)[M_ret_test]]
    E_fgt_test = E_all_norm[M_fgt_test]
    L_fgt_test = [spk2id[d] for d in np.array(labels)[M_fgt_test]]
    E_ret_test_edit = E_edit_norm[M_ret_test]
    E_fgt_test_edit = E_edit_norm[M_fgt_test]

    acc_ret_before = evaluate_classifier(clf, E_ret_test, L_ret_test, device=device)
    acc_fgt_before = evaluate_classifier(clf, E_fgt_test, L_fgt_test, device=device)
    acc_ret_after = evaluate_classifier(clf, E_ret_test_edit, L_ret_test, device=device)
    acc_fgt_after = evaluate_classifier(clf, E_fgt_test_edit, L_fgt_test, device=device)

    print(f"  Classifier Retain: before {acc_ret_before*100:.2f}% after {acc_ret_after*100:.2f}%")
    print(f"  Classifier Forget: before {acc_fgt_before*100:.2f}% after {acc_fgt_after*100:.2f}%")

    results.update({
        "resnet_acc_retain_before": acc_ret_before,
        "resnet_acc_retain_after": acc_ret_after,
        "resnet_acc_forget_before": acc_fgt_before,
        "resnet_acc_forget_after": acc_fgt_after,
    })
    
    # Add Monosemanticity Score results
    results.update({
        "ms_original_mean": ms_original_stats["mean"],
        "ms_original_std": ms_original_stats["std"],
        "ms_original_min": ms_original_stats["min"],
        "ms_original_max": ms_original_stats["max"],
        "ms_sae_mean": ms_sae_stats["mean"],
        "ms_sae_std": ms_sae_stats["std"],
        "ms_sae_min": ms_sae_stats["min"],
        "ms_sae_max": ms_sae_stats["max"],
        "ms_final_mean": ms_final_stats["mean"],
        "ms_final_std": ms_final_stats["std"],
        "ms_final_min": ms_final_stats["min"],
        "ms_final_max": ms_final_stats["max"],
    })

    stats_path = os.path.join(save_dir, "verification_eer_summary.json")
    with open(stats_path, "w") as fh: 
        json.dump(results, fh, indent=2)

    print(f"\nPipeline finished in {human_time(time.time() - t0_all)}. Results saved to {stats_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", type=str, default="/data/veer/testing/new/datasets/voxceleb2/manifest.csv")
    parser.add_argument("--save_dir", type=str, default="./out_voxceleb")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--retain_frac", type=float, default=0.7)
    parser.add_argument("--dev_frac_per_spk", type=float, default=0.7)
    parser.add_argument("--sae_multiplier", type=float, default=5.0)
    parser.add_argument("--sae_variant", type=str, default="topk", choices=["topk", "standard"], help="SAE variant: topk or standard")
    parser.add_argument("--sae_topk", type=int, default=128, help="Top-K value for topk variant (ignored for standard)")
    parser.add_argument("--lambda_sparsity", type=float, default=0.0, help="L1 sparsity penalty for standard SAE variant")
    parser.add_argument("--sae_epochs", type=int, default=150)
    parser.add_argument("--build_U_top", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--no_leace", action="store_true")
    parser.add_argument("--leace_ridge", type=float, default=1e-6)
    parser.add_argument("--noise_sigma", type=float, default=0.02)
    parser.add_argument("--forget_spks", nargs="*", default=["id00017"])
    parser.add_argument("--encoder", type=str, default="ecapa")
    parser.add_argument("--conformer_model", type=str, default="facebook/wav2vec2-conformer-rel-pos-large")
    parser.add_argument("--embeddings_path", type=str, default=None)
    args = parser.parse_args()

    run_pipeline(
        manifest_path=args.manifest_path, save_dir=args.save_dir, device=args.device, seed=args.seed,
        retain_frac=args.retain_frac, dev_frac_per_spk=args.dev_frac_per_spk,
        sae_multiplier=args.sae_multiplier, sae_variant=args.sae_variant, sae_topk=args.sae_topk, lambda_sparsity=args.lambda_sparsity, sae_epochs=args.sae_epochs,
        build_U_top=args.build_U_top, beta=args.beta, leace=not args.no_leace, leace_ridge=args.leace_ridge, 
        noise_sigma=args.noise_sigma, forget_spks=args.forget_spks,
        encoder=args.encoder, embeddings_path=args.embeddings_path,
        wav2vec2_model=args.wav2vec2_model if hasattr(args, 'wav2vec2_model') else "facebook/wav2vec2-base",
        hubert_model=args.hubert_model if hasattr(args, 'hubert_model') else "facebook/hubert-base-ls960",
        conformer_model=args.conformer_model
    )