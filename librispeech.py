from __future__ import annotations
import os, time, json, random, glob
from pathlib import Path
from typing import List, Dict
import argparse
import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression

from common import (
    seed_all, human_time, extract_embeddings, fit_whitener, apply_whitener, 
    invert_whitener, l2_normalize, TopKSAE, train_sae, sae_encode_batch,
    prune_and_normalize_decoder, feature_stats, select_features_from_stats, build_U_from_decoder,
    leace_projector_in_U, apply_subspace_leace, apply_sauce_gating,
    inject_noise_in_subspace, compute_eer, compute_scores,
    train_classifier, evaluate_classifier
)

# -------------------- Manifest --------------------
def build_manifest(data_root: str) -> List[Dict]:
    entries = []
    for sub in ["dev-clean", "dev-other"]:
        pattern = os.path.join(data_root, sub, "*", "*", "*.flac")
        for p in glob.glob(pattern):
            spk = Path(p).parts[-3]  # correctly grab speaker ID from path like dev-clean/84/121123/84-121123-0000.flac
            entries.append({"path": p, "speaker": spk, "subset": sub})
    if len(entries) == 0: raise RuntimeError(f"No entries under {data_root}/dev-clean or dev-other")
    return entries

# -------------------- Linear Probing --------------------
def linear_probe_U(Z_all, labels, U, split_mask, forget_speakers):
    """Train/test a linear probe in reduced coordinates ZU for forget vs not-forget speaker classification."""
    if U.shape[1] == 0:
        return float('nan'), float('nan')
    
    # Binarize labels: 1 = forget_speakers, 0 = others
    y = np.array([1 if l in forget_speakers else 0 for l in labels])
    
    ZU_train = Z_all[split_mask['dev']]
    y_train = y[split_mask['dev']]
    ZU_test = Z_all[split_mask['test']]
    y_test = y[split_mask['test']]
    
    clf = LogisticRegression(max_iter=500)
    clf.fit(ZU_train, y_train)    
    y_pred = clf.predict(ZU_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, clf.decision_function(ZU_test))
    return acc, auc

# -------------------- Orchestration: full run --------------------
def run_pipeline(data_root: str, save_dir: str = "./runs/exp", device: str = "cuda", seed: int = 1337,
                 retain_frac: float = 0.975, dev_frac_per_spk: float = 0.75,
                 sae_hidden_mult: float = 4.0, sae_topk: int = 128, sae_epochs: int = 30,
                 build_U_top: int = 200, beta: float = 0.0, leace: bool = True,
                 leace_ridge: float = 1e-6, noise_sigma: float = 0.02, per_speaker_union: bool = False, 
                 forget_spks: List[str] = None, encoder: str = "ecapa", 
                 wav2vec2_model: str = "facebook/wav2vec2-base", hubert_model: str = "facebook/hubert-base-ls960"):
    seed_all(seed)
    os.makedirs(save_dir, exist_ok=True)
    t0_all = time.time()

    # 1) manifest + splits
    print("[1/11] Building manifest and splits...")
    entries = build_manifest(data_root)

    # group utterances by speaker
    by = {}
    for e in entries:
        by.setdefault(e["speaker"], []).append(e["path"])

    spks = sorted(by.keys())
    if forget_spks is None:
        forget_spks = [random.choice(spks)]
    elif isinstance(forget_spks, str):
        forget_spks = [forget_spks]
    
    # Validate that all requested forget speakers exist
    for spk in forget_spks:
        if spk not in spks:
            raise ValueError(f"Requested forget speaker {spk} not found in dataset")
    
    print(f"  FORGETTING SPEAKERS: {forget_spks}")

    retain_spks = [s for s in spks if s not in forget_spks]

    manifest = []
    rng = random.Random(seed)
    for s, paths in by.items():
        rng.shuffle(paths)
        n_dev = max(1, int(len(paths) * dev_frac_per_spk))
        dev_paths = paths[:n_dev]
        test_paths = paths[n_dev:]
        setlbl = 'forget' if s in forget_spks else 'retain'
        for p in dev_paths:
            manifest.append({"path": p, "speaker": s, "set": setlbl, "split": "dev"})
        for p in test_paths:
            manifest.append({"path": p, "speaker": s, "set": setlbl, "split": "test"})

    # create lists for each split+set
    dev_manifest = [m for m in manifest if m['split']=='dev']
    test_manifest = [m for m in manifest if m['split']=='test']
    dev_retain = [m for m in dev_manifest if m['set']=='retain']
    dev_forget = [m for m in dev_manifest if m['set']=='forget']
    test_retain = [m for m in test_manifest if m['set']=='retain']
    test_forget = [m for m in test_manifest if m['set']=='forget']
    print(f"  retain_spks {len(retain_spks)} forget_spks {len(forget_spks)}")
    print(f"  dev_retain {len(dev_retain)} dev_forget {len(dev_forget)} | test_retain {len(test_retain)} test_forget {len(test_forget)}")

    # 2) ECAPA embeddings
    all_manifest = dev_manifest + test_manifest
    E_all, L_all, P_all = extract_embeddings(all_manifest, encoder=encoder, device=device, wav2vec2_model=wav2vec2_model, hubert_model=hubert_model, label_key='speaker')
    labels = L_all
    # print(labels)
    sets = [f"{m['set']}:{m['split']}" for m in (dev_manifest + test_manifest)]

    # masks
    M_ret_dev = np.array([s == 'retain:dev' for s in sets], dtype=bool)
    M_fgt_dev = np.array([s == 'forget:dev' for s in sets], dtype=bool)
    M_ret_test = np.array([s == 'retain:test' for s in sets], dtype=bool)
    M_fgt_test = np.array([s == 'forget:test' for s in sets], dtype=bool)

    # 3) Whitening on D_r
    print("[3/11] Fitting whitener on Train-retain...")
    E_ret_train = E_all[M_ret_dev]
    if E_ret_train.shape[0] < 2:
        raise RuntimeError("Not enough retain-dev samples to fit whitener")
    mu, W = fit_whitener(E_ret_train)
    Z_all = apply_whitener(E_all, mu, W)

    # 4) SAE training on Z_train (Train-retain)
    print("[4/11] Training SAE on Z_train (train-retain)...")
    Z_train = Z_all[M_ret_dev | M_fgt_dev]
    # split 80/20 for SAE train/val
    perm = np.random.permutation(Z_train.shape[0])
    cut = int(0.8 * len(perm))
    Z_sae_train = Z_train[perm[:cut]]
    Z_sae_val = Z_train[perm[cut:]]
    sae = TopKSAE(input_dim=Z_all.shape[1], hidden_mult=sae_hidden_mult, k=sae_topk)
    sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=sae_epochs, batch_size=4096, lr=1e-3, device=device)

    # 5) SAE encodes for feature mining (train-retain & train-forget)
    print("[5/11] Encoding SAE activations for train-retain and train-forget...")
    Z_ret_dev = Z_all[M_ret_dev]
    Z_fgt_dev = Z_all[M_fgt_dev]
    A_ret = sae_encode_batch(sae, Z_ret_dev, device=device, batch=8192)
    A_fgt = sae_encode_batch(sae, Z_fgt_dev, device=device, batch=8192)
    # print(A_ret, A_fgt)

    # 6) feature stats & selection (supports per-speaker union)
    print("[6/11] Computing feature statistics and selecting features...")
    stats = feature_stats(A_fgt, A_ret)
    feat_idx_global = select_features_from_stats(stats, top_k=build_U_top, fdr=0.05)
    feat_idx = feat_idx_global
    if per_speaker_union:   # TODO
        print("  computing per-forget-speaker feature unions...")
        feat_idx = feat_idx_global

    # 7) prune & normalize decoder & build U
    print("[7/11] Building U subspace...")
    dec_w = sae.dec.weight.detach().cpu().numpy()   # [d, m]
    dec_pruned = prune_and_normalize_decoder(dec_w, prune_threshold=1e-6)  # returns [d, r]
    # Build U from decoder columns corresponding to feat_idx
    U = build_U_from_decoder(dec_w, feat_idx)   # NOTE: ablate the use case of dec_pruned and full dec_w 
    print(f"  U shape: {U.shape}")

    # 8) compute LEACE in U if requested
    P_r = np.eye(U.shape[1])
    if leace:
        print("[8/11] Computing LEACE projector in U...")
        # build Y labels: forget vs rest for all Z_all rows
        mask_dev = np.array([s.endswith(':dev') for s in sets])
        # Create binary labels: 1 for forget speakers, 0 for retain speakers
        Y_dev = np.array([1 if any(spk == m['speaker'] for spk in forget_spks) else 0 
                         for m in all_manifest if mask_dev[all_manifest.index(m)]], dtype=int).reshape(-1,1)
        P_r, mu_r = leace_projector_in_U(Z_all[mask_dev], U, Y_dev, ridge=leace_ridge)
        np.savez(os.path.join(save_dir, "leace_Pr.npz"), U=U, P_r=P_r)
    else:
        print("[8/11] Skipping LEACE (identity)")

    # 9) Apply SAUCE gating (optional) then LEACE and optional noise
    print("[9/11] Applying SAUCE gating (beta) and LEACE edit...")
    Z_edit = Z_all.copy()
    if beta > 0.0:
        print("  applying SAUCE gating with beta", beta)
        Z_edit = apply_sauce_gating(sae, Z_all, feat_idx, beta=beta, device=device, batch=8192)
    # apply LEACE map in U
    if leace:
        Z_edit = apply_subspace_leace(Z_edit, U, P_r, mu_r)
    if noise_sigma and noise_sigma > 0:
        Z_edit = inject_noise_in_subspace(Z_edit, U, sigma=noise_sigma)

    # map back to embedding space and l2 normalize
    E_edit = invert_whitener(Z_edit, mu, W)
    E_all_norm, E_edit_norm = l2_normalize(E_all), l2_normalize(E_edit)

    # -------------------- 11) Per-class EERs --------------------
    print("[10/11] Per-class verification EERs (each class vs all others) ...")
    from itertools import combinations
    all_spks = sorted(set(labels))
    spk2id = {d:i for i,d in enumerate(all_spks)}

    # Build test lists
    E_ret_test = E_all_norm[M_ret_test]
    L_ret_test = [spk2id[d] for d in np.array(labels)[M_ret_test]]
    E_fgt_test = E_all_norm[M_fgt_test]
    L_fgt_test = [spk2id[d] for d in np.array(labels)[M_fgt_test]]
    E_ret_test_edit = E_edit_norm[M_ret_test]
    E_fgt_test_edit = E_edit_norm[M_fgt_test]

    def per_class_eers(emb_before: np.ndarray, emb_after: np.ndarray, labels: List[str], mask: np.ndarray, rng_seed: int = 1337):
        """
        For each class present in `mask`, compute EER using:
            - positives = (pairs of samples both from that class, inside mask)
            - negatives = (pairs where one sample is from that class and the other sample is from any other class in mask)
        Returns dict: class_id -> (eer_before, eer_after)
        """
        rng = np.random.default_rng(rng_seed)
        idx_in_mask = np.where(mask)[0].tolist()
        classes = sorted({labels[i] for i in idx_in_mask})
        per_class = {}
        for c in classes:
            pos_idx = [i for i in idx_in_mask if labels[i] == c]
            other_idx = [i for i in idx_in_mask if labels[i] != c]

            if len(pos_idx) < 2 or len(other_idx) < 1:
                continue

            # all possible positive pairs (i<j)
            pos_pairs_all = list(combinations(pos_idx, 2))
            n_sample = n_pos_all = len(pos_pairs_all)
            if n_pos_all <= n_sample:
                pos_pairs_sel = pos_pairs_all
            else:
                sel_idxs = rng.choice(n_pos_all, size=n_sample, replace=False)
                pos_pairs_sel = [pos_pairs_all[k] for k in sel_idxs]

            n_neg = len(pos_pairs_sel)
            pos_side_choices = rng.choice(pos_idx, size=n_neg, replace=True)
            other_side_choices = rng.choice(other_idx, size=n_neg, replace=True)
            neg_pairs_sel = [(int(a), int(b)) for a, b in zip(pos_side_choices, other_side_choices)]

            # assemble combined trials and labels
            trials = pos_pairs_sel + neg_pairs_sel
            trial_labels = np.array([1] * len(pos_pairs_sel) + [0] * len(neg_pairs_sel), dtype=int)

            # compute scores compute_scores helper
            scores_before = compute_scores(emb_before, trials)
            scores_after = compute_scores(emb_after, trials)

            # compute EER (compute_eer returns (eer, threshold))
            try:
                eer_b, _ = compute_eer(scores_before, trial_labels)
            except Exception:
                eer_b, _ = (float('nan'), float('nan'))
            try:
                eer_a, _ = compute_eer(scores_after, trial_labels)
            except Exception:
                eer_a, _ = (float('nan'), float('nan'))
            per_class[c] = (float(eer_b), float(eer_a))

        return per_class

    # Compute per-class EERs on the TEST partitions
    per_class_test = per_class_eers(E_all_norm, E_edit_norm, labels, mask=M_ret_test | M_fgt_test, rng_seed=seed)
    per_class_ret = {c: vals for c, vals in per_class_test.items() if c in retain_spks}
    per_class_fgt = {c: vals for c, vals in per_class_test.items() if c in forget_spks}

    def avg_eers(per_class_dict):
        if len(per_class_dict) == 0:
            return (float('nan'), float('nan'))
        be = [v[0] for v in per_class_dict.values() if not np.isnan(v[0])]
        ae = [v[1] for v in per_class_dict.values() if not np.isnan(v[1])]
        return float(np.mean(be)) if len(be) > 0 else float('nan'), float(np.mean(ae)) if len(ae) > 0 else float('nan')

    retain_avg_before, retain_avg_after = avg_eers(per_class_ret)
    forget_avg_before, forget_avg_after = avg_eers(per_class_fgt)

    results = {}
    results["retain_EER_before_avg"] = retain_avg_before
    results["retain_EER_after_avg"] = retain_avg_after
    results["forget_EER_before_avg"] = forget_avg_before
    results["forget_EER_after_avg"] = forget_avg_after

    # -------------------- Linear Probing Evaluation --------------------
    print("[11/11] Linear probing evaluation in U subspace (forget vs retain speakers)...")
    
    # Create split masks for linear probing
    split_mask = {"dev": M_ret_dev | M_fgt_dev, "test": M_ret_test | M_fgt_test}
    
    # Linear probe in reduced coordinates U (before edits)
    acc_before, auc_before = linear_probe_U(Z_all, labels, U, split_mask, forget_spks)
    acc_after, auc_after = linear_probe_U(Z_edit, labels, U, split_mask, forget_spks)
    
    print(f'  Linear Probing accuracies and AUC, before: ({acc_before:.4f}, {auc_before:.4f}), after: ({acc_after:.4f}, {auc_after:.4f})')
    
    # Store results
    results["linear_probe_acc_before"] = acc_before
    results["linear_probe_acc_after"] = acc_after
    results["linear_probe_auc_before"] = auc_before
    results["linear_probe_auc_after"] = auc_after

    # Print detailed and summary outputs
    print("\nPer-class EERs (sample):")
    for i, (c, (eb, ea)) in enumerate(sorted(per_class_test.items())): 
        if c in forget_spks:
            print("  [FORGET SPEAKER]")
        print(f"  class {c}: EER_before={eb:.4f}, EER_after={ea:.4f}")

    print("\n--- Average EER summary ---")
    print(f"  Retain avg EER: before {retain_avg_before:.4f}, after {retain_avg_after:.4f}")
    print(f"  Forget avg EER: before {forget_avg_before:.4f}, after {forget_avg_after:.4f}")

    # Save these numbers to disk along with other artifacts
    stats_path = os.path.join(save_dir, "verification_eer_summary.json")
    with open(stats_path, "w") as fh:
        json.dump({
            "per_class_count_test": len(per_class_test),
            "retain_avg": {"before": retain_avg_before, "after": retain_avg_after},
            "forget_avg": {"before": forget_avg_before, "after": forget_avg_after},
            "linear_probe": {
                "acc_before": acc_before, "acc_after": acc_after,
                "auc_before": auc_before, "auc_after": auc_after
            }
        }, fh, indent=2)

    # -------------------- 12) Speaker classifier (ResNet) training (unchanged) --------------------
    print("[12/12] Speaker classification accuracy (ResNet) before/after edits...")
    all_spks = sorted(list(set(labels)))
    spk2id = {s:i for i,s in enumerate(all_spks)}
    
    # Train classifier on retain+forget dev (before edits)
    X_cls_train = E_all_norm[M_ret_dev | M_fgt_dev]
    y_cls_train = [spk2id[s] for i, s in enumerate(labels) if (M_ret_dev | M_fgt_dev)[i]]
    X_cls_val = E_all_norm[M_ret_test | M_fgt_test]
    y_cls_val = [spk2id[s] for i, s in enumerate(labels) if (M_ret_test | M_fgt_test)[i]]
    # print(len(X_cls_train), len(y_cls_train), len(X_cls_val), len(y_cls_val))
    clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(all_spks), device=device, epochs=20, batch_size=512, lr=1e-3)

    # Evaluate BEFORE edits
    acc_ret_before = evaluate_classifier(clf, E_ret_test, L_ret_test, device=device)
    acc_fgt_before = evaluate_classifier(clf, E_fgt_test, L_fgt_test, device=device)

    # Evaluate AFTER edits
    acc_ret_after = evaluate_classifier(clf, E_ret_test_edit, L_ret_test, device=device)
    acc_fgt_after = evaluate_classifier(clf, E_fgt_test_edit, L_fgt_test, device=device)

    print(f"  Classifier Retain: before {acc_ret_before*100:.2f}% after {acc_ret_after*100:.2f}%")
    print(f"  Classifier Forget: before {acc_fgt_before*100:.2f}% after {acc_fgt_after*100:.2f}%")

    # Save artifacts (unchanged)
    print("Saving artifacts...")
    np.savez(os.path.join(save_dir, "artifacts_full.npz"),
             mu=mu, W=W, U=U, P_r=P_r, feat_idx=np.array(feat_idx),
             E_before=E_all_norm, E_after=E_edit_norm,
             labels=np.array(labels, dtype=object), sets=np.array(sets, dtype=object))
    torch.save(sae.state_dict(), os.path.join(save_dir, "sae_state.pt"))
    print("Done. Total time:", human_time(time.time()-t0_all))

# -------------------- CLI --------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./datasets/librispeech_data_dev/LibriSpeech", help="LibriSpeech root containing dev-clean/dev-other")
    p.add_argument("--save_dir", default="./outputs", help="where to store artifacts")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--retain_frac", type=float, default=0.7)
    p.add_argument("--dev_frac_per_spk", type=float, default=0.7)
    p.add_argument("--sae_hidden_mult", type=float, default=10.0)
    p.add_argument("--sae_topk", type=int, default=128)
    p.add_argument("--sae_epochs", type=int, default=100)
    p.add_argument("--build_U_top", type=int, default=200)
    p.add_argument("--beta", type=float, default=0.0)   # for SAUCE, default=don't do
    p.add_argument("--leace", action="store_true")
    p.add_argument("--leace_ridge", type=float, default=1e-6)
    p.add_argument("--noise_sigma", type=float, default=0.02)
    p.add_argument("--forget_spks", nargs='+', default=['174'], help="Speaker IDs to forget (e.g., '174' '175' '176')")
    p.add_argument("--encoder", type=str, default="ecapa", help="Encoder to use - 'ecapa', 'wav2vec2', 'hubert'")
    args = p.parse_args()
    run_pipeline(
        data_root=args.data_root, save_dir=args.save_dir, 
        device=args.device, seed=args.seed,
        retain_frac=args.retain_frac, dev_frac_per_spk=args.dev_frac_per_spk,
        sae_hidden_mult=args.sae_hidden_mult,  sae_topk=args.sae_topk,  sae_epochs=args.sae_epochs,
        build_U_top=args.build_U_top, 
        beta=args.beta, 
        leace=args.leace, leace_ridge=args.leace_ridge, noise_sigma=args.noise_sigma, 
        encoder=args.encoder, forget_spks=args.forget_spks
    )