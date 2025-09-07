from __future__ import annotations
import os, time, json, random, glob
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import numpy as np

from common import (
    seed_all, human_time, extract_embeddings, fit_whitener, apply_whitener, 
    invert_whitener, l2_normalize, TopKSAE, train_sae, sae_encode_batch,
    prune_and_normalize_decoder, feature_stats, select_features_from_stats, build_U_from_decoder,
    leace_projector_in_U, apply_subspace_leace, apply_sauce_gating,
    inject_noise_in_subspace, eer_from_scores, compute_scores,
    train_classifier, evaluate_classifier, linear_probe_U, 
    build_verification_trials, per_class_eers, avg_eers, safe_linear_probe
)

# -------------------- Manifest builder for AudioMNIST --------------------
def build_manifest_from_audiomnist(data_root: str, embeddings_path: str = None, dev_frac_per_digit: float = 0.75, seed: int = 1337) -> List[Dict]:
    """
    Build manifest entries for AudioMNIST recordings folder.
    Expects filenames like: '0_george_0.wav'
    Returns list of dicts: {"path": p, "speaker": spk, "digit": digit, "split": "dev"/"test"}
    """
    if embeddings_path:
        # Load precomputed embeddings case
        data = np.load(embeddings_path, allow_pickle=True)
        labels = data['labels'].tolist() if 'labels' in data else (data['y'].tolist() if 'y' in data else None)
        ids = data['segment_ids'].tolist() if 'segment_ids' in data else (data['ids'].tolist() if 'ids' in data else None)
        if labels is None:
            raise RuntimeError("embeddings .npz must contain 'labels' or 'y' array when embeddings_path is used")
        # For AudioMNIST, assume labels are digits and create dummy entries
        raw_entries = [{'path': f'dummy_{i}', 'digit': labels[i], 'speaker': f'speaker_{i}', 'segment_id': ids[i] if ids is not None else None} for i in range(len(labels))]
    else:
        # Build from audio files
        wavs = glob.glob(os.path.join(data_root, "*.wav"))
        raw_entries = []
        for p in sorted(wavs):
            fname = Path(p).stem
            parts = fname.split("_")
            if len(parts) < 2:
                continue
            digit = parts[0]
            spk = parts[1]
            raw_entries.append({"path": p, "speaker": spk, "digit": digit, "segment_id": fname})
        
    if len(raw_entries) == 0:
        raise RuntimeError(f"No AudioMNIST entries found")

    # Group by digit and create train/test splits (similar to voxceleb's speaker grouping)
    by_digit = {}
    for i, entry in enumerate(raw_entries):
        digit = entry['digit']
        by_digit.setdefault(digit, []).append(i)

    rng = random.Random(seed)
    manifest = []
    for digit, indices in by_digit.items():
        rng.shuffle(indices)
        n_dev = max(1, int(len(indices) * dev_frac_per_digit))
        if len(indices) > 1 and n_dev >= len(indices): 
            n_dev = len(indices) - 1
        dev_indices = indices[:n_dev]
        test_indices = indices[n_dev:]
        
        for idx in dev_indices:
            entry = raw_entries[idx].copy()
            entry['split'] = 'dev'
            manifest.append(entry)
        for idx in test_indices:
            entry = raw_entries[idx].copy()
            entry['split'] = 'test'
            manifest.append(entry)
    
    if len(manifest) == 0:
        raise RuntimeError("No entries found in manifest")
    return manifest

# -------------------- Run full pipeline --------------------
def run_pipeline(
    data_root: str, save_dir: str = "./runs/exp", device: str = "cuda", seed: int = 1337,
    retain_frac: float = 0.975, dev_frac_per_digit: float = 0.75,
    sae_multiplier: float = 4.0, sae_topk: int = 128, sae_epochs: int = 30,
    build_U_top: int = 200, beta: float = 0.0, leace: bool = True,
    leace_ridge: float = 1e-6, noise_sigma: float = 0.02,
    per_speaker_union: bool = False, forget_digits: Optional[List[str]] = None,
    encoder: str = "ecapa", embeddings_path: str = None,
    wav2vec2_model: str = "facebook/wav2vec2-base",
    hubert_model: str = "facebook/hubert-base-ls960",
    conformer_model: str = "facebook/wav2vec2-conformer-rel-pos-large"
):

    seed_all(seed)
    os.makedirs(save_dir, exist_ok=True)
    t0_all = time.time()

    print(f"\n===== Running pipeline (encoder={encoder}) =====")

    # 1) manifest + splits
    print("[1/12] Building manifest and splits...")
    entries = build_manifest_from_audiomnist(data_root, embeddings_path, dev_frac_per_digit, seed)

    # group utterances by digit (use full entries ordering)
    by_digit = {}
    for e in entries:
        by_digit.setdefault(e["digit"], []).append(e)
    digits = sorted(by_digit.keys())
    test_counts = {d: sum(1 for x in by_digit[d] if x.get('split') == 'test') for d in digits}
    rng = random.Random(seed)
    
    if forget_digits is None:
        n_total = len(digits)
        n_retain = max(1, int(round(retain_frac * n_total)))
        n_forget = max(1, n_total - n_retain)
        eligible = [d for d in digits if test_counts.get(d, 0) > 0 and len(by_digit[d]) > 1]
        if len(eligible) >= n_forget:
            forget_digits = rng.sample(eligible, k=n_forget)
        else:
            # fallback to sampling from all digits but warn (this only happens on tiny datasets)
            print(f"  Warning: not enough eligible digits with test samples; sampling from all digits (eligible {len(eligible)} < required {n_forget})")
            forget_digits = rng.sample(digits, k=n_forget)
    elif isinstance(forget_digits, str):
        forget_digits = [forget_digits]
    else:
        forget_digits = list(forget_digits)

    # Validate that all requested forget digits exist
    for digit in forget_digits:
        if digit not in digits:
            raise ValueError(f"Requested forget digit {digit} not found in dataset")

    print(f"  FORGETTING DIGITS ({len(forget_digits)}): {forget_digits}")

    retain_digits = [d for d in digits if d not in forget_digits]

    # add 'set' labels to manifest
    manifest = []
    for entry in entries:
        entry_copy = entry.copy()
        entry_copy['set'] = 'forget' if entry['digit'] in forget_digits else 'retain'
        manifest.append(entry_copy)

    dev_manifest = [m for m in manifest if m['split'] == 'dev']
    test_manifest = [m for m in manifest if m['split'] == 'test']
    dev_retain = [m for m in dev_manifest if m['set'] == 'retain']
    dev_forget = [m for m in dev_manifest if m['set'] == 'forget']
    test_retain = [m for m in test_manifest if m['set'] == 'retain']
    test_forget = [m for m in test_manifest if m['set'] == 'forget']
    print(f"  retain_digits {len(retain_digits)} forget_digits {len(forget_digits)}")
    print(f"  dev_retain {len(dev_retain)} dev_forget {len(dev_forget)} | test_retain {len(test_retain)} test_forget {len(test_forget)}")

    # 2) Embedding extraction or load
    if embeddings_path:
        print("[2/12] Loading precomputed embeddings and aligning to manifest order...")
        data = np.load(embeddings_path, allow_pickle=True)
        embs = data['embeddings']
        # attempt to reorder embeddings to match manifest using segment ids
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
            L_all = [m['digit'] for m in manifest]
            P_all = [m['path'] for m in manifest]
        else:
            # no segment ids: assume order matches manifest (best-effort)
            print("  Warning: embeddings file lacks 'segment_ids' - assuming ordering already matches manifest")
            if embs.shape[0] != len(manifest):
                raise RuntimeError("Embeddings count does not match manifest length and no segment_ids provided for alignment")
            E_all = embs.astype(np.float32)
            L_all = [m['digit'] for m in manifest]
            P_all = [m['path'] for m in manifest]
        print(f"  Loaded {E_all.shape[0]} embeddings, shape {E_all.shape}")
    else:
        print("[2/12] Extracting embeddings...")
        all_manifest = dev_manifest + test_manifest
        E_all, L_all, P_all = extract_embeddings(all_manifest, encoder=encoder, device=device, wav2vec2_model=wav2vec2_model, hubert_model=hubert_model, conformer_model=conformer_model, label_key='digit')
        manifest = list(dev_manifest) + list(test_manifest)

    labels = L_all

    # Create set labels
    sets = [f"{m['set']}:{m['split']}" for m in manifest]
    
    # all masks
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

    # 4) SAE training on dev (retain + forget)
    print("[4/12] Training SAE on Z_train (dev union: retain+forget)...")
    Z_train = Z_all[M_ret_dev | M_fgt_dev]
    perm = np.random.permutation(Z_train.shape[0])
    cut = int(0.8 * len(perm))
    Z_sae_train, Z_sae_val = Z_train[perm[:cut]], Z_train[perm[cut:]]
    sae = TopKSAE(input_dim=Z_all.shape[1], hidden_mult=sae_multiplier, k=sae_topk)
    sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=sae_epochs, batch_size=64, lr=1e-3, device=device)

    # 5) SAE encodes for feature mining
    print("[5/12] Encoding SAE activations for train-retain and train-forget...")
    Z_ret_dev = Z_all[M_ret_dev]
    Z_fgt_dev = Z_all[M_fgt_dev]
    A_ret = sae_encode_batch(sae, Z_ret_dev, device=device, batch=8192)
    A_fgt = sae_encode_batch(sae, Z_fgt_dev, device=device, batch=8192)

    # 6) feature stats & selection
    print("[6/12] Computing feature statistics and selecting features...")
    stats = feature_stats(A_fgt, A_ret)
    feat_idx_global = select_features_from_stats(stats, top_k=build_U_top, fdr=0.05)
    feat_idx = feat_idx_global
    if per_speaker_union:
        print("  computing per-forget-digit feature unions... (not implemented, using global)")
        feat_idx = feat_idx_global

    # 7) prune & normalize decoder & build U
    print("[7/12] Building U subspace...")
    dec_w = sae.dec.weight.detach().cpu().numpy()   # [d, m]
    dec_pruned = prune_and_normalize_decoder(dec_w, prune_threshold=1e-6)  # returns [d, r]
    U = build_U_from_decoder(dec_w, feat_idx)   # TODO: check by passing dec_pruned later
    print(f"  U shape: {U.shape}")

    # 8) compute LEACE in U
    P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
    mu_r = None
    if leace:
        print("[8/12] Computing LEACE projector in U...")
        mask_dev = np.array([s.endswith(':dev') for s in sets], dtype=bool)
        dev_indices = np.where(mask_dev)[0]
        Y_dev = np.array([1 if manifest[i]['digit'] in forget_digits else 0 for i in dev_indices], dtype=int).reshape(-1, 1)
        # safety check: shapes must align
        if Y_dev.shape[0] != Z_all[mask_dev].shape[0]: 
            raise RuntimeError("Mismatch between Y_dev and Z_all[mask_dev] lengths")
        try:
            P_r, mu_r = leace_projector_in_U(Z_all[mask_dev], U, Y_dev, ridge=leace_ridge)
        except Exception as e:
            print(f"  Warning: LEACE computation failed ({e}), falling back to identity projector")
            P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
            mu_r = np.zeros((1, U.shape[1]))
    else:
        print("[8/12] Skipping LEACE (identity)")

    # 9) Apply SAUCE gating (optional) then LEACE and optional noise
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

    results = {}

    # Note: Extended validation checks and diagnostic plots are now handled by run_audiomnist_experiments.py

    # ================= Verification Steps =================
    # 10) Per-class EERs
    print("[10/12] Per-class verification EERs...")
    test_trials, test_labels = build_verification_trials(labels, forget_digits, M_ret_test | M_fgt_test)
    scores_test_before, scores_test_after = compute_scores(E_all_norm, test_trials), compute_scores(E_edit_norm, test_trials)
    eer_before, eer_after = eer_from_scores(scores_test_before, test_labels), eer_from_scores(scores_test_after, test_labels) 
    results["EER_before_raw"], results["EER_after_raw"] = eer_before, eer_after
    print(f"  Global EER: before {eer_before:.4f}, after {eer_after:.4f}")

    per_class_test = per_class_eers(E_all_norm, E_edit_norm, labels, mask=M_ret_test | M_fgt_test, rng_seed=seed)
    per_class_ret = {c: vals for c, vals in per_class_test.items() if c in retain_digits}
    per_class_fgt = {c: vals for c, vals in per_class_test.items() if c in forget_digits}

    retain_avg_before, retain_avg_after = avg_eers(per_class_ret)
    forget_avg_before, forget_avg_after = avg_eers(per_class_fgt)
    results.update({
        "retain_EER_before_avg": retain_avg_before,
        "retain_EER_after_avg": retain_avg_after,
        "forget_EER_before_avg": forget_avg_before,
        "forget_EER_after_avg": forget_avg_after,
    })

    # 11) Linear Probing
    print("[11/12] Linear probing evaluation...")
    split_mask = {"dev": M_ret_dev | M_fgt_dev, "test": M_ret_test | M_fgt_test}
    acc_before, auc_before = safe_linear_probe(Z_all, labels, U, split_mask, forget_classes=forget_digits)
    acc_after, auc_after = safe_linear_probe(Z_edit, labels, U, split_mask, forget_classes=forget_digits)
    results.update({
        "linear_probe_acc_before": acc_before,
        "linear_probe_acc_after": acc_after,
        "linear_probe_auc_before": auc_before,
        "linear_probe_auc_after": auc_after,
    })

    # Print summary
    print("\n--- Average EER summary ---")
    print(f"  Retain avg EER: before {retain_avg_before:.4f}, after {retain_avg_after:.4f}")
    print(f"  Forget avg EER: before {forget_avg_before:.4f}, after {forget_avg_after:.4f}")

    stats_path = os.path.join(save_dir, "verification_eer_summary.json")
    with open(stats_path, "w") as fh: json.dump(results, fh, indent=2)

    # 12) Classifier
    print("[12/12] Digit classification accuracy (ResNet) ...")
    all_digits = sorted(list(set(labels)))
    digit2id = {d: i for i, d in enumerate(all_digits)}

    X_cls_train = E_all_norm[M_ret_dev | M_fgt_dev]
    y_cls_train = [digit2id[d] for i, d in enumerate(labels) if (M_ret_dev | M_fgt_dev)[i]]
    X_cls_val = E_all_norm[M_ret_test | M_fgt_test]
    y_cls_val = [digit2id[d] for i, d in enumerate(labels) if (M_ret_test | M_fgt_test)[i]]

    clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(all_digits), device=device, epochs=20, batch_size=512, lr=1e-3)

    # Evaluate retain/forget sets before and after
    E_ret_test = E_all_norm[M_ret_test]
    L_ret_test = [digit2id[d] for d in np.array(labels)[M_ret_test]]
    E_fgt_test = E_all_norm[M_fgt_test]
    L_fgt_test = [digit2id[d] for d in np.array(labels)[M_fgt_test]]
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

    stats_path = os.path.join(save_dir, "verification_eer_summary.json")
    with open(stats_path, "w") as fh: json.dump(results, fh, indent=2)

    # --- Full validation and interpretability suite ---
    from validation_checks import run_validation_checks, create_all_plots, build_feature_cards, polysemanticity_checks, causal_tests, activation_patching, top_neurons_analysis
    validation_dir = os.path.join(save_dir, 'validation')
    os.makedirs(validation_dir, exist_ok=True)
    mask_dev = np.array([s.endswith(':dev') for s in sets], dtype=bool)
    mask_test = M_ret_test | M_fgt_test
    # Validation checks
    run_validation_checks(
        save_dir=validation_dir,
        Z_all=Z_all, Z_edit=Z_edit,
        E_all_norm=E_all_norm, E_edit_norm=E_edit_norm,
        ZU_mean=(Z_all[mask_dev] @ U).mean(0, keepdims=True) if U.shape[1] > 0 else None,
        U=U, P_r=P_r, mu_r=mu_r, dec_w=dec_w, sae=sae,
        feat_idx=feat_idx, sets=sets, labels=labels,
        mask_dev=mask_dev, mask_test=mask_test,
        y_dev_bin=Y_dev if 'Y_dev' in locals() else None, leace_ridge=leace_ridge,
        mu=mu, W=W, device=device
    )
    # All plots
    Z_before = Z_all[mask_dev]
    Z_after = Z_edit[mask_dev]
    ZU_before = Z_before @ U if U.shape[1] > 0 else Z_before[:, :0]
    ZU_after = Z_after @ U if U.shape[1] > 0 else Z_after[:, :0]
    create_all_plots(
        Z_before, Z_after, ZU_before, ZU_after,
        labels=[labels[i] for i in np.where(mask_dev)[0]],
        forget_classes=forget_digits,
        decoder_weights=dec_w, feat_idx=feat_idx,
        A_retain=A_ret, A_forget=A_fgt,
        save_dir=validation_dir, random_state=seed
    )
    # Interpretability suite
    interp_dir = os.path.join(validation_dir, 'interpretability')
    os.makedirs(interp_dir, exist_ok=True)
    label_binary = np.array([1 if c in forget_digits else 0 for c in labels])
    A_all = sae_encode_batch(sae, Z_all, device=device, batch=8192)
    try:
        build_feature_cards(A_all, label_binary, dec_w, feat_idx, None, interp_dir)
    except Exception as e:
        print('feature card build failed', e)
    try:
        polysemanticity_checks(A_all, np.array(labels), feat_idx, interp_dir)
    except Exception as e:
        print('polysemanticity failed', e)
    try:
        test_trials, test_labels = build_verification_trials(labels, forget_digits, mask_test)
        causal_tests(sae, Z_all, mu, W, test_trials, test_labels, feat_idx, interp_dir, device, max_feats=100)
    except Exception as e:
        print('causal tests failed', e)
    try:
        forget_indices = np.where(M_fgt_test)[0].tolist()
        retain_indices = np.where(M_ret_test)[0].tolist()
        test_trials, test_labels = build_verification_trials(labels, forget_digits, mask_test)
        activation_patching(sae, Z_all, mu, W, test_trials, test_labels, forget_indices, retain_indices, interp_dir, device, topm=8)
    except Exception as e:
        print('patching failed', e)
    try:
        top_neurons_analysis(A_all, labels, manifest, interp_dir)
    except Exception as e:
        print('top neurons analysis failed', e)

    print(f"\nPipeline finished in {human_time(time.time() - t0_all)}. Results saved to {stats_path}")


# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/data/veer/testing/new/datasets/audiomnist_data", help="AudioMNIST root containing recordings/")
    parser.add_argument("--save_dir", default="./outputs_audiomnist", help="where to store artifacts")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--retain_frac", type=float, default=0.7)
    parser.add_argument("--dev_frac_per_digit", type=float, default=0.7)
    parser.add_argument("--sae_multiplier", type=float, default=10.0)
    parser.add_argument("--sae_topk", type=int, default=128)
    parser.add_argument("--sae_epochs", type=int, default=30)
    parser.add_argument("--build_U_top", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--no_leace", action="store_true")
    parser.add_argument("--leace_ridge", type=float, default=1e-6)
    parser.add_argument("--noise_sigma", type=float, default=0.02)
    parser.add_argument("--forget_digits", nargs="*", default=["0"])
    parser.add_argument("--encoder", type=str, default="ecapa")
    parser.add_argument("--conformer_model", type=str, default="facebook/wav2vec2-conformer-rel-pos-large")
    parser.add_argument("--embeddings_path", type=str, default=None)
    args = parser.parse_args()

    run_pipeline(
        data_root=args.data_root, save_dir=args.save_dir, device=args.device, seed=args.seed,
        retain_frac=args.retain_frac, dev_frac_per_digit=args.dev_frac_per_digit,
        sae_multiplier=args.sae_multiplier, sae_topk=args.sae_topk, sae_epochs=args.sae_epochs,
        build_U_top=args.build_U_top, beta=args.beta, leace=not args.no_leace, leace_ridge=args.leace_ridge, noise_sigma=args.noise_sigma,
        forget_digits=args.forget_digits,
        encoder=args.encoder,
        embeddings_path=args.embeddings_path,
        wav2vec2_model=getattr(args, 'wav2vec2_model', "facebook/wav2vec2-base"),
        hubert_model=getattr(args, 'hubert_model', "facebook/hubert-base-ls960"),
        conformer_model=args.conformer_model
    )