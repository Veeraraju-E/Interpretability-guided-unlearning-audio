from __future__ import annotations
import os, time, random, glob
from pathlib import Path
from typing import List, Dict
import numpy as np

from common import (
    seed_all, extract_embeddings, fit_whitener, apply_whitener, 
    invert_whitener, TopKSAE, train_sae, sae_encode_batch,
    feature_stats, select_features_from_stats, build_U_from_decoder,
    leace_projector_in_U, apply_subspace_leace, apply_sauce_gating,
    inject_noise_in_subspace, eer_from_scores, compute_scores,
    train_classifier, evaluate_classifier, linear_probe_U, 
    build_verification_trials, per_class_eers, compute_intra_class_cosine_similarities,
    compute_forget_inter_class_similarities, avg_eers
)

# -------------------- Manifest builder for AudioMNIST --------------------
def build_manifest_from_audiomnist(data_root: str) -> List[Dict]:
    """
    Build manifest entries for AudioMNIST recordings folder.
    Expects filenames like: '0_george_0.wav'
    Returns list of dicts: {"path": p, "speaker": spk, "digit": digit}
    """
    wavs = glob.glob(os.path.join(data_root, "*.wav"))
    entries = []
    for p in sorted(wavs):
        fname = Path(p).stem
        parts = fname.split("_")
        if len(parts) < 2:
            continue
        spk = parts[1]
        digit = parts[0]
        entries.append({"path": p, "speaker": spk, "digit": digit})
    if len(entries) == 0:
        raise RuntimeError(f"No AudioMNIST wav files found under {data_root}")
    return entries

# -------------------- Run full pipeline --------------------
def run_pipeline(data_root: str, save_dir: str = "./runs/exp", device: str = "cuda", seed: int = 1337,
                 dev_frac: float = 0.75,
                 sae_hidden_mult: float = 4.0, sae_k: int = 128, sae_epochs: int = 30,
                 build_U_top: int = 200, beta: float = 0.0, leace: bool = True,
                 leace_ridge: float = 1e-6, noise_sigma: float = 0.02,
                 forget_digits: List[str] = ["0"], encoder: str = "ecapa",
                 wav2vec2_model: str = "facebook/wav2vec2-base",
                 hubert_model: str = "facebook/hubert-base-ls960"):
    """
    Full pipeline that unlearns the specified forget_digits (list of strings, e.g., ["0", "1"]).
    Returns a dict of summary metrics and saves artifacts to save_dir.
    """
    seed_all(seed)
    os.makedirs(save_dir, exist_ok=True)
    t0_all = time.time()

    print(f"\n===== Running pipeline (encoder={encoder}) =====")
    print(f"Forgetting digits: {forget_digits}")

    # 1) manifest + splits
    print("[1/12] Building manifest and splits (AudioMNIST; digit-unlearning)...")
    entries = build_manifest_from_audiomnist(data_root)
    # group by digit
    by_digit = {}
    for e in entries:
        by_digit.setdefault(e["digit"], []).append(e)

    digits = sorted(by_digit.keys())
    for digit in forget_digits:
        if digit not in digits:
            raise ValueError(f"Requested forget digit {digit} not found in dataset")

    manifest = []
    rng = random.Random(seed)
    for d, utts in by_digit.items():
        rng.shuffle(utts)
        n_dev = max(1, int(len(utts) * dev_frac))
        dev_utts = utts[:n_dev]
        test_utts = utts[n_dev:]
        setlbl = "forget" if d in forget_digits else "retain"
        for u in dev_utts:
            manifest.append({**u, "set": setlbl, "split": "dev"})
        for u in test_utts:
            manifest.append({**u, "set": setlbl, "split": "test"})

    dev_manifest = [m for m in manifest if m["split"]=="dev"]
    test_manifest = [m for m in manifest if m["split"]=="test"]
    dev_retain = [m for m in dev_manifest if m["set"]=="retain"]
    dev_forget = [m for m in dev_manifest if m["set"]=="forget"]
    test_retain = [m for m in test_manifest if m["set"]=="retain"]
    test_forget = [m for m in test_manifest if m["set"]=="forget"]

    print(f"  dev_retain {len(dev_retain)} dev_forget {len(dev_forget)} | test_retain {len(test_retain)} test_forget {len(test_forget)}")
    print(f"  Forgetting digits: {forget_digits}")
    print(f"  Retaining digits: {[d for d in digits if d not in forget_digits]}")

    # 2) extract embeddings based on encoder
    E_all, L_all, P_all = extract_embeddings(manifest, encoder=encoder, device=device, wav2vec2_model=wav2vec2_model, hubert_model=hubert_model, label_key='digit')
    E_all = np.asarray(E_all)
    labels_digits = list(L_all)  # list of digit strings per utterance
    sets = [f"{m['split']}" for m in manifest]  # 'dev' or 'test'
    is_dev = np.array([s == 'dev' for s in sets], dtype=bool)
    is_test = np.array([s == 'test' for s in sets], dtype=bool)

    # create binary target: 1 if digit in forget_digits else 0
    y_binary = np.array([1 if d in forget_digits else 0 for d in labels_digits], dtype=int)

    # masks
    M_forget_dev = is_dev & (y_binary == 1)
    M_forget_test = is_test & (y_binary == 1)
    M_retain_dev = is_dev & (y_binary == 0)
    M_retain_test = is_test & (y_binary == 0)

    print(f"  forget-dev: {M_forget_dev.sum()}, forget-test: {M_forget_test.sum()}, retain-dev: {M_retain_dev.sum()}, retain-test: {M_retain_test.sum()}")

    # 3) Whitener on retain-dev (we whiten on non-forget dev samples)
    print("[3/12] Fitting whitener on retain-dev (non-forget dev)...")
    E_ret_train = E_all[M_retain_dev]
    if E_ret_train.shape[0] < 2:
        raise RuntimeError("Not enough retain-dev (non-forget) samples to fit whitener")
    mu, W = fit_whitener(E_ret_train)
    Z_all = apply_whitener(E_all, mu, W)

    # 4) SAE training on Z_dev (retain + forget union, unsupervised)
    print("[4/12] Training SAE on Z_dev (all digits: retain+forget union)...")
    Z_dev = Z_all[is_dev]   # retain + forget
    if Z_dev.shape[0] < 2:
        raise RuntimeError("Not enough dev samples to train SAE")

    perm = np.random.permutation(Z_dev.shape[0])
    cut = max(1, int(0.8 * len(perm)))
    Z_sae_train = Z_dev[perm[:cut]]
    Z_sae_val = Z_dev[perm[cut:]] if cut < len(perm) else Z_dev[perm[:1]]
    sae = TopKSAE(input_dim=Z_all.shape[1], hidden_mult=sae_hidden_mult, k=sae_k)
    sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=sae_epochs, batch_size=min(4096, max(8, Z_sae_train.shape[0])), lr=1e-3, device=device)

    # 5) SAE activations for forget-dev vs retain-dev
    print("[5/12] Encoding SAE activations...")
    Z_ret_dev = Z_all[M_retain_dev]
    Z_fgt_dev = Z_all[M_forget_dev]
    A_ret = sae_encode_batch(sae, Z_ret_dev, device=device, batch=8192)
    A_fgt = sae_encode_batch(sae, Z_fgt_dev, device=device, batch=8192) if Z_fgt_dev.shape[0] > 0 else np.zeros((0, sae.m))

    # 6) feature stats & selection: features that predict forget-digits (A_fgt positive, A_ret negative)
    print("[6/12] Computing feature statistics and selecting discriminative features (forget digits vs rest)...")
    stats = feature_stats(A_fgt, A_ret)
    feat_idx_global = select_features_from_stats(stats, top_k=build_U_top, fdr=0.05)
    feat_idx = feat_idx_global

    # 7) build U from decoder atoms corresponding to selected features
    print("[7/12] Building subspace U from decoder atoms...")
    dec_w = sae.dec.weight.detach().cpu().numpy()  # [d, m]
    U = build_U_from_decoder(dec_w, feat_idx)
    print(f"  U shape: {U.shape}")

    # 8) compute LEACE projector in U to remove predictivity of forget-digits (Y=1 for forget digits)
    P_r = np.eye(U.shape[1])
    mu_r = np.zeros((1, U.shape[1])) if U.shape[1] > 0 else None
    if leace and U.shape[1] > 0:
        print("[8/12] Computing LEACE projector in U (removing component predictive of forget-digits)...")
        mask_dev = is_dev
        Y_dev = y_binary[mask_dev].reshape(-1, 1)
        if mask_dev.sum() == 0:
            raise RuntimeError("No dev samples found for LEACE computation.")
        try:
            P_r, mu_r = leace_projector_in_U(Z_all[mask_dev], U, Y_dev, ridge=leace_ridge)
        except Exception as e:
            print("  LEACE computation failed, falling back to identity. Error:", e)
            P_r = np.eye(U.shape[1])
            mu_r = np.zeros((1, U.shape[1])) if U.shape[1] > 0 else None
    else:
        print("[8/12] Skipping LEACE (identity)")

    # 9) Apply SAUCE gating (optional) then LEACE, optionally add noise
    print("[9/12] Applying SAUCE gating and LEACE edit...")
    Z_edit = Z_all.copy()
    if beta and beta > 0.0 and len(feat_idx) > 0:
        Z_edit = apply_sauce_gating(sae, Z_all, feat_idx, beta=beta, device=device, batch=8192)
    if leace and U.shape[1] > 0:
        Z_edit = apply_subspace_leace(Z_edit, U, P_r, mu_r)
    if noise_sigma and noise_sigma > 0 and U.shape[1] > 0:
        Z_edit = inject_noise_in_subspace(Z_edit, U, sigma=noise_sigma)

    E_edit = invert_whitener(Z_edit, mu, W)
    E_all_norm, E_edit_norm = E_all / (np.linalg.norm(E_all, axis=1, keepdims=True) + 1e-12), E_edit / (np.linalg.norm(E_edit, axis=1, keepdims=True) + 1e-12)

    # 10) Evaluations
    print("[10/12] Digit classification accuracy (ResNet) before/after edits...")
    all_digits = sorted(set(labels_digits))
    digit2id = {d:i for i,d in enumerate(all_digits)}

    # Train classifier on retain+forget dev (before edits)
    X_cls_train = E_all_norm[is_dev]
    y_cls_train = [digit2id[d] for d in np.array(labels_digits)[is_dev]]
    X_cls_val = E_all_norm[is_test]
    y_cls_val = [digit2id[d] for d in np.array(labels_digits)[is_test]]

    clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(all_digits), device=device, epochs=20, batch_size=512, lr=1e-3)

    # Build test sets (retain vs forget)
    E_ret_test = E_all_norm[M_retain_test]
    L_ret_test = [digit2id[d] for d in np.array(labels_digits)[M_retain_test]]
    E_fgt_test = E_all_norm[M_forget_test]
    L_fgt_test = [digit2id[d] for d in np.array(labels_digits)[M_forget_test]]
    E_ret_test_edit = E_edit_norm[M_retain_test]
    E_fgt_test_edit = E_edit_norm[M_forget_test]

    # Evaluate BEFORE edits
    acc_ret_before = evaluate_classifier(clf, E_ret_test, L_ret_test, device=device)
    acc_fgt_before = evaluate_classifier(clf, E_fgt_test, L_fgt_test, device=device)

    # Evaluate AFTER edits (same classifier)
    acc_ret_after = evaluate_classifier(clf, E_ret_test_edit, L_ret_test, device=device)
    acc_fgt_after = evaluate_classifier(clf, E_fgt_test_edit, L_fgt_test, device=device)
    print(f"  Classifier Retain: before {acc_ret_before*100:.2f}% after {acc_ret_after*100:.2f}%")
    print(f"  Classifier Forget: before {acc_fgt_before*100:.2f}% after {acc_fgt_after*100:.2f}%")

    print("[11/12] Verification evaluation (EER, minDCF, Linear Probe).")
    print(f"  Verification trials: positive pairs within forget digits {forget_digits}, negative pairs across forget/retain")
    results = {}

    # --- Build dev/test trials
    test_trials, test_labels = build_verification_trials(labels_digits, forget_digits, is_test)

    # --- Compute scores BEFORE edit ---
    scores_test_before = compute_scores(E_all_norm, test_trials)

    # --- Compute scores AFTER edit ---
    scores_test_after = compute_scores(E_edit_norm, test_trials)

    # --- Compute EER (global forget-vs-rest) ---
    eer_before = eer_from_scores(scores_test_before, test_labels)
    eer_after = eer_from_scores(scores_test_after, test_labels)

    results["EER_before_raw"] = eer_before
    results["EER_after_raw"] = eer_after

    # --- Linear probe in reduced coordinates U (existing) ---
    split_mask = {"dev": is_dev, "test": is_test}
    acc_before, auc_before = linear_probe_U(Z_all, labels_digits, U, split_mask, forget_classes=forget_digits)
    acc_after, auc_after = linear_probe_U(Z_edit, labels_digits, U, split_mask, forget_classes=forget_digits)
    print(f'Linear Probing accuracies and auc, before : {acc_before, auc_before}, after: {acc_after, auc_after}')

    # --- Per-class EERs on TEST partition ---
    per_class_test = per_class_eers(E_all_norm, E_edit_norm, labels_digits, mask=(M_retain_test | M_forget_test), rng_seed=seed)
    
    # --- Intra-class cosine similarities on TEST partition ---
    print("[12/12] Computing intra-class cosine similarities...")
    intra_sim_before = compute_intra_class_cosine_similarities(E_all_norm, labels_digits, mask=(M_retain_test | M_forget_test), rng_seed=seed)
    intra_sim_after = compute_intra_class_cosine_similarities(E_edit_norm, labels_digits, mask=(M_retain_test | M_forget_test), rng_seed=seed)

    # --- Inter-class cosine similarities on TEST partition ---
    inter_sim_before = compute_forget_inter_class_similarities(E_all_norm, labels_digits, mask=(M_retain_test | M_forget_test), forget_classes=forget_digits, rng_seed=seed)
    inter_sim_after = compute_forget_inter_class_similarities(E_edit_norm, labels_digits, mask=(M_retain_test | M_forget_test), forget_classes=forget_digits, rng_seed=seed)

    # Partition per-class dict into retain vs forget classes (for averages)
    all_digits_sorted = sorted(set(labels_digits))
    forget_classes = forget_digits
    retain_classes = [d for d in all_digits_sorted if d not in forget_digits]

    per_class_ret = {c: vals for c, vals in per_class_test.items() if c in retain_classes}
    per_class_fgt = {c: vals for c, vals in per_class_test.items() if c in forget_classes}

    retain_avg_before, retain_avg_after = avg_eers(per_class_ret)
    forget_avg_before, forget_avg_after = avg_eers(per_class_fgt)
    results['retain_EER_before_avg'] = retain_avg_before
    results['retain_EER_after_avg'] = retain_avg_after
    results['forget_EER_before_avg'] = forget_avg_before
    results['forget_EER_after_avg'] = forget_avg_after

    print("\n--- Verification Results ---")
    for k,v in results.items():
        print(f"{k}: {v}")

    print("\nPer-class EERs (sample):")
    for i, (c, (eb, ea)) in enumerate(sorted(per_class_test.items())):
        if c in forget_digits:
            print("  [FORGET DIGIT]")
        print(f"  class {c}: EER_before={eb:.4f}, EER_after={ea:.4f}")
    
    print("\nPer-class Intra-class Cosine Similarities:")
    for c in sorted(intra_sim_before.keys()):
        sim_before = intra_sim_before.get(c, float('nan'))
        sim_after = intra_sim_after.get(c, float('nan'))
        if c in forget_digits:
            print("  [FORGET DIGIT]")
        print(f"  class {c}: similarity_before={sim_before:.4f}, similarity_after={sim_after:.4f}")

    print("\nPer-class Inter-class Cosine Similarities:")
    for (f, o), sim_before in sorted(inter_sim_before.items()):
        sim_after = inter_sim_after.get((f, o), float('nan'))
        print(f"  class {f} vs {o}: similarity_before={sim_before:.4f}, similarity_after={sim_after:.4f}")

    print("\n--- Average EER summary ---")
    print(f"  Retain avg EER: before {retain_avg_before:.4f}, after {retain_avg_after:.4f}")
    print(f"  Forget avg EER: before {forget_avg_before:.4f}, after {forget_avg_after:.4f}")
    print(f"  Forgot digits: {forget_digits}")
    print(f"  Retained digits: {[d for d in digits if d not in forget_digits]}")
    print("Done. Total time:", time.strftime('%H:%M:%S', time.gmtime(time.time() - t0_all)))


# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="../datasets/audiomnist_data", help="AudioMNIST root containing recordings/")
    p.add_argument("--save_dir", default="./outputs_audiomnist", help="where to store artifacts")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dev_frac", type=float, default=0.9)
    p.add_argument("--sae_hidden_mult", type=float, default=10.0)
    p.add_argument("--sae_k", type=int, default=128)
    p.add_argument("--sae_epochs", type=int, default=100)
    p.add_argument("--build_U_top", type=int, default=200)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--leace", action="store_true", help="Compute LEACE projector in U (default: False if not provided)")
    p.add_argument("--leace_ridge", type=float, default=1e-6)
    p.add_argument("--noise_sigma", type=float, default=0.02)
    p.add_argument("--forget_digits", nargs='+', default=["0"], help="Digits to forget, e.g. '0' '1' '2'")
    p.add_argument("--encoder", type=str, default="ecapa", help="Encoder to use - 'ecapa', 'wav2vec2', 'hubert'")
    args = p.parse_args()

    # Run
    run_pipeline(
        data_root=args.data_root, save_dir=args.save_dir, device=args.device, seed=args.seed,
        dev_frac=args.dev_frac,
        sae_hidden_mult=args.sae_hidden_mult, sae_k=args.sae_k, sae_epochs=args.sae_epochs,
        build_U_top=args.build_U_top, beta=args.beta, leace=args.leace, leace_ridge=args.leace_ridge, noise_sigma=args.noise_sigma,
        encoder=args.encoder, forget_digits=args.forget_digits
    )
