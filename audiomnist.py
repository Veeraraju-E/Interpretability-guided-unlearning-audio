"""
AudioMNIST Digit Unlearning Pipeline — "forget digit x"
"""

from __future__ import annotations
import os, time, random, glob
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression

from common import (
    seed_all, extract_embeddings, fit_whitener, apply_whitener, 
    invert_whitener, TopKSAE, train_sae, sae_encode_batch,
    feature_stats, select_features_from_stats, build_U_from_decoder,
    leace_projector_in_U, apply_subspace_leace, apply_sauce_gating,
    inject_noise_in_subspace, compute_eer, compute_scores,
    train_classifier, evaluate_classifier
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

# -------------------- Evaluation helpers --------------------
def actual_dcf(scores, labels, p_tar=0.01, c_miss=1., c_fa=1., threshold=None):
    """
    Compute actual DCF at a fixed threshold.
    If threshold=None, use the EER operating point threshold.
    """
    fpr, tpr, thresh = roc_curve(labels, scores, pos_label=1)
    
    if threshold is None:
        # pick threshold at closest to EER point (FNR ~ FPR)
        fnr = 1 - tpr
        idx = np.argmin(np.abs(fnr - fpr))
        threshold = thresh[idx]
    decisions = (scores >= threshold).astype(int)    
    n_target = sum(labels)
    n_non = len(labels) - n_target
    miss = sum((decisions == 0) & (np.array(labels) == 1)) / max(1, n_target)
    fa = sum((decisions == 1) & (np.array(labels) == 0)) / max(1, n_non)
    
    return c_miss * miss * p_tar + c_fa * fa * (1 - p_tar)


def calibrate_scores(scores_dev: np.ndarray, labels_dev: np.ndarray, scores_test: np.ndarray) -> np.ndarray:
    if len(scores_dev.shape) == 1: scores_dev = scores_dev.reshape(-1,1)
    if len(scores_test.shape) == 1: scores_test = scores_test.reshape(-1,1)
    clf = LogisticRegression(solver='liblinear')
    clf.fit(scores_dev, labels_dev)
    prob = clf.predict_proba(scores_test)[:,1]
    eps = 1e-9
    logits = np.log(prob + eps) - np.log(1 - prob + eps)
    return logits

# -------------------- utility measures --------------------
def compute_intra_class_cosine_similarities(embeddings: np.ndarray, labels: List[str], mask: np.ndarray, rng_seed: int = 1337):
    """
    Compute intra-class cosine similarities for each class present in the mask.
    For each class, computes cosine similarities between all pairs of samples from that class.
    Returns dict: class_label -> (mean_similarity_before, mean_similarity_after)
    """
    rng = np.random.default_rng(rng_seed)
    idx_in_mask = np.where(mask)[0].tolist()
    classes = sorted({labels[i] for i in idx_in_mask})
    per_class_similarities = {}
    
    for c in classes:
        pos_idx = [i for i in idx_in_mask if labels[i] == c]
        
        if len(pos_idx) < 2:
            continue
            
        # all possible positive pairs (i<j) within the same class
        pos_pairs_all = list(combinations(pos_idx, 2))
        
        # compute cosine similarities for all pairs
        similarities = compute_scores(embeddings, pos_pairs_all)
        
        # store mean similarity
        mean_similarity = float(np.mean(similarities))
        per_class_similarities[c] = mean_similarity
    
    return per_class_similarities

def compute_forget_inter_class_similarities(embeddings: np.ndarray, labels: List[str], mask: np.ndarray, forget_digits: List[str], rng_seed: int = 1337, max_pairs_per_class: int = 100):
    """
    Compute inter-class cosine similarities between forget digits and other digits.
    For each forget digit, computes cosine similarities with samples from other classes.
    Returns dict: (forget_digit, other_digit) -> mean_similarity
    """
    rng = np.random.default_rng(rng_seed)
    idx_in_mask = np.where(mask)[0].tolist()
    all_classes = sorted({labels[i] for i in idx_in_mask})
    other_classes = [c for c in all_classes if c not in forget_digits]
    inter_class_similarities = {}
    
    for forget_digit in forget_digits:
        for other_digit in other_classes:
            idx_forget = [j for j in idx_in_mask if labels[j] == forget_digit]
            idx_other = [j for j in idx_in_mask if labels[j] == other_digit]
            
            if len(idx_forget) < 1 or len(idx_other) < 1:
                continue
            
            # sample pairs between forget and other class (limit to avoid too many computations)
            n_pairs = min(max_pairs_per_class, len(idx_forget) * len(idx_other))
            if n_pairs > 0:
                # randomly sample pairs
                pairs = []
                for _ in range(n_pairs):
                    i1 = rng.choice(idx_forget)
                    i2 = rng.choice(idx_other)
                    pairs.append((i1, i2))
                
                # compute cosine similarities
                similarities = compute_scores(embeddings, pairs)
                mean_similarity = float(np.mean(similarities))
                inter_class_similarities[(forget_digit, other_digit)] = mean_similarity
    
    return inter_class_similarities

def build_verification_trials(labels, forget_digits, split_mask, seed=42):
    """
    Build balanced verification trials for a given split (dev/test).
    Returns: list of (i,j,label) where label=1 for target (both in forget_digits), 0 otherwise.
    """
    rng = random.Random(seed)
    indices = [i for i, m in enumerate(split_mask) if m]
    trials, trial_labels = [], []
    
    # separate indices
    forget_indices = [i for i in indices if labels[i] in forget_digits]
    retain_indices = [i for i in indices if labels[i] not in forget_digits]
    
    # positive trials: both digit in forget_digits
    for i in forget_indices:
        others = [j for j in forget_indices if j != i]
        rng.shuffle(others)
        for j in others:
            trials.append((i, j))
            trial_labels.append(1)
    
    # negative trials: one in forget_digits, one not
    for i in forget_indices:
        others = retain_indices.copy()
        rng.shuffle(others)
        for j in others:
            trials.append((i, j))
            trial_labels.append(0)
    
    return trials, np.array(trial_labels)

from itertools import combinations

def per_class_eers(emb_before: np.ndarray, emb_after: np.ndarray, labels: List[str], mask: np.ndarray, rng_seed: int = 1337):
    """
    For each class present in `mask`, compute EER using:
        - positives = (pairs of samples both from that class, inside mask)
        - negatives = (pairs where one sample is from that class and the other sample is from any other class in mask)
    Returns dict: class_label -> (eer_before, eer_after)
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
        # keep all positive pairs (you can subsample if desired)
        pos_pairs_sel = pos_pairs_all

        # sample negatives to match number of positives
        n_neg = len(pos_pairs_sel)
        pos_side_choices = rng.choice(pos_idx, size=n_neg, replace=True)
        other_side_choices = rng.choice(other_idx, size=n_neg, replace=True)
        neg_pairs_sel = [(int(a), int(b)) for a, b in zip(pos_side_choices, other_side_choices)]

        # assemble combined trials and labels
        trials = pos_pairs_sel + neg_pairs_sel
        trial_labels = np.array([1] * len(pos_pairs_sel) + [0] * len(neg_pairs_sel), dtype=int)

        # compute scores using compute_scores helper
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

def linear_probe_U(Z_all, labels, U, split_mask, forget_digits=["0"]):
    """Train/test a linear probe in reduced coordinates ZU for forget vs not-forget classification."""
    if U.shape[1] == 0:
        return float('nan'), float('nan')
    ZU = Z_all @ U  # reduced representation
    
    # Binarize labels: 1 = forget_digits, 0 = others
    y = np.array([1 if l in forget_digits else 0 for l in labels])
    
    ZU_train = ZU[split_mask['dev']]
    y_train = y[split_mask['dev']]
    ZU_test = ZU[split_mask['test']]
    y_test = y[split_mask['test']]
    
    clf = LogisticRegression(max_iter=500)
    clf.fit(ZU_train, y_train)
    
    y_pred = clf.predict(ZU_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, clf.decision_function(ZU_test))
    return acc, auc


# -------------------- Digit classifier (ResNet-like) --------------------
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
    print("[1/11] Building manifest and splits (AudioMNIST; digit-unlearning)...")
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
    print("[3/11] Fitting whitener on retain-dev (non-forget dev)...")
    E_ret_train = E_all[M_retain_dev]
    if E_ret_train.shape[0] < 2:
        raise RuntimeError("Not enough retain-dev (non-forget) samples to fit whitener")
    mu, W = fit_whitener(E_ret_train)
    Z_all = apply_whitener(E_all, mu, W)

    # 4) SAE training on Z_dev (retain + forget union, unsupervised)
    print("[4/11] Training SAE on Z_dev (all digits: retain+forget union)...")
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
    print("[5/11] Encoding SAE activations...")
    Z_ret_dev = Z_all[M_retain_dev]
    Z_fgt_dev = Z_all[M_forget_dev]
    A_ret = sae_encode_batch(sae, Z_ret_dev, device=device, batch=8192)
    A_fgt = sae_encode_batch(sae, Z_fgt_dev, device=device, batch=8192) if Z_fgt_dev.shape[0] > 0 else np.zeros((0, sae.m))

    # 6) feature stats & selection: features that predict forget-digits (A_fgt positive, A_ret negative)
    print("[6/11] Computing feature statistics and selecting discriminative features (forget digits vs rest)...")
    stats = feature_stats(A_fgt, A_ret)
    feat_idx_global = select_features_from_stats(stats, top_k=build_U_top, fdr=0.05)
    feat_idx = feat_idx_global

    # 7) build U from decoder atoms corresponding to selected features
    print("[7/11] Building subspace U from decoder atoms...")
    dec_w = sae.dec.weight.detach().cpu().numpy()  # [d, m]
    U = build_U_from_decoder(dec_w, feat_idx)
    print(f"  U shape: {U.shape}")

    # 8) compute LEACE projector in U to remove predictivity of forget-digits (Y=1 for forget digits)
    P_r = np.eye(U.shape[1])
    if leace and U.shape[1] > 0:
        print("[8/11] Computing LEACE projector in U (removing component predictive of forget-digits)...")
        mask_dev = is_dev
        Y_dev = y_binary[mask_dev].reshape(-1, 1)
        if mask_dev.sum() == 0:
            raise RuntimeError("No dev samples found for LEACE computation.")
        try:
            P_r, mu_r = leace_projector_in_U(Z_all[mask_dev], U, Y_dev, ridge=leace_ridge)
            np.savez(os.path.join(save_dir, "leace_Pr.npz"), U=U, P_r=P_r)
        except Exception as e:
            print("  LEACE computation failed, falling back to identity. Error:", e)
            P_r = np.eye(U.shape[1])
    else:
        print("[8/11] Skipping LEACE (identity)")
        mu_r = np.zeros((1, U.shape[1])) if U.shape[1] > 0 else None

    # 9) Apply SAUCE gating (optional) then LEACE, optionally add noise
    print("[9/11] Applying SAUCE gating and LEACE edit...")
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
    print("[10/11] Digit classification accuracy (ResNet) before/after edits...")
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

    print("[11/11] Verification evaluation (EER, minDCF, Linear Probe).")
    print(f"  Verification trials: positive pairs within forget digits {forget_digits}, negative pairs across forget/retain")
    results = {}
    # --- Build dev/test trials
    dev_trials, dev_labels = build_verification_trials(labels_digits, forget_digits, is_dev)
    test_trials, test_labels = build_verification_trials(labels_digits, forget_digits, is_test)

    # --- Compute scores BEFORE edit ---
    scores_dev_before = compute_scores(E_all_norm, dev_trials)
    scores_test_before = compute_scores(E_all_norm, test_trials)

    # --- Compute scores AFTER edit ---
    scores_dev_after = compute_scores(E_edit_norm, dev_trials)
    scores_test_after = compute_scores(E_edit_norm, test_trials)

    # --- Calibration (single calibrator from dev_before, applied to both) ---
    scores_test_before_cal = calibrate_scores(scores_dev_before, dev_labels, scores_test_before)
    scores_test_after_cal = calibrate_scores(scores_dev_after, dev_labels, scores_test_after)

    # --- Compute EER (global forget-vs-rest) ---
    eer_before = compute_eer(scores_test_before, test_labels)
    eer_after = compute_eer(scores_test_after, test_labels)
    eer_before_cal = compute_eer(scores_test_before_cal, test_labels)
    eer_after_cal = compute_eer(scores_test_after_cal, test_labels)

    results["EER_before_raw"] = eer_before
    results["EER_after_raw"] = eer_after
    results["EER_before_cal"] = eer_before_cal
    results["EER_after_cal"] = eer_after_cal

    # --- Compute minDCF (p_tar=0.01 and 0.001) ---
    for p_tar in [0.01, 0.001]:
        dcf_before = actual_dcf(scores_test_before, test_labels, p_tar=p_tar)
        dcf_after = actual_dcf(scores_test_after, test_labels, p_tar=p_tar)
        results[f"actualDCF_before_{p_tar}"] = dcf_before
        results[f"actualDCF_after_{p_tar}"] = dcf_after

    # --- Linear probe in reduced coordinates U (existing) ---
    split_mask = {"dev": is_dev, "test": is_test}
    acc_before, auc_before = linear_probe_U(Z_all, labels_digits, U, split_mask, forget_digits=forget_digits)
    acc_after, auc_after = linear_probe_U(Z_edit, labels_digits, U, split_mask, forget_digits=forget_digits)

    # --- Per-class EERs on TEST partition ---
    per_class_test = per_class_eers(E_all_norm, E_edit_norm, labels_digits, mask=(M_retain_test | M_forget_test), rng_seed=seed)
    
    # --- Intra-class cosine similarities on TEST partition ---
    print("[12/12] Computing intra-class cosine similarities...")
    intra_sim_before = compute_intra_class_cosine_similarities(E_all_norm, labels_digits, mask=(M_retain_test | M_forget_test), rng_seed=seed)
    intra_sim_after = compute_intra_class_cosine_similarities(E_edit_norm, labels_digits, mask=(M_retain_test | M_forget_test), rng_seed=seed)

    # --- Inter-class cosine similarities on TEST partition ---
    print("[13/13] Computing inter-class cosine similarities...")
    inter_sim_before = compute_forget_inter_class_similarities(E_all_norm, labels_digits, mask=(M_retain_test | M_forget_test), forget_digits=forget_digits, rng_seed=seed)
    inter_sim_after = compute_forget_inter_class_similarities(E_edit_norm, labels_digits, mask=(M_retain_test | M_forget_test), forget_digits=forget_digits, rng_seed=seed)

    # Partition per-class dict into retain vs forget classes (for averages)
    all_digits_sorted = sorted(set(labels_digits))
    forget_classes = forget_digits
    retain_classes = [d for d in all_digits_sorted if d not in forget_digits]

    per_class_ret = {c: vals for c, vals in per_class_test.items() if c in retain_classes}
    per_class_fgt = {c: vals for c, vals in per_class_test.items() if c in forget_classes}

    def avg_eers(per_class_dict):
        if len(per_class_dict) == 0:
            return (float('nan'), float('nan'))
        be = [v[0] for v in per_class_dict.values() if not np.isnan(v[0])]
        ae = [v[1] for v in per_class_dict.values() if not np.isnan(v[1])]
        return (float(np.mean(be)) if len(be) > 0 else float('nan'),
                float(np.mean(ae)) if len(ae) > 0 else float('nan'))

    retain_avg_before, retain_avg_after = avg_eers(per_class_ret)
    forget_avg_before, forget_avg_after = avg_eers(per_class_fgt)

    results["retain_EER_before_avg"] = retain_avg_before
    results["retain_EER_after_avg"] = retain_avg_after
    results["forget_EER_before_avg"] = forget_avg_before
    results["forget_EER_after_avg"] = forget_avg_after

    # --- Print summary ---
    print("\n--- Verification Results ---")
    for k, v in results.items():
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


    # --- Store per-class & summary in final results to be saved ---
    results_bundle = {
        "classification": {
            "acc_ret_before": acc_ret_before,
            "acc_fgt_before": acc_fgt_before,
            "acc_ret_after": acc_ret_after,
            "acc_fgt_after": acc_fgt_after
        },
        "verification": {
            "linear_probe_acc_before": acc_before,
            "linear_probe_acc_after": acc_after,
            "linear_probe_auc_before": auc_before,
            "linear_probe_auc_after": auc_after,
            "per_class_test": per_class_test,
            "retain_avg_eer_before": retain_avg_before,
            "retain_avg_eer_after": retain_avg_after,
            "forget_avg_eer_before": forget_avg_before,
            "forget_avg_eer_after": forget_avg_after
        },
        "intra_class_similarities": {
            "before": intra_sim_before,
            "after": intra_sim_after
        },
        "inter_class_similarities": {
            "before": inter_sim_before,
            "after": inter_sim_after
        }
    }

    np.savez(os.path.join(save_dir, "artifacts_digit_unlearn.npz"),
            mu=mu, W=W, U=U, P_r=P_r, feat_idx=np.array(feat_idx),
            E_before=E_all_norm, E_after=E_edit_norm,
            labels=np.array(labels_digits, dtype=object),
            y_binary=y_binary, sets=np.array(sets, dtype=object),
            results=results_bundle)
    torch.save(sae.state_dict(), os.path.join(save_dir, "sae_state.pt"))

    print("Done. Total time:", time.strftime('%H:%M:%S', time.gmtime(time.time() - t0_all)))


# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./audiomnist_data", help="AudioMNIST root containing recordings/")
    p.add_argument("--save_dir", default="./outputs_audiomnist", help="where to store artifacts")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dev_frac", type=float, default=0.9)
    p.add_argument("--sae_hidden_mult", type=float, default=4.0)
    p.add_argument("--sae_k", type=int, default=128)
    p.add_argument("--sae_epochs", type=int, default=50)
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
