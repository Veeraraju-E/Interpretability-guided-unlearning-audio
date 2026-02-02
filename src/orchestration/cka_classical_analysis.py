"""
CKA analysis: SAE-selected latents vs classical audio features (retain vs forget).

Leakage safeguards: feat_idx is selected on dev; CKA is computed on test only
(speaker means over test utterances). Manifest ordering is aligned with Z_all/A_all.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.core.utils.cka import linear_cka, rbf_cka, bootstrap_cka, perm_test_cka
from src.core.utils.classical_audio_features import (
    FEATURE_DIM, IDX_MFCC, IDX_F0, IDX_FORMANTS, IDX_PROSDY, extract_all_utterances,
)

N_FOLDS_FORGET = 5
N_BOOTSTRAP = 500
N_PERM = 500


def _speaker_means_test(
    A_sel: np.ndarray,
    Y: np.ndarray,
    labels: List,
    set_arr: np.ndarray,
    forget_label: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute speaker-level means on test split only."""
    mask_test = np.array([s.endswith("test") for s in set_arr])
    lab_arr = np.array([str(x) for x in labels])
    lab_test, A_test, Y_test = lab_arr[mask_test], A_sel[mask_test], Y[mask_test]
    
    # Filter valid rows (no NaN in classical features)
    valid = ~np.any(np.isnan(Y_test), axis=1)
    A_test, Y_test, lab_test = A_test[valid], Y_test[valid], lab_test[valid]

    # Retain speakers
    ret_speakers = [x for x in np.unique(lab_test) if x != forget_label]
    X_ret = np.array([A_test[lab_test == s].mean(axis=0) for s in ret_speakers])
    Y_ret = np.array([Y_test[lab_test == s].mean(axis=0) for s in ret_speakers])

    # Forget speaker (fold into pseudo-speakers if enough samples)
    fgt_mask = lab_test == forget_label
    n_fgt = fgt_mask.sum()
    if n_fgt < N_FOLDS_FORGET:
        X_fgt = A_test[fgt_mask].mean(axis=0, keepdims=True)
        Y_fgt = Y_test[fgt_mask].mean(axis=0, keepdims=True)
    else:
        idx = np.where(fgt_mask)[0]
        fold_size = max(1, len(idx) // N_FOLDS_FORGET)
        X_fgt = np.array([A_test[idx[i:i + fold_size]].mean(axis=0) for i in range(0, len(idx), fold_size)])[:N_FOLDS_FORGET]
        Y_fgt = np.array([Y_test[idx[i:i + fold_size]].mean(axis=0) for i in range(0, len(idx), fold_size)])[:N_FOLDS_FORGET]
    
    return X_ret, Y_ret, X_fgt, Y_fgt


def _zscore(X: np.ndarray) -> np.ndarray:
    m, s = X.mean(axis=0, keepdims=True), X.std(axis=0, keepdims=True)
    s[s < 1e-12] = 1.0
    return (X - m) / s


def _compute_cka_stats(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator) -> Dict:
    """Compute linear/RBF CKA with bootstrap CI and permutation p-value."""
    if X.shape[0] < 2:
        return {"linear": None, "rbf": None, "ci": None, "p": None}
    cka_lin = linear_cka(X, Y)
    cka_rbf = rbf_cka(X, Y)
    _, ci_lo, ci_hi = bootstrap_cka(X, Y, n_bootstrap=N_BOOTSTRAP, rng=rng, linear=True)
    _, p = perm_test_cka(X, Y, n_perm=N_PERM, rng=rng, linear=True)
    return {"linear": float(cka_lin), "rbf": float(cka_rbf), "ci": [float(ci_lo), float(ci_hi)], "p": float(p)}


def run_cka_classical_analysis(
    run_save_dir: str,
    cfg: Dict,
    A_all: np.ndarray,
    feat_idx: List[int],
    labels: List,
    sets: np.ndarray,
    manifest: List,
    Z_all: np.ndarray,
    device: str = "cuda",
    seed: int = 1337,
) -> Dict:
    """Run CKA analysis between SAE-selected features and classical audio features."""
    forget_label = str(np.load(os.path.join(run_save_dir, "interpretability_data.npz"), allow_pickle=True)["forget_spks"].item(0))
    A_sel = A_all[:, feat_idx]
    set_arr = np.asarray(sets)

    # Load or extract classical features
    cache_path = os.path.join(run_save_dir, "classical_features_per_utt.npz")
    if os.path.isfile(cache_path):
        Y = np.load(cache_path)["features"]
    else:
        Y = extract_all_utterances(manifest, cache_path=cache_path)
    Y = Y[:, :FEATURE_DIM] if Y.shape[1] > FEATURE_DIM else Y
    
    # Check for valid features
    valid_rows = ~np.all(np.isnan(Y), axis=1)
    if not np.any(valid_rows):
        return {"error": "No valid classical features (all dummy paths or extraction failed)."}

    X_ret, Y_ret, X_fgt, Y_fgt = _speaker_means_test(A_sel, Y, labels, set_arr, forget_label)
    rng = np.random.default_rng(seed)

    # Compute CKA for retain and forget
    results = {
        "forget_label": forget_label,
        "n_retain_speakers": X_ret.shape[0],
        "n_forget_folds": X_fgt.shape[0],
        "feat_idx_len": len(feat_idx),
    }
    
    for name, X, Y_mat in [("retain", X_ret, Y_ret), ("forget", X_fgt, Y_fgt)]:
        stats = _compute_cka_stats(X, Y_mat, rng)
        results[f"cka_linear_{name}"] = stats["linear"]
        results[f"cka_rbf_{name}"] = stats["rbf"]
        results[f"cka_linear_{name}_ci"] = stats["ci"]
        results[f"cka_linear_{name}_p"] = stats["p"]

    # Z-scored CKA
    if X_ret.shape[0] >= 2:
        results["cka_linear_retain_zscored"] = float(linear_cka(_zscore(X_ret), _zscore(Y_ret)))
    if X_fgt.shape[0] >= 2:
        results["cka_linear_forget_zscored"] = float(linear_cka(_zscore(X_fgt), _zscore(Y_fgt)))

    # CKA by feature family
    slices = {"mfcc": IDX_MFCC, "f0": IDX_F0, "formants": IDX_FORMANTS, "prosody": IDX_PROSDY}
    for fam, sl in slices.items():
        Y_ret_f, Y_fgt_f = Y_ret[:, sl], Y_fgt[:, sl]
        if np.any(np.isnan(Y_ret_f)) or np.any(np.isnan(Y_fgt_f)):
            results[f"cka_linear_retain_{fam}"] = None
            results[f"cka_linear_forget_{fam}"] = None
        else:
            if X_ret.shape[0] >= 2:
                results[f"cka_linear_retain_{fam}"] = float(linear_cka(X_ret, Y_ret_f))
            if X_fgt.shape[0] >= 2:
                results[f"cka_linear_forget_{fam}"] = float(linear_cka(X_fgt, Y_fgt_f))

    # Save results
    with open(os.path.join(run_save_dir, "cka_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(os.path.join(run_save_dir, "speaker_averages.npz"), X_ret=X_ret, Y_ret=Y_ret, X_fgt=X_fgt, Y_fgt=Y_fgt)

    # Plot headline CKA
    head_ret, head_fgt = results.get("cka_linear_retain"), results.get("cka_linear_forget")
    if head_ret is not None or head_fgt is not None:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.bar(0, head_ret or 0, width=0.35, label="retain")
        ax.bar(0.35, head_fgt or 0, width=0.35, label="forget")
        ax.set_xticks([0.175])
        ax.set_xticklabels(["combined"])
        ax.set_ylabel("Linear CKA")
        ax.set_title("CKA retain vs forget (test split)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(os.path.join(run_save_dir, f"cka_headline.{ext}"), dpi=150)
        plt.close(fig)

    # Plot CKA by family
    fam_names = ["mfcc", "f0", "formants", "prosody"]
    ret_vals = [results.get(f"cka_linear_retain_{f}") for f in fam_names]
    fgt_vals = [results.get(f"cka_linear_forget_{f}") for f in fam_names]
    if any(x is not None for x in ret_vals + fgt_vals):
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(fam_names))
        w = 0.35
        ax.bar(x - w / 2, [v or 0 for v in ret_vals], w, label="retain")
        ax.bar(x + w / 2, [v or 0 for v in fgt_vals], w, label="forget")
        ax.set_xticks(x)
        ax.set_xticklabels(fam_names)
        ax.set_ylabel("Linear CKA")
        ax.set_title("CKA by feature family (test split)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(os.path.join(run_save_dir, f"cka_by_family.{ext}"), dpi=150)
        plt.close(fig)

    return results
