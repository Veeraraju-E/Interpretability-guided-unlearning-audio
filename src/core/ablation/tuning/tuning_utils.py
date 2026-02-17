"""
Shared utilities for ablation tuning: load pre + U, run one trial, compute metrics.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict

import numpy as np
import torch

from src.core.ablation.base import AblationEraser, apply_ablation_eraser
from src.core.evals.classification import train_classifier, evaluate_classifier
from src.core.evals.probes import safe_linear_probe
from src.core.sae import SAE, train_sae, sae_encode_batch
from src.core.utils.stats import compute_feature_stats, select_features
from src.core.utils.subspace import prune_and_normalize_decoder, build_U_from_decoder
from src.core.utils.seed import seed_all


def get_adapter(dataset_name: str):
    """Return the dataset adapter for the given name (same as cli/main.py ADAPTERS)."""
    from src.datasets.voxceleb_adapter import VoxCelebAdapter
    from src.datasets.speech_commands_adapter import SpeechCommandsAdapter
    from src.datasets.audiomnist_adapter import AudioMNISTAdapter
    from src.datasets.librispeech_adapter import LibriSpeechAdapter
    from src.datasets.lahaja_adapter import LahajaAdapter
    from src.datasets.mls_spanish_adapter import MLSSpanishAdapter
    adapters = {
        "voxceleb": VoxCelebAdapter(),
        "speech_commands": SpeechCommandsAdapter(),
        "audiomnist": AudioMNISTAdapter(),
        "librispeech": LibriSpeechAdapter(),
        "lahaja": LahajaAdapter(),
        "mls_spanish": MLSSpanishAdapter(),
    }
    if dataset_name not in adapters:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(adapters.keys())}")
    return adapters[dataset_name]


def _precompute_from_config(adapter, cfg: Dict):
    """Dispatch precompute like linear_probe_experiment."""
    seed = cfg["experiment"]["seed"]
    device = cfg["experiment"]["device"]
    data_path_param = cfg["dataset"]["root_or_manifest"]
    name = getattr(adapter, "name", None)
    if name in ("audiomnist", "speech_commands"):
        return adapter.precompute(
            data_root=data_path_param,
            embeddings_path=cfg["encoder"].get("embeddings_path"),
            dev_frac=cfg["dataset"]["dev_frac"],
            seed=seed,
            encoder_cfg=cfg["encoder"],
            device=device,
        )
    if name == "librispeech":
        return adapter.precompute(
            root=data_path_param,
            embeddings_path=cfg["encoder"].get("embeddings_path"),
            dev_frac=cfg["dataset"]["dev_frac"],
            seed=seed,
            encoder_cfg=cfg["encoder"],
            device=device,
        )
    if name == "lahaja":
        return adapter.precompute(
            split=data_path_param,
            embeddings_path=cfg["encoder"].get("embeddings_path"),
            dev_frac=cfg["dataset"]["dev_frac"],
            seed=seed,
            encoder_cfg=cfg["encoder"],
            device=device,
        )
    if name == "mls_spanish":
        return adapter.precompute(
            root_or_manifest=data_path_param,
            embeddings_path=cfg["encoder"].get("embeddings_path"),
            dev_frac=cfg["dataset"]["dev_frac"],
            seed=seed,
            encoder_cfg=cfg["encoder"],
            device=device,
        )
    return adapter.precompute(
        root_or_manifest=data_path_param,
        embeddings_path=cfg["encoder"].get("embeddings_path"),
        dev_frac=cfg["dataset"]["dev_frac"],
        seed=seed,
        encoder_cfg=cfg["encoder"],
        device=device,
    )


def precompute_from_config(adapter, cfg: Dict):
    """Run adapter precompute once (embeddings + whitening). Reuse the returned pre across trials."""
    return _precompute_from_config(adapter, cfg)


def get_U_for_sae(pre, cfg: Dict, use_sae: bool) -> np.ndarray:
    """
    Return U for the given pre and use_sae. If use_sae, train SAE and build U from feature mining; else U = I.
    Use with precompute_from_config so embeddings are computed only once per tuning run.
    """
    device = cfg["experiment"]["device"]
    d = pre.Z_all.shape[1]
    if not use_sae:
        return np.eye(d, dtype=pre.Z_all.dtype)
    Z_train = pre.Z_all[pre.mask_dev]
    perm = np.random.permutation(Z_train.shape[0])
    cut = int(0.8 * len(perm)) if len(perm) > 1 else len(perm) // 2
    Z_sae_train = Z_train[perm[:cut]] if cut > 0 else Z_train
    Z_sae_val = Z_train[perm[cut:]] if cut < len(perm) else Z_train
    sae = SAE(
        input_dim=pre.Z_all.shape[1],
        hidden_mult=cfg["sae"]["multiplier"],
        k=cfg["sae"]["topk"],
        variant=cfg["sae"].get("variant", "topk"),
        hidden_dim=cfg["sae"].get("hidden_dim", 0),
    )
    sae = train_sae(
        sae,
        Z_sae_train,
        Z_sae_val,
        epochs=cfg["sae"]["epochs"],
        batch_size=cfg["sae"]["batch_size"],
        lr=1e-3,
        device=device,
        lambda_sparsity=cfg["sae"].get("lambda_sparsity", 0.0),
    )
    A_all = sae_encode_batch(sae, pre.Z_all, device=device, batch=256)
    dec_w = sae.dec.weight.detach().cpu().numpy()
    forget_list = cfg["dataset"].get("forget_labels", [])
    if not forget_list:
        return np.eye(d, dtype=pre.Z_all.dtype)
    forget_label_str = str(forget_list[0])
    sets = [
        "forget" if str(lbl) == forget_label_str else "retain"
        for lbl in pre.labels
    ]
    M_ret_dev = np.array([s == "retain" and pre.mask_dev[i] for i, s in enumerate(sets)])
    M_fgt_dev = np.array([s == "forget" and pre.mask_dev[i] for i, s in enumerate(sets)])
    A_ret = A_all[M_ret_dev]
    A_fgt = A_all[M_fgt_dev]
    if A_fgt.shape[0] < 5 or A_ret.shape[0] < 5:
        return np.eye(d, dtype=pre.Z_all.dtype)
    method = cfg["feature_mining"].get("statistical_method", "cohens_d")
    feat_stats = compute_feature_stats(A_fgt, A_ret, method=method)
    fdr = cfg["feature_mining"].get("fdr", 0.05)
    top_k = cfg["feature_mining"].get("build_U_top", "10%")
    if isinstance(top_k, str) and "%" in top_k:
        pct = int(top_k.replace("%", ""))
        top_k = max(1, (feat_stats["auc"].size * pct) // 100)
    feat_idx = select_features(feat_stats, method=method, top_k=top_k, fdr=fdr)
    if len(feat_idx) == 0:
        return np.eye(d, dtype=pre.Z_all.dtype)
    prune_and_normalize_decoder(dec_w)
    return build_U_from_decoder(dec_w, feat_idx)


def get_pre_U_sae(adapter, cfg: Dict, use_sae: bool):
    """
    Return (pre, U, use_sae). If use_sae, train SAE and build U from feature mining; else U = I.
    For tuning runs, prefer precompute_from_config once then get_U_for_sae(pre, cfg, use_sae) to reuse embeddings.
    """
    seed_all(cfg["experiment"]["seed"])
    pre = _precompute_from_config(adapter, cfg)
    U = get_U_for_sae(pre, cfg, use_sae)
    return pre, U, use_sae


def erb_score(fa_after: float, ra_after: float) -> float:
    """ERB = 2*(100-FA)*RA / ((100-FA)+RA), FA/RA in [0,1] or percentage."""
    fa_pct = float(fa_after) * 100.0 if float(fa_after) <= 1.0 else float(fa_after)
    ra_pct = float(ra_after) * 100.0 if float(ra_after) <= 1.0 else float(ra_after)
    nr = 2.0 * (100.0 - fa_pct) * ra_pct
    dr = (100.0 - fa_pct) + ra_pct
    return float(nr / dr) if dr != 0 else 0.0


def run_one_trial(
    pre,
    U: np.ndarray,
    use_sae: bool,
    forget_label: str,
    eraser: AblationEraser,
    cfg: Dict,
    device: str,
) -> Dict[str, Any]:
    """
    Run one ablation trial: fit eraser on dev, transform all, compute metrics.
    Returns dict with linear_probe_auc_before/after, FA_before/after, RA_before/after, ERB_after, runtime_sec.
    """
    all_labels = pre.labels
    forget_label_str = str(forget_label)
    Y_all = np.array(
        [1 if str(all_labels[i]) == forget_label_str else 0 for i in range(len(all_labels))],
        dtype=np.int64,
    ).reshape(-1, 1)
    mask_dev = pre.mask_dev
    mask_test = pre.mask_test
    R = pre.Z_all @ U if (use_sae and U.shape[1] > 0) else pre.Z_all
    y_flat = Y_all.ravel()
    R_dev = R[mask_dev]
    y_dev = y_flat[mask_dev]
    if np.unique(y_dev).size < 2:
        return {
            "linear_probe_auc_before": float("nan"),
            "linear_probe_auc_after": float("nan"),
            "FA_before": float("nan"),
            "FA_after": float("nan"),
            "RA_before": float("nan"),
            "RA_after": float("nan"),
            "ERB_after": 0.0,
            "runtime_sec": 0.0,
        }
    t0 = time.perf_counter()
    eraser.fit(R_dev, y_dev)
    E_prime = apply_ablation_eraser(pre.Z_all, U, use_sae, eraser, pre.mu, pre.W)
    runtime_sec = time.perf_counter() - t0
    E_all_norm = pre.E_all / (np.linalg.norm(pre.E_all, axis=1, keepdims=True) + 1e-12)
    E_prime_norm = E_prime / (np.linalg.norm(E_prime, axis=1, keepdims=True) + 1e-12)
    split_mask = {"dev": mask_dev, "test": mask_test}
    _, auc_before = safe_linear_probe(
        E_all_norm, all_labels, U, split_mask, [forget_label_str]
    )
    _, auc_after = safe_linear_probe(
        E_prime_norm, all_labels, U, split_mask, [forget_label_str]
    )
    classes = pre.classes
    cls2id = {c: i for i, c in enumerate(classes)}
    # Train one classifier on original (before) embeddings; reuse for both before and after evaluation
    X_cls_train = E_all_norm[mask_dev]
    y_cls_train = [cls2id[pre.labels[i]] for i in range(len(pre.labels)) if mask_dev[i]]
    X_cls_val = E_all_norm[mask_test]
    y_cls_val = [cls2id[pre.labels[i]] for i in range(len(pre.labels)) if mask_test[i]]
    clf = train_classifier(
        X_cls_train,
        y_cls_train,
        X_cls_val,
        y_cls_val,
        num_classes=len(classes),
        device=device,
        epochs=cfg["orchestration"].get("classifier_epochs", 20),
        batch_size=512,
        lr=1e-3,
    )
    M_ret_test = np.array([str(all_labels[i]) != forget_label_str and mask_test[i] for i in range(len(all_labels))])
    M_fgt_test = np.array([str(all_labels[i]) == forget_label_str and mask_test[i] for i in range(len(all_labels))])
    E_ret = E_all_norm[M_ret_test]
    E_fgt = E_all_norm[M_fgt_test]
    E_ret_p = E_prime_norm[M_ret_test]
    E_fgt_p = E_prime_norm[M_fgt_test]
    L_ret = [cls2id[all_labels[i]] for i in range(len(all_labels)) if M_ret_test[i]]
    L_fgt = [cls2id[all_labels[i]] for i in range(len(all_labels)) if M_fgt_test[i]]
    fa_before = evaluate_classifier(clf, E_fgt, L_fgt, device=device)
    ra_before = evaluate_classifier(clf, E_ret, L_ret, device=device)
    fa_after = evaluate_classifier(clf, E_fgt_p, L_fgt, device=device)
    ra_after = evaluate_classifier(clf, E_ret_p, L_ret, device=device)
    erb = erb_score(fa_after, ra_after)
    return {
        "linear_probe_auc_before": auc_before,
        "linear_probe_auc_after": auc_after,
        "FA_before": fa_before,
        "FA_after": fa_after,
        "RA_before": ra_before,
        "RA_after": ra_after,
        "ERB_after": erb,
        "runtime_sec": runtime_sec,
    }


def compute_ablation_metrics(
    pre,
    U: np.ndarray,
    use_sae: bool,
    forget_label: str,
    E_prime: np.ndarray,
    cfg: Dict,
    device: str,
) -> Dict[str, Any]:
    """
    Compute AUC, FA, RA, ERB from pre and E_prime (no fit/apply, no runtime).
    Used by orchestration runner which times the full pipeline including get_pre_U_sae.
    """
    all_labels = pre.labels
    forget_label_str = str(forget_label)
    mask_dev = pre.mask_dev
    mask_test = pre.mask_test
    E_all_norm = pre.E_all / (np.linalg.norm(pre.E_all, axis=1, keepdims=True) + 1e-12)
    E_prime_norm = E_prime / (np.linalg.norm(E_prime, axis=1, keepdims=True) + 1e-12)
    split_mask = {"dev": mask_dev, "test": mask_test}
    _, auc_before = safe_linear_probe(
        E_all_norm, all_labels, U, split_mask, [forget_label_str]
    )
    _, auc_after = safe_linear_probe(
        E_prime_norm, all_labels, U, split_mask, [forget_label_str]
    )
    # Same classifier for before/after: train on original embeddings, evaluate on original and edited
    classes = pre.classes
    cls2id = {c: i for i, c in enumerate(classes)}
    X_cls_train = E_all_norm[mask_dev]
    y_cls_train = [cls2id[pre.labels[i]] for i in range(len(pre.labels)) if mask_dev[i]]
    X_cls_val = E_all_norm[mask_test]
    y_cls_val = [cls2id[pre.labels[i]] for i in range(len(pre.labels)) if mask_test[i]]
    clf = train_classifier(
        X_cls_train,
        y_cls_train,
        X_cls_val,
        y_cls_val,
        num_classes=len(classes),
        device=device,
        epochs=cfg["orchestration"].get("classifier_epochs", 20),
        batch_size=512,
        lr=1e-3,
    )
    M_ret_test = np.array([str(all_labels[i]) != forget_label_str and mask_test[i] for i in range(len(all_labels))])
    M_fgt_test = np.array([str(all_labels[i]) == forget_label_str and mask_test[i] for i in range(len(all_labels))])
    E_ret = E_all_norm[M_ret_test]
    E_fgt = E_all_norm[M_fgt_test]
    E_ret_p = E_prime_norm[M_ret_test]
    E_fgt_p = E_prime_norm[M_fgt_test]
    L_ret = [cls2id[all_labels[i]] for i in range(len(all_labels)) if M_ret_test[i]]
    L_fgt = [cls2id[all_labels[i]] for i in range(len(all_labels)) if M_fgt_test[i]]
    fa_before = evaluate_classifier(clf, E_fgt, L_fgt, device=device)
    ra_before = evaluate_classifier(clf, E_ret, L_ret, device=device)
    fa_after = evaluate_classifier(clf, E_fgt_p, L_fgt, device=device)
    ra_after = evaluate_classifier(clf, E_ret_p, L_ret, device=device)
    erb = erb_score(fa_after, ra_after)
    return {
        "linear_probe_auc_before": auc_before,
        "linear_probe_auc_after": auc_after,
        "FA_before": fa_before,
        "FA_after": fa_after,
        "RA_before": ra_before,
        "RA_after": ra_after,
        "ERB_after": erb,
    }
