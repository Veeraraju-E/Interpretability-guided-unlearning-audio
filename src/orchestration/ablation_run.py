"""
Ablation orchestration: run each method (with_sae / without_sae) with best hyperparams, write results CSV only.
Runtime includes SAE training when sae_enabled.
"""
from __future__ import annotations

import csv
import os
import time
from typing import Any, Dict, List

import numpy as np
import yaml
from tqdm import tqdm

from src.core.ablation.base import apply_ablation_eraser
from src.core.ablation.igbp import IGBPEraser
from src.core.ablation.inlp import INLPEraser
from src.core.ablation.leopard import LEOPARDEraser
from src.core.ablation.sal import SALEraser
from src.core.ablation.tuning.tuning_utils import (
    get_adapter,
    get_pre_U_sae,
    compute_ablation_metrics,
)
from src.core.utils.seed import seed_all


def build_eraser(method_name: str, hparams: Dict[str, Any], seed: int | None) -> Any:
    """Instantiate eraser from method name and hyperparams (from config sae_variants.with_sae/without_sae)."""
    if method_name == "inlp":
        return INLPEraser(
            n_iter=int(hparams.get("n_iter", 50)),
            min_accuracy=float(hparams.get("min_accuracy", 0.0)),
            random_state=seed,
        )
    if method_name == "sal":
        return SALEraser(n_components_remove=int(hparams.get("n_components_remove", 1)))
    if method_name == "igbp":
        pw = hparams.get("probe_width_list", [1])
        if isinstance(pw, str):
            import ast
            pw = ast.literal_eval(pw) if pw.startswith("[") else [1]
        return IGBPEraser(
            n_probes=int(hparams.get("n_probes", 10)),
            probe_width_list=pw,
            random_state=seed,
        )
    if method_name == "leopard":
        return LEOPARDEraser(
            rank=int(hparams.get("rank", 5)),
            random_state=seed,
        )
    raise ValueError(f"Unknown ablation method: {method_name}")


def run_ablation(cfg: Dict[str, Any]) -> str:
    """
    Run all methods × sae_variants from config ablation section.
    Writes one row per (method, sae_enabled) to output_csv. Returns path to CSV.
    No artifacts (no NPZ, no visual abstracts). Runtime includes SAE when sae_enabled.
    """
    ablation_cfg = cfg.get("ablation", {})
    output_csv = ablation_cfg.get("output_csv", "ablation_results.csv")
    methods_cfg = ablation_cfg.get("methods", [])
    if not methods_cfg:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            f.write("method,sae_enabled,linear_probe_auc_before,linear_probe_auc_after,FA_before,FA_after,RA_before,RA_after,ERB_after,runtime_sec\n")
        return output_csv

    adapter = get_adapter(cfg["dataset"]["name"])
    forget_list = cfg["dataset"].get("forget_labels", [])
    forget_label = str(forget_list[0]) if forget_list else None
    if not forget_label:
        raise ValueError("Config must set dataset.forget_labels")
    device = cfg["experiment"]["device"]
    seed = cfg["experiment"].get("seed")

    fieldnames = [
        "method", "sae_enabled", "linear_probe_auc_before", "linear_probe_auc_after",
        "FA_before", "FA_after", "RA_before", "RA_after", "ERB_after", "runtime_sec",
    ]
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    rows: List[Dict[str, Any]] = []

    tasks = []
    for method_entry in methods_cfg:
        name = method_entry.get("name")
        if name not in ("inlp", "sal", "igbp", "leopard"):
            continue
        sae_variants = method_entry.get("sae_variants", {})
        for use_sae, variant_key in [(True, "with_sae"), (False, "without_sae")]:
            hparams = sae_variants.get(variant_key, {})
            if hparams:
                tasks.append((method_entry, name, use_sae, variant_key, hparams))

    for method_entry, name, use_sae, variant_key, hparams in tqdm(tasks, desc="Ablation runs"):
        seed_all(seed)
        t0 = time.perf_counter()
        pre, U, _ = get_pre_U_sae(adapter, cfg, use_sae)
        R = pre.Z_all @ U if (use_sae and U.shape[1] > 0) else pre.Z_all
        all_labels = pre.labels
        forget_label_str = str(forget_label)
        Y_all = np.array(
            [1 if str(all_labels[i]) == forget_label_str else 0 for i in range(len(all_labels))],
            dtype=np.int64,
        ).ravel()
        R_dev = R[pre.mask_dev]
        y_dev = Y_all[pre.mask_dev]
        if np.unique(y_dev).size < 2:
            row = {
                "method": name,
                "sae_enabled": use_sae,
                "linear_probe_auc_before": float("nan"),
                "linear_probe_auc_after": float("nan"),
                "FA_before": float("nan"),
                "FA_after": float("nan"),
                "RA_before": float("nan"),
                "RA_after": float("nan"),
                "ERB_after": 0.0,
                "runtime_sec": time.perf_counter() - t0,
            }
        else:
            eraser = build_eraser(name, hparams, seed)
            eraser.fit(R_dev, y_dev)
            E_prime = apply_ablation_eraser(pre.Z_all, U, use_sae, eraser, pre.mu, pre.W)
            metrics = compute_ablation_metrics(
                pre, U, use_sae, forget_label, E_prime, cfg, device
            )
            metrics["runtime_sec"] = time.perf_counter() - t0
            row = {
                "method": name,
                "sae_enabled": use_sae,
                **metrics,
            }
        rows.append({k: row.get(k, "") for k in fieldnames})

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return output_csv
