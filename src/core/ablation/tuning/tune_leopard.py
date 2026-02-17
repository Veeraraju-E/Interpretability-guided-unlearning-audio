"""
Hyperparameter tuning for L̄EOPARD (with_sae and without_sae). Outputs CSV rows and best YAML snippet.
"""
import argparse
import csv
import os
import sys

import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from src.core.ablation.leopard import LEOPARDEraser
from src.core.ablation.tuning.tuning_utils import (
    get_adapter,
    get_U_for_sae,
    precompute_from_config,
    run_one_trial,
)
from src.core.utils.seed import seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--with_sae", action="store_true")
    parser.add_argument("--without_sae", action="store_true")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if not args.with_sae and not args.without_sae:
        args.with_sae = args.without_sae = True
    adapter = get_adapter(cfg["dataset"]["name"])
    forget_label = str(cfg["dataset"].get("forget_labels", [None])[0])
    if forget_label == "None":
        raise SystemExit("Config must set dataset.forget_labels")
    device = cfg["experiment"]["device"]
    output_dir = args.output_dir or cfg["experiment"].get("save_dir", "./runs/ablation_tuning")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "tune_leopard.csv")
    grid = [
        {"rank": 2},
        {"rank": 5},
        {"rank": 10},
    ]
    best_row = None
    best_score = (-1, -1, 1e9)
    fieldnames = [
        "method", "with_sae", "forget_label", "dataset", "encoder",
        "rank",
        "linear_probe_auc_before", "linear_probe_auc_after",
        "FA_before", "FA_after", "RA_before", "RA_after", "ERB_after", "runtime_sec",
    ]
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    seed_all(cfg["experiment"]["seed"])
    pre = precompute_from_config(adapter, cfg)
    sae_variants = ([True] if args.with_sae else []) + ([False] if args.without_sae else [])
    for use_sae in tqdm(sae_variants, desc="SAE variant"):
        U = get_U_for_sae(pre, cfg, use_sae)
        r_dim = U.shape[1] if use_sae and U.shape[1] > 0 else pre.Z_all.shape[1]
        for hp in tqdm(grid, desc="Hyperparams", leave=False):
            rank = min(hp["rank"], r_dim)
            if rank < 1:
                continue
            eraser = LEOPARDEraser(
                rank=rank,
                random_state=cfg["experiment"].get("seed"),
            )
            try:
                res = run_one_trial(pre, U, use_sae, forget_label, eraser, cfg, device)
            except Exception as e:
                print(f"Failed rank={rank} use_sae={use_sae}: {e}")
                continue
            res["method"] = "leopard"
            res["with_sae"] = use_sae
            res["forget_label"] = forget_label
            res["dataset"] = cfg["dataset"]["name"]
            res["encoder"] = cfg["encoder"].get("name", "")
            res["rank"] = rank
            auc_diff = abs(res["linear_probe_auc_after"] - 0.5)
            score = (-auc_diff, res["ERB_after"], res["runtime_sec"])
            if score > best_score:
                best_score = score
                best_row = res.copy()
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    w.writeheader()
                    write_header = False
                w.writerow({k: res.get(k, "") for k in fieldnames})
    if best_row:
        print("# Best L̄EOPARD hyperparams")
        print("leopard:")
        print("  sae_variants:")
        print("    with_sae:")
        print(f"      rank: {best_row.get('rank', 5)}")
        print("    without_sae:")
        print(f"      rank: {best_row.get('rank', 5)}")
    print(f"Results appended to {csv_path}")


if __name__ == "__main__":
    main()
