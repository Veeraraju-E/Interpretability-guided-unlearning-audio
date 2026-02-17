"""
Hyperparameter tuning for INLP (with_sae and without_sae). Outputs CSV rows and best YAML snippet.
"""
import argparse
import csv
import os
import sys

import yaml
from tqdm import tqdm

# allow running as script (project root so "datasets" = HuggingFace, not src.datasets)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from src.core.ablation.inlp import INLPEraser
from src.core.ablation.tuning.tuning_utils import (
    get_adapter,
    get_U_for_sae,
    precompute_from_config,
    run_one_trial,
)
from src.core.utils.seed import seed_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config")
    parser.add_argument("--with_sae", action="store_true", help="Tune with SAE subspace")
    parser.add_argument("--without_sae", action="store_true", help="Tune without SAE (full space)")
    parser.add_argument("--output_dir", default=None, help="CSV output dir (default: experiment.save_dir)")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if not args.with_sae and not args.without_sae:
        args.with_sae = args.without_sae = True
    dataset_name = cfg["dataset"]["name"]
    adapter = get_adapter(dataset_name)
    forget_list = cfg["dataset"].get("forget_labels", [])
    forget_label = str(forget_list[0]) if forget_list else None
    if not forget_label:
        raise SystemExit("Config must set dataset.forget_labels")
    device = cfg["experiment"]["device"]
    output_dir = args.output_dir or cfg["experiment"].get("save_dir", "./runs/ablation_tuning")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "tune_inlp.csv")

    grid = [
        {"n_iter": 20, "min_accuracy": 0.0},
        {"n_iter": 50, "min_accuracy": 0.0},
        {"n_iter": 100, "min_accuracy": 0.0},
        {"n_iter": 50, "min_accuracy": 0.5},
    ]
    best_row = None
    best_score = (-1, -1, 1e9)
    fieldnames = [
        "method", "with_sae", "forget_label", "dataset", "encoder",
        "n_iter", "min_accuracy",
        "linear_probe_auc_before", "linear_probe_auc_after",
        "FA_before", "FA_after", "RA_before", "RA_after", "ERB_after", "runtime_sec",
    ]
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    seed_all(cfg["experiment"]["seed"])
    pre = precompute_from_config(adapter, cfg)
    sae_variants = ([True] if args.with_sae else []) + ([False] if args.without_sae else [])
    for use_sae in tqdm(sae_variants, desc="SAE variant"):
        U = get_U_for_sae(pre, cfg, use_sae)
        for hp in tqdm(grid, desc="Hyperparams", leave=False):
            eraser = INLPEraser(
                n_iter=hp["n_iter"],
                min_accuracy=hp["min_accuracy"],
                random_state=cfg["experiment"].get("seed"),
            )
            try:
                res = run_one_trial(pre, U, use_sae, forget_label, eraser, cfg, device)
            except Exception as e:
                print(f"Failed {hp} use_sae={use_sae}: {e}")
                continue
            res["method"] = "inlp"
            res["with_sae"] = use_sae
            res["forget_label"] = forget_label
            res["dataset"] = dataset_name
            res["encoder"] = cfg["encoder"].get("name", "")
            res["n_iter"] = hp["n_iter"]
            res["min_accuracy"] = hp["min_accuracy"]
            auc_diff = abs(res["linear_probe_auc_after"] - 0.5)
            erb = res["ERB_after"]
            runtime = res["runtime_sec"]
            score = (-auc_diff, erb, runtime)
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
        print("# Best INLP hyperparams (paste into config ablation.methods)")
        print("inlp:")
        print("  sae_variants:")
        print("    with_sae:")
        print(f"      n_iter: {best_row.get('n_iter', 50)}")
        print(f"      min_accuracy: {best_row.get('min_accuracy', 0.0)}")
        print("    without_sae:")
        print(f"      n_iter: {best_row.get('n_iter', 50)}")
        print(f"      min_accuracy: {best_row.get('min_accuracy', 0.0)}")
    print(f"Results appended to {csv_path}")


if __name__ == "__main__":
    main()
