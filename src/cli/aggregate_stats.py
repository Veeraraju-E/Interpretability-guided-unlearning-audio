import os
import json
import argparse
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import wilcoxon, sem
from typing import Dict


def cohens_dz(before: np.ndarray, after: np.ndarray) -> float:
    """Cohen's dz for paired samples: mean diff / std of diffs."""
    diff = after - before
    return diff.mean() / (diff.std(ddof=1) + 1e-12)


def bootstrap_ci_paired(before: np.ndarray, after: np.ndarray, stat_fn, n_boot: int = 1000, alpha: float = 0.05) -> tuple:
    """Bootstrap CI for a paired statistic."""
    rng = np.random.default_rng(42)
    n = len(before)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot[i] = stat_fn(before[idx], after[idx])
    return float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def load_run_results(run_dir: str) -> Dict:
    """Load metrics from a single run directory."""
    summary_path = os.path.join(run_dir, "verification_eer_summary.json")
    if not os.path.exists(summary_path):
        return None
    with open(summary_path) as f:
        return json.load(f)


def aggregate(results_dir: str, output_dir: str = None):
    """Aggregate results across all seed*_forget_* directories."""
    output_dir = output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all run directories
    run_dirs = sorted(glob(os.path.join(results_dir, "seed*_forget_*")))
    if not run_dirs:
        print(f"No run directories found in {results_dir}")
        return
    
    # Collect results
    records = []
    for run_dir in run_dirs:
        res = load_run_results(run_dir)
        if res is None:
            continue
        basename = os.path.basename(run_dir)
        parts = basename.split("_forget_")
        seed = int(parts[0].replace("seed", ""))
        forget_label = parts[1] if len(parts) > 1 else "unknown"
        records.append({"seed": seed, "forget_label": forget_label, **res})
    
    if not records:
        print("No valid results found.")
        return
    
    df = pd.DataFrame(records)
    
    # Metrics to aggregate (before/after pairs)
    metric_pairs = [
        ("FA_before", "FA_after", "FA"),
        ("RA_before", "RA_after", "RA"),
        ("forget_EER_before", "forget_EER_after", "forget_EER"),
        ("retain_EER_before", "retain_EER_after", "retain_EER"),
        ("linear_probe_auc_before", "linear_probe_auc_after", "probe_AUC"),
    ]
    
    paper_stats = {"n_runs": len(df)}
    summary_rows = []
    
    for col_before, col_after, name in metric_pairs:
        if col_before not in df.columns or col_after not in df.columns:
            continue
        before = df[col_before].dropna().values
        after = df[col_after].dropna().values
        n = min(len(before), len(after))
        if n < 2:
            continue
        before, after = before[:n], after[:n]
        
        # Basic stats
        mean_before, std_before = before.mean(), before.std(ddof=1)
        mean_after, std_after = after.mean(), after.std(ddof=1)
        ci95_before = 1.96 * sem(before)
        ci95_after = 1.96 * sem(after)
        
        # Paired test
        if n >= 5:
            stat, pval = wilcoxon(before, after, alternative='two-sided')
        else:
            stat, pval = np.nan, np.nan
        
        # Effect size
        dz = cohens_dz(before, after)
        dz_ci_lo, dz_ci_hi = bootstrap_ci_paired(before, after, cohens_dz)
        
        paper_stats[name] = {
            "before_mean": float(mean_before),
            "before_std": float(std_before),
            "before_ci95": float(ci95_before),
            "after_mean": float(mean_after),
            "after_std": float(std_after),
            "after_ci95": float(ci95_after),
            "wilcoxon_stat": float(stat) if not np.isnan(stat) else None,
            "wilcoxon_p": float(pval) if not np.isnan(pval) else None,
            "cohens_dz": float(dz),
            "dz_ci_lo": float(dz_ci_lo),
            "dz_ci_hi": float(dz_ci_hi),
        }
        
        summary_rows.append({
            "metric": name,
            "before": f"{mean_before:.4f} ± {std_before:.4f}",
            "after": f"{mean_after:.4f} ± {std_after:.4f}",
            "wilcoxon_p": f"{pval:.4g}" if not np.isnan(pval) else "N/A",
            "cohens_dz": f"{dz:.3f} [{dz_ci_lo:.3f}, {dz_ci_hi:.3f}]",
        })

        print(f"[INFO] paper_stats[{name}]: {paper_stats[name]}\n")
        print(f"[INFO] summary_rows[{name}]: {summary_rows[name]}")
    
    # Save outputs
    with open(os.path.join(output_dir, "paper_stats.json"), "w") as f:
        json.dump(paper_stats, f, indent=2)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "paper_table.csv"), index=False)
    
    print(f"Aggregated {len(df)} runs from {results_dir}")
    print(f"Saved: paper_stats.json, paper_table.csv")
    print("\nSummary:")
    print(summary_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results into paper-ready stats")
    parser.add_argument("results_dir", help="Directory containing seed*_forget_* subdirectories")
    parser.add_argument("-o", "--output", help="Output directory (default: same as results_dir)")
    args = parser.parse_args()
    aggregate(args.results_dir, args.output)


if __name__ == "__main__":
    main()
