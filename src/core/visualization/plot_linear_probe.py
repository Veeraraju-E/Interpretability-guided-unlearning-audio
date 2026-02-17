# Linear probing summary plot: read per-seed CSVs, aggregate by (encoder, dataset), plot one (7, 5) multi-bar figure with CI.

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from .style import apply_publication_style, get_palette, create_figure, save_plot


def load_linear_probe_csvs(output_dir: str, seeds: list = None) -> pd.DataFrame:
    """Load all linear_probe_<seed>.csv from output_dir. If seeds is None, discover from filenames."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return pd.DataFrame()
    if seeds is None:
        files = list(output_dir.glob("linear_probe_*.csv"))
        seeds = []
        for f in files:
            try:
                # linear_probe_1337.csv -> 1337
                seed = int(f.stem.replace("linear_probe_", ""))
                seeds.append(seed)
            except ValueError:
                continue
        seeds = sorted(set(seeds))
    frames = []
    for seed in seeds:
        path = output_dir / f"linear_probe_{seed}.csv"
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _dataset_display_label(name: str) -> str:
    """Convert dataset name to display label: split on underscore into two lines to avoid overlap."""
    return name.replace("_", "\n")


def aggregate_by_encoder_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (encoder, dataset), compute mean accuracy and 95% CI (1.96 * std / sqrt(n))."""
    if df.empty or "accuracy" not in df.columns:
        return pd.DataFrame()
    grouped = df.groupby(["encoder", "dataset"], as_index=False).agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        count=("accuracy", "count"),
    )
    def _ci(r):
        if r["count"] < 2 or not pd.notna(r["std_accuracy"]):
            return 0.0
        return 1.96 * r["std_accuracy"] / np.sqrt(r["count"])

    grouped["ci_95"] = grouped.apply(_ci, axis=1)
    return grouped


def create_linear_probe_summary_plot(
    output_dir: str,
    seeds: list = None,
    output_filename: str = "linear_probe_summary.png",
) -> str:
    """
    Read linear_probe_<seed>.csv from output_dir, aggregate by (encoder, dataset), plot (7, 5) multi-bar with CI.
    Saves figure to output_dir / output_filename. Returns path to saved figure.
    """
    df = load_linear_probe_csvs(output_dir, seeds=seeds)
    if df.empty:
        raise ValueError(f"No linear_probe_*.csv found in {output_dir} or no valid seeds.")

    agg = aggregate_by_encoder_dataset(df)
    if agg.empty:
        raise ValueError("No (encoder, dataset) groups after aggregation.")

    encoders = agg["encoder"].unique().tolist()
    datasets = agg["dataset"].unique().tolist()
    palette = get_palette("accent")
    encoder_colors = {enc: palette[i % len(palette)] for i, enc in enumerate(encoders)}

    apply_publication_style()
    # Wider and less tall figure
    fig, ax = create_figure((12, 4), nrows=1, ncols=1)

    # Grouped bars: x = datasets, groups = encoders (thinner bars)
    n_datasets = len(datasets)
    n_encoders = len(encoders)
    total_group_width = 0.5
    bar_width = total_group_width / n_encoders if n_encoders else 0.25
    x = np.arange(n_datasets, dtype=float)

    for i, encoder in enumerate(encoders):
        means = []
        errs = []
        for ds in datasets:
            row = agg[(agg["encoder"] == encoder) & (agg["dataset"] == ds)]
            if len(row) > 0:
                means.append(row["mean_accuracy"].iloc[0])
                errs.append(row["ci_95"].iloc[0] if pd.notna(row["ci_95"].iloc[0]) else 0)
            else:
                means.append(0)
                errs.append(0)
        offset = (i - (n_encoders - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            means,
            width=bar_width,
            yerr=errs,
            capsize=2,
            color=encoder_colors[encoder],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
            label=encoder,
        )

    label_fontsize = 18
    tick_fontsize = 16
    ax.set_ylabel("Accuracy", fontweight="bold", fontsize=label_fontsize)
    ax.set_xticks(x)
    x_labels = [_dataset_display_label(ds) for ds in datasets]
    ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", frameon=True, fontsize=tick_fontsize)
    ax.grid(True, alpha=0.3, axis="y")

    out_path_png = Path(output_dir) / output_filename
    out_path_pdf = Path(output_dir) / output_filename.replace(".png", ".pdf")
    save_plot(fig, str(out_path_png))
    save_plot(fig, str(out_path_pdf))
    return {"png": str(out_path_png), "pdf": str(out_path_pdf)}


def main():
    parser = argparse.ArgumentParser(description="Plot linear probe summary (mean accuracy + 95% CI) from per-seed CSVs.")
    parser.add_argument("output_dir", type=str, help="Directory containing linear_probe_<seed>.csv files")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Seeds to include (default: all found)")
    parser.add_argument("--output", type=str, default="linear_probe_summary.png", help="Output filename")
    args = parser.parse_args()
    create_linear_probe_summary_plot(args.output_dir, seeds=args.seeds, output_filename=args.output)
    print("[VIZ]: Done")


if __name__ == "__main__":
    main()
