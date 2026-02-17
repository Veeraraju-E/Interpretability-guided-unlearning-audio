"""
Pareto frontier bubble chart for ablation results.
x = RA_after (%), y = 100 - FA_after (%), bubble size ∝ 1/runtime_sec.
Labels: method + sae_enabled. Overlay Pareto frontier (maximize both x and y).
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from .style import apply_publication_style, create_figure, save_plot, get_palette


def _to_pct(val: float) -> float:
    """Convert to percentage if in [0,1]."""
    v = float(val)
    if np.isnan(v):
        return np.nan
    return v * 100.0 if v <= 1.0 else v


def _pareto_frontier_2d(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Points on Pareto frontier (maximize both x and y). Returns sorted x, y for plotting."""
    order = np.lexsort((-y, -x))
    xs, ys = x[order], y[order]
    frontier_x, frontier_y = [xs[0]], [ys[0]]
    for i in range(1, len(xs)):
        if ys[i] > frontier_y[-1]:
            frontier_x.append(xs[i])
            frontier_y.append(ys[i])
    return np.array(frontier_x), np.array(frontier_y)


def plot_pareto_frontier(
    results_csv: str,
    output_png: str | None = None,
    output_pdf: str | None = None,
    bubble_scale: float = 50.0,
    runtime_clip_max_sec: float = 3600.0,
) -> dict:
    """
    Read ablation results CSV, plot Pareto frontier bubble chart.
    x = RA_after (%), y = 100 - FA_after (%), bubble size ∝ 1/runtime_sec (clipped).
    Returns dict with saved paths.
    """
    if not os.path.isfile(results_csv):
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")
    df = pd.read_csv(results_csv)
    for col in ("RA_after", "FA_after", "runtime_sec", "method", "sae_enabled"):
        if col not in df.columns:
            raise ValueError(f"CSV must contain column: {col}")

    ra = np.array([_to_pct(v) for v in df["RA_after"].astype(float)])
    fa = np.array([_to_pct(v) for v in df["FA_after"].astype(float)])
    y_axis = 100.0 - fa
    runtime = np.array(df["runtime_sec"].astype(float))
    runtime = np.clip(runtime, 1e-6, runtime_clip_max_sec)
    sizes = bubble_scale / runtime
    sizes = np.clip(sizes, 10, 800)

    apply_publication_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = get_palette("primary")
    for i, row in df.iterrows():
        label = f"{row['method']} ({'SAE' if row['sae_enabled'] == True or str(row['sae_enabled']).lower() == 'true' else 'no SAE'})"
        ax.scatter(ra[i], y_axis[i], s=sizes[i], c=colors[i % len(colors)], alpha=0.7, label=label, edgecolors="black", linewidths=0.5)
    px, py = _pareto_frontier_2d(ra, y_axis)
    sort_idx = np.argsort(px)
    px, py = px[sort_idx], py[sort_idx]
    ax.plot(px, py, "k--", linewidth=1.5, label="Pareto frontier", zorder=0)
    ax.set_xlabel("RA_after (%)")
    ax.set_ylabel("100 − FA_after (%)")
    ax.set_title("Ablation: Pareto frontier (RA vs unlearning)")
    ax.legend(loc="best", fontsize=5)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    out = {}
    if output_png:
        os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
        save_plot(fig, output_png)
        out["png"] = output_png
    if output_pdf:
        os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
        save_plot(fig, output_pdf)
        out["pdf"] = output_pdf
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot Pareto frontier from ablation results.csv")
    parser.add_argument("results_csv", help="Path to ablation results CSV")
    parser.add_argument("--png", default="pareto_frontier.png", help="Output PNG path")
    parser.add_argument("--pdf", default="pareto_frontier.pdf", help="Output PDF path")
    parser.add_argument("--bubble-scale", type=float, default=50.0)
    parser.add_argument("--runtime-clip-max", type=float, default=3600.0)
    args = parser.parse_args()
    plot_pareto_frontier(
        args.results_csv,
        output_png=args.png,
        output_pdf=args.pdf,
        bubble_scale=args.bubble_scale,
        runtime_clip_max_sec=args.runtime_clip_max,
    )
    print(f"Saved {args.png} and {args.pdf}")


if __name__ == "__main__":
    main()
