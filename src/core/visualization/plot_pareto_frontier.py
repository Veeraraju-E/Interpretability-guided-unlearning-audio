"""
Pareto frontier bubble chart for ablation results.
Axes: x = ERB_after, y = (linear_probe_auc_after - 0.5), bubble size ∝ 1/runtime_sec.
Shapes: igbp=hexagon, inlp=star, leopard=triangle, sal=circle, leace=square.
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.lines as mlines

from .style import apply_publication_style, save_plot, get_palette


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
    bubble_scale: float = 800.0,
    runtime_clip_max_sec: float = 3600.0,
) -> dict:
    """
    Read ablation results CSV, plot Pareto frontier bubble chart.
    Axes: x = ERB_after, y = (linear_probe_auc_after - 0.5), bubble size ∝ 1/runtime_sec (mapped to a visible range).
    Returns dict with saved paths.
    """
    if not os.path.isfile(results_csv):
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")
    df = pd.read_csv(results_csv)
    for col in (
        "linear_probe_auc_after",
        "ERB_after",
        "runtime_sec",
        "method",
        "sae_enabled",
    ):
        if col not in df.columns:
            raise ValueError(f"CSV must contain column: {col}")

    auc_after = np.array(df["linear_probe_auc_after"].astype(float))
    auc_minus_half = auc_after - 0.5
    erb = np.array([_to_pct(v) for v in df["ERB_after"].astype(float)])
    runtime = np.array(df["runtime_sec"].astype(float))
    runtime = np.clip(runtime, 1e-6, runtime_clip_max_sec)
    inv_runtime = 1.0 / runtime
    # Map 1/runtime linearly to [size_min, size_max] so faster runs get larger points
    inv_min, inv_max = inv_runtime.min(), inv_runtime.max()
    if inv_max <= inv_min:
        sizes = np.full_like(runtime, bubble_scale)
    else:
        size_min = bubble_scale * 0.4
        size_max = bubble_scale * 3.0
        sizes = size_min + (size_max - size_min) * (inv_runtime - inv_min) / (inv_max - inv_min)
    sizes = sizes * 0.5

    apply_publication_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))

    def _is_true(v) -> bool:
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        return s in {"1", "true", "t", "yes", "y"}

    method_raw = df["method"].astype(str)
    method_key = method_raw.str.strip().str.lower()
    sae_enabled = df["sae_enabled"].apply(_is_true).astype(bool)
    variant_key = np.where(sae_enabled.values, "sae+" + method_key.values, method_key.values)
    df_plot = df.copy()
    df_plot["__method_raw"] = method_raw.values
    df_plot["__method_key"] = method_key.values
    df_plot["__sae"] = sae_enabled.values
    df_plot["__variant"] = variant_key

    raw_palette = get_palette("primary")
    # Skip orange; replace with teal so we don't duplicate green (palette already has #4CAF50)
    palette = [
        "#00897B" if c.upper() == "#FF5722" else c
        for c in raw_palette
    ]
    method_to_color: dict[str, str] = {}
    unique_methods = list(pd.unique(df_plot["__method_key"]))
    for mi, m in enumerate(unique_methods):
        method_to_color[m] = palette[mi % len(palette)]

    # Method -> marker: igbp=hexagon, inlp=star, leopard=triangle, sal=circle, leace=square
    method_to_marker: dict[str, str] = {
        "igbp": "h",
        "inlp": "*",
        "leopard": "^",
        "sal": "o",
        "leace": "s",
    }
    for m in unique_methods:
        if m not in method_to_marker:
            method_to_marker[m] = "o"

    label_font = {"fontsize": 12, "fontweight": "bold", "fontfamily": "serif"}
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Collect (label, color, marker) per (method, sae) for ordered legend.
    # SAE = filled marker + black edge; no SAE = hollow (white face + method-colored edge).
    # Draw hollow first, then filled so filled (SAE) points appear on top.
    group_info: dict[Tuple[str, bool], Tuple[str, str, str]] = {}
    groups = list(df_plot.groupby(["__method_key", "__sae"], sort=False))
    for (mkey, sae_flag), g in groups:
        label = f"{'SAE+' if sae_flag else ''}{g['__method_raw'].iloc[0]}"
        group_info[(mkey, sae_flag)] = (label, method_to_color[mkey], method_to_marker[mkey])
    for (mkey, sae_flag), g in groups:
        if sae_flag:
            continue
        idx = g.index.to_numpy()
        ax.scatter(
            erb[idx], auc_minus_half[idx], s=sizes[idx], facecolors="white",
            marker=method_to_marker[mkey], edgecolors=method_to_color[mkey],
            linewidths=1.5, zorder=1,
        )
    for (mkey, sae_flag), g in groups:
        if not sae_flag:
            continue
        idx = g.index.to_numpy()
        ax.scatter(
            erb[idx], auc_minus_half[idx], s=sizes[idx], c=method_to_color[mkey],
            marker=method_to_marker[mkey], alpha=1, edgecolors="black",
            linewidths=1.0, zorder=2,
        )

    # Legend: column 1 = inlp, sal, igbp; column 2 = leopard, leace (each with SAE variant)
    legend_order = [
        ("inlp", False), ("inlp", True), ("sal", False), ("sal", True),
        ("igbp", False), ("igbp", True), ("leopard", False), ("leopard", True),
        ("leace", False), ("leace", True),
    ]
    legend_handles: List[mlines.Line2D] = []
    for mkey, sae_flag in legend_order:
        key = (mkey, sae_flag)
        if key not in group_info:
            continue
        label, color, marker = group_info[key]
        if sae_flag:
            leg = mlines.Line2D(
                [0], [0], marker=marker, color="w", markerfacecolor=color,
                markeredgecolor="black", markeredgewidth=1.0, markersize=7, label=label,
            )
        else:
            leg = mlines.Line2D(
                [0], [0], marker=marker, color="w", markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.5, markersize=7, label=label,
            )
        legend_handles.append(leg)

    ax.set_xlabel("ERB (%)", **label_font)
    ax.set_ylabel("AUC separation from 0.5", **label_font)
    ax.legend(handles=legend_handles, loc="upper left", ncol=2, fontsize=11, frameon=True)
    ax.grid(True, alpha=0.5)

    finite_x = erb[np.isfinite(erb)]
    finite_y = auc_minus_half[np.isfinite(auc_minus_half)]
    # if finite_x.size > 0:
    #     xmin, xmax = float(finite_x.min()), float(finite_x.max())
    #     xpad = max(2.0, 0.05 * (xmax - xmin + 1e-12))
    ax.set_xlim(0.0, 100.0)
    if finite_y.size > 0:
        ymin, ymax = float(finite_y.min()), float(finite_y.max())
        ypad = max(0.02, 0.05 * (ymax - ymin + 1e-12))
        ax.set_ylim(ymin - ypad, 0.5)
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
    parser.add_argument("--bubble-scale", type=float, default=800.0)
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
