"""
Grouped ablation bar chart: one color per dataset (primary palette), three conditions
per dataset with distinct fills — whitening off (//), random latents (crosshatch),
original / full pipeline (solid). "Original" matches whitening-on and SAE top-r (AUC-based) results.

Default numbers match the summary table:
| Dataset     | Whitening off | Random latents | Original |
| ----------- | ------------: | -------------: | -------: |
| VoxCeleb    |          95.5 |           58.0 |    99.89 |
| LibriSpeech |          98.0 |           60.0 |   100.00 |
| MLS-Spanish |          91.5 |           55.0 |    95.80 |
| Lahaja      |          74.5 |           49.0 |    82.31 |
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .style import apply_publication_style, create_figure, get_palette, save_plot

# (label, hatch) — None = solid fill
_CONDITIONS: tuple[tuple[str, str | None], ...] = (
    ("No Whitening", "//"),
    ("Random $r$", "xx"),
    ("Original", None),
)

_DEFAULT_DATASETS: tuple[str, ...] = (
    "VoxCeleb",
    "LibriSpeech",
    "MLS-Spanish",
    "Lahaja",
)

# Rows align with _DEFAULT_DATASETS: [whitening_off, random_latents, original]
_DEFAULT_VALUES = np.array(
    [
        [95.5, 58.0, 99.89],
        [98.0, 60.0, 100.0],
        [91.5, 55.0, 95.80],
        [74.5, 49.0, 82.31],
    ],
    dtype=float,
)

# Larger than publication rc defaults (~5–6 pt) for readability on this figure
_AXIS_LABEL_FONTSIZE = 12
_TICK_FONTSIZE = 11


def plot_ablations(
    datasets: Sequence[str] | None = None,
    values: np.ndarray | None = None,
    *,
    output_png: str | None = None,
    output_pdf: str | None = None,
    figsize: tuple[float, float] = (4.0, 2.55),
    ylabel: str = "ERB Score (%)",
    xlabel: str = "Dataset",
    palette_name: str = "primary",
) -> dict[str, Any]:
    """
    Compact multi-bar plot: x = datasets, grouped bars = three ablation conditions.
    Same face color for all bars in a dataset; hatches distinguish conditions.
    """
    if datasets is None:
        datasets = list(_DEFAULT_DATASETS)
    else:
        datasets = list(datasets)

    if values is None:
        arr = _DEFAULT_VALUES.copy()
    else:
        arr = np.asarray(values, dtype=float)
    if arr.shape != (len(datasets), len(_CONDITIONS)):
        raise ValueError(
            f"values must have shape ({len(datasets)}, {len(_CONDITIONS)}); got {arr.shape}"
        )

    apply_publication_style()
    plt.rcParams["hatch.linewidth"] = 0.35

    palette = get_palette(palette_name)
    colors = [palette[i % len(palette)] for i in range(len(datasets))]

    fig, ax = create_figure(figsize, nrows=1, ncols=1)

    n_ds = len(datasets)
    n_cond = len(_CONDITIONS)
    total_w = 0.78
    bar_w = total_w / n_cond
    x_centers = np.arange(n_ds, dtype=float)
    offsets = (np.arange(n_cond, dtype=float) - (n_cond - 1) / 2.0) * bar_w

    for j, (_, hatch) in enumerate(_CONDITIONS):
        xs = x_centers + offsets[j]
        for i in range(n_ds):
            kw: dict = {
                "width": bar_w * 0.92,
                "color": colors[i],
                "alpha": 0.85,
                "edgecolor": "black",
                "linewidth": 0.5,
            }
            if hatch:
                kw["hatch"] = hatch
            ax.bar(xs[i], arr[i, j], **kw)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        list(datasets),
        fontsize=_TICK_FONTSIZE,
        fontweight="bold",
    )
    if xlabel:
        ax.set_xlabel(
            xlabel,
            fontsize=_AXIS_LABEL_FONTSIZE,
            fontweight="bold",
        )
    ax.set_ylabel(
        ylabel,
        fontsize=_AXIS_LABEL_FONTSIZE,
        fontweight="bold",
    )
    ax.tick_params(axis="y", which="major", labelsize=_TICK_FONTSIZE)
    ax.set_ylim(0.0, max(105.0, float(np.nanmax(arr)) * 1.05))
    ax.grid(True, alpha=0.3, axis="y")

    legend_face = "#BDC3C7"
    leg_patches = []
    for label, hatch in _CONDITIONS:
        p_kw: dict = {
            "facecolor": legend_face,
            "edgecolor": "black",
            "linewidth": 0.5,
            "label": label,
        }
        if hatch:
            p_kw["hatch"] = hatch
        leg_patches.append(Patch(**p_kw))
    ax.legend(
        handles=leg_patches,
        loc="upper right",
        frameon=True,
        fancybox=False,
        ncol=3,
        columnspacing=0.7,
        handletextpad=0.4,
    )

    fig.tight_layout(pad=0.35)
    out: dict[str, Any] = {"fig": fig, "ax": ax}
    if output_png:
        os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
        save_plot(fig, output_png)
        out["png"] = output_png
    if output_pdf:
        os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
        save_plot(fig, output_pdf)
        out["pdf"] = output_pdf
    if output_png or output_pdf:
        plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ablation grouped bar chart.")
    parser.add_argument("--png", default="ablations.png", help="Output PNG path")
    parser.add_argument("--pdf", default="", help="Output PDF path (optional)")
    args = parser.parse_args()
    plot_ablations(
        output_png=args.png,
        output_pdf=args.pdf or None,
    )
    print(f"Saved {args.png}" + (f" and {args.pdf}" if args.pdf else ""))


if __name__ == "__main__":
    main()
