"""
Hyperparameter analysis plotting utilities.

Used by `src/orchestration/hyperparameter_analysis.py` to visualize how SAE
hyperparameters (expansion_factor, topk) affect downstream unlearning metrics.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from .style import apply_publication_style, create_figure, save_plot


def _erb_score(fa_after: float, ra_after: float) -> float:
    """
    ERB score used throughout this repo:
        ERB = 2*(100-FA)*RA / ((100-FA)+RA)
    where FA/RA are in percentage points.
    """
    fa_pct = float(fa_after) * 100.0 if float(fa_after) <= 1.0 else float(fa_after)
    ra_pct = float(ra_after) * 100.0 if float(ra_after) <= 1.0 else float(ra_after)
    nr = 2.0 * (100.0 - fa_pct) * ra_pct
    dr = (100.0 - fa_pct) + ra_pct
    return float(nr / dr) if dr != 0 else 0.0


def create_hyperparameter_plot(results_csv_path: str, output_dir: str) -> Dict[str, str]:
    """
    Create a publication-style heatmap figure for hyperparameter analysis.

    Args:
        results_csv_path: Path to `hyperparameter_analysis_results.csv`
        output_dir: Directory where plots will be saved

    Returns:
        Dict with output paths (png/pdf)
    """
    apply_publication_style()

    import pandas as pd
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except Exception as e:  # pragma: no cover
        raise ImportError("seaborn is required for hyperparameter heatmaps. Install seaborn.") from e

    df = pd.read_csv(results_csv_path)
    if df.empty:
        raise ValueError(f"No rows found in results CSV: {results_csv_path}")

    # Basic cleaning
    for col in ("expansion_factor", "topk"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {results_csv_path}")

    # Prefer *_after metrics; fall back gracefully if absent
    required_metrics = ["FA_after", "RA_after"]
    for m in required_metrics:
        if m not in df.columns:
            raise ValueError(f"Missing required metric column '{m}' in {results_csv_path}")

    df = df.copy()
    df["expansion_factor"] = df["expansion_factor"].astype(int)
    df["topk"] = df["topk"].astype(int)

    # Drop failures / NaNs
    df = df.dropna(subset=["FA_after", "RA_after"])
    if df.empty:
        raise ValueError("All hyperparameter rows are NaN after filtering; nothing to plot.")

    df["ERB_after"] = [
        _erb_score(fa, ra) for fa, ra in zip(df["FA_after"].astype(float).values, df["RA_after"].astype(float).values)
    ]
    if "dead_latents_pct" in df.columns:
        df["dead_latents_pct"] = df["dead_latents_pct"].astype(float)

    # Pivots
    erb_pivot = df.pivot_table(index="expansion_factor", columns="topk", values="ERB_after", aggfunc="mean")
    ra_pivot = df.pivot_table(index="expansion_factor", columns="topk", values="RA_after", aggfunc="mean")
    fa_pivot = df.pivot_table(index="expansion_factor", columns="topk", values="FA_after", aggfunc="mean")
    dead_pivot = (
        df.pivot_table(index="expansion_factor", columns="topk", values="dead_latents_pct", aggfunc="mean")
        if "dead_latents_pct" in df.columns
        else None
    )

    # Sort axes (nice order)
    erb_pivot = erb_pivot.sort_index().reindex(sorted(erb_pivot.columns), axis=1)
    ra_pivot = ra_pivot.sort_index().reindex(sorted(ra_pivot.columns), axis=1)
    fa_pivot = fa_pivot.sort_index().reindex(sorted(fa_pivot.columns), axis=1)
    if dead_pivot is not None:
        dead_pivot = dead_pivot.sort_index().reindex(sorted(dead_pivot.columns), axis=1)

    # Figure
    fig, axes = create_figure((12, 9), nrows=2, ncols=2)
    ax00, ax01 = axes[0]  # type: ignore[index]
    ax10, ax11 = axes[1]  # type: ignore[index]

    sns.heatmap(
        erb_pivot,
        ax=ax00,
        cmap="viridis",
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "ERB (after)"},
        linewidths=0.3,
        linecolor="white",
    )
    ax00.set_title("ERB Score (After Unlearning)", fontweight="bold")
    ax00.set_xlabel("topk", fontweight="bold")
    ax00.set_ylabel("expansion_factor", fontweight="bold")

    sns.heatmap(
        ra_pivot * 100.0,
        ax=ax01,
        cmap="YlGn",
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "RA_after (%)"},
        linewidths=0.3,
        linecolor="white",
        vmin=0,
        vmax=100,
    )
    ax01.set_title("Retain Accuracy (After)", fontweight="bold")
    ax01.set_xlabel("topk", fontweight="bold")
    ax01.set_ylabel("")

    sns.heatmap(
        fa_pivot * 100.0,
        ax=ax10,
        cmap="YlOrRd",
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "FA_after (%)"},
        linewidths=0.3,
        linecolor="white",
        vmin=0,
        vmax=100,
    )
    ax10.set_title("Forget Accuracy (After)", fontweight="bold")
    ax10.set_xlabel("topk", fontweight="bold")
    ax10.set_ylabel("expansion_factor", fontweight="bold")

    if dead_pivot is not None:
        sns.heatmap(
            dead_pivot,
            ax=ax11,
            cmap="mako",
            annot=True,
            fmt=".1f",
            cbar_kws={"label": "Dead latents (%)"},
            linewidths=0.3,
            linecolor="white",
            vmin=0,
            vmax=100,
        )
        ax11.set_title("Dead Latents (%)", fontweight="bold")
        ax11.set_xlabel("topk", fontweight="bold")
        ax11.set_ylabel("")
    else:
        # If dead_latents_pct not present, use this panel for before/after AUC delta if available.
        ax11.axis("off")
        ax11.text(
            0.0,
            1.0,
            "Note: `dead_latents_pct` not found in CSV.\n"
            "This panel is intentionally left blank.",
            va="top",
            ha="left",
            fontsize=10,
        )

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "hyperparameter_analysis.png")
    pdf_path = os.path.join(output_dir, "hyperparameter_analysis.pdf")
    save_plot(fig, png_path, dpi=600)
    save_plot(fig, pdf_path, dpi=600)
    plt.close(fig)

    return {"png": png_path, "pdf": pdf_path}

