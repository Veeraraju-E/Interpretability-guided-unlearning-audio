"""
Latent selection / discriminability visualization utilities.

This module provides plots used throughout the unlearning pipeline to visualize
how separable forget vs retain activations are per latent dimension.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .style import apply_publication_style, create_figure, save_plot


def _safe_indices(selected_idx: Optional[Sequence[int]], m: int) -> np.ndarray:
    if not selected_idx:
        return np.array([], dtype=int)
    arr = np.asarray(list(selected_idx), dtype=int)
    arr = arr[(arr >= 0) & (arr < m)]
    if arr.size == 0:
        return arr
    # de-dup while preserving order
    _, first = np.unique(arr, return_index=True)
    return arr[np.sort(first)]


def benjamini_hochberg_qvalues(
    pvals: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute Benjamini–Hochberg FDR-adjusted q-values.

    Args:
        pvals: array of p-values (shape: (m,))
        valid_mask: optional boolean mask of which hypotheses to include in BH.
            If provided, invalid entries get q=1.0.
    """
    pvals = np.asarray(pvals, dtype=float).reshape(-1)
    m = int(pvals.shape[0])
    if m == 0:
        return pvals

    if valid_mask is None:
        valid_mask = np.ones(m, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid_mask.shape[0] != m:
            raise ValueError(f"valid_mask length mismatch: {valid_mask.shape[0]} vs {m}")

    qvals = np.ones(m, dtype=float)
    if not np.any(valid_mask):
        return qvals

    pv = np.clip(pvals[valid_mask], 0.0, 1.0)
    order = np.argsort(pv)
    pv_sorted = pv[order]
    n = int(pv_sorted.shape[0])

    # BH: q_i = min_{j>=i} (n * p_j / j), in sorted order.
    ranks = np.arange(1, n + 1, dtype=float)
    q_sorted = (n * pv_sorted) / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    valid_indices = np.where(valid_mask)[0]
    qvals[valid_indices[order]] = q_sorted
    return qvals


def plot_discriminability_landscape(
    stats: Dict[str, Any],
    selected_idx: Optional[Sequence[int]] = None,
    fdr_alpha: float = 0.05,
    auc_eps: float = 1e-6,
    out_path_pdf: Optional[str] = None,
    out_path_png: Optional[str] = None,
    title: str = "Discriminability Landscape",
    highlight_top_n: int = 20,
) -> None:
    """
    Plot a per-latent discriminability summary.

    Expected stats keys:
        - 'auc': array-like of shape (m,)
        - 'pval': array-like of shape (m,) (optional, but recommended)

    The plot sorts latents by discriminability \(|AUC-0.5|\) and colors points by
    \(-\\log_{10}(q)\) where q are BH-FDR adjusted p-values.
    """
    apply_publication_style()

    auc = np.asarray(stats.get("auc", []), dtype=float).reshape(-1)
    if auc.size == 0:
        raise ValueError("`stats` must contain non-empty key 'auc' for plotting.")

    pvals = stats.get("pval", None)
    pvals_arr = None if pvals is None else np.asarray(pvals, dtype=float).reshape(-1)
    if pvals_arr is not None and pvals_arr.shape[0] != auc.shape[0]:
        pvals_arr = None

    m = int(auc.shape[0])
    disc = np.abs(auc - 0.5)
    valid = disc > float(auc_eps)

    qvals = (
        benjamini_hochberg_qvalues(pvals_arr, valid_mask=valid)
        if pvals_arr is not None
        else np.ones(m, dtype=float)
    )

    # Sort by discriminability (descending)
    order = np.argsort(-disc)
    disc_s = disc[order]
    q_s = qvals[order]
    idx_s = np.arange(m, dtype=int)[order]

    neglogq = -np.log10(np.clip(q_s, 1e-10, 1.0))
    vmax = float(min(np.nanmax(neglogq), 10.0)) if neglogq.size else 1.0

    sel = _safe_indices(selected_idx, m)
    sel_set = set(sel.tolist())
    sel_mask = np.array([i in sel_set for i in idx_s], dtype=bool)

    fig, ax = create_figure((10, 4))
    x = np.arange(m, dtype=int)

    sc = ax.scatter(
        x,
        disc_s,
        c=neglogq,
        cmap="viridis",
        s=10,
        alpha=0.75,
        edgecolors="none",
        vmin=0.0,
        vmax=vmax,
    )

    if sel_mask.any():
        ax.scatter(
            x[sel_mask],
            disc_s[sel_mask],
            c=neglogq[sel_mask],
            cmap="viridis",
            s=35,
            alpha=0.95,
            edgecolors="black",
            linewidths=0.7,
            vmin=0.0,
            vmax=vmax,
            zorder=5,
            label=f"Selected (r={int(sel_mask.sum())})",
        )

    # Optional highlight of top-N ranks (outline), excluding selected
    topn = int(max(0, highlight_top_n))
    top_only_any = False
    if topn > 0:
        top_mask = np.zeros(m, dtype=bool)
        top_mask[: min(topn, m)] = True
        top_only = top_mask & ~sel_mask
        if top_only.any():
            top_only_any = True
            ax.scatter(
                x[top_only],
                disc_s[top_only],
                c=neglogq[top_only],
                cmap="viridis",
                s=18,
                alpha=0.8,
                edgecolors="orange",
                linewidths=0.8,
                vmin=0.0,
                vmax=vmax,
                zorder=4,
                label=f"Top-{min(topn, m)}",
            )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(r"Latent rank (sorted by $|AUC-0.5|$)", fontweight="bold")
    ax.set_ylabel(r"Discriminability  $|AUC-0.5|$", fontweight="bold")
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"$-\log_{10}(q)$", fontweight="bold")

    # Summary textbox
    n_valid = int(valid.sum())
    n_sig = int((qvals <= float(fdr_alpha)).sum()) if pvals_arr is not None else 0
    r = int(len(sel))
    summary = f"m={m} | valid={n_valid} | q≤{fdr_alpha:g}: {n_sig} | r={r}"
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    if sel_mask.any() or top_only_any:
        ax.legend(loc="best", frameon=True, fancybox=False, shadow=False)

    plt.tight_layout()

    for out_path in (out_path_pdf, out_path_png):
        if not out_path:
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_plot(fig, out_path, dpi=600)

    plt.close(fig)

