"""
Visual abstract generation.

The original repository workflow expects a "visual abstract" panel to be saved
alongside other plots for each forget run. This implementation is intentionally
robust: if pre-rendered plots exist (UMAP comparison / discriminability PNGs),
they are embedded; otherwise, it falls back to lightweight plots from the
provided arrays.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .style import apply_publication_style, create_figure, save_plot


def _exists(path: Optional[str]) -> bool:
    return bool(path) and os.path.exists(path) and os.path.isfile(path)


def _safe_selected(selected_idx: Sequence[int], m: int) -> np.ndarray:
    arr = np.array(list(selected_idx), dtype=int) if selected_idx is not None else np.array([], dtype=int)
    arr = arr[(arr >= 0) & (arr < m)]
    if arr.size == 0:
        return arr
    _, first = np.unique(arr, return_index=True)
    return arr[np.sort(first)]


def _get_forget_retain_masks(
    labels: Sequence[Any],
    forget_label: str,
    sets: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (forget_mask, retain_mask). Prefers `sets` (forget/retain:test) when
    provided; otherwise derives from labels.
    """
    n = len(labels)
    if sets is not None and len(sets) == n:
        sets_arr = np.asarray(sets, dtype=str)
        # Prefer test split if present
        is_test = np.char.endswith(sets_arr, ":test")
        if bool(is_test.any()):
            forget_mask = is_test & np.char.startswith(sets_arr, "forget")
            retain_mask = is_test & np.char.startswith(sets_arr, "retain")
        else:
            forget_mask = np.char.startswith(sets_arr, "forget")
            retain_mask = np.char.startswith(sets_arr, "retain")
        # If malformed sets, fall back to label-based masks
        if bool((forget_mask | retain_mask).any()):
            return forget_mask.astype(bool), retain_mask.astype(bool)

    labels_str = np.asarray([str(l) for l in labels], dtype=str)
    forget_mask = labels_str == str(forget_label)
    retain_mask = ~forget_mask
    return forget_mask.astype(bool), retain_mask.astype(bool)


def create_visual_abstract(
    *,
    Z_all: np.ndarray,
    E_edit: np.ndarray,
    E_before: np.ndarray,
    E_after: np.ndarray,
    mu: np.ndarray,
    W: np.ndarray,
    U: np.ndarray,
    labels: Sequence[Any],
    forget_label: str,
    stats: Dict[str, Any],
    selected_idx: Sequence[int],
    dec_w: np.ndarray,
    A_all: np.ndarray,
    save_dir: str,
    ridge: float = 1e-6,
    max_points: int = 1500,
    k_plot: int = 5,
    seed: int = 0,
    overlay_top_n: int = 8,
    hist_top_n: int = 8,
    ablation: bool = True,
    overlay_xlim: Optional[Tuple[float, float]] = None,
    overlay_ylim: Optional[Tuple[float, float]] = None,
    sets: Optional[Sequence[str]] = None,
    umap_comparison_path: Optional[str] = None,
    discriminability_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Create and save a compact "visual abstract" figure.

    Notes:
        Many arguments are included for API compatibility with existing
        orchestration code. Not all are used in this lightweight panel.
    """
    # Keep API compatibility; not all inputs are needed for this condensed panel
    _ = (Z_all, E_edit, mu, W, U, dec_w, ridge, k_plot, ablation, overlay_xlim, overlay_ylim)

    apply_publication_style()
    os.makedirs(save_dir, exist_ok=True)

    n = len(labels)
    if E_before.shape[0] != n or E_after.shape[0] != n:
        raise ValueError("`labels`, `E_before`, and `E_after` must have the same number of rows.")
    if A_all.shape[0] != n:
        raise ValueError("`A_all` must have the same number of rows as `labels`.")

    forget_mask, retain_mask = _get_forget_retain_masks(labels, forget_label, sets=sets)

    # Selected feature ranking
    auc = np.asarray(stats.get("auc", []), dtype=float)
    cohens_d = np.asarray(stats.get("cohens_d", []), dtype=float) if "cohens_d" in stats else None
    m = int(A_all.shape[1])
    sel = _safe_selected(selected_idx, m)

    if auc.size == m:
        disc = np.abs(auc - 0.5)
        score = disc
        score_name = r"$|AUC-0.5|$"
    elif cohens_d is not None and cohens_d.size == m:
        score = np.abs(cohens_d)
        score_name = r"$|d|$"
    else:
        score = np.zeros(m, dtype=float)
        score_name = "score"

    if sel.size > 0:
        sel_scores = score[sel]
        order = np.argsort(-sel_scores)
        sel_sorted = sel[order]
    else:
        sel_sorted = np.array([], dtype=int)

    top_n = int(max(1, overlay_top_n))
    top_sel = sel_sorted[: min(top_n, sel_sorted.size)]

    # Activation summaries for selected latents
    # Prefer a small panel using the top few latents (hist_top_n)
    hist_n = int(max(1, hist_top_n))
    hist_sel = sel_sorted[: min(hist_n, sel_sorted.size)]
    if hist_sel.size == 0 and m > 0:
        hist_sel = np.array([int(np.argmax(score))], dtype=int)

    # Subsample points for any in-panel scatter (fallback case)
    rng = np.random.default_rng(int(seed))
    max_points = int(max(100, max_points))
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
    else:
        idx = np.arange(n, dtype=int)

    # Figure: 2x2
    fig, axes = create_figure((13, 9), nrows=2, ncols=2)
    ax00, ax01 = axes[0]  # type: ignore[index]
    ax10, ax11 = axes[1]  # type: ignore[index]

    # Panel A: UMAP comparison (embed if available)
    if _exists(umap_comparison_path):
        import matplotlib.image as mpimg

        img = mpimg.imread(str(umap_comparison_path))
        ax00.imshow(img)
        ax00.axis("off")
        ax00.set_title("UMAP (Before vs After)", fontweight="bold")
    else:
        # Fallback: quick 2D scatter of AFTER embeddings (first two dims)
        ax00.scatter(
            E_after[idx][retain_mask[idx], 0],
            E_after[idx][retain_mask[idx], 1],
            s=8,
            alpha=0.6,
            label="Retain",
            color="#0066CC",
        )
        ax00.scatter(
            E_after[idx][forget_mask[idx], 0],
            E_after[idx][forget_mask[idx], 1],
            s=10,
            alpha=0.8,
            label="Forget",
            color="#FF0000",
        )
        ax00.set_title("Embeddings (After) – fallback", fontweight="bold")
        ax00.set_xlabel("dim 1", fontweight="bold")
        ax00.set_ylabel("dim 2", fontweight="bold")
        ax00.grid(True, alpha=0.3)
        ax00.legend(loc="best", frameon=True, fancybox=False, shadow=False)

    # Panel B: Discriminability landscape (embed if available)
    if _exists(discriminability_path):
        import matplotlib.image as mpimg

        img = mpimg.imread(str(discriminability_path))
        ax01.imshow(img)
        ax01.axis("off")
        ax01.set_title("Discriminability Landscape", fontweight="bold")
    else:
        # Fallback: plot score across latents
        x = np.arange(m)
        ax01.scatter(x, score, s=8, alpha=0.6, color="#3F51B5")
        if sel.size > 0:
            ax01.scatter(sel, score[sel], s=16, alpha=0.9, color="#E91E63", edgecolors="black", linewidths=0.3)
        ax01.set_title("Discriminability (fallback)", fontweight="bold")
        ax01.set_xlabel("latent index", fontweight="bold")
        ax01.set_ylabel(score_name, fontweight="bold")
        ax01.grid(True, alpha=0.3)

    # Panel C: top selected feature scores
    if top_sel.size > 0:
        y = score[top_sel]
        ax10.bar(np.arange(top_sel.size), y, color="#E91E63", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax10.set_xticks(np.arange(top_sel.size))
        ax10.set_xticklabels([str(int(i)) for i in top_sel], rotation=0)
        ax10.set_title(f"Top selected latents by {score_name}", fontweight="bold")
        ax10.set_xlabel("latent index", fontweight="bold")
        ax10.set_ylabel(score_name, fontweight="bold")
        ax10.grid(True, axis="y", alpha=0.3)
    else:
        ax10.axis("off")
        ax10.text(0.0, 1.0, "No selected latents provided.", va="top", ha="left")

    # Panel D: forget vs retain activation means for a few latents
    if hist_sel.size > 0 and (forget_mask.any() and retain_mask.any()):
        f_means = np.nanmean(A_all[forget_mask][:, hist_sel], axis=0)
        r_means = np.nanmean(A_all[retain_mask][:, hist_sel], axis=0)
        x = np.arange(hist_sel.size, dtype=float)
        w = 0.38
        ax11.bar(x - w / 2, r_means, width=w, color="#0066CC", alpha=0.85, edgecolor="black", linewidth=0.5, label="Retain")
        ax11.bar(x + w / 2, f_means, width=w, color="#FF0000", alpha=0.85, edgecolor="black", linewidth=0.5, label="Forget")
        ax11.set_xticks(x)
        ax11.set_xticklabels([str(int(i)) for i in hist_sel])
        ax11.set_title("Mean activation (test if available)", fontweight="bold")
        ax11.set_xlabel("latent index", fontweight="bold")
        ax11.set_ylabel("mean activation", fontweight="bold")
        ax11.grid(True, axis="y", alpha=0.3)
        ax11.legend(loc="best", frameon=True, fancybox=False, shadow=False)
    else:
        ax11.axis("off")
        summary = [
            f"Forget label: {forget_label}",
            f"Selected latents: {int(sel.size)}",
            f"Max points: {int(max_points)}",
        ]
        ax11.text(0.0, 1.0, "\n".join(summary), va="top", ha="left")

    fig.suptitle(f"Visual Abstract – forget={forget_label}", fontweight="bold", y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    png_path = os.path.join(save_dir, "visual_abstract.png")
    pdf_path = os.path.join(save_dir, "visual_abstract.pdf")
    save_plot(fig, png_path, dpi=600)
    save_plot(fig, pdf_path, dpi=600)
    plt.close(fig)

    return {"png": png_path, "pdf": pdf_path}

