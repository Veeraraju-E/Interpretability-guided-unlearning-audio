"""
Embedding visualization functions for unlearning evaluation.

Creates 2D visualizations comparing embeddings before and after unlearning.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .style import apply_publication_style, create_figure, save_plot


def _try_compute_umap_2d(
    combined_embeddings: np.ndarray, *, seed: int = 42
) -> Tuple[Optional[np.ndarray], Optional[Exception]]:
    """
    Compute UMAP(2D) coordinates for a combined embedding matrix.

    Notes:
        We keep the UMAP import inside this function. Some environments have a
        broken `umap` import due to optional parametric UMAP / torchvision
        dependency issues; importing lazily prevents module import failures.
    """
    try:
        from umap import UMAP  # type: ignore

        reducer = UMAP(
            n_components=2,
            random_state=seed,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
        )
        coords = reducer.fit_transform(combined_embeddings)
        return coords, None
    except Exception as e:  # pragma: no cover
        return None, e


def _compute_pca_2d(combined_embeddings: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    return PCA(n_components=2).fit_transform(combined_embeddings)


def _scatter_forget_retain(
    ax: plt.Axes,
    coords_2d: np.ndarray,
    retain_mask: np.ndarray,
    forget_mask: np.ndarray,
    *,
    s_retain: float,
    s_forget: float,
) -> None:
    ax.scatter(
        coords_2d[retain_mask, 0],
        coords_2d[retain_mask, 1],
        c="#0066CC",
        label="Retain",
        alpha=0.6,
        s=s_retain,
        edgecolors="black",
        linewidths=0.5,
    )
    ax.scatter(
        coords_2d[forget_mask, 0],
        coords_2d[forget_mask, 1],
        c="#FF0000",
        label="Forget",
        alpha=0.8,
        s=s_forget,
        edgecolors="black",
        linewidths=0.5,
    )


def create_embedding_visualizations(
    embeddings_before: np.ndarray,
    embeddings_after: np.ndarray,
    labels: List,
    forget_labels: List,
    save_dir: str,
    title_prefix: str = "",
    *,
    seed: int = 42,
    fallback_to_pca: bool = True,
) -> Dict[str, str]:
    """
    Create 2D visualizations comparing embeddings before and after unlearning.

    Outputs:
        - `umap_comparison.png` (+ individual before/after) if UMAP is available
        - otherwise, if `fallback_to_pca=True`, falls back to PCA and writes
          `pca_comparison.png` (+ individual before/after)

    Returns:
        Dictionary mapping plot names to file paths
    """
    apply_publication_style()

    os.makedirs(save_dir, exist_ok=True)
    forget_labels_set = {str(fl) for fl in forget_labels}

    # Create binary labels: Forget vs Retain
    binary_labels = ["Forget" if str(l) in forget_labels_set else "Retain" for l in labels]
    forget_mask = np.array([l == "Forget" for l in binary_labels], dtype=bool)
    retain_mask = np.array([l == "Retain" for l in binary_labels], dtype=bool)

    # Fit projection on combined embeddings for consistent coordinates
    print(f"[VIZ] Fitting 2D projection on {len(embeddings_before)} samples...")
    combined_embeddings = np.vstack([embeddings_before, embeddings_after])

    coords_2d, umap_err = _try_compute_umap_2d(combined_embeddings, seed=seed)
    if coords_2d is not None:
        method = "umap"
        xlab, ylab = "UMAP Dimension 1", "UMAP Dimension 2"
        comparison_fname = "umap_comparison.png"
        before_fname = "umap_before.png"
        after_fname = "umap_after.png"
    else:
        if not fallback_to_pca:
            raise ImportError(
                "Failed to import/compute UMAP for embedding visualization. "
                "Install/fix `umap-learn`, or call with `fallback_to_pca=True`."
            ) from umap_err
        print(f"[VIZ] UMAP unavailable ({umap_err}); falling back to PCA(2D).")
        coords_2d = _compute_pca_2d(combined_embeddings)
        method = "pca"
        xlab, ylab = "PCA Component 1", "PCA Component 2"
        comparison_fname = "pca_comparison.png"
        before_fname = "pca_before.png"
        after_fname = "pca_after.png"

    coords_before = coords_2d[: len(embeddings_before)]
    coords_after = coords_2d[len(embeddings_before) :]

    # -------- Comparison figure --------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    _scatter_forget_retain(
        ax1, coords_before, retain_mask, forget_mask, s_retain=20, s_forget=30
    )
    ax1.set_title(f"{title_prefix} - Before Unlearning", fontsize=15, fontweight="bold")
    ax1.set_xlabel(xlab, fontsize=15, fontweight="bold")
    ax1.set_ylabel(ylab, fontsize=15, fontweight="bold")
    ax1.tick_params(axis="both", labelsize=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=13, frameon=True, fancybox=False, shadow=False)

    _scatter_forget_retain(
        ax2, coords_after, retain_mask, forget_mask, s_retain=20, s_forget=30
    )
    ax2.set_title(f"{title_prefix} - After Unlearning", fontsize=15, fontweight="bold")
    ax2.set_xlabel(xlab, fontsize=15, fontweight="bold")
    ax2.set_ylabel(ylab, fontsize=15, fontweight="bold")
    ax2.tick_params(axis="both", labelsize=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=13, frameon=True, fancybox=False, shadow=False)

    plt.tight_layout()

    comparison_path = os.path.join(save_dir, comparison_fname)
    save_plot(fig, comparison_path, dpi=600)
    plt.close(fig)

    # -------- Individual: Before --------
    fig_before, ax_before = create_figure((7, 6))
    _scatter_forget_retain(
        ax_before, coords_before, retain_mask, forget_mask, s_retain=20, s_forget=30
    )
    ax_before.set_title(f"{title_prefix} - Before Unlearning", fontsize=15, fontweight="bold")
    ax_before.set_xlabel(xlab, fontsize=15, fontweight="bold")
    ax_before.set_ylabel(ylab, fontsize=15, fontweight="bold")
    ax_before.tick_params(axis="both", labelsize=15)
    ax_before.grid(True, alpha=0.3)
    ax_before.legend(loc="best", fontsize=13, frameon=True, fancybox=False, shadow=False)
    plt.tight_layout()

    before_path = os.path.join(save_dir, before_fname)
    save_plot(fig_before, before_path, dpi=300)
    plt.close(fig_before)

    # -------- Individual: After --------
    fig_after, ax_after = create_figure((7, 6))
    _scatter_forget_retain(
        ax_after, coords_after, retain_mask, forget_mask, s_retain=20, s_forget=30
    )
    ax_after.set_title(f"{title_prefix} - After Unlearning", fontsize=15, fontweight="bold")
    ax_after.set_xlabel(xlab, fontsize=15, fontweight="bold")
    ax_after.set_ylabel(ylab, fontsize=15, fontweight="bold")
    ax_after.tick_params(axis="both", labelsize=15)
    ax_after.grid(True, alpha=0.3)
    ax_after.legend(loc="best", fontsize=13, frameon=True, fancybox=False, shadow=False)
    plt.tight_layout()

    after_path = os.path.join(save_dir, after_fname)
    save_plot(fig_after, after_path, dpi=300)
    plt.close(fig_after)

    return {
        "summary": comparison_path,
        "comparison": comparison_path,
        "before": before_path,
        "after": after_path,
        # extra metadata
        "method": method,
    }


def create_embedding_summary_stats(
    embeddings_before: np.ndarray,
    embeddings_after: np.ndarray,
    labels: List,
    forget_labels: List,
    save_dir: str,
) -> Dict:
    """
    Compute summary statistics for embeddings before and after unlearning.

    Args:
        embeddings_before: Normalized embeddings before unlearning (N x d)
        embeddings_after: Normalized embeddings after unlearning (N x d)
        labels: List of class labels for each sample (length N)
        forget_labels: List of forget class labels
        save_dir: Directory (not used but kept for API consistency)

    Returns:
        Dictionary with summary statistics
    """
    _ = save_dir  # kept for API consistency

    forget_labels_set = {str(fl) for fl in forget_labels}
    labels_str = [str(l) for l in labels]

    # Separate forget and retain samples
    forget_mask = np.array([l in forget_labels_set for l in labels_str], dtype=bool)
    retain_mask = ~forget_mask

    stats: Dict[str, float | int] = {}

    # Overall statistics
    stats["n_samples"] = int(len(embeddings_before))
    stats["n_forget"] = int(forget_mask.sum())
    stats["n_retain"] = int(retain_mask.sum())
    stats["embedding_dim"] = int(embeddings_before.shape[1])

    def compute_mean_pairwise_distance(embeddings: np.ndarray) -> float:
        """Compute mean pairwise cosine distance within a set of embeddings."""
        if len(embeddings) < 2:
            return 0.0
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        similarity = np.dot(normalized, normalized.T)
        distance = 1 - similarity
        triu_indices = np.triu_indices(len(distance), k=1)
        return float(distance[triu_indices].mean())

    # Before unlearning
    stats["forget_intra_class_dist_before"] = (
        compute_mean_pairwise_distance(embeddings_before[forget_mask]) if forget_mask.any() else 0.0
    )
    stats["retain_intra_class_dist_before"] = (
        compute_mean_pairwise_distance(embeddings_before[retain_mask]) if retain_mask.any() else 0.0
    )

    # After unlearning
    stats["forget_intra_class_dist_after"] = (
        compute_mean_pairwise_distance(embeddings_after[forget_mask]) if forget_mask.any() else 0.0
    )
    stats["retain_intra_class_dist_after"] = (
        compute_mean_pairwise_distance(embeddings_after[retain_mask]) if retain_mask.any() else 0.0
    )

    # Mean norms
    stats["mean_norm_before"] = float(np.linalg.norm(embeddings_before, axis=1).mean())
    stats["mean_norm_after"] = float(np.linalg.norm(embeddings_after, axis=1).mean())

    # Change in forget class spread (higher = more scattered after unlearning)
    if forget_mask.sum() > 1:
        forget_spread_before = float(np.std(embeddings_before[forget_mask], axis=0).mean())
        forget_spread_after = float(np.std(embeddings_after[forget_mask], axis=0).mean())
        stats["forget_spread_before"] = forget_spread_before
        stats["forget_spread_after"] = forget_spread_after
        stats["forget_spread_change"] = forget_spread_after - forget_spread_before
    else:
        stats["forget_spread_before"] = 0.0
        stats["forget_spread_after"] = 0.0
        stats["forget_spread_change"] = 0.0

    return stats

