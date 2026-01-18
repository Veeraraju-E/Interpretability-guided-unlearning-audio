"""
Attribute direction analysis module.
Implements Step 6 of the interpretation protocol.
"""

import numpy as np
from typing import Dict, List
from scipy.stats import wasserstein_distance


def compute_attribute_directions(probes: Dict, which: str = 'pre') -> np.ndarray:
    """
    Compute normalized attribute direction vectors from probe weights.
    
    Args:
        probes: Dictionary from fit_trait_probes
        which: 'pre' or 'post' to select which probes to use
    
    Returns:
        (n_traits, d) array of normalized direction vectors
    """
    probe_list = probes[f'probes_{which}']
    n_traits = len(probe_list)
    d = len(probe_list[0]['w'])
    
    directions = np.zeros((n_traits, d))
    
    for k in range(n_traits):
        w = probe_list[k]['w']
        norm = np.linalg.norm(w)
        if norm > 1e-10:
            directions[k] = w / norm
        else:
            directions[k] = w
    
    return directions


def compute_projections(embeddings: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """
    Compute scalar projections of embeddings onto attribute directions.
    
    Args:
        embeddings: (n_samples, d) embeddings
        directions: (n_traits, d) attribute direction vectors
    
    Returns:
        (n_samples, n_traits) projection values
    """
    return embeddings @ directions.T


def compute_wasserstein_distances(
    projections_pre: np.ndarray,
    projections_post: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Compute 1-Wasserstein distances between pre and post projection distributions.
    
    Args:
        projections_pre: (n_samples, n_traits) pre-edit projections
        projections_post: (n_samples, n_traits) post-edit projections
        mask: (n_samples,) boolean mask for samples to include
    
    Returns:
        (n_traits,) array of Wasserstein distances
    """
    n_traits = projections_pre.shape[1]
    w1_distances = np.zeros(n_traits)
    
    proj_pre_masked = projections_pre[mask]
    proj_post_masked = projections_post[mask]
    
    for k in range(n_traits):
        dist_pre = proj_pre_masked[:, k]
        dist_post = proj_post_masked[:, k]
        w1_distances[k] = wasserstein_distance(dist_pre, dist_post)
    
    return w1_distances
