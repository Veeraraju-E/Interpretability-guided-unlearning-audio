"""Centered Kernel Alignment (CKA) for representation similarity (Kornblith et al., 2019)."""

import numpy as np
from typing import Tuple, Optional


def linear_cka(X: np.ndarray, Y: np.ndarray, center: bool = True) -> float:
    """
    Linear CKA: CKA_lin(X,Y) = ||X_c^T Y_c||_F^2 / sqrt(||X_c^T X_c||_F^2 * ||Y_c^T Y_c||_F^2).
    Samples are rows; center columns by default.
    """
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows (samples).")
    if center:
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
    xtx = X.T @ X
    yty = Y.T @ Y
    xty = X.T @ Y
    n = np.trace(xtx @ xtx)
    d = np.trace(yty @ yty)
    c = (xty ** 2).sum()
    if n <= 0 or d <= 0:
        return 0.0
    return float(c / np.sqrt(n * d))


def rbf_cka(X: np.ndarray, Y: np.ndarray, sigma_X: Optional[float] = None, sigma_Y: Optional[float] = None, center: bool = True) -> float:
    """
    RBF-kernel CKA. Gram matrices on rows; bandwidth via median heuristic if not provided.
    """
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError("X and Y must have the same number of rows.")
    from sklearn.metrics.pairwise import rbf_kernel
    if sigma_X is None:
        pairwise = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
        sigma_X = np.median(pairwise[np.triu_indices(n, k=1)]) if n > 1 else 1.0
        if sigma_X <= 0:
            sigma_X = 1.0
    if sigma_Y is None:
        pairwise = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))
        sigma_Y = np.median(pairwise[np.triu_indices(n, k=1)]) if n > 1 else 1.0
        if sigma_Y <= 0:
            sigma_Y = 1.0
    K = rbf_kernel(X, X, gamma=1.0 / (2 * sigma_X ** 2))
    L = rbf_kernel(Y, Y, gamma=1.0 / (2 * sigma_Y ** 2))
    if center:
        H = np.eye(n) - np.ones((n, n)) / n
        K = H @ K @ H
        L = H @ L @ H
    c = np.trace(K @ L)
    nk = np.trace(K @ K)
    nl = np.trace(L @ L)
    if nk <= 0 or nl <= 0:
        return 0.0
    return float(c / np.sqrt(nk * nl))


def bootstrap_cka(
    X: np.ndarray,
    Y: np.ndarray,
    n_bootstrap: int = 500,
    rng: Optional[np.random.Generator] = None,
    linear: bool = True,
) -> Tuple[float, float, float]:
    """
    CKA and bootstrap CI. Resamples rows (samples). Returns (cka, ci_low, ci_high).
    """
    rng = rng or np.random.default_rng()
    n = X.shape[0]
    cka_fn = linear_cka if linear else (lambda x, y: rbf_cka(x, y))
    point = cka_fn(X, Y)
    vals = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        vals.append(cka_fn(X[idx], Y[idx]))
    vals = np.array(vals)
    ci_low = float(np.percentile(vals, 2.5))
    ci_high = float(np.percentile(vals, 97.5))
    return point, ci_low, ci_high


def perm_test_cka(
    X: np.ndarray,
    Y: np.ndarray,
    n_perm: int = 500,
    rng: Optional[np.random.Generator] = None,
    linear: bool = True,
) -> Tuple[float, float]:
    """
    CKA and p-value under null that X and Y are unrelated (permute rows of Y).
    Returns (cka, p_value).
    """
    rng = rng or np.random.default_rng()
    cka_fn = linear_cka if linear else (lambda x, y: rbf_cka(x, y))
    point = cka_fn(X, Y)
    nulls = []
    for _ in range(n_perm):
        perm = rng.permutation(Y.shape[0])
        nulls.append(cka_fn(X, Y[perm]))
    nulls = np.array(nulls)
    p = float(np.mean(nulls >= point))
    return point, p
