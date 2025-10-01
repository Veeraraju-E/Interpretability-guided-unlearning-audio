import numpy as np
from sklearn.covariance import LedoitWolf

def fit_whitener(X: np.ndarray, eps: float = 1e-8, use_shrinkage: bool = True):
    X = np.asarray(X)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples to fit whitener")
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    if use_shrinkage:
        try:
            lw = LedoitWolf(store_precision=False)
            lw.fit(Xc)
            C = lw.covariance_
        except Exception:
            C = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    else:
        C = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, eps, None)
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return mu, W

def apply_whitener(X: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (X - mu) @ W

def invert_whitener(Z: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    W_inv = np.linalg.pinv(W)
    return (Z @ W_inv.T) + mu