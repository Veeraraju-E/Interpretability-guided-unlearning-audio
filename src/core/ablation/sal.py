"""
SAL (Spectral Attribute Removal) wrapper. Projects onto directions with reduced covariance with the guarded attribute.
Faithful to the formulation in Shao et al., EACL 2023: matrix decomposition to remove directions that correlate with Y.
"""
from __future__ import annotations

import numpy as np


class SALEraser:
    """Spectral Attribute Removal: project out top-r directions that correlate with the attribute (SVD of cross-covariance)."""

    def __init__(self, n_components_remove: int = 1, center: bool = True):
        """
        n_components_remove: number of singular vectors (directions of max covariance with Y) to remove.
        For binary Y, 1 is often enough; increase for stronger removal.
        """
        self.n_components_remove = n_components_remove
        self.center = center
        self.P_: np.ndarray | None = None
        self._mean_R: np.ndarray | None = None
        self._mean_y: np.ndarray | None = None

    def fit(self, R_dev: np.ndarray, y_dev: np.ndarray, **kwargs) -> SALEraser:
        R = np.asarray(R_dev, dtype=np.float64)
        y = np.asarray(y_dev).reshape(-1, 1).astype(np.float64)
        if self.center:
            self._mean_R = R.mean(axis=0, keepdims=True)
            self._mean_y = y.mean(axis=0, keepdims=True)
            Rc = R - self._mean_R
            yc = y - self._mean_y
        else:
            self._mean_R = np.zeros((1, R.shape[1]))
            self._mean_y = np.zeros((1, 1))
            Rc = R
            yc = y
        n = Rc.shape[0]
        Sigma_ZY = (Rc.T @ yc) / max(1, n - 1)
        U, S, _ = np.linalg.svd(Sigma_ZY, full_matrices=False)
        r = min(self.n_components_remove, U.shape[1], U.shape[0])
        if r <= 0:
            self.P_ = np.eye(R.shape[1])
            return self
        U_r = U[:, :r]
        P_remove = U_r @ U_r.T
        self.P_ = np.eye(R.shape[1]) - P_remove
        return self

    def transform(self, R_all: np.ndarray) -> np.ndarray:
        if self.P_ is None:
            raise RuntimeError("SALEraser not fitted")
        R = np.asarray(R_all, dtype=np.float64)
        if self.center and self._mean_R is not None:
            R = R - self._mean_R
        out = (R @ self.P_.T).astype(R_all.dtype)
        if self.center and self._mean_R is not None:
            out = out + self._mean_R
        return out
