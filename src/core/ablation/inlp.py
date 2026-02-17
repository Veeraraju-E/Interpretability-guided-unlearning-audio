"""
INLP (Iterative Nullspace Projection) wrapper. Faithful to shauli-ravfogel/nullspace_projection.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier

from src.core.ablation._vendor.inlp.debias import get_debiasing_projection


class INLPEraser:
    """Iterative Nullspace Projection: train linear classifiers, project to intersection of nullspaces."""

    def __init__(
        self,
        n_iter: int = 50,
        min_accuracy: float = 0.0,
        is_autoregressive: bool = True,
        classifier_params: dict | None = None,
        random_state: int | None = None,
    ):
        self.n_iter = n_iter
        self.min_accuracy = min_accuracy
        self.is_autoregressive = is_autoregressive
        self.classifier_params = classifier_params or {}
        self.random_state = random_state
        self.P_: np.ndarray | None = None

    def fit(self, R_dev: np.ndarray, y_dev: np.ndarray, **kwargs) -> INLPEraser:
        if self.random_state is not None:
            np.random.seed(self.random_state)
        y_flat = np.asarray(y_dev).ravel().astype(np.int64)
        if np.unique(y_flat).size < 2:
            self.P_ = np.eye(R_dev.shape[1])
            return self
        input_dim = R_dev.shape[1]
        P, _, _ = get_debiasing_projection(
            SGDClassifier,
            self.classifier_params,
            self.n_iter,
            input_dim,
            self.is_autoregressive,
            self.min_accuracy,
            R_dev,
            y_flat,
            R_dev,
            y_flat,
            by_class=False,
            dropout_rate=0,
        )
        self.P_ = P
        return self

    def transform(self, R_all: np.ndarray) -> np.ndarray:
        if self.P_ is None:
            raise RuntimeError("INLPEraser not fitted")
        return (R_all @ self.P_.T).astype(R_all.dtype)
