"""
Common interfaces and lift-back utilities for ablation erasers.
Faithful to the existing pipeline: Span(U) replacement then inverse whitening.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class AblationEraser(Protocol):
    """Protocol for ablation concept erasers. Fit on dev split, transform any R."""

    def fit(self, R_dev: np.ndarray, y_dev: np.ndarray, **kwargs) -> "AblationEraser":
        """Fit the eraser on dev representations and binary labels (0=retain, 1=forget)."""
        ...

    def transform(self, R_all: np.ndarray) -> np.ndarray:
        """Return R_all' of same shape as R_all (edited representations)."""
        ...


def lift_back_to_Z(
    Z_all: np.ndarray,
    R_prime: np.ndarray,
    U: np.ndarray,
    use_sae: bool,
) -> np.ndarray:
    """
    Lift edited representations R' back to whitened space Z.
    Mirrors src.core.edits.apply_edits logic (lines 66-69):
    - If use_sae: Z' = Z - Z U U^T + R' U^T  (replace only Span(U) component).
    - If not use_sae: Z' = R' (R was full Z).
    """
    if use_sae and U is not None and U.shape[1] > 0:
        ZU = Z_all @ U
        Z_edit = Z_all - (ZU @ U.T) + (R_prime @ U.T)
        return Z_edit
    return R_prime


def apply_ablation_eraser(
    Z_all: np.ndarray,
    U: np.ndarray,
    use_sae: bool,
    eraser: AblationEraser,
    mu: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """
    Build R = Z_all @ U (or Z_all if U is identity), run eraser.transform(R),
    lift back to Z, then invert whitening to get E' (embedding space).
    """
    from src.core.utils.whitening import invert_whitener

    if use_sae and U is not None and U.shape[1] > 0:
        R = Z_all @ U
    else:
        R = Z_all

    R_prime = eraser.transform(R)
    Z_prime = lift_back_to_Z(Z_all, R_prime, U, use_sae)
    E_prime = invert_whitener(Z_prime, mu, W)
    return E_prime
