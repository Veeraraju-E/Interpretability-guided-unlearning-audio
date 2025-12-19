"""
Core editing module for applying LEACE and noise injection to embeddings.
"""

import numpy as np
from src.core.utils.whitening import invert_whitener


def inject_noise_in_subspace(Z: np.ndarray, U: np.ndarray, sigma: float = 0.02, rng=None) -> np.ndarray:
    rng = np.random if rng is None else rng
    ZU = Z @ U
    noise = rng.randn(*ZU.shape) * sigma
    return Z + (noise @ U.T)


def leace_projector_in_U(Z: np.ndarray, U: np.ndarray, Y: np.ndarray, ridge: float = 1e-6):
    # ensure Z and Y are for same data split
    assert Z.shape[0] == Y.shape[0], "Z and Y must have the same number of samples"

    if U.shape[1] == 0:
        return np.eye(0), np.zeros((1, 0))

    ZU = Z @ U
    mu_r = ZU.mean(axis=0, keepdims=True)
    ZUc = ZU - mu_r  # ZU_centered
    Yc = Y - Y.mean(axis=0, keepdims=True)  # Y_centered
    N = ZU.shape[0]
    Sig = (ZUc.T @ ZUc) / max(1, N - 1)  # empirical covariance of ZUc
    Sig_ZY = (ZUc.T @ Yc) / max(1, N - 1)  # empirical cross-covariance between ZUc and Yc
    Sig_gamma = Sig + ridge * np.eye(Sig.shape[0])
    evals, evecs = np.linalg.eigh(Sig_gamma)
    evals = np.clip(evals, 1e-12, None)
    W_r = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T   # Sig_gamma^{-1/2} using moore-penrose pseudo-inverse
    W_r_pinv = evecs @ np.diag(np.sqrt(evals)) @ evecs.T    # Sig_gamma^{1/2}
    M = W_r @ Sig_ZY
    U_M, _, _ = np.linalg.svd(M, full_matrices=False)   # SVD equivalent for P_c = M * (M^T M)^{-1} * M^T
    P_c = U_M @ U_M.T if U_M.size > 0 else np.zeros((Sig.shape[0], Sig.shape[0]))
    P_r = np.eye(U.shape[1]) - (W_r_pinv @ P_c @ W_r) # LEACE editor
    return P_r, mu_r


def apply_edits(
    Z_all: np.ndarray,
    sae,
    feat_idx,
    U,
    mu,
    W,
    beta: float,
    do_leace: bool,
    leace_params: dict,
    noise_sigma: float,
) -> tuple[np.ndarray, dict]:
    """
    Applies r_LEACE in U: r(z) = z - W_r^+ P W_r (z - mu_r), implemented as:
    ZU' = (ZU - mu_r) @ P_r.T + mu_r, then maps back via the whitener inverse.
    """
    Z_edit = Z_all.copy()
    info: dict = {}

    if beta and beta > 0.0:
        raise NotImplementedError("SAUCE edits are not implemented yet")

    if do_leace:
        P_r, mu_r = leace_projector_in_U(Z_all, U, leace_params['Y_dev'], ridge=float(leace_params.get('ridge', 1e-6)))
        ZU = Z_edit @ U
        ZU_prime = (ZU - mu_r) @ P_r.T + mu_r
        Z_edit = Z_edit - (ZU @ U.T) + (ZU_prime @ U.T)
        info['P_r'], info['mu_r'] = P_r, mu_r

    if noise_sigma and noise_sigma > 0:
        Z_edit = inject_noise_in_subspace(Z_edit, U, sigma=noise_sigma)

    E_edit = invert_whitener(Z_edit, mu, W)
    return E_edit, info