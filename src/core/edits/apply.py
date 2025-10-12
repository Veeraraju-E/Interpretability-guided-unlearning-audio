import numpy as np
from src.core.edits.leace import leace_projector_in_U
from src.core.edits.noise import inject_noise_in_subspace
from src.core.utils.whitening import invert_whitener

def apply_edits(Z_all: np.ndarray, sae, feat_idx, U, mu, W, *, beta: float, do_leace: bool, leace_params: dict, noise_sigma: float, device: str) -> tuple[np.ndarray, dict]:
    """
    Applies r_LEACE in U: r(z) = z - W_r^+ P W_r (z - mu_r), implemented as:
    ZU' = (ZU - mu_r) @ P_r.T + mu_r,
    Then map back.

    Args:
        For LEACE edits: Z_all, U, mu, W, leace_params, noise_sigma
    """
    Z_edit = Z_all.copy()
    info = {}
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