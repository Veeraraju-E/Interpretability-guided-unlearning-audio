import numpy as np

def inject_noise_in_subspace(Z: np.ndarray, U: np.ndarray, sigma: float = 0.02, rng=None) -> np.ndarray:
    rng = np.random if rng is None else rng
    ZU = Z @ U
    noise = rng.randn(*ZU.shape) * sigma
    return Z + (noise @ U.T)


