import numpy as np

def leace_projector_in_U(Z: np.ndarray, U: np.ndarray, Y: np.ndarray, ridge: float = 1e-6):
    if U.shape[1] == 0:
        return np.eye(0), np.zeros((1, 0))
    ZU = Z @ U
    mu_r = ZU.mean(axis=0, keepdims=True)
    ZUc = ZU - mu_r
    Yc = Y - Y.mean(axis=0, keepdims=True)
    N = ZU.shape[0]
    Sig = (ZUc.T @ ZUc) / max(1, N - 1)
    Sig_ZY = (ZUc.T @ Yc) / max(1, N - 1)
    evals, evecs = np.linalg.eigh(Sig + ridge * np.eye(Sig.shape[0]))
    evals = np.clip(evals, 1e-12, None)
    W_r = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    W_r_pinv = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
    M = W_r @ Sig_ZY
    U_M, _, _ = np.linalg.svd(M, full_matrices=False)
    if U_M.size == 0:
        P = np.zeros((Sig.shape[0], Sig.shape[0]))
    else:
        P = U_M @ U_M.T
    P_r = np.eye(U.shape[1]) - (W_r_pinv @ P @ W_r)
    return P_r, mu_r


