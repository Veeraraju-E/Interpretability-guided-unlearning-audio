import numpy as np

def leace_projector_in_U(Z: np.ndarray, U: np.ndarray, Y: np.ndarray, ridge: float = 1e-6):
    # ensure Z and Y are for same data split
    assert Z.shape[0] == Y.shape[0], "Z and Y must have the same number of samples"

    if U.shape[1] == 0: return np.eye(0), np.zeros((1, 0))
    ZU = Z @ U
    mu_r = ZU.mean(axis=0, keepdims=True)
    ZUc = ZU - mu_r # ZU_centered
    Yc = Y - Y.mean(axis=0, keepdims=True)  # Y_centered
    N = ZU.shape[0]
    Sig = (ZUc.T @ ZUc) / max(1, N - 1) # emperical cross-covariance bw ZUc and itself
    print(f"in leace.py, Sig: {Sig.shape}, Yc: {Yc.shape}, ZUc: {ZUc.shape}")   # [192, 192]; [25071, 912]; [25071, 192]
    print(f"in leace.py, Sig: {Sig}")
    print(f"in leace.py, ZUc: {ZUc}")
    print(f"in leace.py, Yc: {Yc}")
    Sig_ZY = (ZUc.T @ Yc) / max(1, N - 1) # emperical cross-covariance bw ZUc and Yc
    evals, evecs = np.linalg.eigh(Sig + ridge * np.eye(Sig.shape[0]))   # EVD on noise-added emperical cross-covariance
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