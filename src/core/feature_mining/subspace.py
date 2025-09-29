import numpy as np

def prune_and_normalize_decoder(dec_weight: np.ndarray, prune_threshold: float = 1e-6) -> np.ndarray:
    cols = dec_weight.T
    norms = np.linalg.norm(cols, axis=1)
    keep = norms > prune_threshold
    if keep.sum() < 1:
        keep = norms.argsort()[::-1][:max(1, int(len(norms) * 0.1))]
        cols_kept = cols[keep]
    else:
        cols_kept = cols[keep]
    cols_kept = cols_kept / (np.linalg.norm(cols_kept, axis=1, keepdims=True) + 1e-12)
    return cols_kept.T

def build_U_from_decoder(dec_weight: np.ndarray, feat_idx: list[int]) -> np.ndarray:
    if len(feat_idx) == 0:
        return np.zeros((dec_weight.shape[0], 0))
    D = dec_weight.copy()
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    D = D / (norms + 1e-12)
    D_S = D[:, feat_idx]
    Uu, svals, _ = np.linalg.svd(D_S, full_matrices=False)
    rank = (svals > 1e-12).sum()
    if rank == 0:
        return np.zeros((D_S.shape[0], 0))
    return Uu[:, :rank]


