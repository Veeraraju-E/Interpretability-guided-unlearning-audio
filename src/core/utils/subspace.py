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
    """
    Build subspace U from decoder weights and selected feature indices.   
    """
    if len(feat_idx) == 0:
        return np.zeros((dec_weight.shape[0], 0))
    
    r = len(feat_idx)
    d = dec_weight.shape[0]
    m = dec_weight.shape[1]
    print(f"Debug: build_U_from_decoder - dec_weight.shape: {dec_weight.shape}, length of feat_idx: {len(feat_idx)}, r: {r}, d: {d}, m: {m}")
    if max(feat_idx) >= m or min(feat_idx) < 0:
        raise ValueError(f"Feature indices must be in [0, {m-1}] for SAE latent space, got indices in [{min(feat_idx)}, {max(feat_idx)}]")
        
    D = dec_weight.copy()
    D_S = D[:, feat_idx]
    U, _, _ = np.linalg.svd(D_S, full_matrices=False)
    
    # if Uu.shape[1] >= r:
    #     U = Uu[:, :r]
    # else:
    #     # NOTE: this should not happen in practice.
    #     raise ValueError(f"bad things happening with build_U_from_decoder, r: {r} < rank(D_S): {Uu.shape[1]}...")
    
    return U