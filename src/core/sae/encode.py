import numpy as np
import torch
from tqdm import tqdm
from .model import SAE

@torch.no_grad()
def sae_encode_batch(model: SAE, Z: np.ndarray, device: str = 'cuda', batch: int = 8192, verbose: bool = False) -> np.ndarray:
    model = model.to(device).eval()
    Z = np.asarray(Z)
    if Z.ndim > 2:
        Z = Z.reshape(Z.shape[0], -1)
    
    if verbose:
        print(f"DEBUG: SAE encoding - input Z shape: {Z.shape}")
        print(f"DEBUG: SAE model - input_dim: {model.input_dim}, hidden_dim: {model.m}, k: {model.k}")
        print(f"DEBUG: Input Z stats - mean: {Z.mean():.6f}, std: {Z.std():.6f}, min: {Z.min():.6f}, max: {Z.max():.6f}")
    
    A = []
    for i in tqdm(range(0, Z.shape[0], batch), desc="SAE encode"):
        z = torch.from_numpy(Z[i:i+batch]).float().to(device)
        a = model.encode(z)
        A.append(a.cpu().numpy())
    
    A_all = np.vstack(A) if len(A) > 0 else np.zeros((0, model.m))
    
    if verbose:
        print(f"DEBUG: SAE output A shape: {A_all.shape}")
        print(f"DEBUG: A stats - mean: {A_all.mean():.6f}, std: {A_all.std():.6f}, min: {A_all.min():.6f}, max: {A_all.max():.6f}")
        print(f"DEBUG: A sparsity - non-zero elements: {(A_all > 1e-6).sum()}, total elements: {A_all.size}")
        print(f"DEBUG: A sparsity ratio: {(A_all > 1e-6).sum() / A_all.size:.6f}")
        
        # Check for dead features
        feature_activity = (A_all > 1e-6).sum(axis=0)
        dead_features = (feature_activity == 0)
        print(f"DEBUG: Dead features (never activated): {dead_features.sum()} out of {len(dead_features)}")
    
    return A_all


