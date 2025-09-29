from typing import Iterable
import numpy as np
import torch
from tqdm import tqdm
from .model import SAE

@torch.no_grad()
def sae_encode_batch(model: SAE, Z: np.ndarray, device: str = 'cuda', batch: int = 8192) -> np.ndarray:
    model = model.to(device).eval()
    Z = np.asarray(Z)
    if Z.ndim > 2:
        Z = Z.reshape(Z.shape[0], -1)
    A = []
    for i in tqdm(range(0, Z.shape[0], batch), desc="SAE encode"):
        z = torch.from_numpy(Z[i:i+batch]).float().to(device)
        a = model.encode(z)
        A.append(a.cpu().numpy())
    return np.vstack(A) if len(A) > 0 else np.zeros((0, model.m))


