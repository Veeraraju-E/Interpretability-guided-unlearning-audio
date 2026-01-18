"""
SAE module for monosemantic basis.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange


class SAE(nn.Module):
    def __init__(self, input_dim: int, hidden_mult: float = 8.0, k: int = 32, hidden_dim: int = 0, variant: str = "topk"):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_mult = float(hidden_mult)
        self.m = int(hidden_dim) if hidden_dim else int(round(self.input_dim * self.hidden_mult))
        self.k = int(k)
        self.variant = variant.lower()
        if self.variant not in ["topk", "standard"]:
            raise ValueError(f"SAE variant must be 'topk' or 'standard', got '{variant}'")
        self.enc = nn.Linear(self.input_dim, self.m, bias=True)
        self.dec = nn.Linear(self.m, self.input_dim, bias=False)
        nn.init.xavier_uniform_(self.enc.weight)
        nn.init.zeros_(self.enc.bias)
        nn.init.xavier_uniform_(self.dec.weight)
        print(f"Debug from sae.py: enc.weight.shape: {self.enc.weight.shape}, dec.weight.shape: {self.dec.weight.shape}")

    def _ensure_2d(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 3 and z.size(1) == 1:
            z = z.squeeze(1)
        if z.ndim > 2:
            z = z.view(z.size(0), -1)
        return z

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        z = self._ensure_2d(z)
        a = F.relu(self.enc(z))
        if self.variant == "topk" and 0 < self.k < a.shape[1]:
            topk_vals, _ = torch.topk(a, self.k, dim=1)
            kth = topk_vals[:, -1].unsqueeze(1)
            mask = (a >= kth).float()
            a = a * mask
        return a

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        return self.dec(a)

    def forward(self, z: torch.Tensor):
        z = self._ensure_2d(z)
        a = self.encode(z)
        z_hat = self.decode(a)
        return z_hat, a


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
    A_all = np.vstack(A) if len(A) > 0 else np.zeros((0, model.m))
    return A_all


def train_sae(
    model: SAE,
    Z_train: np.ndarray,
    Z_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 1e-3,
    device: str = 'cuda',
    lambda_sparsity: float = 0.0,
    return_loss_history: bool = False,
):
    Z_train = np.asarray(Z_train)
    Z_val = np.asarray(Z_val)
    if Z_train.ndim > 2:
        Z_train = Z_train.reshape(Z_train.shape[0], -1)
    if Z_val.ndim > 2:
        Z_val = Z_val.reshape(Z_val.shape[0], -1)
    if Z_train.shape[1] != model.input_dim:
        raise RuntimeError(f"SAE input dim mismatch: model.input_dim={model.input_dim} but Z_train has dim={Z_train.shape[1]}")

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    dl = DataLoader(TensorDataset(torch.from_numpy(Z_train).float()), batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(TensorDataset(torch.from_numpy(Z_val).float()), batch_size=batch_size, shuffle=False)

    best_val = float('inf'); best_state = None
    loss_history = {
        'epochs': [], 'train_loss': [], 'val_loss': [], 'train_l1_loss': [], 'val_l1_loss': [], 'dead_latent_fraction': []
    } if return_loss_history else None

    pbar = trange(1, epochs + 1, desc="Epoch", position=0)
    for epoch in pbar:
        model.train(); acc_loss = 0.0; acc_l1 = 0.0; n = 0
        for (z_batch,) in dl:
            z_batch = z_batch.to(device)
            z_hat, a = model(z_batch)
            loss = F.mse_loss(z_hat, z_batch)
            l1p = 0.0
            if model.variant == "standard" and lambda_sparsity > 0:
                l1p = lambda_sparsity * torch.mean(torch.abs(a))
                loss = loss + l1p
            opt.zero_grad(); loss.backward(); opt.step()
            acc_loss += loss.item() * z_batch.size(0)
            acc_l1 += (float(l1p) if isinstance(l1p, float) else l1p.item()) * z_batch.size(0)
            n += z_batch.size(0)
        train_loss = acc_loss / max(1, n)
        train_l1 = acc_l1 / max(1, n)

        model.eval(); acc_v = 0.0; acc_v_l1 = 0.0; n_v = 0; all_a = []
        with torch.no_grad():
            for (z_batch,) in dl_val:
                z_batch = z_batch.to(device)
                z_hat, a = model(z_batch)
                val_loss = F.mse_loss(z_hat, z_batch).item()
                val_l1 = 0.0
                if model.variant == "standard" and lambda_sparsity > 0:
                    val_l1 = (lambda_sparsity * torch.mean(torch.abs(a))).item()
                    val_loss += val_l1
                acc_v += val_loss * z_batch.size(0)
                acc_v_l1 += val_l1 * z_batch.size(0)
                n_v += z_batch.size(0)
                all_a.append(a.cpu())
        val_loss = acc_v / max(1, n_v)
        val_l1 = acc_v_l1 / max(1, n_v)

        pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}"
        })

        A = torch.cat(all_a, dim=0)
        dead_mask = (A.abs().mean(dim=0) < 1e-5)
        dead_frac = dead_mask.float().mean().item()

        if return_loss_history:
            loss_history['epochs'].append(epoch)
            loss_history['train_loss'].append(train_loss)
            loss_history['val_loss'].append(val_loss)
            loss_history['train_l1_loss'].append(train_l1)
            loss_history['val_l1_loss'].append(val_l1)
            loss_history['dead_latent_fraction'].append(dead_frac)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return (model, loss_history) if return_loss_history else model


