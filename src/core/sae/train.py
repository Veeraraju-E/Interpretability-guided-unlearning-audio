import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
from .model import SAE

def train_sae(model: SAE, Z_train: np.ndarray, Z_val: np.ndarray, epochs: int = 30, batch_size: int = 4096, lr: float = 1e-3, device: str = 'cuda', lambda_sparsity: float = 0.0, return_loss_history: bool = False):
    Z_train = np.asarray(Z_train)
    Z_val = np.asarray(Z_val)
    if Z_train.ndim > 2: Z_train = Z_train.reshape(Z_train.shape[0], -1)
    if Z_val.ndim > 2: Z_val = Z_val.reshape(Z_val.shape[0], -1)
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

        # Update tqdm bar with only epoch, train_loss, val_loss
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


