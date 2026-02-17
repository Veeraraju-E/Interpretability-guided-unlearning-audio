"""
IGBP (Iterative Gradient-Based Projection) wrapper. Faithful to technion-cs-nlp/igbp_nonlinear-removal.
Iteratively train nonlinear classifier to predict attribute, then project representations so classifier is oblivious.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _build_probe(input_size: int, num_classes: int, width_list: list, device: torch.device) -> nn.Module:
    """MLP probe; width_list e.g. [1] = one hidden layer of size input_size."""
    if len(width_list) == 0:
        return nn.Sequential(nn.Linear(input_size, num_classes)).to(device)
    layers = [nn.Linear(input_size, width_list[0] * input_size), nn.ReLU()]
    for i in range(len(width_list) - 1):
        layers.append(nn.Linear(width_list[i] * input_size, width_list[i + 1] * input_size))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(width_list[-1] * input_size, num_classes))
    return nn.Sequential(*layers).to(device)


def _train_probe(
    probe: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float = 0.0002,
    batch_size: int = 256,
) -> float:
    probe.train()
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    dl = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(probe(xb), yb)
            loss.backward()
            opt.step()
    probe.eval()
    with torch.no_grad():
        acc = (probe(X).argmax(1) == y).float().mean().item()
    return acc


def _projection_step(
    R_batch: torch.Tensor,
    probe: nn.Module,
    loss_fn_name: str,
) -> torch.Tensor:
    """One IGBP projection step: move R so probe confidence drops (hypersurface projection)."""
    R_batch = R_batch.detach().clone().requires_grad_(True)
    probe.eval()
    out = probe(R_batch)
    if loss_fn_name == "projection":
        sftmx = torch.softmax(out, dim=1)
        pt = sftmx.max(dim=1)[0].unsqueeze(1)
        loss = (0.5 * (torch.log(pt) - torch.log(1 - pt + 1e-12)).pow(2)).mean()
    else:
        loss = nn.functional.cross_entropy(out, out.argmax(1))
    probe.zero_grad()
    loss.backward()
    grad = R_batch.grad.detach()
    grad = torch.nan_to_num(grad)
    if loss_fn_name == "projection":
        sftmx = torch.softmax(out.detach(), dim=1)
        pt = sftmx.max(dim=1)[0].unsqueeze(1).clamp(1e-6, 1 - 1e-6)
        scalar = -torch.log(1 / pt - 1)
        theta = grad / (scalar + 1e-12)
        theta_norm2 = (theta * theta).sum(1, keepdim=True).clamp(min=1e-12)
        pert = torch.nan_to_num(grad / theta_norm2)
    else:
        pert = -0.02 * grad
    return (R_batch.detach() - pert).detach()


class IGBPEraser:
    """Iterative Gradient-Based Projection: nonlinear probe + gradient-based projection to hypersurface."""

    def __init__(
        self,
        n_probes: int = 10,
        probe_width_list: list | None = None,
        max_epochs: int = 10,
        batch_size: int = 32,
        loss_fn: str = "projection",
        device: str | None = None,
        random_state: int | None = None,
    ):
        self.n_probes = n_probes
        self.probe_width_list = probe_width_list or [1]
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self._probe_list: list[nn.Module] = []
        self._input_size: int | None = None

    def fit(self, R_dev: np.ndarray, y_dev: np.ndarray, **kwargs) -> IGBPEraser:
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        R = np.asarray(R_dev, dtype=np.float32)
        y = np.asarray(y_dev).ravel().astype(np.int64)
        if np.unique(y).size < 2:
            self._input_size = R.shape[1]
            self._probe_list = []
            return self
        dev = torch.device(self.device)
        X = torch.from_numpy(R).float().to(dev)
        Y = torch.from_numpy(y).long().to(dev)
        input_size = R.shape[1]
        num_classes = int(y.max()) + 1
        self._input_size = input_size

        X_cur = X.clone()
        Y_cur = Y.clone()
        majority = (y == 1).mean()

        for _ in range(self.n_probes):
            probe = _build_probe(input_size, num_classes, self.probe_width_list, dev)
            _train_probe(probe, X_cur, Y_cur, self.max_epochs, batch_size=self.batch_size)
            self._probe_list.append(probe)

            X_cleaned = []
            for i in range(0, X_cur.size(0), self.batch_size):
                batch = X_cur[i : i + self.batch_size]
                X_cleaned.append(_projection_step(batch, probe, self.loss_fn))
            X_cur = torch.cat(X_cleaned, dim=0)
            with torch.no_grad():
                acc = (probe(X_cur).argmax(1) == Y_cur).float().mean().item()
            if acc <= majority + 0.02:
                break
        return self

    def transform(self, R_all: np.ndarray) -> np.ndarray:
        if not self._probe_list:
            return np.asarray(R_all, dtype=np.float64)
        dev = torch.device(self.device)
        X = torch.from_numpy(np.asarray(R_all, dtype=np.float32)).float().to(dev)
        for probe in self._probe_list:
            X_cleaned = []
            for i in range(0, X.size(0), self.batch_size):
                batch = X[i : i + self.batch_size]
                X_cleaned.append(_projection_step(batch, probe, self.loss_fn))
            X = torch.cat(X_cleaned, dim=0)
        return X.cpu().numpy().astype(R_all.dtype)
