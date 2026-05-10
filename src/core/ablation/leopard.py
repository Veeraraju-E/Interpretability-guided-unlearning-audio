"""
L̄EOPARD wrapper: learn orthogonal projection P = U U^T to minimize MMD between class-conditional distributions.
Faithful to Saillenfest & Lemberger, ECAI 2025 (density matching via MMD).
Uses batching/subsampling so large datasets (e.g. N~10k) do not OOM.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Max samples for sigma heuristic (pairwise diff is sample_size^2 * d)
_SIGMA_SAMPLE_SIZE = 512
# Max samples per class for MMD per step (kernel matrices are max_mmd_samples^2)
_MAX_MMD_SAMPLES = 1024


def _sigma_heuristic_batched(X_all: torch.Tensor, sample_size: int = _SIGMA_SAMPLE_SIZE) -> float:
    """Estimate sigma from mean squared pairwise distance using a subset to avoid O(N^2) memory."""
    n, d = X_all.shape
    if n <= 1:
        return 1.0
    size = min(sample_size, n)
    idx = torch.randperm(n, device=X_all.device)[:size]
    X = X_all[idx]
    diff = (X.unsqueeze(0) - X.unsqueeze(1)).reshape(-1, d)
    sigma_sq = (diff ** 2).sum(1).mean().item() + 1e-12
    return float(np.sqrt(sigma_sq))


def _mmd2_gaussian(X: torch.Tensor, Y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Unbiased MMD^2 with Gaussian kernel. Use on subsampled X,Y when full size would OOM."""
    n, m = X.size(0), Y.size(0)
    if n < 2 or m < 2:
        return torch.tensor(0.0, device=X.device)
    sigma_sq = sigma ** 2 + 1e-12
    XX = (X * X).sum(1)
    YY = (Y * Y).sum(1)
    Kxx = torch.exp(-(XX.unsqueeze(0) + XX.unsqueeze(1) - 2 * (X @ X.T)) / (2 * sigma_sq))
    Kyy = torch.exp(-(YY.unsqueeze(0) + YY.unsqueeze(1) - 2 * (Y @ Y.T)) / (2 * sigma_sq))
    Kxy = torch.exp(-(XX.unsqueeze(1) + YY.unsqueeze(0) - 2 * (X @ Y.T)) / (2 * sigma_sq))
    ix = ~torch.eye(n, dtype=torch.bool, device=X.device)
    iy = ~torch.eye(m, dtype=torch.bool, device=Y.device)
    term_xx = Kxx[ix].sum() / (n * (n - 1))
    term_yy = Kyy[iy].sum() / (m * (m - 1))
    term_xy = -2 * Kxy.sum() / (n * m)
    return term_xx + term_yy + term_xy


class LEOPARDEraser:
    """L̄EOPARD: orthogonal projection learned to minimize MMD between class-conditional distributions."""

    def __init__(
        self,
        rank: int,
        n_epochs: int = 200,
        lr: float = 1e-2,
        sigma_heuristic: bool = True,
        gamma_proj: float = 0.1,
        device: str | None = None,
        random_state: int | None = None,
    ):
        self.rank = rank
        self.n_epochs = n_epochs
        self.lr = lr
        self.sigma_heuristic = sigma_heuristic
        self.gamma_proj = gamma_proj
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self._U: np.ndarray | None = None
        self._input_dim: int | None = None

    def fit(self, R_dev: np.ndarray, y_dev: np.ndarray, **kwargs) -> LEOPARDEraser:
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        R = np.asarray(R_dev, dtype=np.float32)
        y = np.asarray(y_dev).ravel()
        if np.unique(y).size < 2:
            self._input_dim = R.shape[1]
            self._U = np.eye(R.shape[1], min(self.rank, R.shape[1]))
            return self
        d = R.shape[1]
        r = min(self.rank, d)
        dev = torch.device(self.device)
        U = torch.randn(d, r, device=dev) * 0.01
        nn.init.orthogonal_(U)
        U.requires_grad_(True)
        opt = torch.optim.Adam([U], lr=self.lr)
        mask0 = torch.from_numpy(y == 0).to(dev)
        mask1 = torch.from_numpy(y == 1).to(dev)
        X_all = torch.from_numpy(R).float().to(dev)
        n0 = mask0.sum().item()
        n1 = mask1.sum().item()
        if n0 < 2 or n1 < 2:
            self._U = np.eye(d, r) if r <= d else np.eye(d)
            self._input_dim = d
            return self
        if self.sigma_heuristic:
            sigma = _sigma_heuristic_batched(X_all)
        else:
            sigma = 1.0
        max_mmd = _MAX_MMD_SAMPLES
        for _ in range(self.n_epochs):
            opt.zero_grad()
            P = U @ U.T
            X_proj = X_all @ P.T
            X0 = X_proj[mask0]
            X1 = X_proj[mask1]
            n0_cur, n1_cur = X0.size(0), X1.size(0)
            if n0_cur > max_mmd or n1_cur > max_mmd:
                i0 = torch.randperm(n0_cur, device=dev)[: min(n0_cur, max_mmd)]
                i1 = torch.randperm(n1_cur, device=dev)[: min(n1_cur, max_mmd)]
                X0 = X0[i0]
                X1 = X1[i1]
            loss_mmd = _mmd2_gaussian(X0, X1, sigma)
            loss_orth = ((U.T @ U) - torch.eye(r, device=dev)).pow(2).mean()
            loss = loss_mmd + self.gamma_proj * loss_orth
            loss.backward()
            opt.step()
            with torch.no_grad():
                U.data = U.data / (U.norm(dim=0, keepdim=True) + 1e-12)
        with torch.no_grad():
            _, _, Vh = torch.linalg.svd(U @ U.T)
            Ep = Vh[:r].T
            self._U = (Ep.cpu().numpy()).astype(np.float64)
        self._input_dim = d
        return self

    def transform(self, R_all: np.ndarray) -> np.ndarray:
        if self._U is None:
            raise RuntimeError("LEOPARDEraser not fitted")
        P = self._U @ self._U.T
        return (np.asarray(R_all) @ P.T).astype(R_all.dtype)
