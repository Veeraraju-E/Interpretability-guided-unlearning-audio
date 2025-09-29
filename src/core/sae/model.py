import torch
import torch.nn as nn
import torch.nn.functional as F

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


