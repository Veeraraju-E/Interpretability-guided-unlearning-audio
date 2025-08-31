from __future__ import annotations
import math
import random
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Union

# optional dependencies
try:
    from statsmodels.stats.multitest import multipletests as _multipletests
    multipletests = _multipletests
except Exception:
    multipletests = None

try:
    from sklearn.covariance import LedoitWolf
    _HAS_LED = True
except Exception:
    LedoitWolf = None
    _HAS_LED = False

try:
    from transformers import AutoFeatureExtractor, AutoModel
    _HAS_TRANSFORMERS = True
except Exception:
    AutoFeatureExtractor = AutoModel = None
    _HAS_TRANSFORMERS = False

try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

try:
    import torchaudio
except Exception:
    torchaudio = None

try:
    from speechbrain.pretrained import EncoderClassifier
except Exception:
    EncoderClassifier = None

from typing import List, Dict, Optional, Tuple

# -------------------- Utilities --------------------

def seed_all(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def human_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


# -------------------- Audio helpers --------------------

def _to_mono_and_resample(wav: torch.Tensor, sr: int, target_sr: int, device: str):
    if wav.ndim == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        if torchaudio is None:
            raise RuntimeError("torchaudio required for resampling")
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.to(device)


# -------------------- ECAPA embedding extraction --------------------

def extract_ecapa_embeddings(entries, device='cuda', label_key='speaker'):
    if EncoderClassifier is None:
        raise RuntimeError("speechbrain not installed or EncoderClassifier unavailable")
    encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
    E, labels, paths = [], [], []
    for e in tqdm(entries, desc="Extracting ECAPA"):
        wav, sr = torchaudio.load(e['path'])
        if wav.ndim == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(device)
        with torch.no_grad():
            try:
                emb = encoder.encode_batch(wav).squeeze().cpu().numpy().reshape(-1)
            except Exception:
                emb = encoder.encode_batch(wav[:1]).squeeze().cpu().numpy().reshape(-1)
        E.append(emb)
        labels.append(e.get(label_key, "NA"))
        paths.append(e['path'])
    E = np.vstack(E)
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    E = E / norms
    return E, labels, paths


# -------------------- HF wav2vec2 / hubert extraction --------------------
@torch.no_grad()
def _extract_hf_embeddings(entries: list, model_name: str, device: str = 'cuda', target_sr: int = 16000, label_key='speaker'):
    if not _HAS_TRANSFORMERS:
        raise RuntimeError("transformers is not installed. Please `pip install transformers` to enable wav2vec2/hubert.")

    processor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    E, labels, paths = [], [], []
    for e in tqdm(entries, desc=f"Extracting {model_name.split('/')[-1]}"):
        wav, sr = torchaudio.load(e['path'])
        wav = _to_mono_and_resample(wav, sr, target_sr, device)
        inp = processor(wav.squeeze().cpu().numpy(), sampling_rate=target_sr, return_tensors="pt")
        input_values = inp["input_values"].to(device)
        out = model(input_values)
        # Take mean pooling of last hidden states
        last = out.last_hidden_state
        emb = last.mean(dim=1).squeeze(0).detach().cpu().numpy()
        E.append(emb)
        labels.append(e.get(label_key, 'NA'))
        paths.append(e['path'])
    E = np.vstack(E)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    return E, labels, paths


# Unified switch
def extract_embeddings(entries: list, encoder: str, device: str = 'cuda', wav2vec2_model: str = "facebook/wav2vec2-base", hubert_model: str = "facebook/hubert-base-ls960", label_key='speaker'):
    enc = encoder.strip().lower()
    if enc in {"ecapa", "ecapa-tdnn", "speechbrain-ecapa"}:
        print("[2/11] Extracting ECAPA embeddings...")
        return extract_ecapa_embeddings(entries, device=device, label_key=label_key)
    elif enc in {"wav2vec", "wav2vec2", "w2v2"}:
        print(f"[2/11] Extracting wav2vec2 embeddings ({wav2vec2_model})...")
        return _extract_hf_embeddings(entries, wav2vec2_model, device=device, target_sr=16000, label_key=label_key)
    elif enc in {"hubert"}:
        print(f"[2/11] Extracting HuBERT embeddings ({hubert_model})...")
        return _extract_hf_embeddings(entries, hubert_model, device=device, target_sr=16000, label_key=label_key)
    else:
        raise ValueError(f"Unknown encoder '{encoder}'. Choose from: ecapa | wav2vec2 | hubert")


# -------------------- Whitener --------------------
def fit_whitener(X: np.ndarray, eps: float = 1e-8, use_shrinkage: bool = True, shrinkage_alpha: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a whitening transform. By default, attempts Ledoit-Wolf shrinkage if sklearn available.
    Returns (mu, W) where W is the whitening matrix such that (X - mu) @ W is whitened.
    """
    X = np.array(X)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples to fit whitener")
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    N = Xc.shape[0]

    if use_shrinkage and _HAS_LED and LedoitWolf is not None:
        try:
            lw = LedoitWolf(store_precision=False)
            lw.fit(Xc)
            C = lw.covariance_
        except Exception:
            C = (Xc.T @ Xc) / max(1, N - 1)
    else:
        C = (Xc.T @ Xc) / max(1, N - 1)

    # regularize
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, eps, None)
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return mu, W


def apply_whitener(X: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (X - mu) @ W


def invert_whitener(Z: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    # W is symmetric in our construction, but use pinv for numerical safety
    W_inv = np.linalg.pinv(W)
    return (Z @ W_inv.T) + mu


def l2_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n


# -------------------- Top-K SAE --------------------
class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, hidden_mult: float = 8.0, k: int = 32, hidden_dim: int | None = None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_mult = float(hidden_mult)
        self.m = int(hidden_dim) if hidden_dim is not None else int(round(self.input_dim * self.hidden_mult))
        self.k = int(k)
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
        assert z.size(1) == self.input_dim, f"Encoder expected in_features={self.input_dim}, got {z.size(1)}"
        a = F.relu(self.enc(z))
        if 0 < self.k < a.shape[1]:
            # top-k thresholding (keeps possible ties above kth)
            topk_vals, _ = torch.topk(a, self.k, dim=1)
            kth = topk_vals[:, -1].unsqueeze(1)
            mask = (a >= kth).float()
            a = a * mask
        return a

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        assert a.ndim == 2 and a.size(1) == self.m
        z_hat = self.dec(a)
        return z_hat

    def forward(self, z: torch.Tensor):
        z = self._ensure_2d(z)
        a = self.encode(z)
        z_hat = self.decode(a)
        return z_hat, a


# -------------------- Training SAE --------------------
def train_sae(model: TopKSAE, Z_train: np.ndarray, Z_val: np.ndarray, epochs: int = 30, batch_size: int = 4096, lr: float = 1e-3, device: str = 'cuda') -> TopKSAE:
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
    ds = TensorDataset(torch.from_numpy(Z_train).float())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    ds_val = TensorDataset(torch.from_numpy(Z_val).float())
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    best_val = float('inf'); best_state = None
    for ep in range(1, epochs+1):
        model.train()
        acc_loss, n = 0.0, 0
        for (z_batch,) in dl:
            z_batch = z_batch.to(device)
            z_hat, a = model(z_batch)
            loss = F.mse_loss(z_hat, z_batch)
            opt.zero_grad(); loss.backward(); opt.step()
            acc_loss += loss.item() * z_batch.size(0); n += z_batch.size(0)
        train_loss = acc_loss / max(1, n)

        model.eval(); acc_val = 0.0; nv = 0
        all_a = []
        with torch.no_grad():
            for (z_batch,) in dl_val:
                z_batch = z_batch.to(device)
                z_hat, a = model(z_batch)
                acc_val += F.mse_loss(z_hat, z_batch).item() * z_batch.size(0); nv += z_batch.size(0)
                all_a.append(a.cpu())
        val_loss = acc_val / max(1, nv)

        A = torch.cat(all_a, dim=0)  # shape: [N_val, latent_dim]
        mean_abs_per_latent = A.abs().mean(dim=0)  # [latent_dim]
        dead_mask = (mean_abs_per_latent < 1e-3)
        dead_frac = dead_mask.float().mean().item()
        avg_dead_latents = int(dead_mask.sum().item())

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(f"[SAE] ep {ep}/{epochs}, train {train_loss:.6f} val {val_loss:.6f}, dead_latents {avg_dead_latents}/{A.size(1)}, ({100*dead_frac:.1f}%)")
    if best_state is not None: model.load_state_dict(best_state)
    return model


@torch.no_grad()
def sae_encode_batch(model: TopKSAE, Z: np.ndarray, device: str = 'cuda', batch: int = 8192) -> np.ndarray:
    model = model.to(device).eval()
    Z = np.asarray(Z)
    if Z.ndim > 2:
        Z = Z.reshape(Z.shape[0], -1)
    A = []
    for i in tqdm(range(0, Z.shape[0], batch), desc="SAE encode"):
        z = torch.from_numpy(Z[i:i+batch]).float().to(device)
        a = model.encode(z)
        A.append(a.cpu().numpy())
    return np.vstack(A)


# -------------------- Decoder pruning & normalization --------------------
def prune_and_normalize_decoder(dec_weight: np.ndarray, prune_threshold: float = 1e-6) -> np.ndarray:
    cols = dec_weight.T  # [m, d]
    norms = np.linalg.norm(cols, axis=1)
    keep = norms > prune_threshold
    if keep.sum() < 1:
        keep = norms.argsort()[::-1][:max(1, int(len(norms)*0.1))]
    cols_kept = cols[keep]
    cols_kept = cols_kept / (np.linalg.norm(cols_kept, axis=1, keepdims=True) + 1e-12)
    return cols_kept.T  # [d, r]


# -------------------- Feature mining statistics --------------------

def cohens_d(x_pos: np.ndarray, x_neg: np.ndarray) -> float:
    n1, n2 = len(x_pos), len(x_neg)
    m1, m2 = x_pos.mean(), x_neg.mean()
    s1, s2 = x_pos.var(ddof=1), x_neg.var(ddof=1)
    s_pooled = math.sqrt(((n1-1)*s1 + (n2-1)*s2) / max(1, n1+n2-2))
    if s_pooled == 0: return 0.0
    return float((m1 - m2) / s_pooled)


def mann_whitney_u_p(x_pos: np.ndarray, x_neg: np.ndarray) -> tuple[float,float]:
    x = np.concatenate([x_pos, x_neg])
    labels = np.concatenate([np.ones(len(x_pos)), np.zeros(len(x_neg))])
    ranks = np.argsort(np.argsort(x)) + 1
    r1 = ranks[labels==1].sum()
    n1, n2 = len(x_pos), len(x_neg)
    U1 = r1 - n1*(n1+1)/2
    auc = U1 / (n1*n2 + 1e-12)
    mu = n1*n2/2
    sigma = math.sqrt(n1*n2*(n1+n2+1)/12)
    z = (U1 - mu) / (sigma + 1e-12)
    from math import erf, sqrt
    p = 2*(1 - 0.5*(1+erf(abs(z)/sqrt(2))))
    return auc, p


def kl_gaussian_1d(p: np.ndarray, q: np.ndarray) -> float:
    mp, sp = p.mean(), p.var(ddof=1)
    mq, sq = q.mean(), q.var(ddof=1)
    sp = max(sp, 1e-9); sq = max(sq, 1e-9)
    return 0.5 * (math.log(sq/sp) + (sp + (mp-mq)**2)/sq - 1.0)


def feature_stats(A_pos: np.ndarray, A_neg: np.ndarray):
    m = A_pos.shape[1]
    stats = {"cohens_d": np.zeros(m), "auc": np.zeros(m), "pval": np.ones(m), "kl": np.zeros(m)}
    for j in range(m):
        x_pos = A_pos[:, j]; x_neg = A_neg[:, j]
        stats["cohens_d"][j] = cohens_d(x_pos, x_neg)
        try:
            stats["auc"][j] = float(roc_auc_score(np.concatenate([np.ones(len(x_pos)), np.zeros(len(x_neg))]),np.concatenate([x_pos, x_neg])))
        except Exception:
            stats["auc"][j] = 0.5
        auc, p = mann_whitney_u_p(x_pos, x_neg)
        stats["pval"][j] = p
        stats["kl"][j] = kl_gaussian_1d(x_pos, x_neg)
    return stats


def select_features_from_stats(stats: dict, top_k: int = 200, fdr: float = 0.05):
    pvals = stats["pval"]
    if multipletests is not None:
        rej, p_corr, _, _ = multipletests(pvals, alpha=fdr, method='fdr_bh')
        idx = np.where(rej)[0]
        if len(idx) == 0:
            print(f'Using pvals fallback to |AUC - 0.5| scores')
            idx = np.argsort(-np.abs(stats["auc"] - 0.5))[:top_k]
        else:
            print(f'Using FDR-corrected features, sorted by |AUC-0.5|')
            idx = sorted(list(idx), key=lambda j: -abs(stats["auc"][j] - 0.5))
            idx = idx[:top_k]
    else:
        print(f'Using AUC - 0.5 scores (statsmodels not available)')
        idx = np.argsort(-np.abs(stats["auc"] - 0.5))[:top_k]
    return list(idx)


# -------------------- Build U from decoder --------------------
def build_U_from_decoder(dec_weight: np.ndarray, feat_idx: list[int]) -> np.ndarray:
    if len(feat_idx) == 0:
        return np.zeros((dec_weight.shape[0], 0))
    D = dec_weight.copy()                              # [d, m]
    norms = np.linalg.norm(D, axis=0, keepdims=True)   # [1, m]
    D = D / (norms + 1e-12)
    D_S = D[:, feat_idx]                               # [d, r]
    # SVD and return orthonormal basis of actual rank
    Uu, svals, Vt = np.linalg.svd(D_S, full_matrices=False)
    rank = (svals > 1e-12).sum()
    if rank == 0:
        return np.zeros((D_S.shape[0], 0))
    return Uu[:, :rank]


# -------------------- LEACE in U --------------------
def leace_projector_in_U(Z: np.ndarray, U: np.ndarray, Y: np.ndarray, ridge: float = 1e-6):
    if U.shape[1] == 0:
        return np.eye(0), np.zeros((1, 0))
    ZU = Z @ U                       # [N, r]
    mu_r = ZU.mean(axis=0, keepdims=True)
    ZUc = ZU - mu_r
    Yc  = Y - Y.mean(axis=0, keepdims=True)
    N = ZU.shape[0]
    Sig = (ZUc.T @ ZUc) / max(1, N - 1)           # [r, r]
    Sig_ZY = (ZUc.T @ Yc) / max(1, N - 1)         # [r, k]
    evals, evecs = np.linalg.eigh(Sig + ridge * np.eye(Sig.shape[0]))
    evals = np.clip(evals, 1e-12, None)
    W_r     = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T   # Σ^{-1/2}
    W_r_pinv = evecs @ np.diag(np.sqrt(evals)) @ evecs.T        # Σ^{+1/2}
    M = W_r @ Sig_ZY
    # Projection onto column-space of M via SVD
    U_M, _, _ = np.linalg.svd(M, full_matrices=False)
    if U_M.size == 0:
        P = np.zeros((Sig.shape[0], Sig.shape[0]))
    else:
        P = U_M @ U_M.T
    P_r = np.eye(U.shape[1]) - (W_r_pinv @ P @ W_r)
    return P_r, mu_r


def apply_subspace_leace(Z: np.ndarray, U: np.ndarray, P_r: np.ndarray, mu_r: np.ndarray | None = None) -> np.ndarray:
    if U.shape[1] == 0: return Z
    ZU = Z @ U
    if mu_r is None: mu_r = np.zeros((1, U.shape[1]), dtype=ZU.dtype)
    ZU_prime = (ZU - mu_r) @ P_r.T + mu_r
    Z_prime  = Z - (ZU @ U.T) + (ZU_prime @ U.T)
    return Z_prime


# -------------------- SAUCE gating --------------------
@torch.no_grad()
def apply_sauce_gating(sae_model: TopKSAE, Z: np.ndarray, feat_idx: list[int], beta: float = 0.0, device: str = 'cuda', batch: int = 8192) -> np.ndarray:
    assert 0.0 <= beta <= 1.0
    sae_model = sae_model.to(device).eval()
    Z_out = []
    for i in tqdm(range(0, Z.shape[0], batch), desc="SAUCE gating"):
        z = torch.from_numpy(Z[i:i+batch]).float().to(device)
        a = sae_model.encode(z)
        if len(feat_idx) > 0:
            a[:, feat_idx] = a[:, feat_idx] * beta
        z_hat = sae_model.decode(a)
        Z_out.append(z_hat.cpu().numpy())
    return np.vstack(Z_out)


# -------------------- Noise injection --------------------
def inject_noise_in_subspace(Z: np.ndarray, U: np.ndarray, sigma: float = 0.02, rng=None) -> np.ndarray:
    rng = np.random if rng is None else rng
    ZU = Z @ U  # [N, r]
    noise = rng.randn(*ZU.shape) * sigma
    return Z + (noise @ U.T)


from sklearn.metrics import roc_curve

def compute_scores(embeddings, trials):
    scores = []
    for i, j in trials:
        scores.append(float(np.dot(embeddings[i], embeddings[j])))
    return np.array(scores)


def eer_from_scores(scores: np.ndarray, labels: np.ndarray, return_threshold: bool = False):
    """
    Compute Equal Error Rate (EER) from precomputed scores and trial labels.
    - scores: array of similarity scores
    - labels: array of 0/1 (1=positive, 0=negative)
    """
    from sklearn.metrics import roc_curve
    if len(scores) == 0 or len(labels) == 0 or np.all(labels == 0) or np.all(labels == 1):
        return (float('nan'), float('nan')) if return_threshold else float('nan')
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    abs_diffs = np.abs(fnr - fpr)
    idx = np.argmin(abs_diffs)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    thr = thresholds[idx]
    return (eer, thr) if return_threshold else eer


# -------------------- Speaker classifier --------------------
class ResNetClassifier(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(emb_dim,512), nn.BatchNorm1d(512), nn.ReLU())
        self.res = nn.Sequential(nn.Linear(512,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512,512), nn.BatchNorm1d(512))
        self.relu = nn.ReLU()
        self.out = nn.Linear(512, num_classes)
    def forward(self, x):
        o = self.layer1(x); r = self.res(o); o = self.relu(o + r); return self.out(o)


def train_classifier(X_train: np.ndarray, y_train: list[int], X_val: np.ndarray, y_val: list[int], num_classes: int, device: str = 'cuda', epochs: int = 20, batch_size: int = 512, lr: float = 1e-3):
    model = ResNetClassifier(X_train.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    ds = TensorDataset(torch.from_numpy(X_train).float(), torch.tensor(y_train).long())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    best_acc = 0.0; best_state = None
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb); loss = loss_fn(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(torch.from_numpy(X_val).float().to(device)).argmax(1).cpu().numpy()
        acc = accuracy_score(y_val, preds)
        print(f"[Classifier] ep {ep} val acc {acc*100:.2f}%")
        if acc > best_acc:
            best_acc = acc; best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    return model


def evaluate_classifier(model: ResNetClassifier, X: np.ndarray, y: list[int], device: str = 'cuda'):
    model.eval().to(device)
    with torch.no_grad():
        preds = model(torch.from_numpy(X).float().to(device)).argmax(1).cpu().numpy()
    acc = accuracy_score(y, preds)
    return acc


# -------------------- New helpers: effective rank & participation  --------------------

def effective_rank(svals: np.ndarray, threshold: float = 0.01) -> int:
    svals = np.array(svals)
    if svals.size == 0:
        return 0
    mx = svals.max()
    return int((svals > (threshold * mx)).sum())


def participation_ratio(energy: np.ndarray) -> float:
    num = (np.sum(energy) ** 2)
    den = np.sum(energy ** 2) + 1e-12
    return float(num / den)


# -------------------- Jacobian sensitivity estimate (finite differences) --------------------
def estimate_jacobian_alignment(sae: TopKSAE, Z_samples: np.ndarray, deltas: np.ndarray, eps: float = 1e-4, device: str = 'cpu') -> np.ndarray:
    sae = sae.to(device).eval()
    Z = Z_samples.copy()
    alignments = []
    batch = 64
    for i in range(0, Z.shape[0], batch):
        z = torch.from_numpy(Z[i:i+batch]).float().to(device)
        delta = torch.from_numpy(deltas[i:i+batch]).float().to(device)
        with torch.no_grad():
            a0 = sae.encode(z)  # [B, m]
            a_eps = sae.encode((z + eps * delta))
        jac_v = (a_eps - a0) / (eps + 1e-12)  # [B, m]
        num = torch.norm(jac_v, dim=1).cpu().numpy()
        den = torch.norm(delta, dim=1).cpu().numpy() + 1e-12
        alignments.append(num / den)
    return np.concatenate(alignments, axis=0)


# -------------------- Plotting helpers --------------------

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_whitening_spectrum(evals: np.ndarray, outpath: str):
    ensure_dir(os.path.dirname(outpath) or ".")
    plt.figure(figsize=(10,6))
    plt.plot(np.arange(1, len(evals)+1), evals, marker='o')
    plt.axhline(1.0, color='red', linestyle='--', label='1.0')
    plt.xlabel('Eigen-index'); plt.ylabel('Eigenvalue')
    plt.title('Whitening eigen-spectrum')
    plt.legend()
    plt.grid(True)
    plt.tight_layout(); plt.savefig(outpath); plt.close()


def plot_decoder_overlap_matrix(D_selected: np.ndarray, outpath: str, top_n: int = 50):
    ensure_dir(os.path.dirname(outpath) or ".")
    cols = D_selected.copy()
    cols = cols / (np.linalg.norm(cols, axis=0, keepdims=True) + 1e-12)
    S = cols.T @ cols
    np.fill_diagonal(S, 0.0)
    plt.figure(figsize=(8,8))
    im = plt.imshow(S, cmap='bwr', vmin=-1, vmax=1)
    plt.title('Decoder column cosine similarities (selected atoms)')
    plt.colorbar(im)
    plt.tight_layout(); plt.savefig(outpath); plt.close()


def plot_U_svd_energy(D_selected: np.ndarray, outpath: str):
    ensure_dir(os.path.dirname(outpath) or ".")
    Uu, svals, Vt = np.linalg.svd(D_selected, full_matrices=False)
    cum = np.cumsum(svals) / (svals.sum() + 1e-12)
    plt.figure(figsize=(10,6))
    plt.plot(np.arange(1, len(svals)+1), svals, marker='o', label='singular values')
    plt.twinx()
    plt.plot(np.arange(1, len(svals)+1), cum, color='orange', marker='x', label='cumulative energy')
    plt.ylabel('cumulative energy (right)')
    plt.title('SVD of selected decoder atoms')
    plt.grid(True)
    plt.tight_layout(); plt.savefig(outpath); plt.close()


def plot_edit_magnitude_histogram(edit_norms: np.ndarray, outpath: str):
    ensure_dir(os.path.dirname(outpath) or ".")
    plt.figure(figsize=(8,5))
    plt.hist(edit_norms, bins=100)
    plt.title('Edit magnitude per sample (L2 norm)')
    plt.xlabel('||Z_edit - Z||'); plt.ylabel('Count')
    plt.tight_layout(); plt.savefig(outpath); plt.close()


def plot_activation_change_histogram(A_before: np.ndarray, A_after: np.ndarray, feat_idx: List[int], outpath: str):
    ensure_dir(os.path.dirname(outpath) or ".")
    if len(feat_idx) == 0:
        return
    sel_before = A_before[:, feat_idx]
    sel_after = A_after[:, feat_idx]
    abs_change = np.mean(np.abs(sel_after - sel_before), axis=0)
    sign_flip_frac = np.mean((np.sign(sel_before) != np.sign(sel_after)).astype(float), axis=0)
    plt.figure(figsize=(10,6))
    plt.bar(np.arange(len(feat_idx)), abs_change)
    plt.title('Mean absolute activation change for selected atoms')
    plt.xlabel('Selected atom index'); plt.ylabel('mean |a_after - a_before|')
    plt.tight_layout(); plt.savefig(outpath); plt.close()
    plt.figure(figsize=(10,6))
    plt.bar(np.arange(len(feat_idx)), sign_flip_frac)
    plt.title('Sign flip fraction for selected atoms')
    plt.xlabel('Selected atom index'); plt.ylabel('sign flip fraction')
    plt.tight_layout(); plt.savefig(outpath.replace('.png', '_signflip.png')); plt.close()


def plot_jacobian_alignment_histogram(alignments: np.ndarray, outpath: str):
    ensure_dir(os.path.dirname(outpath) or ".")
    plt.figure(figsize=(8,5))
    plt.hist(alignments, bins=100)
    plt.title('Jacobian alignment histogram: ||J v|| / ||v|| (approx)')
    plt.xlabel('alignment'); plt.ylabel('count')
    plt.tight_layout(); plt.savefig(outpath); plt.close()


def plot_superposition_energy_sorted(proj_coeffs: np.ndarray, outpath: str):
    ensure_dir(os.path.dirname(outpath) or ".")
    energy = (proj_coeffs ** 2)
    order = np.argsort(energy)[::-1]
    sorted_energy = energy[order]
    cum = np.cumsum(sorted_energy) / (sorted_energy.sum() + 1e-12)
    plt.figure(figsize=(10,6))
    plt.bar(np.arange(len(sorted_energy)), sorted_energy)
    plt.plot(np.arange(len(sorted_energy)), cum, color='orange', linewidth=2)
    plt.title('Energy in decoder atoms for label direction (sorted)')
    plt.xlabel('atom rank'); plt.ylabel('energy (squared projection)')
    plt.tight_layout(); plt.savefig(outpath); plt.close()


def plot_embedding_tsne_before_after(Z_before: np.ndarray, Z_after: np.ndarray, labels: List[str], outpath: str, sample: int = 2000):
    ensure_dir(os.path.dirname(outpath) or ".")
    try:
        from sklearn.manifold import TSNE
    except Exception:
        print("tsne not available; skipping")
        return
    N = min(sample, Z_before.shape[0])
    idx = np.random.choice(Z_before.shape[0], N, replace=False)
    X = np.vstack([Z_before[idx], Z_after[idx]])
    ts = TSNE(n_components=2, init='pca', perplexity=30, n_iter=1000, random_state=0).fit_transform(X)
    ts_b = ts[:N]; ts_a = ts[N:]
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.scatter(ts_b[:,0], ts_b[:,1], c=[1 if labels[i] == 'forget' else 0 for i in idx], s=6)
    plt.title('pre-edit tSNE (sample)')
    plt.subplot(1,2,2)
    plt.scatter(ts_a[:,0], ts_a[:,1], c=[1 if labels[i] == 'forget' else 0 for i in idx], s=6)
    plt.title('post-edit tSNE (sample)')
    plt.tight_layout(); plt.savefig(outpath); plt.close()


def plot_participation_ratio_hist(part_ratios: np.ndarray, outpath: str):
    ensure_dir(os.path.dirname(outpath) or ".")
    plt.figure(figsize=(8,5))
    plt.hist(part_ratios, bins=100)
    plt.title('Per-sample participation ratio (energy spread across atoms)')
    plt.xlabel('participation ratio'); plt.ylabel('count')
    plt.tight_layout(); plt.savefig(outpath); plt.close()


# -------------------- MAIN diagnostics (fixed & extended) --------------------
def compute_numerical_diagnostics(Z_all: np.ndarray, Z_edit: np.ndarray, U: np.ndarray,
                                 sae: TopKSAE, feat_idx: List[int], y_binary: np.ndarray,
                                 is_dev: np.ndarray, is_test: np.ndarray,
                                 labels: List[str], forget_digits: List[str],
                                 mu: np.ndarray, W: np.ndarray, P_r: np.ndarray, mu_r: np.ndarray,
                                 E_all: np.ndarray, E_edit: np.ndarray,
                                 plot_dir: str = "./diagnostics_plots") -> Dict:
    """
    Extended diagnostics with corrected logic and extra checks (effective rank, Jacobian alignment, participation ratio).
    Saves plots into plot_dir and returns a diagnostics dict.
    """
    ensure_dir(plot_dir)
    diagnostics = {}
    # ---------------- Whitening ----------------
    print("  Computing whitening sanity checks...")
    Z_centered = Z_all - Z_all.mean(axis=0, keepdims=True)
    C = (Z_centered.T @ Z_centered) / max(1, Z_all.shape[0] - 1)
    evals, evecs = np.linalg.eigh(C)
    diagnostics['whitening_eigenvalues'] = evals
    diagnostics['whitening_total_variance'] = float(np.trace(C))
    diagnostics['whitening_condition_number'] = float(evals.max() / (evals.min() + 1e-12))
    plot_whitening_spectrum(evals, os.path.join(plot_dir, "whitening_spectrum.png"))

    # ---------------- Decoder geometry ----------------
    print("  Computing decoder geometry analysis...")
    if len(feat_idx) > 0:
        dec_w = sae.dec.weight.detach().cpu().numpy()  # [d, m]
        D_selected = dec_w[:, feat_idx]  # [d, r]
        col_norms = np.linalg.norm(D_selected, axis=0)
        diagnostics['decoder_col_norms_mean'] = float(np.mean(col_norms))
        diagnostics['decoder_col_norms_std'] = float(np.std(col_norms))
        D_norm = D_selected / (col_norms.reshape(1, -1) + 1e-12)
        similarities = D_norm.T @ D_norm
        np.fill_diagonal(similarities, 0)
        diagnostics['decoder_mean_offdiag_similarity'] = float(np.mean(similarities))
        diagnostics['decoder_max_overlap'] = float(np.max(similarities))
        diagnostics['decoder_top5_overlaps'] = np.sort(similarities.flatten())[-5:].tolist()
        plot_decoder_overlap_matrix(D_selected, os.path.join(plot_dir, "decoder_overlap.png"))

    # ---------------- U construction ----------------
    print("  Computing U construction checks...")
    if U.shape[1] > 0:
        diagnostics['U_dimension'] = U.shape[1]
        ZU = Z_all @ U
        ZU_var = np.var(ZU, axis=0)
        total_var = np.var(Z_all, axis=0).sum()
        diagnostics['U_variance_explained'] = float(ZU_var.sum() / (total_var + 1e-12))
        ZU_energy = np.sum(ZU**2, axis=1)
        Z_total_energy = np.sum(Z_all**2, axis=1)
        U_energy_fraction = ZU_energy / (Z_total_energy + 1e-12)
        diagnostics['U_energy_fraction_mean'] = float(np.mean(U_energy_fraction))
        diagnostics['U_energy_fraction_std'] = float(np.std(U_energy_fraction))

        # Effective rank of selected decoder atoms (use D_selected if available)
        if len(feat_idx) > 0:
            D_selected = sae.dec.weight.detach().cpu().numpy()[:, feat_idx]
            Uu, svals, Vt = np.linalg.svd(D_selected, full_matrices=False)
            diagnostics['D_selected_svals'] = svals
            diagnostics['D_selected_effective_rank'] = effective_rank(svals, threshold=0.01)
            plot_U_svd_energy(D_selected, os.path.join(plot_dir, "D_selected_svd.png"))

    # ---------------- Label correlation ----------------
    print("  Computing label correlation analysis...")
    if U.shape[1] > 0:
        Z_dev = Z_all[is_dev]
        Y_dev = y_binary[is_dev].reshape(-1, 1)
        Z_centered_dev = Z_dev - Z_dev.mean(axis=0, keepdims=True)
        Y_centered = Y_dev - Y_dev.mean(axis=0, keepdims=True)
        C_ZY_full = (Z_centered_dev.T @ Y_centered) / max(1, Z_dev.shape[0] - 1)
        diagnostics['label_covariance_norm_full'] = float(np.linalg.norm(C_ZY_full))
        ZU_dev = Z_dev @ U
        ZU_centered = ZU_dev - ZU_dev.mean(axis=0, keepdims=True)
        C_ZY_U = (ZU_centered.T @ Y_centered) / max(1, ZU_dev.shape[0] - 1)
        diagnostics['label_covariance_norm_U'] = float(np.linalg.norm(C_ZY_U))
        if diagnostics['label_covariance_norm_full'] > 0:
            diagnostics['label_covariance_fraction_in_U'] = float(diagnostics['label_covariance_norm_U'] / diagnostics['label_covariance_norm_full'])
        else:
            diagnostics['label_covariance_fraction_in_U'] = 0.0

    # ---------------- LEACE operator ----------------
    print("  Computing LEACE operator checks...")
    if U.shape[1] > 0 and P_r.shape[0] == U.shape[1]:
        diagnostics['leace_Pr_frobenius_norm'] = float(np.linalg.norm(P_r, 'fro'))
        diagnostics['leace_Pr_spectral_norm'] = float(np.linalg.norm(P_r, 2))
        if 'label_covariance_norm_U' in diagnostics:
            ZU_edit_dev = Z_edit[is_dev] @ U
            ZU_edit_centered = ZU_edit_dev - ZU_edit_dev.mean(axis=0, keepdims=True)
            C_ZY_U_after = (ZU_edit_centered.T @ Y_centered) / max(1, ZU_edit_dev.shape[0] - 1)
            C_ZY_U_after_norm = float(np.linalg.norm(C_ZY_U_after))
            diagnostics['label_covariance_norm_U_after'] = C_ZY_U_after_norm
            if diagnostics['label_covariance_norm_U'] > 0:
                diagnostics['leace_covariance_reduction_pct'] = float((1 - C_ZY_U_after_norm / diagnostics['label_covariance_norm_U']) * 100)
            else:
                diagnostics['leace_covariance_reduction_pct'] = 0.0

    # ---------------- Post-edit diagnostics ----------------
    print("  Computing post-edit diagnostics...")
    edit_magnitude = np.linalg.norm(Z_edit - Z_all, axis=1)
    diagnostics['edit_magnitude_mean'] = float(np.mean(edit_magnitude))
    diagnostics['edit_magnitude_std'] = float(np.std(edit_magnitude))
    diagnostics['edit_magnitude_max'] = float(np.max(edit_magnitude))
    plot_edit_magnitude_histogram(edit_magnitude, os.path.join(plot_dir, "edit_magnitude_hist.png"))

    if U.shape[1] > 0:
        ZU_edit = Z_edit @ U
        ZU_edit_energy = np.sum(ZU_edit**2, axis=1)
        Z_edit_energy = np.sum((Z_edit - Z_all)**2, axis=1)
        U_edit_fraction = ZU_edit_energy / (Z_edit_energy + 1e-12)
        diagnostics['edit_energy_fraction_in_U_mean'] = float(np.mean(U_edit_fraction))
        diagnostics['edit_energy_fraction_in_U_std'] = float(np.std(U_edit_fraction))

    # ---------------- Selected atom activations ----------------
    print("  Computing selected atom activation changes...")
    if len(feat_idx) > 0:
        Z_dev_before = Z_all[is_dev]
        Z_dev_after = Z_edit[is_dev]
        A_before = sae_encode_batch(sae, Z_dev_before, device='cpu', batch=8192)
        A_after = sae_encode_batch(sae, Z_dev_after, device='cpu', batch=8192)
        A_selected_before = A_before[:, feat_idx]
        A_selected_after = A_after[:, feat_idx]
        diagnostics['selected_atoms_mean_activation_before'] = float(np.mean(A_selected_before))
        diagnostics['selected_atoms_mean_activation_after'] = float(np.mean(A_selected_after))
        activation_change = A_selected_after - A_selected_before
        diagnostics['selected_atoms_activation_change_mean'] = float(np.mean(activation_change))
        diagnostics['selected_atoms_activation_change_std'] = float(np.std(activation_change))
        sign_flips = np.sum(np.sign(A_selected_before) != np.sign(A_selected_after))
        total_activations = A_selected_before.size
        diagnostics['selected_atoms_sign_flip_fraction'] = float(sign_flips / (total_activations + 1e-12))
        plot_activation_change_histogram(A_before, A_after, feat_idx, os.path.join(plot_dir, "activation_changes.png"))

    # ---------------- Superposition quantification ----------------
    print("  Computing superposition quantification...")
    if len(feat_idx) > 0 and U.shape[1] > 0:
        labels_arr = np.array(labels)
        retain_mask = np.logical_not(np.isin(labels_arr, forget_digits))
        forget_mask = np.isin(labels_arr, forget_digits)
        if retain_mask.sum() > 0 and forget_mask.sum() > 0:
            retain_mean = Z_all[retain_mask].mean(axis=0)
            forget_mean = Z_all[forget_mask].mean(axis=0)
            label_direction = forget_mean - retain_mean
            label_direction = label_direction / (np.linalg.norm(label_direction) + 1e-12)
            D_selected = sae.dec.weight.detach().cpu().numpy()[:, feat_idx]
            D_norm = D_selected / (np.linalg.norm(D_selected, axis=0, keepdims=True) + 1e-12)
            proj_coeffs = label_direction @ D_norm
            diagnostics['superposition_proj_coeffs'] = proj_coeffs
            energy_distribution = proj_coeffs ** 2
            sorted_indices = np.argsort(energy_distribution)[::-1]
            sorted_energy = energy_distribution[sorted_indices]
            total_energy = np.sum(sorted_energy)
            if total_energy > 0:
                for k in [1,5,10,20,50]:
                    k1 = min(k, len(sorted_energy))
                    diagnostics[f'superposition_top{k1}_energy_fraction'] = float(np.sum(sorted_energy[:k1]) / (total_energy + 1e-12))
                diagnostics['superposition_participation_ratio'] = participation_ratio(sorted_energy)
                p = sorted_energy / (total_energy + 1e-12)
                p = p[p > 0]
                if len(p) > 0:
                    entropy = -np.sum(p * np.log(p))
                    diagnostics['superposition_entropy'] = float(entropy)
                    diagnostics['superposition_effective_components'] = float(np.exp(entropy))
                else:
                    diagnostics['superposition_entropy'] = 0.0
                    diagnostics['superposition_effective_components'] = 0.0
                plot_superposition_energy_sorted(proj_coeffs, os.path.join(plot_dir, "superposition_energy.png"))

    # ---------------- Jacobian alignment ----------------
    print("  Computing Jacobian alignment (approx finite-diff)...")
    try:
        deltas = (Z_edit - Z_all)
        N = deltas.shape[0]
        sample_idx = np.random.choice(N, min(512, N), replace=False)
        alignments = estimate_jacobian_alignment(sae, Z_all[sample_idx], deltas[sample_idx], eps=1e-4, device='cpu')
        diagnostics['jacobian_alignment_samples'] = alignments
        diagnostics['jacobian_alignment_median'] = float(np.median(alignments))
        plot_jacobian_alignment_histogram(alignments, os.path.join(plot_dir, "jacobian_alignment.png"))
    except Exception as e:
        print("  Jacobian alignment estimation failed:", e)

    # ---------------- Participation ratio per sample ----------------
    print("  Computing participation ratio per sample (using selected atoms)...")
    if len(feat_idx) > 0:
        A_all = sae_encode_batch(sae, Z_all, device='cpu', batch=8192)
        E_sel = (A_all[:, feat_idx] ** 2)
        part_ratios = []
        for i in range(E_sel.shape[0]):
            part_ratios.append(participation_ratio(E_sel[i]))
        part_ratios = np.array(part_ratios)
        diagnostics['participation_ratio_mean'] = float(np.mean(part_ratios))
        diagnostics['participation_ratio_std'] = float(np.std(part_ratios))
        plot_participation_ratio_hist(part_ratios, os.path.join(plot_dir, "participation_ratio_hist.png"))

    # ---------------- t-SNE pre/post ----------------
    print("  Saving tSNE pre/post plots (sample)...")
    try:
        binary_labels = ['forget' if (l in forget_digits) else 'retain' for l in labels]
        plot_embedding_tsne_before_after(Z_all, Z_edit, binary_labels, os.path.join(plot_dir, "tsne_before_after.png"), sample=2000)
    except Exception as e:
        print("  tSNE plot skipped:", e)

    # ---------------- Summary (corrected logic) ----------------
    print("\n========== NUMERICAL DIAGNOSTICS SUMMARY ==========")
    evals = diagnostics.get('whitening_eigenvalues', np.array([]))
    cond = diagnostics.get('whitening_condition_number', np.nan)
    total_var = diagnostics.get('whitening_total_variance', np.nan)
    if evals.size > 0:
        eig_mean = np.mean(evals)
        eig_std = np.std(evals)
        eig_min = np.min(evals)
        eig_max = np.max(evals)
        print(f"Whitening eigenvalues: mean={eig_mean:.3f}, std={eig_std:.3f}, min={eig_min:.3f}, max={eig_max:.3f}")
        if eig_std < 0.1 and abs(eig_mean - 1) < 0.1:
            print("  [good] Eigenvalues close to 1 (well-whitened)")
        else:
            print("  [warning] Whitening not ideal — consider shrinkage/reg parameter")
    print(f"Whitening condition number: {cond:.2f}")
    if cond < 2:
        print("  [good] Condition number near 1")
    elif cond < 5:
        print("  [ok] Condition number moderate")
    else:
        print("  [warning] Condition number high (ill-conditioned)")

    col_mean = diagnostics.get('decoder_col_norms_mean', np.nan)
    col_std = diagnostics.get('decoder_col_norms_std', np.nan)
    mean_sim = diagnostics.get('decoder_mean_offdiag_similarity', np.nan)
    max_ov = diagnostics.get('decoder_max_overlap', np.nan)
    print(f"Decoder column norm: mean={col_mean:.3f}, std={col_std:.3f}")
    if abs(col_mean - 1) < 0.2:
        print("  [ok] Decoder columns reasonably normalized")
    else:
        print("  [warning] Decoder columns not normalized")
    print(f"Decoder mean off-diag similarity: {mean_sim:.3f}")
    if abs(mean_sim) < 0.05:
        print("  [good] Low average overlap (atoms fairly separated)")
    elif abs(mean_sim) < 0.15:
        print("  [caution] Moderate overlap")
    else:
        print("  [warning] High average overlap (superposition present)")
    print(f"Decoder max overlap: {max_ov:.3f}")
    if max_ov < 0.3:
        print("  [good] No highly overlapping atom pairs")
    elif max_ov < 0.6:
        print("  [caution] Some pairs overlap")
    else:
        print("  [warning] Strong overlapping atoms (superposition)")

    u_dim = diagnostics.get('U_dimension', 0)
    u_var = diagnostics.get('U_variance_explained', 0)
    u_energy = diagnostics.get('U_energy_fraction_mean', 0)
    print(f"U dimension: {u_dim}, variance explained: {u_var:.3f}, mean energy fraction: {u_energy:.3f}")
    if u_dim == 0:
        print("  [warning] U is empty")
    if u_var >= 0.7:
        print("  [good] U captures a large fraction of variance (targetable)")
    elif u_var >= 0.4:
        print("  [ok] U captures moderate variance")
    else:
        print("  [warning] U captures little of data variance — may not contain label direction")

    frac_in_U = diagnostics.get('label_covariance_fraction_in_U', 0)
    print(f"Label covariance fraction in U: {frac_in_U:.3f}")
    if frac_in_U > 0.7:
        print("  [good] Majority of label covariance lies in U — suitable for subspace LEACE")
    elif frac_in_U > 0.4:
        print("  [caution] Some label covariance in U")
    else:
        print("  [warning] Little label covariance in U — LEACE in U may be ineffective")

    leace_red = diagnostics.get('leace_covariance_reduction_pct', 0)
    print(f"LEACE covariance reduction in U: {leace_red:.1f}%")
    if leace_red > 80:
        print("  [good] LEACE removed most linear label covariance in U")
    elif leace_red > 50:
        print("  [ok] LEACE partially removed covariance")
    else:
        print("  [warning] LEACE not effective — consider U change or gamma tuning")

    edit_mean = diagnostics.get('edit_magnitude_mean', 0)
    edit_std = diagnostics.get('edit_magnitude_std', 0)
    edit_max = diagnostics.get('edit_magnitude_max', 0)
    edit_energy = diagnostics.get('edit_energy_fraction_in_U_mean', 0)
    print(f"Edit magnitude: mean={edit_mean:.4f}, std={edit_std:.4f}, max={edit_max:.4f}")
    print(f"Edit energy fraction in U: {edit_energy:.3f}")
    if edit_energy > 0.5:
        print("  [good] Majority of edit energy lies in U (targeted edit)")
    else:
        print("  [warning] Edits leak outside U — could be non-targeted or indicate U too small")

    sign_flip = diagnostics.get('selected_atoms_sign_flip_fraction', 0)
    mean_act_change = diagnostics.get('selected_atoms_activation_change_mean', 0)
    print(f"Selected atom sign flip fraction: {sign_flip:.3f}")
    print(f"Selected atom mean activation change: {mean_act_change:.6f}")
    if abs(mean_act_change) < 1e-3 and sign_flip < 0.05:
        print("  [warning] Minimal activation changes — edits may not be affecting SAE activations (ineffective for unlearning)")
    elif abs(mean_act_change) < 0.05:
        print("  [caution] Small activation changes")
    else:
        print("  [good] Significant activation changes observed for selected atoms")

    top1 = diagnostics.get('superposition_top1_energy_fraction', 0)
    top5 = diagnostics.get('superposition_top5_energy_fraction', 0)
    part_ratio = diagnostics.get('superposition_participation_ratio', 0)
    entropy = diagnostics.get('superposition_entropy', 0)
    eff_comp = diagnostics.get('superposition_effective_components', 0)
    print(f"Superposition: top-1={top1:.3f}, top-5={top5:.3f}, participation ratio={part_ratio:.3f}, entropy={entropy:.3f}, effective components={eff_comp:.3f}")
    if part_ratio > 10:
        print("  [warning] High superposition: label energy spread over many atoms (hard to target)")
    elif top1 > 0.5:
        print("  [ok] Some concentration in top atom(s)")
    else:
        print("  [ok] Energy not dominated by a single atom")

    print("====================================================\n")
    # persist diagnostics
    try:
        with open(os.path.join(plot_dir, 'diagnostics_summary.json'), 'w') as f:
            json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in diagnostics.items()}, f)
    except Exception:
        pass
    return diagnostics
