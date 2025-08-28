#!/usr/bin/env python3
"""
AudioMNIST Digit Unlearning Pipeline — "forget digit 0"

This script adapts the SAE-first unlearning pipeline to perform *digit* unlearning
on the AudioMNIST dataset (./audiomnist_data/recordings). The target digit to
forget is configurable (default '0'). The pipeline:

  1. Builds manifests from AudioMNIST filenames (speaker_digit_rep.wav -> speaker, digit)
  2. Extracts ECAPA embeddings (speechbrain pretrained model)
  3. Whitens embeddings on retain-dev samples (non-forget digit)
  4. Trains a Top-K Sparse AutoEncoder (SAE) on retain-dev whitened embeddings
  5. Mines SAE features that discriminate the forget digit vs others
  6. Builds subspace U from decoder atoms corresponding to discriminative features
  7. Computes LEACE projector in U to remove signal predictive of the forget digit
  8. Applies SAUCE gating (optional), LEACE mapping and noise injection in subspace
  9. Inverts whitener -> gets edited embeddings, evaluates digit-detection and classification
 10. Evaluates "attacker retrain" ability to recover the forgotten digit from edited embeddings

Usage:
  pip install numpy torch torchaudio speechbrain scikit-learn statsmodels tqdm
  python audiomnist_digit_unlearn.py --data_root ./audiomnist_data --save_dir ./out --device cuda

Notes:
  - Make sure ./audiomnist_data/recordings contains AudioMNIST wavs with names like '001_0_0.wav'
  - This is compute-heavy (ECAPA extraction + SAE training). Adjust batch sizes/epochs for your GPU.
"""

from __future__ import annotations
import os
import sys
import time
import math
import json
import random
import glob
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    multipletests = None

import torchaudio
from speechbrain.pretrained import EncoderClassifier

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

# -------------------- Manifest builder for AudioMNIST --------------------
def build_manifest_from_audiomnist(data_root: str) -> List[Dict]:
    """
    Build manifest entries for AudioMNIST recordings folder.
    Expects filenames like: '0_george_0.wav'
    Returns list of dicts: {"path": p, "speaker": spk, "digit": digit}
    """
    wavs = glob.glob(os.path.join(data_root, "*.wav"))
    entries = []
    for p in sorted(wavs):
        fname = Path(p).stem
        parts = fname.split("_")
        if len(parts) < 2:
            continue
        spk = parts[1]
        digit = parts[0]
        entries.append({"path": p, "speaker": spk, "digit": digit})
    if len(entries) == 0:
        raise RuntimeError(f"No AudioMNIST wav files found under {data_root}")
    return entries

# -------------------- ECAPA embedding extraction --------------------
def extract_ecapa_embeddings(entries, device='cuda'):
    """
    Given a list of manifest entries (with 'path'), return:
      E: [N, D] np.array of normalized embeddings
      labels: list of digit labels (strings)
      paths: list of file paths
    """
    encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                            run_opts={"device": device})
    E, labels, paths = [], [], []
    for e in tqdm(entries, desc="Extracting ECAPA"):
        wav, sr = torchaudio.load(e['path'])
        # speechbrain enc expects waveform shape [batch, time] or [time]
        if wav.ndim == 2 and wav.size(0) > 1:
            # mixdown to mono
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(device)
        with torch.no_grad():
            try:
                emb = encoder.encode_batch(wav).squeeze().cpu().numpy().reshape(-1)
            except Exception:
                # fallback: use wav[0:1] to ensure shape
                emb = encoder.encode_batch(wav[:1]).squeeze().cpu().numpy().reshape(-1)
        E.append(emb)
        labels.append(e.get('digit', "NA"))
        paths.append(e['path'])
    E = np.vstack(E)
    # l2 normalize
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    E = E / norms
    return E, labels, paths

# -------------------- Whitener --------------------
def fit_whitener(X: np.ndarray, eps: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array(X)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples to fit whitener")
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    C = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    evals, evecs = np.linalg.eigh(C)
    evals = np.clip(evals, eps, None)
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return mu, W

def apply_whitener(X: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (X - mu) @ W

def invert_whitener(Z: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    W_inv = np.linalg.pinv(W)
    return (Z @ W_inv.T) + mu

def l2_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

# -------------------- Top-K SAE --------------------
class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, hidden_mult: float = 8.0, k: int = 32, hidden_dim: Optional[int] = None):
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
        assert z.size(1) == self.input_dim
        a = F.relu(self.enc(z))
        if 0 < self.k < a.shape[1]:
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
        raise RuntimeError("SAE input dim mismatch")
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

        A = torch.cat(all_a, dim=0) if len(all_a) > 0 else torch.zeros((0, model.m))
        if A.size(0) > 0:
            mean_abs_per_latent = A.abs().mean(dim=0)
            dead_mask = (mean_abs_per_latent < 1e-3)
            dead_frac = dead_mask.float().mean().item()
            avg_dead_latents = dead_mask.sum().item()
        else:
            dead_frac = 0.0; avg_dead_latents = 0

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(f"[SAE] ep {ep}/{epochs}, train {train_loss:.6f} val {val_loss:.6f}, dead_latents {avg_dead_latents}/{model.m} ({100*dead_frac:.1f}%)")
    if best_state is not None:
        model.load_state_dict(best_state)
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
    if len(A) == 0:
        return np.zeros((0, model.m))
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
    if s_pooled == 0:
        return 0.0
    return float((m1 - m2) / s_pooled)

def mann_whitney_u_p(x_pos: np.ndarray, x_neg: np.ndarray) -> Tuple[float,float]:
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
    m = A_pos.shape[1] if A_pos.ndim == 2 else (A_neg.shape[1] if A_neg.ndim == 2 else 0)
    stats = {"cohens_d": np.zeros(m), "auc": np.zeros(m), "pval": np.ones(m), "kl": np.zeros(m)}
    for j in range(m):
        x_pos = A_pos[:, j] if A_pos.shape[0] > 0 else np.array([])
        x_neg = A_neg[:, j] if A_neg.shape[0] > 0 else np.array([])
        if x_pos.size == 0 or x_neg.size == 0:
            stats["cohens_d"][j] = 0.0
            stats["auc"][j] = 0.5
            stats["pval"][j] = 1.0
            stats["kl"][j] = 0.0
            continue
        stats["cohens_d"][j] = cohens_d(x_pos, x_neg)
        try:
            stats["auc"][j] = float(roc_auc_score(np.concatenate([np.ones(len(x_pos)), np.zeros(len(x_neg))]),
                                                   np.concatenate([x_pos, x_neg])))
        except Exception:
            stats["auc"][j] = 0.5
        auc, p = mann_whitney_u_p(x_pos, x_neg)
        stats["pval"][j] = p
        stats["kl"][j] = kl_gaussian_1d(x_pos, x_neg)
    return stats

def select_features_from_stats(stats: Dict, top_k: int = 200, fdr: float = 0.05):
    pvals = stats["pval"]
    if multipletests is not None and len(pvals) > 0:
        try:
            rej, p_corr, _, _ = multipletests(pvals, alpha=fdr, method='fdr_bh')
            idx = np.where(rej)[0]
            if len(idx) == 0:
                idx = np.argsort(-np.abs(stats["auc"] - 0.5))[:top_k]
            else:
                idx = sorted(list(idx), key=lambda j: -abs(stats["auc"][j] - 0.5))
                idx = idx[:top_k]
        except Exception:
            idx = np.argsort(-np.abs(stats["auc"] - 0.5))[:top_k]
    else:
        idx = np.argsort(-np.abs(stats["auc"] - 0.5))[:top_k]
    return list(idx)

# -------------------- Build U from decoder --------------------
def build_U_from_decoder(dec_weight: np.ndarray, feat_idx: List[int], whiten_metric: Optional[np.ndarray] = None):
    """
    dec_weight: [d, m], feat_idx: indices into m (SAE hidden units)
    Returns U: [d, r] orthonormal columns (SVD of the selected decoder columns)
    """
    if len(feat_idx) == 0:
        return np.zeros((dec_weight.shape[0], 0))
    D_S = dec_weight[:, feat_idx]  # [d, r]
    U, s, Vt = np.linalg.svd(D_S, full_matrices=False)
    U = U[:, :D_S.shape[1]]
    return U

# -------------------- LEACE in U --------------------
def leace_projector_in_U(Z: np.ndarray, U: np.ndarray, Y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """
    Z: [N, d] whitened embeddings
    U: [d, r] subspace
    Y: [N, k] one-hot or numeric labels (here k=1 for binary: forget-digit indicator)
    returns P_r: [r, r] projector to remove predictive directions
    """
    ZU = Z @ U  # [N, r]
    N = ZU.shape[0]
    ZUc = ZU - ZU.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    C_ZY = (ZUc.T @ Yc) / max(1, N - 1)  # [r, k]
    C_YY = (Yc.T @ Yc) / max(1, N - 1)  # [k, k]
    C_YY_reg = C_YY + ridge * np.eye(C_YY.shape[0])
    C_YY_pinv = np.linalg.pinv(C_YY_reg)
    B = C_ZY @ C_YY_pinv @ C_ZY.T  # [r, r]
    P_r = np.eye(U.shape[1]) - B
    P_r = 0.5 * (P_r + P_r.T)
    return P_r

def apply_subspace_leace(Z: np.ndarray, U: np.ndarray, P_r: np.ndarray) -> np.ndarray:
    ZU = Z @ U
    ZU_prime = ZU @ P_r.T
    Z_prime = Z - (ZU @ U.T) + (ZU_prime @ U.T)
    return Z_prime

# -------------------- SAUCE gating --------------------
@torch.no_grad()
def apply_sauce_gating(sae_model: TopKSAE, Z: np.ndarray, feat_idx: List[int], beta: float = 0.0, device: str = 'cuda', batch: int = 8192) -> np.ndarray:
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
    if len(Z_out) == 0:
        return np.zeros_like(Z)
    return np.vstack(Z_out)

# -------------------- Noise injection --------------------
def inject_noise_in_subspace(Z: np.ndarray, U: np.ndarray, sigma: float = 0.02, rng=None) -> np.ndarray:
    rng = np.random if rng is None else rng
    if U.shape[1] == 0:
        return Z
    ZU = Z @ U
    noise = rng.randn(*ZU.shape) * sigma
    return Z + (noise @ U.T)

# -------------------- Evaluation helpers --------------------
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fnr[idx] + fpr[idx]) / 2.0
    return float(eer), float(thr[idx])

def min_dcf(scores: np.ndarray, labels: np.ndarray, p_tar: float = 0.01, c_miss: float = 1., c_fa: float = 1.) -> float:
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    c_det = c_miss * p_tar * fnr + c_fa * (1 - p_tar) * fpr
    return float(np.min(c_det))

def calibrate_scores(scores_dev: np.ndarray, labels_dev: np.ndarray, scores_test: np.ndarray) -> np.ndarray:
    if len(scores_dev.shape) == 1: scores_dev = scores_dev.reshape(-1,1)
    if len(scores_test.shape) == 1: scores_test = scores_test.reshape(-1,1)
    clf = LogisticRegression(solver='liblinear')
    clf.fit(scores_dev, labels_dev)
    prob = clf.predict_proba(scores_test)[:,1]
    eps = 1e-9
    logits = np.log(prob + eps) - np.log(1 - prob + eps)
    return logits

# -------------------- Digit classifier (ResNet-like) --------------------
class ResNetClassifier(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(emb_dim,512), nn.BatchNorm1d(512), nn.ReLU())
        self.res = nn.Sequential(nn.Linear(512,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512,512), nn.BatchNorm1d(512))
        self.relu = nn.ReLU()
        self.out = nn.Linear(512, num_classes)
    def forward(self, x):
        o = self.layer1(x)
        r = self.res(o)
        o = self.relu(o + r)
        return self.out(o)

def train_classifier(X_train: np.ndarray, y_train: List[int], X_val: np.ndarray, y_val: List[int],
                     num_classes: int, device: str = 'cuda', epochs: int = 20, batch_size: int = 512, lr: float = 1e-3):
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
        # val
        model.eval()
        with torch.no_grad():
            preds = model(torch.from_numpy(X_val).float().to(device)).argmax(1).cpu().numpy()
        acc = accuracy_score(y_val, preds)
        print(f"[Classifier] ep {ep} val acc {acc*100:.2f}%")
        if acc > best_acc:
            best_acc = acc; best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def evaluate_classifier(model: ResNetClassifier, X: np.ndarray, y: List[int], device: str = 'cuda'):
    model.eval().to(device)
    with torch.no_grad():
        preds = model(torch.from_numpy(X).float().to(device)).argmax(1).cpu().numpy()
    acc = accuracy_score(y, preds)
    return acc

# -------------------- Orchestration: full pipeline for digit-unlearning --------------------
def run_pipeline(data_root: str, save_dir: str = "./runs/exp", device: str = "cuda", seed: int = 1337,
                 dev_frac: float = 0.75,
                 sae_hidden_mult: float = 4.0, sae_k: int = 128, sae_epochs: int = 30,
                 build_U_top: int = 200, beta: float = 0.0, leace: bool = True,
                 leace_ridge: float = 1e-6, noise_sigma: float = 0.02,
                 forget_digit: str = "0"):
    """
    Full pipeline that unlearns the specified forget_digit (string, e.g., "0").
    Returns a dict of summary metrics and saves artifacts to save_dir.
    """
    seed_all(seed)
    os.makedirs(save_dir, exist_ok=True)
    t0_all = time.time()

    # 1) manifest + splits
    print("[1/11] Building manifest and splits (AudioMNIST; digit-unlearning)...")
    entries = build_manifest_from_audiomnist(data_root)

    # group by digit
    by_digit = {}
    for e in entries:
        by_digit.setdefault(e["digit"], []).append(e)

    digits = sorted(by_digit.keys())
    print(digits)
    if forget_digit not in digits:
        raise ValueError(f"Requested forget_digit {forget_digit} not found in dataset")

    manifest = []
    rng = random.Random(seed)
    for d, utts in by_digit.items():
        rng.shuffle(utts)
        n_dev = max(1, int(len(utts) * dev_frac))
        dev_utts = utts[:n_dev]
        test_utts = utts[n_dev:]
        setlbl = "forget" if d == forget_digit else "retain"
        for u in dev_utts:
            manifest.append({**u, "set": setlbl, "split": "dev"})
        for u in test_utts:
            manifest.append({**u, "set": setlbl, "split": "test"})

    dev_manifest = [m for m in manifest if m["split"]=="dev"]
    test_manifest = [m for m in manifest if m["split"]=="test"]
    dev_retain = [m for m in dev_manifest if m["set"]=="retain"]
    dev_forget = [m for m in dev_manifest if m["set"]=="forget"]
    test_retain = [m for m in test_manifest if m["set"]=="retain"]
    test_forget = [m for m in test_manifest if m["set"]=="forget"]

    print(f"  dev_retain {len(dev_retain)} dev_forget {len(dev_forget)} | test_retain {len(test_retain)} test_forget {len(test_forget)}")


    # 2) extract ECAPA embeddings
    print("[2/11] Extracting ECAPA embeddings...")
    E_all, L_all, P_all = extract_ecapa_embeddings(manifest, device=device)
    E_all = np.asarray(E_all)
    labels_digits = list(L_all)  # list of digit strings per utterance
    sets = [f"{m['split']}" for m in manifest]  # 'dev' or 'test'
    is_dev = np.array([s == 'dev' for s in sets], dtype=bool)
    is_test = np.array([s == 'test' for s in sets], dtype=bool)

    # create binary target: 1 if digit == forget_digit else 0
    y_binary = np.array([1 if d == forget_digit else 0 for d in labels_digits], dtype=int)

    # masks
    M_forget_dev = is_dev & (y_binary == 1)
    M_forget_test = is_test & (y_binary == 1)
    M_retain_dev = is_dev & (y_binary == 0)
    M_retain_test = is_test & (y_binary == 0)

    print(f"  forget-dev: {M_forget_dev.sum()}, forget-test: {M_forget_test.sum()}, retain-dev: {M_retain_dev.sum()}, retain-test: {M_retain_test.sum()}")

    # 3) Whitener on retain-dev (we whiten on non-forget dev samples)
    print("[3/11] Fitting whitener on retain-dev (non-forget dev)...")
    E_ret_train = E_all[M_retain_dev]
    if E_ret_train.shape[0] < 2:
        raise RuntimeError("Not enough retain-dev (non-forget) samples to fit whitener")
    mu, W = fit_whitener(E_ret_train)
    Z_all = apply_whitener(E_all, mu, W)

    # 4) SAE training on Z_train (retain-dev)
    print("[4/11] Training SAE on Z_train (retain-dev)...")
    Z_train = Z_all[M_retain_dev]
    if Z_train.shape[0] < 2:
        raise RuntimeError("Not enough retain-dev samples to train SAE")
    perm = np.random.permutation(Z_train.shape[0])
    cut = max(1, int(0.8 * len(perm)))
    Z_sae_train = Z_train[perm[:cut]]
    Z_sae_val = Z_train[perm[cut:]] if cut < len(perm) else Z_train[perm[:1]]
    sae = TopKSAE(input_dim=Z_all.shape[1], hidden_mult=sae_hidden_mult, k=sae_k)
    sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=sae_epochs, batch_size=min(4096, max(8, Z_sae_train.shape[0])), lr=1e-3, device=device)

    # 5) SAE activations for forget-dev vs retain-dev
    print("[5/11] Encoding SAE activations for forget-dev and retain-dev...")
    Z_ret_dev = Z_all[M_retain_dev]
    Z_fgt_dev = Z_all[M_forget_dev]
    A_ret = sae_encode_batch(sae, Z_ret_dev, device=device, batch=8192)
    A_fgt = sae_encode_batch(sae, Z_fgt_dev, device=device, batch=8192) if Z_fgt_dev.shape[0] > 0 else np.zeros((0, sae.m))

    # 6) feature stats & selection: features that predict forget-digit (A_fgt positive, A_ret negative)
    print("[6/11] Computing feature statistics and selecting discriminative features (forget digit vs rest)...")
    if A_fgt.shape[0] == 0:
        print("  Warning: no forget-dev samples found; selecting top activations from retain-dev as fallback.")
        feat_idx_global = list(np.argsort(-np.abs(A_ret).mean(axis=0))[:build_U_top])
    else:
        stats = feature_stats(A_fgt, A_ret)
        feat_idx_global = select_features_from_stats(stats, top_k=build_U_top, fdr=0.05)
    feat_idx = feat_idx_global

    # 7) build U from decoder atoms corresponding to selected features
    print("[7/11] Building subspace U from decoder atoms...")
    dec_w = sae.dec.weight.detach().cpu().numpy()  # [d, m]
    U = build_U_from_decoder(dec_w, feat_idx)
    print(f"  U shape: {U.shape}")

    # 8) compute LEACE projector in U to remove predictivity of forget-digit (Y=1 for forget digit)
    P_r = np.eye(U.shape[1])
    if leace and U.shape[1] > 0:
        print("[8/11] Computing LEACE projector in U (removing component predictive of forget-digit)...")
        mask_dev = is_dev
        Y_dev = y_binary[mask_dev].reshape(-1, 1)
        if mask_dev.sum() == 0:
            raise RuntimeError("No dev samples found for LEACE computation.")
        try:
            P_r = leace_projector_in_U(Z_all[mask_dev], U, Y_dev, ridge=leace_ridge)
            np.savez(os.path.join(save_dir, "leace_Pr.npz"), U=U, P_r=P_r)
        except Exception as e:
            print("  LEACE computation failed, falling back to identity. Error:", e)
            P_r = np.eye(U.shape[1])
    else:
        print("[8/11] Skipping LEACE (identity)")

    # 9) Apply SAUCE gating (optional) then LEACE, optionally add noise
    print("[9/11] Applying SAUCE gating and LEACE edit...")
    Z_edit = Z_all.copy()
    if beta and beta > 0.0 and len(feat_idx) > 0:
        Z_edit = apply_sauce_gating(sae, Z_all, feat_idx, beta=beta, device=device, batch=8192)
    if U.shape[1] > 0:
        Z_edit = apply_subspace_leace(Z_edit, U, P_r)
    if noise_sigma and noise_sigma > 0 and U.shape[1] > 0:
        Z_edit = inject_noise_in_subspace(Z_edit, U, sigma=noise_sigma)

    E_edit = invert_whitener(Z_edit, mu, W)
    E_all_norm = l2_normalize(E_all)
    E_edit_norm = l2_normalize(E_edit)

    # 10) Evaluations
    print("[10/11] Digit classification accuracy (ResNet) before/after edits...")
    all_digits = sorted(set(labels_digits))
    digit2id = {d:i for i,d in enumerate(all_digits)}

    # Train classifier on retain+forget dev (before edits)
    X_cls_train = E_all_norm[is_dev]
    y_cls_train = [digit2id[d] for d in np.array(labels_digits)[is_dev]]
    X_cls_val = E_all_norm[is_test]
    y_cls_val = [digit2id[d] for d in np.array(labels_digits)[is_test]]

    clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(all_digits), device=device,epochs=10, batch_size=512, lr=1e-3)

    # Build test sets (retain vs forget)
    E_ret_test = E_all_norm[M_retain_test]
    L_ret_test = [digit2id[d] for d in np.array(labels_digits)[M_retain_test]]
    E_fgt_test = E_all_norm[M_forget_test]
    L_fgt_test = [digit2id[d] for d in np.array(labels_digits)[M_forget_test]]
    E_ret_test_edit = E_edit_norm[M_retain_test]
    E_fgt_test_edit = E_edit_norm[M_forget_test]

    # Evaluate BEFORE edits
    acc_ret_before = evaluate_classifier(clf, E_ret_test, L_ret_test, device=device)
    acc_fgt_before = evaluate_classifier(clf, E_fgt_test, L_fgt_test, device=device)

    # Evaluate AFTER edits (same classifier)
    acc_ret_after = evaluate_classifier(clf, E_ret_test_edit, L_ret_test, device=device)
    acc_fgt_after = evaluate_classifier(clf, E_fgt_test_edit, L_fgt_test, device=device)
    print(f"  Classifier Retain: before {acc_ret_before*100:.2f}% after {acc_ret_after*100:.2f}%")
    print(f"  Classifier Forget: before {acc_fgt_before*100:.2f}% after {acc_fgt_after*100:.2f}%")

    results = {
        "classification": {
            "acc_ret_before": acc_ret_before,
            "acc_fgt_before": acc_fgt_before,
            "acc_ret_after": acc_ret_after,
            "acc_fgt_after": acc_fgt_after
        }
    }


    # Save artifacts
    print("Saving artifacts...")
    np.savez(os.path.join(save_dir, "artifacts_digit_unlearn.npz"),
             mu=mu, W=W, U=U, P_r=P_r, feat_idx=np.array(feat_idx),
             E_before=E_all_norm, E_after=E_edit_norm,
             labels=np.array(labels_digits, dtype=object),
             y_binary=y_binary, sets=np.array(sets, dtype=object),
             results=results)
    torch.save(sae.state_dict(), os.path.join(save_dir, "sae_state.pt"))

    print("Done. Total time:", human_time(time.time() - t0_all))
    return results

# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./audiomnist_data", help="AudioMNIST root containing recordings/")
    p.add_argument("--save_dir", default="./outputs_audiomnist", help="where to store artifacts")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dev_frac", type=float, default=0.9)
    p.add_argument("--sae_hidden_mult", type=float, default=4.0)
    p.add_argument("--sae_k", type=int, default=128)
    p.add_argument("--sae_epochs", type=int, default=30)
    p.add_argument("--build_U_top", type=int, default=200)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--leace", action="store_true", help="Compute LEACE projector in U (default: False if not provided)")
    p.add_argument("--leace_ridge", type=float, default=1e-6)
    p.add_argument("--noise_sigma", type=float, default=0.02)
    p.add_argument("--forget_digit", type=str, default="0", help="Digit to forget, e.g. '0'")
    args = p.parse_args()

    # Run
    metrics = run_pipeline(data_root=args.data_root, save_dir=args.save_dir, device=args.device, seed=args.seed,
                           dev_frac=args.dev_frac,
                           sae_hidden_mult=args.sae_hidden_mult, sae_k=args.sae_k, sae_epochs=args.sae_epochs,
                           build_U_top=args.build_U_top, beta=args.beta, leace=args.leace,
                           leace_ridge=args.leace_ridge, noise_sigma=args.noise_sigma,
                           forget_digit=args.forget_digit)
    print("Summary metrics:", json.dumps(metrics, indent=2))
