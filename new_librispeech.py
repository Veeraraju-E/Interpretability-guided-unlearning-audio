from __future__ import annotations
import os, time, math, json, random, glob
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import argparse
import numpy as np
import torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from statsmodels.stats.multitest import multipletests
from speechbrain.pretrained import EncoderClassifier

# Optional HF backends for wav2vec2 / hubert
try:
    from transformers import Wav2Vec2Processor, AutoModel
    _HAS_TRANSFORMERS = True
except Exception:
    Wav2Vec2Processor = AutoModel = None
    _HAS_TRANSFORMERS = False

# -------------------- Utilities --------------------
def seed_all(seed: int = 1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def human_time(s: float) -> str:
    m, s = divmod(int(s), 60); h, m = divmod(m, 60); return f"{h:d}:{m:02d}:{s:02d}"

# -------------------- Manifest --------------------
def build_manifest(data_root: str) -> List[Dict]:
    entries = []
    for sub in ["dev-clean", "dev-other"]:
        pattern = os.path.join(data_root, sub, "*", "*", "*.flac")
        for p in glob.glob(pattern):
            spk = Path(p).parts[-3]  # correctly grab speaker ID from path like dev-clean/84/121123/84-121123-0000.flac
            entries.append({"path": p, "speaker": spk, "subset": sub})
    if len(entries) == 0: raise RuntimeError(f"No entries under {data_root}/dev-clean or dev-other")
    return entries

# -------------------- Audio helpers --------------------
def _to_mono_and_resample(wav: torch.Tensor, sr: int, target_sr: int, device: str):
    if wav.ndim == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.to(device)

# -------------------- ECAPA embedding extraction --------------------
def extract_ecapa_embeddings(entries, device='cuda'):
    """
    Given a list of manifest entries (with 'path'), return:
      E: [N, D] np.array of normalized embeddings
      labels: list of digit labels (strings)
      paths: list of file paths
    """
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
        labels.append(e.get('speaker', "NA"))
        paths.append(e['path'])
    E = np.vstack(E)
    # l2 normalize
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    E = E / norms
    return E, labels, paths

# -------------------- HF wav2vec2 / hubert extraction --------------------
@torch.no_grad()
def _extract_hf_embeddings(entries: List[Dict], model_name: str, device: str = 'cuda', target_sr: int = 16000):
    """
    Generic extractor for HF audio encoders (wav2vec2, hubert). Pools last hidden state over time.
    Returns L2-normalized embeddings.
    """
    if not _HAS_TRANSFORMERS:
        raise RuntimeError(
            "transformers is not installed. Please `pip install transformers` to enable wav2vec2/hubert.")

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    E, labels, paths = [], [], []
    for e in tqdm(entries, desc=f"Extracting {Path(model_name).name}"):
        wav, sr = torchaudio.load(e['path'])
        wav = _to_mono_and_resample(wav, sr, target_sr, device)
        inp = processor(wav.squeeze().cpu().numpy(), sampling_rate=target_sr, return_tensors="pt")
        input_values = inp["input_values"].to(device)
        out = model(input_values)
        # mean-pool last hidden state: [B, T, C] -> [C]
        last = out.last_hidden_state
        emb = last.mean(dim=1).squeeze(0).detach().cpu().numpy()
        E.append(emb)
        labels.append(e.get('speaker', 'NA'))
        paths.append(e['path'])
    E = np.vstack(E)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    return E, labels, paths

# Unified switch

def extract_embeddings(entries: List[Dict], encoder: str, device: str = 'cuda', wav2vec2_model: str = "facebook/wav2vec2-base", hubert_model: str = "facebook/hubert-base-ls960"):
    enc = encoder.strip().lower()
    if enc in {"ecapa", "ecapa-tdnn", "speechbrain-ecapa"}:
        print("[2/11] Extracting ECAPA embeddings...")
        return extract_ecapa_embeddings(entries, device=device)
    elif enc in {"wav2vec", "wav2vec2", "w2v2"}:
        print(f"[2/11] Extracting wav2vec2 embeddings ({wav2vec2_model})...")
        return _extract_hf_embeddings(entries, wav2vec2_model, device=device, target_sr=16000)
    elif enc in {"hubert"}:
        print(f"[2/11] Extracting HuBERT embeddings ({hubert_model})...")
        return _extract_hf_embeddings(entries, hubert_model, device=device, target_sr=16000)
    else:
        raise ValueError(f"Unknown encoder '{encoder}'. Choose from: ecapa | wav2vec2 | hubert")

# -------------------- Whitener --------------------
def fit_whitener(X: np.ndarray, eps: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array(X)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if X.shape[0] < 2:
        raise ValueError(f"Need at least 2 samples to fit whitener, got {X.shape[0]}")
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    print("Whitener input shapes:", X.shape, mu.shape, Xc.shape)
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
        # encoder: in_features=input_dim, out_features=m
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

# -------------------- Training SAE (shape-checked) --------------------
def train_sae(model: TopKSAE, Z_train: np.ndarray, Z_val: np.ndarray, epochs: int = 30, batch_size: int = 4096, lr: float = 1e-3, device: str = 'cuda') -> TopKSAE:
    Z_train = np.asarray(Z_train)
    Z_val = np.asarray(Z_val)
    if Z_train.ndim > 2:
        Z_train = Z_train.reshape(Z_train.shape[0], -1)
    if Z_val.ndim > 2:
        Z_val = Z_val.reshape(Z_val.shape[0], -1)
    # verify dims
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

        # concatenate activations across validation
        A = torch.cat(all_a, dim=0)  # shape: [N_val, latent_dim]
        mean_abs_per_latent = A.abs().mean(dim=0)  # [latent_dim]
        dead_mask = (mean_abs_per_latent < 1e-3)   # threshold for “dead”
        dead_frac = dead_mask.float().mean().item()
        avg_dead_latents = dead_mask.sum().item()

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
        Z = Z.reshape(Z.shape[0], -1)   # TODO: pooling? consider a particular sequence?
    A = []
    for i in tqdm(range(0, Z.shape[0], batch), desc="SAE encode"):
        z = torch.from_numpy(Z[i:i+batch]).float().to(device)
        a = model.encode(z)
        A.append(a.cpu().numpy())
    return np.vstack(A)

# -------------------- Decoder pruning & normalization --------------------
def prune_and_normalize_decoder(dec_weight: np.ndarray, prune_threshold: float = 1e-6) -> np.ndarray:
    # dec_weight: [d, m]
    # compute norms per column (m)
    cols = dec_weight.T  # [m, d]
    norms = np.linalg.norm(cols, axis=1)
    keep = norms > prune_threshold
    if keep.sum() < 1:
        keep = norms.argsort()[::-1][:max(1, int(len(norms)*0.1))]
    cols_kept = cols[keep]
    cols_kept = cols_kept / (np.linalg.norm(cols_kept, axis=1, keepdims=True) + 1e-12) # normalize columns
    return cols_kept.T  # [d, r]

# -------------------- Feature mining statistics --------------------
def cohens_d(x_pos: np.ndarray, x_neg: np.ndarray) -> float:
    n1, n2 = len(x_pos), len(x_neg)
    m1, m2 = x_pos.mean(), x_neg.mean()
    s1, s2 = x_pos.var(ddof=1), x_neg.var(ddof=1)
    # pooled sd
    s_pooled = math.sqrt(((n1-1)*s1 + (n2-1)*s2) / max(1, n1+n2-2))
    if s_pooled == 0: return 0.0
    return float((m1 - m2) / s_pooled)

def mann_whitney_u_p(x_pos: np.ndarray, x_neg: np.ndarray) -> Tuple[float,float]:
    # returns (auc, p) approximate via ranks
    x = np.concatenate([x_pos, x_neg])
    labels = np.concatenate([np.ones(len(x_pos)), np.zeros(len(x_neg))])
    ranks = np.argsort(np.argsort(x)) + 1
    r1 = ranks[labels==1].sum()
    n1, n2 = len(x_pos), len(x_neg)
    U1 = r1 - n1*(n1+1)/2
    auc = U1 / (n1*n2 + 1e-12)
    # p-value approximate using normal approx
    mu = n1*n2/2
    sigma = math.sqrt(n1*n2*(n1+n2+1)/12)
    z = (U1 - mu) / (sigma + 1e-12)
    from math import erf, sqrt
    p = 2*(1 - 0.5*(1+erf(abs(z)/sqrt(2))))
    return auc, p

def kl_gaussian_1d(p: np.ndarray, q: np.ndarray) -> float:
    # fit gaussian N(mu,sigma^2) to p and q arrays and compute KL(Np||Nq)
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
    if multipletests:
        rej, p_corr, _, _ = multipletests(pvals, alpha=fdr, method='fdr_bh')
        idx = np.where(rej)[0]
        if len(idx) == 0:
            # fallback to top by |auc-0.5|
            print(f'Using pvals sorted by effetc size viz |AUC - 0.5| scores')
            idx = np.argsort(-np.abs(stats["auc"] - 0.5))[:top_k]
        else:
            # sort idx by effect size
            print(f'Using AUC - 0.5 scores')
            idx = sorted(list(idx), key=lambda j: -abs(stats["auc"][j] - 0.5))
            idx = idx[:top_k]
    else:
        print(f'Using AUC - 0.5 scores')
        idx = np.argsort(-np.abs(stats["auc"] - 0.5))[:top_k]
    return list(idx)

# -------------------- Build U from decoder --------------------
def build_U_from_decoder(dec_weight: np.ndarray, feat_idx: List[int]) -> np.ndarray:
    """
    dec_weight: [d, m], feat_idx: indices into m (SAE hidden units)
    Returns U: [d, r] with orthonormal columns (SVD of the selected, normalized decoder columns)
    """
    if len(feat_idx) == 0:
        return np.zeros((dec_weight.shape[0], 0))

    # 1) Column-normalize ALL decoder atoms so selection is scale-invariant
    D = dec_weight.copy()                              # [d, m]
    norms = np.linalg.norm(D, axis=0, keepdims=True)   # [1, m]
    D = D / (norms + 1e-12)

    # 2) Select the mined features
    D_S = D[:, feat_idx]                               # [d, r]

    # 3) Orthonormal basis for span(D_S)
    U, _, _ = np.linalg.svd(D_S, full_matrices=False)  # U: [d, r]
    U = U[:, :D_S.shape[1]]
    return U

# -------------------- LEACE in U --------------------
def leace_projector_in_U(Z: np.ndarray, U: np.ndarray, Y: np.ndarray, ridge: float = 1e-6):
    """
    Z: [N, d]   -- whitened embeddings (already (X - mu) @ W upstream)
    U: [d, r]   -- subspace basis (orthonormal columns)
    Y: [N, k]   -- labels for the attribute to erase (k=1 for binary)
    Returns: (P_r, mu_r) where
       P_r is the LEACE linear map in U (I - W_r^+ P W_r) and
       mu_r is the mean of Z projected into U used for the affine term.
    """
    # Work in U coordinates
    ZU = Z @ U                       # [N, r]
    mu_r = ZU.mean(axis=0, keepdims=True)
    ZUc = ZU - mu_r
    Yc  = Y - Y.mean(axis=0, keepdims=True)

    N = ZU.shape[0]
    # Covariances in U
    Sig = (ZUc.T @ ZUc) / max(1, N - 1)           # Σ_{ZZ} in U  [r, r]
    Sig_ZY = (ZUc.T @ Yc) / max(1, N - 1)         # Σ_{ZY}       [r, k]

    # Whiten in U: W_r = (Σ^{1/2})^+ ; W_r^+ = Σ^{1/2}
    evals, evecs = np.linalg.eigh(Sig + ridge * np.eye(Sig.shape[0]))
    evals = np.clip(evals, 1e-12, None)
    W_r     = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T   # Σ^{-1/2}
    W_r_pinv = evecs @ np.diag(np.sqrt(evals)) @ evecs.T        # Σ^{+1/2}

    # Column space of W_r Σ_{ZY}
    M = W_r @ Sig_ZY                                            # [r, k]

    # Orthogonal projector onto colsp(M): P = U_M U_M^T
    U_M, _, _ = np.linalg.svd(M, full_matrices=False)           # U_M: [r, q]
    P = U_M @ U_M.T                                            # [r, r]

    # LEACE linear map in U: P_r = I - W_r^+ P W_r
    P_r = np.eye(U.shape[1]) - (W_r_pinv @ P @ W_r)
    return P_r, mu_r


def apply_subspace_leace(Z: np.ndarray, U: np.ndarray, P_r: np.ndarray, mu_r: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Applies r_LEACE in U: r(z) = z - W_r^+ P W_r (z - mu_r), implemented as:
      ZU' = (ZU - mu_r) @ P_r.T + mu_r, then map back.
    """
    if U.shape[1] == 0: return Z
    ZU = Z @ U
    if mu_r is None: mu_r = np.zeros((1, U.shape[1]), dtype=ZU.dtype)
    ZU_prime = (ZU - mu_r) @ P_r.T + mu_r
    Z_prime  = Z - (ZU @ U.T) + (ZU_prime @ U.T)
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
    return np.vstack(Z_out)

# -------------------- Noise injection --------------------
def inject_noise_in_subspace(Z: np.ndarray, U: np.ndarray, sigma: float = 0.02, rng=None) -> np.ndarray:
    rng = np.random if rng is None else rng
    ZU = Z @ U  # [N, r]
    noise = rng.randn(*ZU.shape) * sigma
    return Z + (noise @ U.T)

# -------------------- Speaker Verification helpers --------------------
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fnr[idx] + fpr[idx]) / 2.0
    return float(eer), float(thr[idx])

def compute_scores(embeddings, trials):
    """Compute cosine scores for all pairs in trials."""
    scores = []
    for i, j in trials:
        scores.append(float(np.dot(embeddings[i], embeddings[j])))
    return np.array(scores)

# -------------------- Speaker classifier (ResNet) --------------------
class ResNetClassifier(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(emb_dim,512), nn.BatchNorm1d(512), nn.ReLU())
        self.res = nn.Sequential(nn.Linear(512,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512,512), nn.BatchNorm1d(512))
        self.relu = nn.ReLU()
        self.out = nn.Linear(512, num_classes)
    def forward(self, x):
        o = self.layer1(x); r = self.res(o); o = self.relu(o + r); return self.out(o)

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
    if best_state is not None: model.load_state_dict(best_state)
    return model

def evaluate_classifier(model: ResNetClassifier, X: np.ndarray, y: List[int], device: str = 'cuda'):
    model.eval().to(device)
    with torch.no_grad():
        preds = model(torch.from_numpy(X).float().to(device)).argmax(1).cpu().numpy()
    acc = accuracy_score(y, preds)
    return acc

# -------------------- Orchestration: full run --------------------
def run_pipeline(data_root: str, save_dir: str = "./runs/exp", device: str = "cuda", seed: int = 1337,
                 retain_frac: float = 0.975, dev_frac_per_spk: float = 0.75,
                 sae_hidden_mult: float = 4.0, sae_topk: int = 128, sae_epochs: int = 30,
                 build_U_top: int = 200, beta: float = 0.0, leace: bool = True,
                 leace_ridge: float = 1e-6, noise_sigma: float = 0.02, per_speaker_union: bool = False, forget_spk: int = None,
                 encoder: str = "ecapa", wav2vec2_model: str = "facebook/wav2vec2-base", hubert_model: str = "facebook/hubert-base-ls960"):
    seed_all(seed)
    os.makedirs(save_dir, exist_ok=True)
    t0_all = time.time()

    # 1) manifest + splits
    print("[1/11] Building manifest and splits...")
    entries = build_manifest(data_root)

    # group utterances by speaker
    by = {}
    for e in entries:
        by.setdefault(e["speaker"], []).append(e["path"])

    spks = sorted(by.keys())
    if forget_spk is None:
        forget_spk = random.choice(spks)
    if forget_spk not in spks:
        raise ValueError(f"Requested forget_spk {forget_spk} not found in dataset")
    print(f"  FORGETTING SPEAKER: {forget_spk}")

    retain_spks = [s for s in spks if s != forget_spk]
    forget_spks = [forget_spk]

    manifest = []
    rng = random.Random(seed)
    for s, paths in by.items():
        rng.shuffle(paths)
        n_dev = max(1, int(len(paths) * dev_frac_per_spk))
        dev_paths = paths[:n_dev]
        test_paths = paths[n_dev:]
        setlbl = 'forget' if s == forget_spk else 'retain'
        for p in dev_paths:
            manifest.append({"path": p, "speaker": s, "set": setlbl, "split": "dev"})
        for p in test_paths:
            manifest.append({"path": p, "speaker": s, "set": setlbl, "split": "test"})

    # create lists for each split+set
    dev_manifest = [m for m in manifest if m['split']=='dev']
    test_manifest = [m for m in manifest if m['split']=='test']
    dev_retain = [m for m in dev_manifest if m['set']=='retain']
    dev_forget = [m for m in dev_manifest if m['set']=='forget']
    test_retain = [m for m in test_manifest if m['set']=='retain']
    test_forget = [m for m in test_manifest if m['set']=='forget']
    print(f"  retain_spks {len(retain_spks)} forget_spks {len(forget_spks)}")
    print(f"  dev_retain {len(dev_retain)} dev_forget {len(dev_forget)} | test_retain {len(test_retain)} test_forget {len(test_forget)}")

    # 2) ECAPA embeddings
    print("[2/11] Extracting ECAPA embeddings (this may take a while)...")
    all_manifest = dev_manifest + test_manifest
    E_all, L_all, P_all = extract_embeddings(all_manifest, encoder=encoder, device=device, wav2vec2_model=wav2vec2_model, hubert_model=hubert_model)
    labels = L_all
    # print(labels)
    sets = [f"{m['set']}:{m['split']}" for m in (dev_manifest + test_manifest)]

    # masks
    M_ret_dev = np.array([s == 'retain:dev' for s in sets], dtype=bool)
    M_fgt_dev = np.array([s == 'forget:dev' for s in sets], dtype=bool)
    M_ret_test = np.array([s == 'retain:test' for s in sets], dtype=bool)
    M_fgt_test = np.array([s == 'forget:test' for s in sets], dtype=bool)

    # 3) Whitening on D_r
    print("[3/11] Fitting whitener on Train-retain...")
    E_ret_train = E_all[M_ret_dev]
    if E_ret_train.shape[0] < 2:
        raise RuntimeError("Not enough retain-dev samples to fit whitener")
    mu, W = fit_whitener(E_ret_train)
    Z_all = apply_whitener(E_all, mu, W)

    # 4) SAE training on Z_train (Train-retain)
    print("[4/11] Training SAE on Z_train (train-retain)...")
    Z_train = Z_all[M_ret_dev | M_fgt_dev]
    # split 80/20 for SAE train/val
    perm = np.random.permutation(Z_train.shape[0])
    cut = int(0.8 * len(perm))
    Z_sae_train = Z_train[perm[:cut]]
    Z_sae_val = Z_train[perm[cut:]]
    sae = TopKSAE(input_dim=Z_all.shape[1], hidden_mult=sae_hidden_mult, k=sae_topk)
    sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=sae_epochs, batch_size=4096, lr=1e-3, device=device)

    # 5) SAE encodes for feature mining (train-retain & train-forget)
    print("[5/11] Encoding SAE activations for train-retain and train-forget...")
    Z_ret_dev = Z_all[M_ret_dev]
    Z_fgt_dev = Z_all[M_fgt_dev]
    A_ret = sae_encode_batch(sae, Z_ret_dev, device=device, batch=8192)
    A_fgt = sae_encode_batch(sae, Z_fgt_dev, device=device, batch=8192)
    # print(A_ret, A_fgt)

    # 6) feature stats & selection (supports per-speaker union)
    print("[6/11] Computing feature statistics and selecting features...")
    stats = feature_stats(A_fgt, A_ret)
    feat_idx_global = select_features_from_stats(stats, top_k=build_U_top, fdr=0.05)
    feat_idx = feat_idx_global
    if per_speaker_union:   # TODO
        print("  computing per-forget-speaker feature unions...")
        feat_idx = feat_idx_global

    # 7) prune & normalize decoder & build U
    print("[7/11] Building U subspace...")
    dec_w = sae.dec.weight.detach().cpu().numpy()   # [d, m]
    dec_pruned = prune_and_normalize_decoder(dec_w, prune_threshold=1e-6)  # returns [d, r]
    # Build U from decoder columns corresponding to feat_idx
    U = build_U_from_decoder(dec_w, feat_idx)   # NOTE: ablate the use case of dec_pruned and full dec_w 
    print(f"  U shape: {U.shape}")

    # 8) compute LEACE in U if requested
    P_r = np.eye(U.shape[1])
    if leace:
        print("[8/11] Computing LEACE projector in U...")
        # build Y labels: forget vs rest for all Z_all rows (same as )
        mask_dev = np.array([s.endswith(':dev') for s in sets])
        Y_dev = (np.array([1 if s.startswith('forget') else 0 for s in sets])[mask_dev]).reshape(-1,1)
        P_r, mu_r = leace_projector_in_U(Z_all[mask_dev], U, Y_dev, ridge=leace_ridge)
        np.savez(os.path.join(save_dir, "leace_Pr.npz"), U=U, P_r=P_r)
    else:
        print("[8/11] Skipping LEACE (identity)")

    # 9) Apply SAUCE gating (optional) then LEACE and optional noise
    print("[9/11] Applying SAUCE gating (beta) and LEACE edit...")
    Z_edit = Z_all.copy()
    if beta > 0.0:
        print("  applying SAUCE gating with beta", beta)
        Z_edit = apply_sauce_gating(sae, Z_all, feat_idx, beta=beta, device=device, batch=8192)
    # apply LEACE map in U
    if leace:
        Z_edit = apply_subspace_leace(Z_edit, U, P_r, mu_r)
    if noise_sigma and noise_sigma > 0:
        Z_edit = inject_noise_in_subspace(Z_edit, U, sigma=noise_sigma)

    # map back to embedding space and l2 normalize
    E_edit = invert_whitener(Z_edit, mu, W)
    E_all_norm, E_edit_norm = l2_normalize(E_all), l2_normalize(E_edit)

    # -------------------- 11) Per-class EERs --------------------
    print("[10/11] Per-class verification EERs (each class vs all others) ...")
    from itertools import combinations
    all_spks = sorted(set(labels))
    spk2id = {d:i for i,d in enumerate(all_spks)}

    # Build test lists
    E_ret_test = E_all_norm[M_ret_test]
    L_ret_test = [spk2id[d] for d in np.array(labels)[M_ret_test]]
    E_fgt_test = E_all_norm[M_fgt_test]
    L_fgt_test = [spk2id[d] for d in np.array(labels)[M_fgt_test]]
    E_ret_test_edit = E_edit_norm[M_ret_test]
    E_fgt_test_edit = E_edit_norm[M_fgt_test]

    def per_class_eers(emb_before: np.ndarray, emb_after: np.ndarray, labels: List[str], mask: np.ndarray, rng_seed: int = 1337):
        """
        For each class present in `mask`, compute EER using:
            - positives = (pairs of samples both from that class, inside mask)
            - negatives = (pairs where one sample is from that class and the other sample is from any other class in mask)
        Returns dict: class_id -> (eer_before, eer_after)
        """
        rng = np.random.default_rng(rng_seed)
        idx_in_mask = np.where(mask)[0].tolist()
        classes = sorted({labels[i] for i in idx_in_mask})
        per_class = {}
        for c in classes:
            pos_idx = [i for i in idx_in_mask if labels[i] == c]
            other_idx = [i for i in idx_in_mask if labels[i] != c]

            if len(pos_idx) < 2 or len(other_idx) < 1:
                continue

            # all possible positive pairs (i<j)
            pos_pairs_all = list(combinations(pos_idx, 2))
            n_sample = n_pos_all = len(pos_pairs_all)
            if n_pos_all <= n_sample:
                pos_pairs_sel = pos_pairs_all
            else:
                sel_idxs = rng.choice(n_pos_all, size=n_sample, replace=False)
                pos_pairs_sel = [pos_pairs_all[k] for k in sel_idxs]

            n_neg = len(pos_pairs_sel)
            pos_side_choices = rng.choice(pos_idx, size=n_neg, replace=True)
            other_side_choices = rng.choice(other_idx, size=n_neg, replace=True)
            neg_pairs_sel = [(int(a), int(b)) for a, b in zip(pos_side_choices, other_side_choices)]

            # assemble combined trials and labels
            trials = pos_pairs_sel + neg_pairs_sel
            trial_labels = np.array([1] * len(pos_pairs_sel) + [0] * len(neg_pairs_sel), dtype=int)

            # compute scores compute_scores helper
            scores_before = compute_scores(emb_before, trials)
            scores_after = compute_scores(emb_after, trials)

            # compute EER (compute_eer returns (eer, threshold))
            try:
                eer_b, _ = compute_eer(scores_before, trial_labels)
            except Exception:
                eer_b, _ = (float('nan'), float('nan'))
            try:
                eer_a, _ = compute_eer(scores_after, trial_labels)
            except Exception:
                eer_a, _ = (float('nan'), float('nan'))
            per_class[c] = (float(eer_b), float(eer_a))

        return per_class

    # Compute per-class EERs on the TEST partitions
    per_class_test = per_class_eers(E_all_norm, E_edit_norm, labels, mask=M_ret_test | M_fgt_test, rng_seed=seed)
    per_class_ret = {c: vals for c, vals in per_class_test.items() if c in retain_spks}
    per_class_fgt = {c: vals for c, vals in per_class_test.items() if c in forget_spks}

    def avg_eers(per_class_dict):
        if len(per_class_dict) == 0:
            return (float('nan'), float('nan'))
        be = [v[0] for v in per_class_dict.values() if not np.isnan(v[0])]
        ae = [v[1] for v in per_class_dict.values() if not np.isnan(v[1])]
        return float(np.mean(be)) if len(be) > 0 else float('nan'), float(np.mean(ae)) if len(ae) > 0 else float('nan')

    retain_avg_before, retain_avg_after = avg_eers(per_class_ret)
    forget_avg_before, forget_avg_after = avg_eers(per_class_fgt)

    results = {}
    results["retain_EER_before_avg"] = retain_avg_before
    results["retain_EER_after_avg"] = retain_avg_after
    results["forget_EER_before_avg"] = forget_avg_before
    results["forget_EER_after_avg"] = forget_avg_after

    # Print detailed and summary outputs
    print("\nPer-class EERs (sample):")
    for i, (c, (eb, ea)) in enumerate(sorted(per_class_test.items())): 
        if c == forget_spk:
            print("  [FORGET SPEAKER]")
        print(f"  class {c}: EER_before={eb:.4f}, EER_after={ea:.4f}")

    print("\n--- Average EER summary ---")
    print(f"  Retain avg EER: before {retain_avg_before:.4f}, after {retain_avg_after:.4f}")
    print(f"  Forget avg EER: before {forget_avg_before:.4f}, after {forget_avg_after:.4f}")

    # Save these numbers to disk along with other artifacts
    stats_path = os.path.join(save_dir, "verification_eer_summary.json")
    with open(stats_path, "w") as fh:
        json.dump({
            "per_class_count_test": len(per_class_test),
            "retain_avg": {"before": retain_avg_before, "after": retain_avg_after},
            "forget_avg": {"before": forget_avg_before, "after": forget_avg_after}
        }, fh, indent=2)

    # -------------------- 12) Speaker classifier (ResNet) training (unchanged) --------------------
    print("[11/11] Speaker classification accuracy (ResNet) before/after edits...")
    all_spks = sorted(list(set(labels)))
    spk2id = {s:i for i,s in enumerate(all_spks)}
    
    # Train classifier on retain+forget dev (before edits)
    X_cls_train = E_all_norm[M_ret_dev | M_fgt_dev]
    y_cls_train = [spk2id[s] for i, s in enumerate(labels) if (M_ret_dev | M_fgt_dev)[i]]
    X_cls_val = E_all_norm[M_ret_test | M_fgt_test]
    y_cls_val = [spk2id[s] for i, s in enumerate(labels) if (M_ret_test | M_fgt_test)[i]]
    # print(len(X_cls_train), len(y_cls_train), len(X_cls_val), len(y_cls_val))
    clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(all_spks), device=device, epochs=10, batch_size=512, lr=1e-3)

    # Evaluate BEFORE edits
    acc_ret_before = evaluate_classifier(clf, E_ret_test, L_ret_test, device=device)
    acc_fgt_before = evaluate_classifier(clf, E_fgt_test, L_fgt_test, device=device)

    # Evaluate AFTER edits
    acc_ret_after = evaluate_classifier(clf, E_ret_test_edit, L_ret_test, device=device)
    acc_fgt_after = evaluate_classifier(clf, E_fgt_test_edit, L_fgt_test, device=device)

    print(f"  Classifier Retain: before {acc_ret_before*100:.2f}% after {acc_ret_after*100:.2f}%")
    print(f"  Classifier Forget: before {acc_fgt_before*100:.2f}% after {acc_fgt_after*100:.2f}%")

    # Save artifacts (unchanged)
    print("Saving artifacts...")
    np.savez(os.path.join(save_dir, "artifacts_full.npz"),
             mu=mu, W=W, U=U, P_r=P_r, feat_idx=np.array(feat_idx),
             E_before=E_all_norm, E_after=E_edit_norm,
             labels=np.array(labels, dtype=object), sets=np.array(sets, dtype=object))
    torch.save(sae.state_dict(), os.path.join(save_dir, "sae_state.pt"))
    print("Done. Total time:", human_time(time.time()-t0_all))

# -------------------- CLI --------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./librispeech_data_dev/LibriSpeech", help="LibriSpeech root containing dev-clean/dev-other")
    p.add_argument("--save_dir", default="./outputs", help="where to store artifacts")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--retain_frac", type=float, default=0.7)
    p.add_argument("--dev_frac_per_spk", type=float, default=0.7)
    p.add_argument("--sae_hidden_mult", type=float, default=10.0)
    p.add_argument("--sae_topk", type=int, default=128)
    p.add_argument("--sae_epochs", type=int, default=100)
    p.add_argument("--build_U_top", type=int, default=200)
    p.add_argument("--beta", type=float, default=0.0)   # for SAUCE, default=don't do
    p.add_argument("--leace", action="store_true")
    p.add_argument("--leace_ridge", type=float, default=1e-6)
    p.add_argument("--noise_sigma", type=float, default=0.02)
    p.add_argument("--forget_spk", type=str, default='174', help="Speaker ID to forget (if not given, random)")
    p.add_argument("--encoder", type=str, default="ecapa", help="Encoder to use - 'ecapa', 'wav2vec2', 'hubert'")
    args = p.parse_args()
    run_pipeline(
        data_root=args.data_root, save_dir=args.save_dir, 
        device=args.device, seed=args.seed,
        retain_frac=args.retain_frac, dev_frac_per_spk=args.dev_frac_per_spk,
        sae_hidden_mult=args.sae_hidden_mult,  sae_topk=args.sae_topk,  sae_epochs=args.sae_epochs,
        build_U_top=args.build_U_top, 
        beta=args.beta, 
        leace=args.leace, leace_ridge=args.leace_ridge, noise_sigma=args.noise_sigma, 
        encoder=args.encoder, forget_spk=args.forget_spk
    )