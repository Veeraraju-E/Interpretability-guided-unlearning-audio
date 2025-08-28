from __future__ import annotations
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from statsmodels.stats.multitest import multipletests
import torchaudio
from speechbrain.pretrained import EncoderClassifier

try:
    from transformers import AutoFeatureExtractor, AutoModel
    _HAS_TRANSFORMERS = True
except Exception:
    AutoFeatureExtractor = AutoModel = None
    _HAS_TRANSFORMERS = False

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
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.to(device)

# -------------------- ECAPA embedding extraction --------------------
def extract_ecapa_embeddings(entries, device='cuda', label_key='speaker'):
    """
    Given a list of manifest entries (with 'path'), return:
      E: [N, D] np.array of normalized embeddings
      labels: list of labels (strings)
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
        labels.append(e.get(label_key, "NA"))
        paths.append(e['path'])
    E = np.vstack(E)
    # l2 normalize
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    E = E / norms
    return E, labels, paths

# -------------------- HF wav2vec2 / hubert extraction --------------------
@torch.no_grad()
def _extract_hf_embeddings(entries: list, model_name: str, device: str = 'cuda', target_sr: int = 16000, label_key='speaker'):
    """
    Generic extractor for HF audio encoders (wav2vec2, hubert). Pools last hidden state over time.
    Returns L2-normalized embeddings.
    """
    if not _HAS_TRANSFORMERS:
        raise RuntimeError(
            "transformers is not installed. Please `pip install transformers` to enable wav2vec2/hubert.")

    processor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    E, labels, paths = [], [], []
    for e in tqdm(entries, desc=f"Extracting {model_name.split('/')[-1]}"):
        wav, sr = torchaudio.load(e['path'])
        wav = _to_mono_and_resample(wav, sr, target_sr, device)
        inp = processor(wav.squeeze().cpu().numpy(), sampling_rate=target_sr, return_tensors="pt")
        input_values = inp["input_values"].to(device)
        out = model(input_values)
        # mean-pool last hidden state: [B, T, C] -> [C]
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
def fit_whitener(X: np.ndarray, eps: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
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
    def __init__(self, input_dim: int, hidden_mult: float = 8.0, k: int = 32, hidden_dim: int | None = None):
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
        dead_mask = (mean_abs_per_latent < 1e-3)   # threshold for "dead"
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

def mann_whitney_u_p(x_pos: np.ndarray, x_neg: np.ndarray) -> tuple[float,float]:
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

def select_features_from_stats(stats: dict, top_k: int = 200, fdr: float = 0.05):
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
def build_U_from_decoder(dec_weight: np.ndarray, feat_idx: list[int]) -> np.ndarray:
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


def apply_subspace_leace(Z: np.ndarray, U: np.ndarray, P_r: np.ndarray, mu_r: np.ndarray | None = None) -> np.ndarray:
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

# -------------------- Speaker Verification helpers --------------------
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
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

def evaluate_classifier(model: ResNetClassifier, X: np.ndarray, y: list[int], device: str = 'cuda'):
    model.eval().to(device)
    with torch.no_grad():
        preds = model(torch.from_numpy(X).float().to(device)).argmax(1).cpu().numpy()
    acc = accuracy_score(y, preds)
    return acc
