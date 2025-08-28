#!/usr/bin/env python3
"""
run_voxceleb_unlearn.py

VoxCeleb1 targeted speaker unlearning using final pooled embeddings from:
 - ECAPA-TDNN (SpeechBrain)
 - wav2vec2 (HuggingFace)
 - HuBERT (HuggingFace)

This variant includes the ResNetClassifier (copied from new_librispeech.py)
and runs classification checks (retain/forget accuracies) before and after editing.
"""
import os
import sys
import time
import json
import argparse
import math
from typing import List, Tuple, Dict, Any
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Optional third-party imports
try:
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except Exception:
    SPEECHBRAIN_AVAILABLE = False

try:
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, AutoFeatureExtractor, HubertModel
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# -------------------------
# Small utilities
# -------------------------
def human_time(s: float) -> str:
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"

def l2_norm_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

# -------------------------
# Manifest + audio loading
# -------------------------
def read_manifest(manifest_path: str) -> List[Dict[str,Any]]:
    manifest = []
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            seg = {
                'segment_id': parts[0],
                'wav_path': parts[1],
                'start': float(parts[2]) if len(parts) > 2 and parts[2] != '' else 0.0,
                'end': float(parts[3]) if len(parts) > 3 and parts[3] != '' else None,
                'speaker_id': parts[4] if len(parts) > 4 else os.path.basename(os.path.dirname(parts[1])),
                'session_id': parts[5] if len(parts) > 5 else None
            }
            manifest.append(seg)
    return manifest

def load_audio(path: str, start: float = 0.0, end: float = None, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    data, sr = sf.read(path, dtype='float32')
    if data.ndim > 1:
        data = data.mean(axis=1)
    if end is not None:
        start_frame = int(round(start * sr))
        end_frame = int(round(end * sr))
        data = data[start_frame:end_frame]
    elif start > 0:
        start_frame = int(round(start * sr))
        data = data[start_frame:]
    if sr != target_sr:
        data = torchaudio.functional.resample(torch.from_numpy(data).unsqueeze(0), orig_freq=sr, new_freq=target_sr).squeeze(0).numpy()
        sr = target_sr
    wav = torch.from_numpy(data).float()
    return wav, sr

# -------------------------
# Embedding extractors
# -------------------------
def init_speechbrain_ecapa(device: str = 'cuda'):
    if not SPEECHBRAIN_AVAILABLE:
        raise RuntimeError("speechbrain not installed. pip install speechbrain")
    savedir = "pretrained_models/spkrec_ecapa"
    cls = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=savedir)
    if device.startswith('cuda'): cls.to(device)
    return cls

def extract_ecapa_embeddings(classifier, wavs: List[torch.Tensor], device: str='cuda', batch_size: int = 32) -> np.ndarray:
    embeddings = []
    classifier.eval()
    with torch.no_grad():
        for i in range(0, len(wavs), batch_size):
            batch = wavs[i:i+batch_size]
            max_len = max([w.shape[0] for w in batch])
            padded = torch.stack([torch.cat([w, torch.zeros(max_len - w.shape[0])]) for w in batch]).to(classifier.device)
            try:
                emb = classifier.encode_batch(padded)
            except Exception:
                # fallback—encode one by one
                emb_list = []
                for w in batch:
                    emb_s = classifier.encode_batch(w.unsqueeze(0).to(classifier.device))
                    emb_list.append(emb_s.cpu())
                emb = torch.cat(emb_list, dim=0)
            embeddings.append(emb.cpu().numpy())
    embeddings = np.vstack(embeddings)
    return l2_norm_rows(embeddings.astype(np.float32))

def init_wav2vec2(device: str='cuda', model_name: str='facebook/wav2vec2-large-960h'):
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not installed. pip install transformers")
    feat = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return feat, model

def extract_wav2vec2_embeddings(feature_extractor, model, wavs: List[torch.Tensor], device: str='cuda', batch_size: int=8) -> np.ndarray:
    embs = []
    with torch.no_grad():
        for i in range(0, len(wavs), batch_size):
            batch = wavs[i:i+batch_size]
            arr = [w.numpy() for w in batch]
            inputs = feature_extractor(arr, sampling_rate=16000, return_tensors='pt', padding=True)
            input_values = inputs['input_values'].to(device)
            out = model(input_values).last_hidden_state
            pooled = out.mean(dim=1).cpu().numpy()
            embs.append(pooled)
    embs = np.vstack(embs)
    return l2_norm_rows(embs.astype(np.float32))

def init_hubert(device: str='cuda', model_name: str='facebook/hubert-large-ls960-ft'):
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not installed. pip install transformers")
    feat = AutoFeatureExtractor.from_pretrained(model_name)
    model = HubertModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return feat, model

def extract_hubert_embeddings(feature_extractor, model, wavs: List[torch.Tensor], device: str='cuda', batch_size: int=8) -> np.ndarray:
    embs = []
    with torch.no_grad():
        for i in range(0, len(wavs), batch_size):
            batch = wavs[i:i+batch_size]
            arr = [w.numpy() for w in batch]
            inputs = feature_extractor(arr, sampling_rate=16000, return_tensors='pt', padding=True)
            input_values = inputs['input_values'].to(device)
            out = model(input_values).last_hidden_state
            pooled = out.mean(dim=1).cpu().numpy()
            embs.append(pooled)
    embs = np.vstack(embs)
    return l2_norm_rows(embs.astype(np.float32))

def extract_embeddings(manifest: List[Dict[str,Any]], encoder: str, device: str='cuda', batch_size: int=32, concat_list: List[str]=None) -> Tuple[np.ndarray, List[str], List[str]]:
    wavs = []
    labels = []
    ids = []
    for m in manifest:
        wav, sr = load_audio(m['wav_path'], m['start'], m['end'], target_sr=16000)
        wavs.append(wav)
        labels.append(m['speaker_id'])
        ids.append(m['segment_id'])
    if encoder == 'ecapa':
        cls = init_speechbrain_ecapa(device=device)
        E = extract_ecapa_embeddings(cls, wavs, device=device, batch_size=batch_size)
        return E, labels, ids
    if encoder == 'wav2vec2':
        feat, model = init_wav2vec2(device=device)
        E = extract_wav2vec2_embeddings(feat, model, wavs, device=device, batch_size=min(batch_size,8))
        return E, labels, ids
    if encoder == 'hubert':
        feat, model = init_hubert(device=device)
        E = extract_hubert_embeddings(feat, model, wavs, device=device, batch_size=min(batch_size,8))
        return E, labels, ids
    if encoder == 'concat':
        assert concat_list and len(concat_list)>0
        emb_list = []
        for enc in concat_list:
            E_sub, _, _ = extract_embeddings(manifest, enc, device=device, batch_size=batch_size)
            emb_list.append(E_sub)
        E = np.concatenate(emb_list, axis=1)
        return l2_norm_rows(E.astype(np.float32)), labels, ids
    raise ValueError("Unknown encoder")

# -------------------------
# Whitening, TopKSAE, LEACE
# -------------------------
def compute_whitening(embeddings: np.ndarray, n_components: int = None) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(embeddings, dtype=np.float64)
    mu = np.mean(X, axis=0)
    Xc = X - mu
    if n_components is None:
        n_components = Xc.shape[1]
    pca = PCA(n_components=n_components, svd_solver='full', whiten=True)
    pca.fit(Xc)
    components = pca.components_
    explained = pca.explained_variance_
    W = (components.T / np.sqrt(explained + 1e-12))
    return mu.astype(np.float32), W.astype(np.float32)

def apply_whitening(embeddings: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    X = np.asarray(embeddings, dtype=np.float32)
    Xc = X - mu
    Xw = np.dot(Xc, W)
    Xw = l2_norm_rows(Xw)
    return Xw

class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512,256]
        dims = [input_dim] + hidden_dims
        enc_layers = []
        for i in range(len(dims)-1):
            enc_layers.append(nn.Linear(dims[i], dims[i+1]))
            enc_layers.append(nn.ReLU())
        enc_layers.append(nn.Linear(dims[-1], latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        dec_dims = [latent_dim] + list(reversed(hidden_dims)) + [input_dim]
        dec_layers = []
        for i in range(len(dec_dims)-1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i+1]))
            if i < len(dec_dims)-2:
                dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

def train_sae_on_embeddings(sae: TopKSAE, embeddings: np.ndarray, mask: np.ndarray, device='cuda', epochs: int = 50, batch_size: int = 512, lr: float = 1e-3, lambda_sparsity: float = 1e-4):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError("No samples in TopKSAE training mask.")
    X = embeddings[idx].astype(np.float32)
    tensorX = torch.from_numpy(X)
    dataset = torch.utils.data.TensorDataset(tensorX)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    sae.to(device)
    opt = optim.Adam(sae.parameters(), lr=lr)
    mse = nn.MSELoss()
    for ep in range(epochs):
        sae.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon, z = sae(batch)
            loss_rec = mse(recon, batch)
            loss_sparse = lambda_sparsity * torch.mean(torch.abs(z))
            loss = loss_rec + loss_sparse
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * batch.size(0)
        avg_loss = total_loss / len(dataset)
        if (ep % 10) == 0 or ep == epochs-1:
            print(f" TopKSAE epoch {ep+1}/{epochs} avg_loss={avg_loss:.6f}")
    sae.eval()
    return sae

def compute_leace_projector(E: np.ndarray, labels: List[str], target_speakers: List[str], mask: np.ndarray, ridge: float = 1e-3):
    X = np.asarray(E, dtype=np.float64)
    labels_arr = np.array([1 if (lab in target_speakers) else 0 for lab in labels], dtype=int)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError("Empty mask for projector computation.")
    Xtr = X[idx]
    ytr = labels_arr[idx]
    clf = LogisticRegression(penalty='l2', C=1.0/(ridge+1e-12), solver='lbfgs', max_iter=1000)
    clf.fit(Xtr, ytr)
    w = clf.coef_.ravel()
    if np.linalg.norm(w) == 0:
        return np.eye(X.shape[1], dtype=np.float32)
    w = w / (np.linalg.norm(w) + 1e-12)
    P = np.outer(w, w)
    I = np.eye(P.shape[0], dtype=np.float32)
    return I - P

def apply_projector(E: np.ndarray, projector: np.ndarray) -> np.ndarray:
    return np.dot(E, projector.astype(np.float32))

# -------------------------
# Speaker Verification helpers
# -------------------------
def compute_scores(embeddings: np.ndarray, trials: List[Tuple[int,int]]) -> np.ndarray:
    mat = l2_norm_rows(embeddings)
    scores = np.array([float(np.dot(mat[i], mat[j])) for (i,j) in trials], dtype=float)
    return scores

def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    idx = np.argmin(abs_diffs)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thresholds[idx])

def calibrate_scores(dev_scores: np.ndarray, dev_labels: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    dev_scores = dev_scores.reshape(-1,1)
    test_scores = test_scores.reshape(-1,1)
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(dev_scores, dev_labels)
    probs = clf.predict_proba(test_scores)[:,1]
    return probs

def actual_dcf(scores: np.ndarray, labels: np.ndarray, p_tar: float=0.01, c_miss: float=1.0, c_fa: float=1.0) -> float:
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    thresholds = np.concatenate(([scores.min()-1e-6], np.sort(scores), [scores.max()+1e-6]))
    min_cost = float('inf')
    for thr in thresholds:
        decisions = (scores >= thr).astype(int)
        fn = np.sum((decisions == 0) & (labels == 1))
        fp = np.sum((decisions == 1) & (labels == 0))
        n_p = np.sum(labels == 1)
        n_n = np.sum(labels == 0)
        p_miss = fn / n_p if n_p > 0 else 0.0
        p_fa = fp / n_n if n_n > 0 else 0.0
        cost = c_miss * p_miss * p_tar + c_fa * p_fa * (1 - p_tar)
        if cost < min_cost: min_cost = cost
    return float(min_cost)

def build_verification_trials(labels: List[str], target_speaker: str, mask: np.ndarray) -> Tuple[List[Tuple[int,int]], np.ndarray]:
    labels = np.array(labels)
    idx_in_mask = np.where(mask)[0].tolist()
    pos_idx = [i for i in idx_in_mask if labels[i] == target_speaker]
    ok_idx = [i for i in idx_in_mask if labels[i] != target_speaker]
    trials = []
    trial_labels = []
    for (i,j) in combinations(pos_idx, 2):
        trials.append((i,j)); trial_labels.append(1)
    n_pos = max(1, len(trials))
    if len(ok_idx) == 0 or len(pos_idx) == 0:
        return trials, np.array(trial_labels, dtype=int) if len(trial_labels)>0 else ([], np.array([], dtype=int))
    import numpy as _np
    rng = _np.random.RandomState(1337)
    neg_pairs = []
    for _ in range(n_pos):
        a = rng.choice(pos_idx)
        b = rng.choice(ok_idx)
        neg_pairs.append((a,b))
    trials.extend(neg_pairs)
    trial_labels.extend([0]*len(neg_pairs))
    return trials, np.array(trial_labels, dtype=int)

def per_class_eers(emb_before: np.ndarray, emb_after: np.ndarray, labels: List[str], mask: np.ndarray, rng_seed: int = 1337):
    rng = np.random.default_rng(rng_seed)
    idx_in_mask = np.where(mask)[0].tolist()
    classes = sorted({labels[i] for i in idx_in_mask})
    per_class = {}
    for c in classes:
        pos_idx = [i for i in idx_in_mask if labels[i] == c]
        other_idx = [i for i in idx_in_mask if labels[i] != c]
        if len(pos_idx) < 2 or len(other_idx) < 1:
            continue
        pos_pairs_all = list(combinations(pos_idx, 2))
        pos_pairs_sel = pos_pairs_all
        n_neg = len(pos_pairs_sel)
        pos_side_choices = rng.choice(pos_idx, size=n_neg, replace=True)
        other_side_choices = rng.choice(other_idx, size=n_neg, replace=True)
        neg_pairs_sel = [(int(a), int(b)) for a,b in zip(pos_side_choices, other_side_choices)]
        trials = pos_pairs_sel + neg_pairs_sel
        trial_labels = np.array([1]*len(pos_pairs_sel) + [0]*len(neg_pairs_sel), dtype=int)
        scores_before = compute_scores(emb_before, trials)
        scores_after = compute_scores(emb_after, trials)
        try:
            eer_b, _ = compute_eer(scores_before, trial_labels)
        except Exception:
            eer_b = float('nan')
        try:
            eer_a, _ = compute_eer(scores_after, trial_labels)
        except Exception:
            eer_a = float('nan')
        per_class[c] = (float(eer_b), float(eer_a))
    return per_class

# -------------------------
# Splits & masks
# -------------------------
def build_train_dev_test_splits(labels: List[str], test_size: float=0.1, dev_size: float=0.1, seed: int=1337) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n = len(labels)
    sets = np.array(['train'] * n, dtype=object)
    label_to_indices = {}
    for i, lab in enumerate(labels):
        label_to_indices.setdefault(lab, []).append(i)
    for lab, idxs in label_to_indices.items():
        k = len(idxs)
        if k < 3:
            for i in idxs:
                sets[i] = 'train'
            continue
        shuffled = list(idxs); rng.shuffle(shuffled)
        n_test = max(1, int(math.ceil(test_size * k)))
        n_dev = max(1, int(math.ceil(dev_size * k)))
        for i in shuffled[:n_test]:
            sets[i] = 'test'
        for i in shuffled[n_test:n_test+n_dev]:
            sets[i] = 'dev'
        for i in shuffled[n_test+n_dev:]:
            sets[i] = 'train'
    return sets

def build_masks_by_forget_speaker(labels: List[str], sets: np.ndarray, forget_speaker: str) -> Dict[str, np.ndarray]:
    labels = np.array(labels)
    M_forget_train = (labels == forget_speaker) & (sets == 'train')
    M_forget_dev = (labels == forget_speaker) & (sets == 'dev')
    M_forget_test = (labels == forget_speaker) & (sets == 'test')
    M_retain_train = (labels != forget_speaker) & (sets == 'train')
    M_retain_dev = (labels != forget_speaker) & (sets == 'dev')
    M_retain_test = (labels != forget_speaker) & (sets == 'test')
    return {
        'M_forget_train': M_forget_train,
        'M_forget_dev': M_forget_dev,
        'M_forget_test': M_forget_test,
        'M_retain_train': M_retain_train,
        'M_retain_dev': M_retain_dev,
        'M_retain_test': M_retain_test
    }

# -------------------------
# ResNet classifier
# -------------------------
class ResNetClassifier(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(emb_dim,512), nn.BatchNorm1d(512), nn.ReLU())
        self.res = nn.Sequential(nn.Linear(512,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512,512), nn.BatchNorm1d(512))
        self.relu = nn.ReLU()
        self.out = nn.Linear(512, num_classes)
    def forward(self, x):
        o = self.layer1(x); r = self.res(o); o = self.relu(o + r); return self.out(o)

def train_classifier(X_train: np.ndarray, y_train: List[int], X_val: np.ndarray, y_val: List[int], num_classes: int, device: str = 'cuda', epochs: int = 20, batch_size: int = 512, lr: float = 1e-3):
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
            best_acc = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    return model

def evaluate_classifier(model: ResNetClassifier, X: np.ndarray, y: List[int], device: str = 'cuda'):
    model.eval().to(device)
    with torch.no_grad():
        preds = model(torch.from_numpy(X).float().to(device)).argmax(1).cpu().numpy()
    acc = accuracy_score(y, preds)
    return acc

# -------------------------
# Linear probe helper (binary speaker vs rest)
# -------------------------
def linear_probe_U(embeddings: np.ndarray, labels: List[str], U: np.ndarray, split_masks: Dict[str, np.ndarray], target_speaker: str):
    X = embeddings if U is None else np.dot(embeddings, U)
    labels_arr = np.array(labels)
    y = (labels_arr == target_speaker).astype(int)
    train_idx = np.where(split_masks['train'])[0]
    dev_idx = np.where(split_masks['dev'])[0]
    test_idx = np.where(split_masks['test'])[0]
    if len(train_idx) == 0:
        train_idx = dev_idx
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X[train_idx], y[train_idx])
    if len(test_idx) == 0:
        return float('nan'), float('nan')
    probs = clf.predict_proba(X[test_idx])[:,1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y[test_idx], preds)
    fpr, tpr, _ = roc_curve(y[test_idx], probs, pos_label=1)
    auc_score = auc(fpr, tpr)
    return float(acc), float(auc_score)

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="VoxCeleb targeted speaker unlearning (ECAPA / wav2vec2 / HuBERT).")
    p.add_argument('--manifest', type=str, required=True, help='CSV manifest file')
    p.add_argument('--embeddings', type=str, default=None, help='Load precomputed embeddings .npz (keys:embeddings,labels,segment_ids)')
    p.add_argument('--encoder', type=str, default='ecapa', choices=['ecapa','wav2vec2','hubert','concat'], help="Which encoder to use; 'concat' requires --concat-list")
    p.add_argument('--concat-list', type=str, default=None, help="Comma-separated list when using --encoder concat e.g. ecapa,wav2vec2")
    p.add_argument('--save-dir', type=str, default='./vox_unlearn_out', help='Save dir')
    p.add_argument('--forget-speaker', type=str, required=True, help='speaker id to forget')
    p.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sae-latent', type=int, default=128)
    p.add_argument('--sae-epochs', type=int, default=60)
    p.add_argument('--sae-batch', type=int, default=512)
    p.add_argument('--sae-lr', type=float, default=1e-3)
    p.add_argument('--lambda-sparse', type=float, default=1e-4)
    p.add_argument('--whiten-ncomp', type=int, default=None)
    p.add_argument('--classifier-epochs', type=int, default=10)
    p.add_argument('--classifier-batch', type=int, default=512)
    return p.parse_args()

def load_precomputed_embeddings(emb_path: str):
    data = np.load(emb_path, allow_pickle=True)
    embeddings = data['embeddings']
    labels = data['labels'].tolist() if isinstance(data['labels'], np.ndarray) else list(data['labels'])
    ids = data['segment_ids'].tolist() if isinstance(data['segment_ids'], np.ndarray) else list(data['segment_ids'])
    return embeddings.astype(np.float32), labels, ids

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    t0_all = time.time()
    print("VoxCeleb unlearning pipeline starting. Device:", args.device)

    # 1) manifest & embeddings
    print("[1/11] Extracting Manifest and Computing Manifest")
    manifest = read_manifest(args.manifest)
    print("Manifest entries:", len(manifest))
    if args.embeddings is not None:
        E_raw, labels, ids = load_precomputed_embeddings(args.embeddings)
    else:
        if args.encoder == 'concat':
            if not args.concat_list:
                raise ValueError("When --encoder concat you must provide --concat-list")
            concat_list = [s.strip() for s in args.concat_list.split(',')]
            E_raw, labels, ids = extract_embeddings(manifest, 'concat', device=args.device, concat_list=concat_list)
        else:
            E_raw, labels, ids = extract_embeddings(manifest, args.encoder, device=args.device)
    print("Raw embeddings shape:", E_raw.shape)

    # 2) splits & masks
    print("[2/11] Extracting Dataset Splits, Retain and Forget Masks")
    sets = build_train_dev_test_splits(labels, test_size=0.1, dev_size=0.1, seed=args.seed)
    masks = build_masks_by_forget_speaker(labels, sets, args.forget_speaker)
    M_forget_train = masks['M_forget_train']; M_forget_dev = masks['M_forget_dev']; M_forget_test = masks['M_forget_test']
    M_retain_train = masks['M_retain_train']; M_retain_dev = masks['M_retain_dev']; M_retain_test = masks['M_retain_test']
    print("Counts: forget train/dev/test:", int(M_forget_train.sum()), int(M_forget_dev.sum()), int(M_forget_test.sum()))
    print("Counts: retain train/dev/test:", int(M_retain_train.sum()), int(M_retain_dev.sum()), int(M_retain_test.sum()))

    # 3) whitening
    print("[3/11] Applying Whitening on Embeddings")
    mu, W = compute_whitening(E_raw[sets=='train'], n_components=args.whiten_ncomp)
    E_all_norm = apply_whitening(E_raw, mu, W)
    print("E_all_norm shape:", E_all_norm.shape)

    # -------------------------
    # Speaker Classifier - Train it before editing
    # -------------------------
    print("[4/11] Running ResNet classifier")
    all_spks = sorted(list(set(labels)))
    spk2id = {s:i for i,s in enumerate(all_spks)}
    # Build classifier train/val sets (train on dev as per librispeech style)
    cls_train_mask = (M_retain_dev | M_forget_dev)
    cls_val_mask = (M_retain_test | M_forget_test)
    X_cls_train = E_all_norm[cls_train_mask]
    y_cls_train = [spk2id[s] for i,s in enumerate(labels) if cls_train_mask[i]]
    X_cls_val = E_all_norm[cls_val_mask]
    y_cls_val = [spk2id[s] for i,s in enumerate(labels) if cls_val_mask[i]]
    print("Classifier: train samples:", len(y_cls_train), "val samples:", len(y_cls_val))
    if len(y_cls_train) == 0 or len(y_cls_val) == 0:
        print("Warning: empty train or val for classifier; skipping classification step.")
        clf = None
        acc_ret_before = acc_fgt_before = acc_ret_after = acc_fgt_after = float('nan')
    else:
        clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(all_spks),
                               device=args.device, epochs=args.classifier_epochs, batch_size=args.classifier_batch, lr=1e-3)
        # Prepare group test arrays (retain and forget in test)
        E_ret_test = E_all_norm[M_retain_test]
        L_ret_test = [spk2id[s] for i,s in enumerate(labels) if M_retain_test[i]]
        E_fgt_test = E_all_norm[M_forget_test]
        L_fgt_test = [spk2id[s] for i,s in enumerate(labels) if M_forget_test[i]]
        acc_ret_before = evaluate_classifier(clf, E_ret_test, L_ret_test, device=args.device) if len(L_ret_test)>0 else float('nan')
        acc_fgt_before = evaluate_classifier(clf, E_fgt_test, L_fgt_test, device=args.device) if len(L_fgt_test)>0 else float('nan')
        print(f"  Classifier Retain BEFORE: {acc_ret_before*100:.2f}%  Forget BEFORE: {acc_fgt_before*100:.2f}%")

    # 4) TopKSAE training (train on retain training mask)
    print("[5/11] Training TopKSAE on retained training samples (D_r train)...")
    input_dim = E_all_norm.shape[1]
    sae = TopKSAE(input_dim=input_dim, latent_dim=args.sae_latent, hidden_dims=[512,256])
    device = args.device
    sae = train_sae_on_embeddings(sae, E_all_norm, M_retain_train | M_forget_train, device=device, epochs=args.sae_epochs, batch_size=args.sae_batch, lr=args.sae_lr, lambda_sparsity=args.lambda_sparse)
    torch.save(sae.state_dict(), os.path.join(args.save_dir, "sae_state.pt"))

    with torch.no_grad():
        sae.to(device)
        X_tensor = torch.from_numpy(E_all_norm.astype(np.float32)).to(device)
        batch_sz = 1024
        zs = []
        for i in range(0, X_tensor.shape[0], batch_sz):
            b = X_tensor[i:i+batch_sz]
            _, z = sae(b)
            zs.append(z.cpu().numpy())
        Z_all = np.vstack(zs)
    Z_all = l2_norm_rows(Z_all.astype(np.float32))

    # 6) compute projector in TopKSAE latent space and apply
    project_mask = (M_forget_train | M_retain_train | M_retain_dev)
    projector = compute_leace_projector(Z_all, labels, [args.forget_speaker], mask=project_mask)
    Z_edit = apply_projector(Z_all, projector)
    Z_edit = l2_norm_rows(Z_edit)

    # 7) decode Z_edit back to embedding space via TopKSAE.decoder
    with torch.no_grad():
        Zedt = torch.from_numpy(Z_edit.astype(np.float32)).to(device)
        recs = []
        batch_sz = 1024
        for i in range(0, Zedt.shape[0], batch_sz):
            b = Zedt[i:i+batch_sz]
            rec = sae.decoder(b)
            recs.append(rec.cpu().numpy())
        E_edit = np.vstack(recs).astype(np.float32)
    E_edit_norm = l2_norm_rows(E_edit)

    # 8) If classifier was trained earlier, evaluate ON EDITED embeddings (same classifier)
    if clf is not None:
        E_ret_test_edit = E_edit_norm[M_retain_test]
        L_ret_test = [spk2id[s] for i,s in enumerate(labels) if M_retain_test[i]]
        E_fgt_test_edit = E_edit_norm[M_forget_test]
        L_fgt_test = [spk2id[s] for i,s in enumerate(labels) if M_forget_test[i]]
        acc_ret_after = evaluate_classifier(clf, E_ret_test_edit, L_ret_test, device=args.device) if len(L_ret_test)>0 else float('nan')
        acc_fgt_after = evaluate_classifier(clf, E_fgt_test_edit, L_fgt_test, device=args.device) if len(L_fgt_test)>0 else float('nan')
        print(f"  Classifier Retain AFTER: {acc_ret_after*100:.2f}%  Forget AFTER: {acc_fgt_after*100:.2f}%")
    else:
        acc_ret_after = acc_fgt_after = float('nan')

    # 9) verification & per-class evaluation (same as earlier)
    mask_dev = (sets == 'dev')
    mask_test = (sets == 'test')
    dev_trials, dev_labels = build_verification_trials(labels, args.forget_speaker, mask_dev)
    test_trials, test_labels = build_verification_trials(labels, args.forget_speaker, mask_test)
    scores_dev_before = compute_scores(E_all_norm, dev_trials) if len(dev_trials)>0 else np.array([])
    scores_test_before = compute_scores(E_all_norm, test_trials) if len(test_trials)>0 else np.array([])
    scores_dev_after = compute_scores(E_edit_norm, dev_trials) if len(dev_trials)>0 else np.array([])
    scores_test_after = compute_scores(E_edit_norm, test_trials) if len(test_trials)>0 else np.array([])

    if len(scores_dev_before) > 0:
        scores_test_before_cal = calibrate_scores(scores_dev_before, dev_labels, scores_test_before)
        scores_test_after_cal = calibrate_scores(scores_dev_after, dev_labels, scores_test_after)
    else:
        scores_test_before_cal = scores_test_before
        scores_test_after_cal = scores_test_after

    eer_before_raw = compute_eer(scores_test_before, test_labels) if len(scores_test_before)>0 else (float('nan'), float('nan'))
    eer_after_raw = compute_eer(scores_test_after, test_labels) if len(scores_test_after)>0 else (float('nan'), float('nan'))
    eer_before_cal = compute_eer(scores_test_before_cal, test_labels) if len(scores_test_before_cal)>0 else (float('nan'), float('nan'))
    eer_after_cal = compute_eer(scores_test_after_cal, test_labels) if len(scores_test_after_cal)>0 else (float('nan'), float('nan'))

    results = {
        'EER_before_raw': eer_before_raw,
        'EER_after_raw': eer_after_raw,
        'EER_before_cal': eer_before_cal,
        'EER_after_cal': eer_after_cal,
        'classifier_retain_before': acc_ret_before,
        'classifier_forget_before': acc_fgt_before,
        'classifier_retain_after': acc_ret_after,
        'classifier_forget_after': acc_fgt_after
    }

    for p_tar in [0.01, 0.001]:
        if len(scores_test_before) > 0:
            results[f'actualDCF_before_{p_tar}'] = dcf(scores_test_before, test_labels, p_tar=p_tar)
            results[f'actualDCF_after_{p_tar}'] = dcf(scores_test_after, test_labels, p_tar=p_tar)
        else:
            results[f'actualDCF_before_{p_tar}'] = float('nan')
            results[f'actualDCF_after_{p_tar}'] = float('nan')

    split_masks = {'train': (sets=='train'), 'dev': (sets=='dev'), 'test': (sets=='test')}
    acc_before, auc_before = linear_probe_U(Z_all, labels, None, split_masks, target_speaker=args.forget_speaker)
    acc_after, auc_after = linear_probe_U(Z_edit, labels, None, split_masks, target_speaker=args.forget_speaker)
    results['linear_probe_acc_before'] = acc_before
    results['linear_probe_acc_after'] = acc_after
    results['linear_probe_auc_before'] = auc_before
    results['linear_probe_auc_after'] = auc_after

    per_class_test = per_class_eers(E_all_norm, E_edit_norm, labels, mask_test, rng_seed=args.seed)
    per_class_ret = {c:vals for c,vals in per_class_test.items() if c != args.forget_speaker}
    per_class_fgt = {c:vals for c,vals in per_class_test.items() if c == args.forget_speaker}
    def avg_eers(per_class_dict):
        if len(per_class_dict) == 0:
            return (float('nan'), float('nan'))
        be = [v[0] for v in per_class_dict.values() if not np.isnan(v[0])]
        ae = [v[1] for v in per_class_dict.values() if not np.isnan(v[1])]
        return (float(np.mean(be)) if len(be)>0 else float('nan'), float(np.mean(ae)) if len(ae)>0 else float('nan'))
    retain_avg_before, retain_avg_after = avg_eers(per_class_ret)
    forget_avg_before, forget_avg_after = avg_eers(per_class_fgt)
    results['retain_EER_before_avg'] = retain_avg_before
    results['retain_EER_after_avg'] = retain_avg_after
    results['forget_EER_before_avg'] = forget_avg_before
    results['forget_EER_after_avg'] = forget_avg_after

    print("\n--- Verification Results ---")
    for k,v in results.items():
        print(f"{k}: {v}")

    # Save artifacts
    out_npz = os.path.join(args.save_dir, "artifacts_voxceleb_unlearn.npz")
    np.savez(out_npz,
             mu=mu, W=W, embeddings_before=E_all_norm, embeddings_after=E_edit_norm,
             labels=np.array(labels, dtype=object), segment_ids=np.array(ids, dtype=object),
             sets=np.array(sets, dtype=object), results=results, per_class_test=json.dumps(per_class_test))
    np.save(os.path.join(args.save_dir, "Z_all.npy"), Z_all)
    np.save(os.path.join(args.save_dir, "Z_edit.npy"), Z_edit)
    if clf is not None:
        torch.save(clf.state_dict(), os.path.join(args.save_dir, "resnet_classifier.pt"))
    torch.save(sae.state_dict(), os.path.join(args.save_dir, "sae_state.pt"))

    print("Saved results to", args.save_dir)
    print("Total time:", human_time(time.time()-t0_all))

if __name__ == '__main__':
    main()
