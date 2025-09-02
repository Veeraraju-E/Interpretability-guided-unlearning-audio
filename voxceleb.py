#!/usr/bin/env python3
"""
run_voxceleb_unlearn.py

VoxCeleb1 targeted speaker unlearning using final pooled embeddings from:
 - ECAPA-TDNN (SpeechBrain)
 - wav2vec2 (HuggingFace)
 - HuBERT (HuggingFace)
"""
import os
import time
import json
import argparse
import math
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf

from common import (
    seed_all, human_time, extract_embeddings, TopKSAE, 
    train_sae, train_classifier, evaluate_classifier, per_class_eers, avg_eers,
    feature_stats, select_features_from_stats, build_U_from_decoder, linear_probe_U,
    l2_normalize, leace_projector_in_U, apply_subspace_leace,
    fit_whitener, apply_whitener, invert_whitener
)

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

def init_speechbrain_ecapa(device: str = 'cuda'):
    if not SPEECHBRAIN_AVAILABLE:
        raise RuntimeError("speechbrain not installed. pip install speechbrain")
    savedir = "pretrained_models/spkrec_ecapa"
    cls = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=savedir)
    cls.device = torch.device(device)
    cls = cls.to(device)
    return cls

def extract_ecapa_embeddings(classifier, wavs: List[torch.Tensor], device: str='cuda', batch_size: int = 32) -> np.ndarray:
    from tqdm import tqdm
    embeddings = []
    classifier.eval()
    device_obj = torch.device(device)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(wavs), batch_size), desc="Extracting ECAPA embeddings"):
            batch = wavs[i:i+batch_size]
            max_len = max([w.shape[0] for w in batch])
            
            padded_batch = []
            for w in batch:
                if w.shape[0] < max_len:
                    padded = torch.cat([w, torch.zeros(max_len - w.shape[0])], dim=0)
                else:
                    padded = w
                padded_batch.append(padded)
            
            padded = torch.stack(padded_batch).to(device_obj)
            
            try:
                emb = classifier.encode_batch(padded)
                if hasattr(emb, 'squeeze'):
                    emb = emb.squeeze()
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
            except Exception as e:
                print(f"Batch encoding failed, trying individual encoding: {e}")
                emb_list = []
                for w in batch:
                    w_input = w.unsqueeze(0).to(device_obj)
                    emb_s = classifier.encode_batch(w_input)
                    if hasattr(emb_s, 'squeeze'):
                        emb_s = emb_s.squeeze()
                    if emb_s.dim() == 1:
                        emb_s = emb_s.unsqueeze(0)
                    emb_list.append(emb_s.cpu())
                emb = torch.cat(emb_list, dim=0)
            
            embeddings.append(emb.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    return l2_normalize(embeddings.astype(np.float32))

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
    device_obj = torch.device(device)
    
    with torch.no_grad():
        for i in range(0, len(wavs), batch_size):
            batch = wavs[i:i+batch_size]
            arr = [w.numpy() for w in batch]
            inputs = feature_extractor(arr, sampling_rate=16000, return_tensors='pt', padding=True)
            input_values = inputs['input_values'].to(device_obj)
            out = model(input_values).last_hidden_state
            pooled = out.mean(dim=1).cpu().numpy()
            embs.append(pooled)
    
    embs = np.vstack(embs)
    return l2_normalize(embs.astype(np.float32))

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
    device_obj = torch.device(device)
    
    with torch.no_grad():
        for i in range(0, len(wavs), batch_size):
            batch = wavs[i:i+batch_size]
            arr = [w.numpy() for w in batch]
            inputs = feature_extractor(arr, sampling_rate=16000, return_tensors='pt', padding=True)
            input_values = inputs['input_values'].to(device_obj)
            out = model(input_values).last_hidden_state
            pooled = out.mean(dim=1).cpu().numpy()
            embs.append(pooled)
    
    embs = np.vstack(embs)
    return l2_normalize(embs.astype(np.float32))

def extract_embeddings(manifest: List[Dict[str,Any]], encoder: str, device: str='cuda', batch_size: int=32, concat_list: List[str]=None) -> Tuple[np.ndarray, List[str], List[str]]:
    wavs = []
    labels = []
    ids = []
    
    for m in manifest:
        wav, sr = load_audio(m['wav_path'], m['start'], m['end'], target_sr=16000)
        wavs.append(wav)
        labels.append(m['speaker_id'])
        ids.append(m['segment_id'])
    # print(wavs, labels, ids)
    if encoder == 'ecapa':
        cls = init_speechbrain_ecapa(device=device)
        E = extract_ecapa_embeddings(cls, wavs, device=device, batch_size=batch_size)
        return E, labels, ids
    elif encoder == 'wav2vec2':
        feat, model = init_wav2vec2(device=device)
        E = extract_wav2vec2_embeddings(feat, model, wavs, device=device, batch_size=min(batch_size,8))
        return E, labels, ids
    elif encoder == 'hubert':
        feat, model = init_hubert(device=device)
        E = extract_hubert_embeddings(feat, model, wavs, device=device, batch_size=min(batch_size,8))
        return E, labels, ids
    elif encoder == 'concat':
        assert concat_list and len(concat_list)>0
        emb_list = []
        for enc in concat_list:
            E_sub, _, _ = extract_embeddings(manifest, enc, device=device, batch_size=batch_size)
            emb_list.append(E_sub)
        E = np.concatenate(emb_list, axis=1)
        return l2_normalize(E.astype(np.float32)), labels, ids
    else:
        raise ValueError("Unknown encoder")

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
        
        shuffled = list(idxs)
        rng.shuffle(shuffled)
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

def load_precomputed_embeddings(emb_path: str):
    data = np.load(emb_path, allow_pickle=True)
    embeddings = data['embeddings']
    labels = data['labels'].tolist() if isinstance(data['labels'], np.ndarray) else list(data['labels'])
    ids = data['segment_ids'].tolist() if isinstance(data['segment_ids'], np.ndarray) else list(data['segment_ids'])
    return embeddings.astype(np.float32), labels, ids

def parse_args():
    p = argparse.ArgumentParser(description="VoxCeleb targeted speaker unlearning (ECAPA / wav2vec2 / HuBERT).")
    p.add_argument('--manifest', type=str, default="/data/veer/testing/new/datasets/voxceleb2/manifest.csv", help='CSV manifest file')
    p.add_argument('--embeddings', type=str, default=None, help='Load precomputed embeddings .npz (keys:embeddings,labels,segment_ids)')
    p.add_argument('--encoder', type=str, default='ecapa', choices=['ecapa','wav2vec2','hubert','concat'], help="Which encoder to use; 'concat' requires --concat-list")
    p.add_argument('--concat-list', type=str, default=None, help="Comma-separated list when using --encoder concat e.g. ecapa,wav2vec2")
    p.add_argument('--save-dir', type=str, default='./vox_unlearn_out', help='Save dir')
    p.add_argument('--forget-speaker', type=str, default="id00017", help='speaker id to forget')
    p.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sae-topk', type=int, default=128)
    p.add_argument('--sae-epochs', type=int, default=100)
    p.add_argument('--sae-batch', type=int, default=512)
    p.add_argument('--hidden-mult', type=float, default=10.0)
    p.add_argument('--sae-lr', type=float, default=1e-3)
    p.add_argument('--lambda-sparse', type=float, default=1e-4)
    p.add_argument('--whiten-ncomp', type=int, default=None)
    p.add_argument('--classifier-epochs', type=int, default=10)
    p.add_argument('--classifier-batch', type=int, default=512)
    return p.parse_args()

def run_pipeline(manifest_path: str, save_dir: str = "./vox_unlearn_out", device: str = "cuda", seed: int = 42,
                 forget_speaker: str = "id00017", encoder: str = "ecapa", concat_list: str = None,
                 embeddings: str = None, sae_topk: int = 128, sae_epochs: int = 100, sae_batch: int = 512,
                 hidden_mult: float = 10.0, sae_lr: float = 1e-3, lambda_sparse: float = 1e-4,
                 whiten_ncomp: int = None, classifier_epochs: int = 10, classifier_batch: int = 512):
    seed_all(seed)
    os.makedirs(save_dir, exist_ok=True)
    t0_all = time.time()
    print("VoxCeleb unlearning pipeline starting. Device:", device)

    print("[1/12] Building manifest and splits...")
    manifest = read_manifest(manifest_path)
    print("Manifest entries:", len(manifest))
    
    if embeddings is not None:
        E_raw, labels, ids = load_precomputed_embeddings(embeddings)
    else:
        if encoder == 'concat':
            if not concat_list:
                raise ValueError("When encoder='concat' you must provide concat_list")
            concat_list_parsed = [s.strip() for s in concat_list.split(',')]
            E_raw, labels, ids = extract_embeddings(manifest, 'concat', device=device, concat_list=concat_list_parsed)
        else:
            E_raw, labels, ids = extract_embeddings(manifest, encoder, device=device)
    
    print("Raw embeddings shape:", E_raw.shape)

    # Create dataset splits and masks
    sets = build_train_dev_test_splits(labels, test_size=0.1, dev_size=0.1, seed=seed)
    masks = build_masks_by_forget_speaker(labels, sets, forget_speaker)
    M_forget_train = masks['M_forget_train']
    M_forget_dev = masks['M_forget_dev']
    M_forget_test = masks['M_forget_test']
    M_retain_train = masks['M_retain_train']
    M_retain_dev = masks['M_retain_dev']
    M_retain_test = masks['M_retain_test']
    
    print("Counts: forget train/dev/test:", int(M_forget_train.sum()), int(M_forget_dev.sum()), int(M_forget_test.sum()))
    print("Counts: retain train/dev/test:", int(M_retain_train.sum()), int(M_retain_dev.sum()), int(M_retain_test.sum()))

    # 3) Whitening on retain-dev only (like librispeech.py)
    print("[3/12] Fitting whitener on Train-retain...")
    E_ret_train = E_raw[M_retain_train]
    if E_ret_train.shape[0] < 2:
        raise RuntimeError("Not enough retain-train samples to fit whitener")
    
    if whiten_ncomp is not None:
        # If n_components is specified, use PCA-based whitening
        from sklearn.decomposition import PCA
        mu = np.mean(E_ret_train, axis=0)
        Xc = E_ret_train - mu
        pca = PCA(n_components=whiten_ncomp, svd_solver='full', whiten=True)
        pca.fit(Xc)
        components = pca.components_
        explained = pca.explained_variance_
        W = (components.T / np.sqrt(explained + 1e-12))
        mu = mu.astype(np.float32)
        W = W.astype(np.float32)
    else:
        # Use the default whitening from common.py
        mu, W = fit_whitener(E_ret_train)
    
    Z_all = apply_whitener(E_raw, mu, W)

    # 4) SAE training on Z_train (train-retain + train-forget)
    print("[4/12] Training SAE on Z_train (train-retain + train-forget)...")
    Z_train = Z_all[M_retain_train | M_forget_train]
    # split 80/20 for SAE train/val
    perm = np.random.permutation(Z_train.shape[0])
    cut = int(0.8 * len(perm))
    Z_sae_train = Z_train[perm[:cut]]
    Z_sae_val = Z_train[perm[cut:]]
    sae = TopKSAE(input_dim=Z_all.shape[1], hidden_mult=hidden_mult, k=sae_topk)
    sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=sae_epochs, batch_size=sae_batch, lr=sae_lr, device=device)
    
    # 5) SAE encodes for feature mining (train-retain & train-forget)
    print("[5/12] Encoding SAE activations for train-retain and train-forget...")
    Z_ret_train = Z_all[M_retain_train]
    Z_fgt_train = Z_all[M_forget_train]
    from common import sae_encode_batch
    A_ret = sae_encode_batch(sae, Z_ret_train, device=device, batch=8192)
    A_fgt = sae_encode_batch(sae, Z_fgt_train, device=device, batch=8192)

    # 6) feature stats & selection
    print("[6/12] Computing feature statistics and selecting features...")
    stats = feature_stats(A_fgt, A_ret)
    feat_idx = select_features_from_stats(stats, top_k=200, fdr=0.05)

    # 7) prune & normalize decoder & build U
    print("[7/12] Building U subspace...")
    dec_w = sae.dec.weight.detach().cpu().numpy()   # [d, m]
    # Build U from decoder columns corresponding to feat_idx
    U = build_U_from_decoder(dec_w, feat_idx)
    print(f"  U shape: {U.shape}")

    # 8) compute LEACE in U
    P_r = np.eye(U.shape[1])
    print("[8/12] Computing LEACE projector in U...")
    # build Y labels: forget vs rest for all Z_all rows
    mask_dev = (sets == 'dev')
    # Create binary labels: 1 for forget speaker, 0 for retain speakers
    Y_dev = np.array([1 if l == forget_speaker else 0 for l in labels], dtype=int).reshape(-1,1)
    
    if mask_dev.sum() == 0:
        raise RuntimeError("No dev samples found for LEACE computation.")
    try:
        P_r, mu_r = leace_projector_in_U(Z_all[mask_dev], U, Y_dev[mask_dev], ridge=1e-6)
    except Exception as e:
        print("  LEACE computation failed, falling back to identity. Error:", e)
        P_r = np.eye(U.shape[1])
        mu_r = np.zeros((1, U.shape[1]))

    # 9) Apply LEACE edit
    print("[9/12] Applying LEACE edit...")
    Z_edit = apply_subspace_leace(Z_all, U, P_r, mu_r)

    # map back to embedding space and l2 normalize
    E_edit = invert_whitener(Z_edit, mu, W)
    E_all_norm, E_edit_norm = l2_normalize(E_raw), l2_normalize(E_edit)

    # -------------------- 10) Per-class EERs --------------------
    print("[10/12] Per-class verification EERs (each class vs all others) ...")
    from itertools import combinations
    all_spks = sorted(set(labels))
    spk2id = {d:i for i,d in enumerate(all_spks)}

    # Compute per-class EERs on the TEST partitions
    per_class_test = per_class_eers(E_all_norm, E_edit_norm, labels, mask=M_retain_test | M_forget_test, rng_seed=seed)
    per_class_ret = {c: vals for c, vals in per_class_test.items() if c != forget_speaker}
    per_class_fgt = {c: vals for c, vals in per_class_test.items() if c == forget_speaker}

    retain_avg_before, retain_avg_after = avg_eers(per_class_ret)
    forget_avg_before, forget_avg_after = avg_eers(per_class_fgt)

    results = {}
    results["retain_EER_before_avg"] = retain_avg_before
    results["retain_EER_after_avg"] = retain_avg_after
    results["forget_EER_before_avg"] = forget_avg_before
    results["forget_EER_after_avg"] = forget_avg_after

    # -------------------- 11) Linear Probing Evaluation --------------------
    print("[11/12] Linear probing evaluation in U subspace (forget vs retain speakers)...")
    
    # Create split masks for linear probing
    split_mask = {"dev": M_retain_dev | M_forget_dev, "test": M_retain_test | M_forget_test}
    
    # Linear probe in reduced coordinates U (before edits)
    acc_before, auc_before = linear_probe_U(Z_all, labels, U, split_mask, forget_classes=[forget_speaker])
    acc_after, auc_after = linear_probe_U(Z_edit, labels, U, split_mask, forget_classes=[forget_speaker])
    
    print(f'  Linear Probing accuracies and AUC, before: ({acc_before:.4f}, {auc_before:.4f}), after: ({acc_after:.4f}, {auc_after:.4f})')
    
    # Store results
    results["linear_probe_acc_before"] = acc_before
    results["linear_probe_acc_after"] = acc_after
    results["linear_probe_auc_before"] = auc_before
    results["linear_probe_auc_after"] = auc_after

    # Print detailed and summary outputs
    print("\nPer-class EERs (sample):")
    for i, (c, (eb, ea)) in enumerate(sorted(per_class_test.items())): 
        if c == forget_speaker:
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
            "forget_avg": {"before": forget_avg_before, "after": forget_avg_after},
            "linear_probe": {
                "acc_before": acc_before, "acc_after": acc_after,
                "auc_before": auc_before, "auc_after": auc_after
            }
        }, fh, indent=2)

    # -------------------- 12) Speaker classifier (ResNet) training --------------------
    print("[12/12] Speaker classification accuracy (ResNet) before/after edits...")
    all_spks = sorted(list(set(labels)))
    spk2id = {s:i for i,s in enumerate(all_spks)}
    
    # Train classifier on retain+forget train (before edits)
    X_cls_train = E_all_norm[M_retain_train | M_forget_train]
    y_cls_train = [spk2id[s] for i, s in enumerate(labels) if (M_retain_train | M_forget_train)[i]]
    X_cls_val = E_all_norm[M_retain_test | M_forget_test]
    y_cls_val = [spk2id[s] for i, s in enumerate(labels) if (M_retain_test | M_forget_test)[i]]
    
    if len(y_cls_train) == 0 or len(y_cls_val) == 0:
        print("Warning: empty train or val for classifier; skipping classification step.")
        clf = None
        acc_ret_before = acc_fgt_before = acc_ret_after = acc_fgt_after = float('nan')
    else:
        clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(all_spks),
                               device=device, epochs=classifier_epochs, batch_size=classifier_batch, lr=1e-3)
        
        # Evaluate BEFORE edits
        E_ret_test = E_all_norm[M_retain_test]
        L_ret_test = [spk2id[s] for i,s in enumerate(labels) if M_retain_test[i]]
        E_fgt_test = E_all_norm[M_forget_test]
        L_fgt_test = [spk2id[s] for i,s in enumerate(labels) if M_forget_test[i]]
        
        acc_ret_before = evaluate_classifier(clf, E_ret_test, L_ret_test, device=device) if len(L_ret_test)>0 else float('nan')
        acc_fgt_before = evaluate_classifier(clf, E_fgt_test, L_fgt_test, device=device) if len(L_fgt_test)>0 else float('nan')

        # Evaluate AFTER edits
        E_ret_test_edit = E_edit_norm[M_retain_test]
        E_fgt_test_edit = E_edit_norm[M_forget_test]
        
        acc_ret_after = evaluate_classifier(clf, E_ret_test_edit, L_ret_test, device=device) if len(L_ret_test)>0 else float('nan')
        acc_fgt_after = evaluate_classifier(clf, E_fgt_test_edit, L_fgt_test, device=device) if len(L_fgt_test)>0 else float('nan')

        print(f"  Classifier Retain: before {acc_ret_before*100:.2f}% after {acc_ret_after*100:.2f}%")
        print(f"  Classifier Forget: before {acc_fgt_before*100:.2f}% after {acc_fgt_after*100:.2f}%")

    # Save artifacts
    if clf is not None:
        torch.save(clf.state_dict(), os.path.join(save_dir, "resnet_classifier.pt"))    
    torch.save(sae.state_dict(), os.path.join(save_dir, "sae_state.pt"))
    print("Done. Total time:", human_time(time.time()-t0_all))

def main():
    args = parse_args()
    run_pipeline(
        manifest_path=args.manifest,
        save_dir=args.save_dir,
        device=args.device,
        seed=args.seed,
        forget_speaker=args.forget_speaker,
        encoder=args.encoder,
        concat_list=args.concat_list,
        embeddings=args.embeddings,
        sae_topk=args.sae_topk,
        sae_epochs=args.sae_epochs,
        sae_batch=args.sae_batch,
        hidden_mult=args.hidden_mult,
        sae_lr=args.sae_lr,
        lambda_sparse=args.lambda_sparse,
        whiten_ncomp=args.whiten_ncomp,
        classifier_epochs=args.classifier_epochs,
        classifier_batch=args.classifier_batch
    )

if __name__ == '__main__':
    main()