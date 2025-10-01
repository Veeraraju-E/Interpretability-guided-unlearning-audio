from typing import List, Dict, Tuple
import numpy as np
import torch
from tqdm import tqdm

try:
    import torchaudio
except Exception:
    torchaudio = None

try:
    from speechbrain.pretrained import EncoderClassifier
except Exception:
    EncoderClassifier = None

from transformers import AutoFeatureExtractor, AutoModel

def _to_mono_and_resample(wav: torch.Tensor, sr: int, target_sr: int, device: str):
    if wav.ndim == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        if torchaudio is None:
            raise RuntimeError("torchaudio required for resampling")
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.to(device)

def extract_ecapa_embeddings(entries: List[Dict], device='cuda', label_key='label') -> Tuple[np.ndarray, List[str], List[str]]:
    if EncoderClassifier is None:
        raise RuntimeError("speechbrain not installed or EncoderClassifier unavailable")
    encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})
    embeddings, labels, paths = [], [], []
    for entry in tqdm(entries, desc="Extracting ECAPA"):
        wav, sr = torchaudio.load(entry['path'])
        if wav.ndim == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(device)
        with torch.no_grad():
            try:
                emb = encoder.encode_batch(wav).squeeze().cpu().numpy().reshape(-1)
            except Exception:
                emb = encoder.encode_batch(wav[:1]).squeeze().cpu().numpy().reshape(-1)
        embeddings.append(emb)
        labels.append(entry.get(label_key, "NA"))
        paths.append(entry['path'])
    embeddings = np.vstack(embeddings)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    return embeddings, labels, paths

@torch.no_grad()
def extract_hf_embeddings(entries: List[Dict], model_name: str, device: str = 'cuda', target_sr: int = 16000, label_key='label') -> Tuple[np.ndarray, List[str], List[str]]:
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    embeddings, labels, paths = [], [], []
    for entry in tqdm(entries, desc=f"Extracting {model_name.split('/')[-1]}"):
        wav, sr = torchaudio.load(entry['path'])
        wav = _to_mono_and_resample(wav, sr, target_sr, device)
        inp = processor(wav.squeeze().cpu().numpy(), sampling_rate=target_sr, return_tensors="pt")
        input_values = inp["input_values"].to(device)
        out = model(input_values)
        last = out.last_hidden_state
        emb = last.mean(dim=1).squeeze(0).detach().cpu().numpy()
        embeddings.append(emb)
        labels.append(entry.get(label_key, 'NA'))
        paths.append(entry['path'])
    embeddings = np.vstack(embeddings)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    return embeddings, labels, paths


