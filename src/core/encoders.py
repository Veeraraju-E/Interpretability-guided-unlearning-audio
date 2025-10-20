"""
Encoder module for extracting embeddings.
"""

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
        if 'audio_array' in entry and 'sampling_rate' in entry:
            wav = torch.tensor(entry['audio_array'], dtype=torch.float32)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
        else:
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
        paths.append(entry.get('path', 'in_memory_audio'))
    embeddings = np.vstack(embeddings)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    return embeddings, labels, paths


@torch.no_grad()
def extract_hf_embeddings(entries: List[Dict], model_name: str, device: str = 'cuda', target_sr: int = 16000, label_key='label') -> Tuple[np.ndarray, List[str], List[str]]:
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    embeddings, labels, paths = [], [], []
    for entry in tqdm(entries, desc=f"Extracting {model_name.split('/')[-1]}"):
        if 'audio_array' in entry and 'sampling_rate' in entry:
            wav = torch.tensor(entry['audio_array'], dtype=torch.float32)
            sr = entry['sampling_rate']
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
        else:
            wav, sr = torchaudio.load(entry['path'])
        wav = _to_mono_and_resample(wav, sr, target_sr, device)
        inp = processor(wav.squeeze().cpu().numpy(), sampling_rate=target_sr, return_tensors="pt")
        input_values = inp["input_values"].to(device)
        out = model(input_values)
        last = out.last_hidden_state
        emb = last.mean(dim=1).squeeze(0).detach().cpu().numpy()
        embeddings.append(emb)
        labels.append(entry.get(label_key, 'NA'))
        paths.append(entry.get('path', 'in_memory_audio'))
    embeddings = np.vstack(embeddings)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    return embeddings, labels, paths


# Maintain registry of encoders
_REGISTRY = {}


def register(name: str):
    def deco(fn):
        _REGISTRY[name.strip().lower()] = fn
        return fn
    return deco


def extract_embeddings(
    entries: List[Dict],
    encoder: str,
    device: str = 'cuda',
    wav2vec2_model: str = "facebook/wav2vec2-base",
    hubert_model: str = "facebook/hubert-base-ls960",
    conformer_model: str = "facebook/wav2vec2-conformer-rel-pos-large",
    label_key: str = 'label',
) -> Tuple[np.ndarray, List[str], List[str]]:
    key = encoder.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown encoder '{encoder}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[key](
        entries,
        device=device,
        label_key=label_key,
        wav2vec2_model=wav2vec2_model,
        hubert_model=hubert_model,
        conformer_model=conformer_model,
    )


@register('ecapa')
def _ecapa(entries, device='cuda', label_key='label', wav2vec2_model=None, hubert_model=None, conformer_model=None):
    return extract_ecapa_embeddings(entries, device=device, label_key=label_key)


@register('wav2vec2')
@register('wav2vec')
@register('w2v2')
def _w2v2(entries, device='cuda', label_key='label', wav2vec2_model: str = "facebook/wav2vec2-base", hubert_model=None, conformer_model=None):
    return extract_hf_embeddings(entries, model_name=wav2vec2_model, device=device, label_key=label_key)


@register('hubert')
def _hubert(entries, device='cuda', label_key='label', hubert_model: str = "facebook/hubert-base-ls960", wav2vec2_model=None, conformer_model=None):
    return extract_hf_embeddings(entries, model_name=hubert_model, device=device, label_key=label_key)


@register('conformer')
@register('wav2vec2-conformer')
def _conformer(entries, device='cuda', label_key='label', conformer_model: str = "facebook/wav2vec2-conformer-rel-pos-large", wav2vec2_model=None, hubert_model=None):
    return extract_hf_embeddings(entries, model_name=conformer_model, device=device, label_key=label_key)


