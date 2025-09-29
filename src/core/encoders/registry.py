from typing import Tuple, List, Dict
import numpy as np
from . import extractors

_REGISTRY = {}

def register(name: str):
    def deco(fn):
        _REGISTRY[name.strip().lower()] = fn
        return fn
    return deco

def extract_embeddings(entries: List[Dict], encoder: str, device: str = 'cuda', wav2vec2_model: str = "facebook/wav2vec2-base", hubert_model: str = "facebook/hubert-base-ls960", conformer_model: str = "facebook/wav2vec2-conformer-rel-pos-large", label_key: str = 'label') -> Tuple[np.ndarray, List[str], List[str]]:
    key = encoder.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown encoder '{encoder}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[key](entries, device=device, label_key=label_key, wav2vec2_model=wav2vec2_model, hubert_model=hubert_model, conformer_model=conformer_model)

@register('ecapa')
def _ecapa(entries, device='cuda', label_key='label', **kwargs):
    return extractors.extract_ecapa_embeddings(entries, device=device, label_key=label_key)

@register('wav2vec2')
@register('wav2vec')
@register('w2v2')
def _w2v2(entries, device='cuda', label_key='label', wav2vec2_model: str = "facebook/wav2vec2-base", **_):
    return extractors.extract_hf_embeddings(entries, model_name=wav2vec2_model, device=device, label_key=label_key)

@register('hubert')
def _hubert(entries, device='cuda', label_key='label', hubert_model: str = "facebook/hubert-base-ls960", **_):
    return extractors.extract_hf_embeddings(entries, model_name=hubert_model, device=device, label_key=label_key)

@register('conformer')
@register('wav2vec2-conformer')
def _conformer(entries, device='cuda', label_key='label', conformer_model: str = "facebook/wav2vec2-conformer-rel-pos-large", **_):
    return extractors.extract_hf_embeddings(entries, model_name=conformer_model, device=device, label_key=label_key)


