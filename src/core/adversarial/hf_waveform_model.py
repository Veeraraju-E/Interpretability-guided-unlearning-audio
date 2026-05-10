"""Differentiable waveform -> HF encoder embeddings -> logits (frozen encoder + classifier)."""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

try:
    import torchaudio
except Exception:
    torchaudio = None

from transformers import AutoFeatureExtractor, AutoModel

from src.core.encoders import _to_mono_and_resample
from src.core.evals.classification import SimpleClassifier


def resolve_hf_model_name(encoder_cfg: Dict) -> str:
    name = (encoder_cfg.get("name") or "").strip().lower()
    if name in ("wav2vec2", "wav2vec", "w2v2"):
        path = encoder_cfg.get("wav2vec2_model", "facebook/wav2vec2-base")
        if not path:
            raise ValueError("encoder.wav2vec2_model is required for wav2vec2 encoder.")
        return path
    if name == "hubert":
        path = encoder_cfg.get("hubert_model", "facebook/hubert-base-ls960")
        if not path:
            raise ValueError("encoder.hubert_model is required for HuBERT encoder.")
        return path
    if name in ("conformer", "wav2vec2-conformer"):
        path = encoder_cfg.get("conformer_model", "facebook/wav2vec2-conformer-rel-pos-large")
        if not path:
            raise ValueError("encoder.conformer_model is required for conformer encoder.")
        return path
    raise RuntimeError(
        "Waveform attacks require Hugging Face encoders (wav2vec2, hubert, conformer); "
        f"got encoder name '{encoder_cfg.get('name')}'. ECAPA/SpeechBrain is not supported for differentiable PGD/APGD."
    )


def hf_preprocess_torch_mono(wav_1_bt: torch.Tensor, do_normalize: bool) -> torch.Tensor:
    """Match Wav2Vec2FeatureExtractor zero_mean_unit_var_norm for a single waveform (..., T)."""
    if wav_1_bt.dim() != 2:
        raise ValueError(f"Expected (batch, length) waveform tensor, got shape {wav_1_bt.shape}")
    if do_normalize:
        mean = wav_1_bt.mean(dim=-1, keepdim=True)
        var = wav_1_bt.var(dim=-1, keepdim=True, unbiased=False)
        wav_1_bt = (wav_1_bt - mean) / torch.sqrt(var + 1e-7)
    return wav_1_bt


class HFWaveformToLogits(nn.Module):
    """encoder(wav_mono) -> mean-pooled hidden -> row L2-normalize -> classifier -> logits."""

    def __init__(self, model_name: str, classifier: nn.Module, device: str):
        super().__init__()
        if torchaudio is None:
            raise RuntimeError("torchaudio is required for waveform attacks/resampling.")
        self.device = torch.device(device)
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.classifier = classifier.to(self.device).eval()
        for p in self.classifier.parameters():
            p.requires_grad_(False)
        self.model_name = model_name
        self.target_sr = int(self.processor.sampling_rate)

    def forward(self, wav_mono: torch.Tensor) -> torch.Tensor:
        """wav_mono: (1, T) float tensor on same device as model, already resampled to target_sr."""
        if wav_mono.dim() != 2 or wav_mono.size(0) != 1:
            raise ValueError(f"Expected wav_mono shape (1, T), got {wav_mono.shape}")
        x = hf_preprocess_torch_mono(wav_mono, bool(self.processor.do_normalize))
        out = self.encoder(x)
        emb = out.last_hidden_state.mean(dim=1)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
        return self.classifier(emb)


def load_mono_waveform_tensor(
    entry: Dict,
    *,
    device: torch.device,
    target_sr: int,
) -> Tuple[torch.Tensor, Optional[int]]:
    """
    Load (1, T) mono waveform tensor. Returns (wav, sr_if_from_array_else_None).
    If entry has sampling_rate alongside audio_array, returns that sr so caller can still resample to target_sr.
    """
    if torchaudio is None:
        raise RuntimeError("torchaudio is required.")
    if "audio_array" in entry:
        wav = torch.tensor(entry["audio_array"], dtype=torch.float32)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        sr = entry.get("sampling_rate")
        wav = wav.to(device)
        if sr is None:
            raise RuntimeError("Manifest entry has 'audio_array' but missing required 'sampling_rate'.")
        return wav, int(sr)
    path = entry.get("path")
    if path is None or str(path).startswith("dummy"):
        raise RuntimeError(
            "Cannot load waveform: entry has no usable 'audio_array' and path is missing or synthetic (embedding-only)."
        )
    wav, sr = torchaudio.load(path)
    wav = wav.to(device)
    return wav, int(sr)


def prepare_mono_waveform_for_model(
    entry: Dict,
    *,
    device: torch.device,
    target_sr: int,
) -> torch.Tensor:
    wav, sr = load_mono_waveform_tensor(entry, device=device, target_sr=target_sr)
    if sr is None:
        sr = target_sr
    return _to_mono_and_resample(wav, sr, target_sr, str(device))


def manifest_entries_have_audio(entries: List[Dict]) -> bool:
    """True if every entry can yield a raw waveform (for attacks)."""
    for e in entries:
        try:
            if "audio_array" in e:
                if "sampling_rate" not in e:
                    return False
                continue
            p = e.get("path")
            if not p or str(p).startswith("dummy"):
                return False
        except Exception:
            return False
    return True


def simple_classifier_clone_on_device(clf: SimpleClassifier, device: str) -> SimpleClassifier:
    other = SimpleClassifier(clf.layer1[0].in_features, clf.out.out_features).to(device)
    other.load_state_dict(clf.state_dict())
    other.eval()
    return other
