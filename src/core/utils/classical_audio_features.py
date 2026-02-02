"""
Classical audio feature extraction (MFCC, F0, F1-F5, prosody) per utterance.
Aligned to manifest order; supports optional segment (start/end) for VoxCeleb.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import parselmouth
from parselmouth.praat import call

SR = 16000
N_FFT = 512
HOP = 160
WIN = 400
N_MFCC = 13
N_FORMANTS = 5
DIM_MFCC = N_MFCC * 2
DIM_F0 = 6
DIM_FORMANTS = N_FORMANTS * 3
DIM_PROSDY = 4
FEATURE_DIM = DIM_MFCC + DIM_F0 + DIM_FORMANTS + DIM_PROSDY
IDX_MFCC = slice(0, DIM_MFCC)
IDX_F0 = slice(DIM_MFCC, DIM_MFCC + DIM_F0)
IDX_FORMANTS = slice(DIM_MFCC + DIM_F0, DIM_MFCC + DIM_F0 + DIM_FORMANTS)
IDX_PROSDY = slice(DIM_MFCC + DIM_F0 + DIM_FORMANTS, FEATURE_DIM)


def _load_segment(path: str, start: Optional[float] = None, end: Optional[float] = None) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        sr = SR
    if start is not None and start > 0:
        wav = wav[int(start * sr):]
    if end is not None and end > (start or 0):
        end_samp = int(end * sr) - (int(start * sr) if start else 0)
        wav = wav[:end_samp]
    return wav, sr


def _mfcc_utterance(wav: np.ndarray, sr: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def _f0_utterance(wav: np.ndarray, sr: int) -> np.ndarray:
    f0, voiced_flag, _ = librosa.pyin(wav, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr, hop_length=HOP)
    f0_voiced = np.nan_to_num(f0, nan=0.0)
    mean_f0 = float(np.mean(f0_voiced[voiced_flag]) if np.any(voiced_flag) else 0.0)
    std_f0 = float(np.std(f0_voiced[voiced_flag]) if np.sum(voiced_flag) > 1 else 0.0)
    median_f0 = float(np.median(f0_voiced[voiced_flag]) if np.any(voiced_flag) else 0.0)
    voiced_frac = float(np.mean(voiced_flag))
    low, high = np.percentile(f0_voiced[voiced_flag], [5, 95]) if np.sum(voiced_flag) > 2 else (0.0, 0.0)
    return np.array([mean_f0, std_f0, median_f0, voiced_frac, float(low), float(high)])


def _formants_utterance(wav: np.ndarray, sr: int) -> np.ndarray:
    snd = parselmouth.Sound(wav, sampling_frequency=sr)
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    formant = call(snd, "To Formant (burg)", 0.0, N_FORMANTS, 5500.0, 0.025, 50.0)
    n_frames = call(point_process, "Get number of points")
    out = []
    for fi in range(1, N_FORMANTS + 1):
        vals = []
        for i in range(1, n_frames + 1):
            t = call(point_process, "Get time from index", i)
            f = call(formant, "Get value at time", fi, t, "Hertz", "Linear")
            if not np.isnan(f) and f != 0:
                vals.append(f)
        vals = np.array(vals) if vals else np.zeros(1)
        out.extend([float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else 0.0, float(np.median(vals))])
    return np.array(out)


def _prosody_utterance(wav: np.ndarray, sr: int) -> np.ndarray:
    rms = librosa.feature.rms(y=wav, hop_length=HOP, frame_length=WIN)[0]
    rms_mean, rms_std = float(rms.mean()), float(rms.std())
    duration = len(wav) / sr
    f0, voiced_flag, _ = librosa.pyin(wav, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr, hop_length=HOP)
    f0_voiced = np.nan_to_num(f0, nan=0.0)
    pitch_range = float(np.ptp(f0_voiced[voiced_flag])) if np.sum(voiced_flag) > 1 else 0.0
    return np.array([rms_mean, rms_std, duration, pitch_range])


def extract_utterance_features(path: str, start: Optional[float] = None, end: Optional[float] = None) -> np.ndarray:
    """Extract and concatenate all classical features for a single utterance."""
    wav, sr = _load_segment(path, start, end)
    mfcc = _mfcc_utterance(wav, sr)
    f0 = _f0_utterance(wav, sr)
    formants = _formants_utterance(wav, sr)
    prosody = _prosody_utterance(wav, sr)
    return np.concatenate([mfcc, f0, formants, prosody])


def extract_all_utterances(manifest: List[Dict], cache_path: Optional[str] = None) -> np.ndarray:
    """
    Extract classical features for each manifest entry. Each row = one utterance.
    manifest[i] must have 'path'; may have 'start', 'end' (seconds).
    Returns matrix (N, FEATURE_DIM) - rows with dummy paths are NaN.
    """
    out = np.full((len(manifest), FEATURE_DIM), np.nan, dtype=np.float32)
    for i, entry in enumerate(tqdm(manifest, desc="Extracting classical features")):
        ent = entry if isinstance(entry, dict) else dict(entry)
        path = ent.get("path", "")
        if path.startswith("dummy_"):
            continue
        vec = extract_utterance_features(path, ent.get("start"), ent.get("end"))
        out[i, :vec.size] = vec
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez_compressed(cache_path, features=out, allow_pickle=False)
    return out
