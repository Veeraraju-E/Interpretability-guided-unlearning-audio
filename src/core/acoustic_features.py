"""
Acoustic feature extraction module for speech characteristics.
Implements Step 1 of the interpretation protocol.
"""

import numpy as np
from typing import Dict, Tuple
import torch
import torchaudio
import librosa

from scipy import signal
from scipy.fft import rfft, rfftfreq


def load_audio(entry: Dict, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio from entry dict, returning waveform and sample rate."""
    if 'audio_array' in entry and 'sampling_rate' in entry:
        wav = entry['audio_array']
        sr = entry['sampling_rate']
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        if wav.ndim > 1:
            wav = wav.mean(axis=0) if wav.ndim == 2 else wav.squeeze()
    else:
        wav, sr = torchaudio.load(entry['path'])
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
    
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    return wav, sr


def extract_f0_trajectory(wav: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Extract F0 trajectory and voicing decisions per frame."""
    f0, voiced_flag, _ = librosa.pyin(wav, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), 
                                       frame_length=frame_length, hop_length=hop_length)
    voiced_mask = (~np.isnan(voiced_flag)) & (voiced_flag > 0.5) if voiced_flag is not None else np.zeros(len(f0), dtype=bool)
    return f0, voiced_mask


def extract_formants(wav: np.ndarray, sr: int, f0: np.ndarray, voiced_mask: np.ndarray, 
                     frame_length: int = 2048, hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract F1, F2, F3 formant trajectories on voiced frames."""
    n_frames = len(f0)
    F1 = np.zeros(n_frames)
    F2 = np.zeros(n_frames)
    F3 = np.zeros(n_frames)
    
    for i in range(0, len(wav) - frame_length, hop_length):
        frame_idx = i // hop_length
        if frame_idx >= n_frames:
            break
        if not voiced_mask[frame_idx]:
            continue
        
        frame = wav[i:i+frame_length]
        frame = frame * np.hanning(len(frame))
        
        spectrum = np.abs(rfft(frame))
        freqs = rfftfreq(len(frame), 1/sr)
        
        if len(spectrum) < 3:
            continue
        
        peaks, props = signal.find_peaks(spectrum, height=np.max(spectrum) * 0.1, distance=int(sr / 2000))
        peak_freqs = freqs[peaks]
        peak_heights = spectrum[peaks]
        
        sorted_peaks = sorted(zip(peak_freqs, peak_heights), key=lambda x: x[1], reverse=True)
        
        if len(sorted_peaks) >= 1:
            F1[frame_idx] = sorted_peaks[0][0]
        if len(sorted_peaks) >= 2:
            F2[frame_idx] = sorted_peaks[1][0]
        if len(sorted_peaks) >= 3:
            F3[frame_idx] = sorted_peaks[2][0]
    
    return F1, F2, F3


def extract_energy(wav: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Extract frame-level short-time energy."""
    n_frames = (len(wav) - frame_length) // hop_length + 1
    energy = np.zeros(n_frames)
    
    for i in range(0, len(wav) - frame_length, hop_length):
        frame = wav[i:i+frame_length]
        energy[i // hop_length] = np.sum(frame ** 2)
    
    return energy


def extract_mfccs(wav: np.ndarray, sr: int, n_mfcc: int = 13, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Extract MFCC features per frame."""
    mfccs = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_length)
    return mfccs.T


def detect_phones_syllables(wav: np.ndarray, sr: int, energy_threshold: float = 0.01) -> int:
    """Detect phone-like or syllable-nuclei events using energy-based approach."""
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    energy = extract_energy(wav, frame_length=frame_length, hop_length=hop_length)
    
    threshold = np.percentile(energy, 20)
    above_threshold = energy > threshold
    
    events = 0
    in_event = False
    for val in above_threshold:
        if val and not in_event:
            events += 1
            in_event = True
        elif not val:
            in_event = False
    
    return events


def detect_pauses(wav: np.ndarray, sr: int, energy_threshold: float = 0.01, min_duration: float = 0.1) -> int:
    """Detect silence segments (pauses) above threshold and minimum duration."""
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    energy = extract_energy(wav, frame_length=frame_length, hop_length=hop_length)
    
    threshold = np.percentile(energy, 10)
    below_threshold = energy < threshold
    
    frame_duration = hop_length / sr
    min_frames = int(min_duration / frame_duration)
    
    pauses = 0
    pause_length = 0
    for val in below_threshold:
        if val:
            pause_length += 1
        else:
            if pause_length >= min_frames:
                pauses += 1
            pause_length = 0
    
    if pause_length >= min_frames:
        pauses += 1
    
    return pauses


def compute_snr_proxy(wav: np.ndarray) -> float:
    """Compute SNR proxy using signal-to-noise ratio estimation."""
    signal_power = np.var(wav)
    noise_estimate = np.percentile(np.abs(wav - np.mean(wav)), 25)
    noise_power = noise_estimate ** 2 + 1e-10
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def compute_spectral_flatness(wav: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> float:
    """Compute spectral flatness statistic."""
    n_frames = (len(wav) - frame_length) // hop_length + 1
    flatness_vals = []
    
    for i in range(0, len(wav) - frame_length, hop_length):
        frame = wav[i:i+frame_length]
        frame = frame * np.hanning(len(frame))
        spectrum = np.abs(rfft(frame))
        spectrum = spectrum + 1e-10
        
        geometric_mean = np.exp(np.mean(np.log(spectrum)))
        arithmetic_mean = np.mean(spectrum)
        flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0
        flatness_vals.append(flatness)
    
    return np.mean(flatness_vals) if flatness_vals else 0.0


def extract_acoustic_features(entry: Dict, target_sr: int = 16000, n_mfcc: int = 13, 
                             frame_length: int = 2048, hop_length: int = 512) -> Dict[str, float]:
    """
    Extract all acoustic features for an utterance (Step 1).
    Returns a dictionary mapping feature names to scalar values.
    """
    wav, sr = load_audio(entry, target_sr=target_sr)
    duration = len(wav) / sr
    
    f0, voiced_mask = extract_f0_trajectory(wav, sr, frame_length=frame_length, hop_length=hop_length)
    T_v = np.where(voiced_mask)[0]
    T_all = np.arange(len(f0))
    
    features = {}
    
    if len(T_v) > 0:
        f0_voiced = f0[T_v]
        features['F0-mean'] = np.mean(f0_voiced)
        features['F0-std'] = np.std(f0_voiced, ddof=1) if len(f0_voiced) > 1 else 0.0
        features['F0-range'] = np.max(f0_voiced) - np.min(f0_voiced)
    else:
        features['F0-mean'] = 0.0
        features['F0-std'] = 0.0
        features['F0-range'] = 0.0
    
    features['voicing-rate'] = len(T_v) / len(T_all) if len(T_all) > 0 else 0.0
    
    energy = extract_energy(wav, frame_length=frame_length, hop_length=hop_length)
    features['energy-mean'] = np.mean(energy)
    features['energy-std'] = np.std(energy, ddof=1) if len(energy) > 1 else 0.0
    
    n_phones = detect_phones_syllables(wav, sr)
    features['rate'] = n_phones / duration if duration > 0 else 0.0
    
    n_pauses = detect_pauses(wav, sr)
    features['pause-rate'] = n_pauses / duration if duration > 0 else 0.0
    
    if len(T_v) > 0:
        F1, F2, F3 = extract_formants(wav, sr, f0, voiced_mask, frame_length=frame_length, hop_length=hop_length)
        F1_voiced = F1[T_v]
        F2_voiced = F2[T_v]
        F3_voiced = F3[T_v]
        features['F1-mean'] = np.mean(F1_voiced[F1_voiced > 0])
        features['F2-mean'] = np.mean(F2_voiced[F2_voiced > 0])
        features['F3-mean'] = np.mean(F3_voiced[F3_voiced > 0])
        features['F2-F1'] = features['F2-mean'] - features['F1-mean']
    else:
        features['F1-mean'] = 0.0
        features['F2-mean'] = 0.0
        features['F3-mean'] = 0.0
        features['F2-F1'] = 0.0
    
    mfccs = extract_mfccs(wav, sr, n_mfcc=n_mfcc, frame_length=frame_length, hop_length=hop_length)
    mfcc_mean = np.mean(mfccs, axis=0)
    mfcc_var = np.var(mfccs, axis=0, ddof=1) if mfccs.shape[0] > 1 else np.zeros(n_mfcc)
    
    for i in range(n_mfcc):
        features[f'mfcc-mean-{i}'] = mfcc_mean[i]
        features[f'mfcc-var-{i}'] = mfcc_var[i]
    
    features['snr'] = compute_snr_proxy(wav)
    features['flat'] = compute_spectral_flatness(wav, frame_length=frame_length, hop_length=hop_length)
    
    return features


def extract_acoustic_features_batch(entries: list[Dict], target_sr: int = 16000, n_mfcc: int = 13,
                                    frame_length: int = 2048, hop_length: int = 512) -> Tuple[np.ndarray, list[str]]:
    """
    Extract acoustic features for a batch of entries.
    Returns (feature_matrix, feature_names) where feature_matrix is (n_samples, n_features).
    """
    all_features = []
    feature_names = None
    
    for entry in entries:
        feat_dict = extract_acoustic_features(entry, target_sr=target_sr, n_mfcc=n_mfcc,
                                             frame_length=frame_length, hop_length=hop_length)
        if feature_names is None:
            feature_names = sorted(feat_dict.keys())
        feat_vec = np.array([feat_dict[name] for name in feature_names])
        all_features.append(feat_vec)
    
    return np.array(all_features), feature_names
