#!/usr/bin/env python3
"""
hf_download_and_preprocess_voxceleb1.py

Downloads VoxCeleb1 dev/test via Hugging Face dataset "ProgramComputer/voxceleb" (vox1 config),
extracts audio, runs VAD + segmentation, and writes a manifest for downstream processing.

Usage:
  python hf_download_and_preprocess_voxceleb1.py --cache_dir /data/hf_cache --out-manifest vox1_manifest.csv
"""

import os
import argparse
import csv
import tempfile
import zipfile
import pathlib

import numpy as np
import soundfile as sf
import webrtcvad
from datasets import load_dataset, DownloadConfig
from pathlib import Path

# VAD helper definitions
class Frame:
    __slots__ = ("bytes", "timestamp", "duration")
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_ms, audio, sr):
    n = int(sr * (frame_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = n / sr / 2.0
    audio_bytes = audio.tobytes()
    while offset + n <= len(audio_bytes):
        yield Frame(audio_bytes[offset:offset+n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sr, frame_ms, pad_ms, vad, frames):
    num_pad = int(pad_ms / frame_ms)
    ring = []
    triggered = False
    voiced_start = None
    voiced = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sr)
        if not triggered:
            ring.append((frame, is_speech))
            if sum(s for _, s in ring) > 0.9 * len(ring):
                triggered = True
                voiced_start = ring[0][0].timestamp
                ring = []
        else:
            ring.append((frame, is_speech))
            if sum(not s for _, s in ring) > 0.9 * len(ring):
                voiced_end = frame.timestamp + frame.duration
                voiced.append((voiced_start, voiced_end))
                triggered = False
                ring = []
    if triggered:
        voiced_end = frame.timestamp + frame.duration
        voiced.append((voiced_start, voiced_end))
    return voiced

def load_and_extract_zips(dataset, save_dir):
    """Downloads ZIP files via HF dataset and extracts dev/test WAV archives."""
    os.makedirs(save_dir, exist_ok=True)
    local_paths = {}
    for split in ['dev', 'test']:
        key = f"vox1_{split}_wav"
        content = dataset[f"{split}_wav"]
        out_zip = os.path.join(save_dir, f"{key}.zip")
        if not os.path.exists(out_zip):
            print(f"Downloading {key}...")
            dataset.download_and_extract(split, output_dir=save_dir, download_config=DownloadConfig(use_auth_token=True))
        # Extract
        print(f"Extracting {key} to {save_dir}/{split}/...")
        extract_dir = os.path.join(save_dir, split)
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(out_zip, 'r') as z:
            z.extractall(extract_dir)
        local_paths[split] = extract_dir
    return local_paths

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache_dir', default="./cache", help='Local directory to cache HF dataset')
    p.add_argument('--out_manifests_dir', default='./hf_vox1_processed', help='Where to write manifest CSVs')
    p.add_argument('--segment_len', type=float, default=3.0)
    p.add_argument('--hop', type=float, default=1.5)
    p.add_argument('--min_duration', type=float, default=0.5)
    p.add_argument('--min_rms', type=float, default=0.01)
    p.add_argument('--frame_ms', type=int, default=30)
    p.add_argument('--pad_ms', type=int, default=300)
    p.add_argument('--vad_mode', type=int, default=2, choices=[0,1,2,3])
    args = p.parse_args()

    print("Loading HF dataset ProgramComputer/voxceleb (vox1 config)...")
    ds = load_dataset("ProgramComputer/voxceleb", name="default", cache_dir=args.cache_dir)

    os.makedirs(args.out_manifests_dir, exist_ok=True)
    save_dir = os.path.join(args.cache_dir, "vox1_audio")
    splits = load_and_extract_zips(ds, save_dir)

    for split, wav_dir in splits.items():
        manifest_path = os.path.join(args.out_manifests_dir, f"vox1_{split}_manifest.csv")
        print(f"Processing split '{split}' WAV files under {wav_dir} → manifest: {manifest_path}")
        vad = webrtcvad.Vad(args.vad_mode)
        with open(manifest_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for root, _, files in os.walk(wav_dir):
                for fn in files:
                    if fn.lower().endswith('.wav'):
                        speaker_id = Path(root).name
                        wav_path = os.path.join(root, fn)
                        data, sr = sf.read(wav_path, dtype='float32')
                        if data.ndim > 1:
                            data = data.mean(axis=1)
                        frames = frame_generator(args.frame_ms, (data*32767).astype('<i2'), sr)
                        voiced = vad_collector(sr, args.frame_ms, args.pad_ms, vad, list(frames))
                        for (vstart, vend) in voiced:
                            if vend - vstart < args.min_duration:
                                continue
                            t = vstart
                            while t + args.segment_len <= vend + 1e-6:
                                start = t; end = t + args.segment_len
                                s_frame = int(round(start * sr)); e_frame = int(round(end * sr))
                                seg = data[s_frame:e_frame]
                                rms = float(np.sqrt(np.mean(seg**2)+1e-12)) if seg.size>0 else 0
                                if rms >= args.min_rms:
                                    seg_id = f"{speaker_id}_{fn}_{start:.3f}_{end:.3f}"
                                    writer.writerow([seg_id, wav_path, f"{start:.3f}", f"{end:.3f}", speaker_id, fn])
                                t += args.hop
        print(f"Manifest for '{split}' written: {manifest_path}")

if __name__ == "__main__":
    main()
