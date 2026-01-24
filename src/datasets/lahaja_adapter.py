from typing import List, Dict, Optional
import os, random
import numpy as np
from datasets import load_dataset
from .adapter_base import DatasetAdapter, Precomp
from src.core.encoders import extract_embeddings
from src.core.utils.whitening import fit_whitener, apply_whitener
from rich.console import Console

console = Console()

class LahajaAdapter(DatasetAdapter):
    name = "lahaja"
    label_key = "label"

    def _extract_audio_info_from_item(self, item: Dict, index: int = 0) -> Optional[Dict]:
        audio_info = {}
        
        # Lahaja dataset uses 'audio_filepath' key with AudioDecoder object
        if 'audio_filepath' in item:
            audio_decoder = item['audio_filepath']
            # The AudioDecoder object has array and sampling_rate attributes directly
            try:
                # Access the audio data directly from the decoder
                audio_array = audio_decoder['array']
                sampling_rate = audio_decoder['sampling_rate']
                
                audio_info['array'] = audio_array
                audio_info['sampling_rate'] = sampling_rate
                audio_info['path'] = f"lahaja_audio_{index}"
                return audio_info
            except Exception as e:
                print(f"Warning: Could not decode audio for item {index}: {e}")
                return None
        
        # Fallback: try other possible audio keys
        audio_keys = ['audio', 'audio_path', 'file', 'path']
        for key in audio_keys:
            if key in item:
                val = item[key]
                if isinstance(val, dict):
                    # Handle audio dict with array and sampling_rate
                    if 'array' in val and 'sampling_rate' in val:
                        audio_info['array'] = val['array']
                        audio_info['sampling_rate'] = val['sampling_rate']
                        audio_info['path'] = val.get('path', f"lahaja_audio_{index}")
                        return audio_info
                    # Handle audio dict with path
                    elif 'path' in val and isinstance(val['path'], str):
                        audio_info['path'] = val['path']
                        return audio_info
                elif isinstance(val, str):
                    # Handle direct path string
                    audio_info['path'] = val
                    return audio_info
        
        return None

    def build_manifest(self, split: str, embeddings_path: Optional[str], dev_frac: float, seed: int) -> List[Dict]:
        if embeddings_path is not None:
            data = np.load(embeddings_path, allow_pickle=True)
            labels = data['labels'].tolist() if 'labels' in data else (data['y'].tolist() if 'y' in data else None)
            print(f"\n[Debug] labels: {labels}")
            ids = data['segment_ids'].tolist() if 'segment_ids' in data else (data['ids'].tolist() if 'ids' in data else None)
            if labels is None:
                raise RuntimeError("embeddings .npz must contain 'labels' or 'y' array when embeddings_path is used")
            raw_entries = [{'path': f'dummy_{i}', 'label': labels[i], 'segment_id': ids[i] if ids is not None else None} for i in range(len(labels))]
        else:
            print(f"Loading Lahaja dataset with split: {split}")
            try:
                ds = load_dataset("ai4bharat/Lahaja", split=split, streaming=True)
                print("Dataset loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load Lahaja dataset: {e}. Please ensure you have access to the dataset and have accepted the terms.")
            
            raw_entries = []
            speaker_ids = set()
            for i, item in enumerate(ds):
                # if i % 100 == 0:  # Progress indicator
                #     print(f"Processing item {i}...")
                
                audio_info = self._extract_audio_info_from_item(item, i)
                if audio_info is None:
                    console.print(f"[red] Warning: Could not extract audio info for item {i}")
                    continue
                    
                # Extract speaker ID - Lahaja uses 'sp_id' field
                speaker_id = item.get('sp_id')
                if speaker_id is None:
                    console.print(f"[red] Warning: No speaker ID found for item {i}, skipping...")
                    continue
                
                speaker_ids.add(str(speaker_id))
                raw_entries.append({
                    'path': audio_info.get('path', f"lahaja_{i}"),
                    'label': str(speaker_id),  # Convert to string for consistency
                    'segment_id': f"lahaja_{i}",
                    'audio_array': audio_info.get('array'),  # Store the actual audio data
                    'sampling_rate': audio_info.get('sampling_rate')  # Store the sampling rate
                })
            
            console.print(f"[green] Found {len(speaker_ids)} unique speakers: {sorted(list(speaker_ids))[:10]}...")  # Show first 10 speakers
        valid_manifest = [m for m in raw_entries if m.get('label') is not None]
        by = {}
        for i, m in enumerate(valid_manifest):
            by.setdefault(m['label'], []).append(i)
        rng = random.Random(seed)
        manifest: List[Dict] = []
        for speaker, indices in by.items():
            rng.shuffle(indices)
            n_dev = max(1, int(len(indices) * dev_frac))
            if len(indices) > 1 and n_dev >= len(indices):
                n_dev = len(indices) - 1
            for idx in indices[:n_dev]:
                e = valid_manifest[idx].copy(); e['split'] = 'dev'; manifest.append(e)
            for idx in indices[n_dev:]:
                e = valid_manifest[idx].copy(); e['split'] = 'test'; manifest.append(e)
        if len(manifest) == 0:
            raise RuntimeError("No entries found in manifest")
        return manifest

    def precompute(self, split: str, embeddings_path: Optional[str], dev_frac: float, seed: int, encoder_cfg: Dict, device: str) -> Precomp:
        entries = self.build_manifest(split, embeddings_path, dev_frac, seed)
        dev_manifest = [m for m in entries if m['split'] == 'dev']
        test_manifest = [m for m in entries if m['split'] == 'test']
        manifest_ordered = list(dev_manifest) + list(test_manifest)
        if embeddings_path:
            data = np.load(embeddings_path, allow_pickle=True)
            embs = data['embeddings']
            seg_ids = None
            if 'segment_ids' in data:
                seg_ids = list(map(str, data['segment_ids']))
            elif 'ids' in data:
                seg_ids = list(map(str, data['ids']))
            if seg_ids is not None:
                id_to_idx = {sid: i for i, sid in enumerate(seg_ids)}
                E_all = np.zeros((len(manifest_ordered), embs.shape[1]), dtype=np.float32)
                for i, entry in enumerate(manifest_ordered):
                    sid = entry.get('segment_id', None)
                    if sid is None:
                        raise RuntimeError("Embeddings provided but manifest entries lack 'segment_id' to align")
                    sid_s = str(sid)
                    if sid_s not in id_to_idx:
                        raise RuntimeError(f"segment_id {sid} not found in embeddings file keys")
                    E_all[i] = embs[id_to_idx[sid_s]]
                labels = [m['label'] for m in manifest_ordered]
            else:
                if embs.shape[0] != len(manifest_ordered):
                    raise RuntimeError("Embeddings count does not match manifest length and no segment_ids provided for alignment")
                E_all = embs.astype(np.float32)
                labels = [m['label'] for m in manifest_ordered]
        else:
            E_all, labels, _ = extract_embeddings(manifest_ordered, encoder=encoder_cfg['name'], device=device, wav2vec2_model=encoder_cfg.get('wav2vec2_model', 'facebook/wav2vec2-base'), hubert_model=encoder_cfg.get('hubert_model', 'facebook/hubert-base-ls960'), conformer_model=encoder_cfg.get('conformer_model', 'facebook/wav2vec2-conformer-rel-pos-large'), label_key='label')
        mask_dev = np.array([m['split'] == 'dev' for m in manifest_ordered], dtype=bool)
        mask_test = np.array([m['split'] == 'test' for m in manifest_ordered], dtype=bool)
        mu, W = fit_whitener(E_all[mask_dev])
        Z_all = apply_whitener(E_all, mu, W)
        classes = sorted(list(set(labels)))
        return Precomp(manifest=manifest_ordered, labels=labels, classes=classes, E_all=E_all, Z_all=Z_all, mu=mu, W=W, mask_dev=mask_dev, mask_test=mask_test)