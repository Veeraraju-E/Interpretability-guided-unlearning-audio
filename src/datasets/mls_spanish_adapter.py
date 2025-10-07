from typing import List, Dict, Optional
import os, glob, random
import numpy as np
from pathlib import Path
from .adapter_base import DatasetAdapter, Precomp
from src.core.encoders.registry import extract_embeddings
from src.core.preprocess.whitening import fit_whitener, apply_whitener

class MLSSpanishAdapter(DatasetAdapter):
    name = "mls_spanish"
    label_key = "label"

    def _read_raw(self, root: str) -> List[Dict]:
        """Read raw MLS Spanish dataset from directory structure."""
        raw = []
        # MLS Spanish structure: /dev/audio/{speaker_id}/{subfolder_id}/*.flac
        pattern = os.path.join(root, "*", "*.flac")
        for p in glob.glob(pattern):
            parts = Path(p).parts
            # Extract speaker ID from path: .../audio/{speaker_id}/{subfolder_id}/file.flac
            if len(parts) >= 2:
                speaker_id = parts[-3]  # speaker folder name
                raw.append({"path": p, "label": speaker_id, "segment_id": os.path.basename(p)})
        return raw

    def build_manifest(self, root: str, embeddings_path: Optional[str], dev_frac: float, seed: int) -> List[Dict]:
        """Build manifest for MLS Spanish dataset."""
        if embeddings_path is not None:
            data = np.load(embeddings_path, allow_pickle=True)
            labels = data.get('labels')
            if labels is None:
                labels = data.get('y')
            ids = data.get('segment_ids')
            if ids is None:
                ids = data.get('ids')
            if labels is None:
                raise RuntimeError("embeddings .npz must contain 'labels' or 'y'")
            labels = list(labels)
            ids = list(ids) if ids is not None else [None] * len(labels)
            raw = [{'path': f'dummy_{i}', 'label': labels[i], 'segment_id': ids[i]} for i in range(len(labels))]
        else:
            raw = self._read_raw(root)
        
        # Group by speaker
        by_speaker = {}
        for i, m in enumerate(raw):
            by_speaker.setdefault(m['label'], []).append(i)
        
        rng = random.Random(seed)
        manifest: List[Dict] = []
        
        for speaker, indices in by_speaker.items():
            rng.shuffle(indices)
            n_dev = max(1, int(len(indices) * dev_frac))
            if len(indices) > 1 and n_dev >= len(indices):
                n_dev = len(indices) - 1
            
            dev_idx = indices[:n_dev]
            test_idx = indices[n_dev:]
            
            for idx in dev_idx:
                e = raw[idx].copy()
                e['split'] = 'dev'
                manifest.append(e)
            for idx in test_idx:
                e = raw[idx].copy()
                e['split'] = 'test'
                manifest.append(e)
        
        if len(manifest) == 0:
            raise RuntimeError("No entries found in manifest")
        return manifest

    def precompute(self, root: str, embeddings_path: Optional[str], dev_frac: float, seed: int, encoder_cfg: Dict, device: str) -> Precomp:
        """Precompute embeddings and other data for MLS Spanish dataset."""
        manifest = self.build_manifest(root, embeddings_path, dev_frac, seed)
        
        # Sort manifest for consistent ordering
        manifest_ordered = sorted(manifest, key=lambda x: (x['label'], x['path']))
        
        if embeddings_path is not None:
            # Load precomputed embeddings
            data = np.load(embeddings_path, allow_pickle=True)
            E_all = data['embeddings'] if 'embeddings' in data else data['E_all']
            labels = [m['label'] for m in manifest_ordered]
        else:
            # Extract embeddings from audio files
            E_all, labels, _ = extract_embeddings(
                manifest_ordered, 
                encoder=encoder_cfg['name'], 
                device=device, 
                wav2vec2_model=encoder_cfg.get('wav2vec2_model', 'facebook/wav2vec2-base'), 
                hubert_model=encoder_cfg.get('hubert_model', 'facebook/hubert-base-ls960'), 
                conformer_model=encoder_cfg.get('conformer_model', 'facebook/wav2vec2-conformer-rel-pos-large'), 
                label_key='label'
            )
        
        mask_dev = np.array([m['split'] == 'dev' for m in manifest_ordered], dtype=bool)
        mask_test = np.array([m['split'] == 'test' for m in manifest_ordered], dtype=bool)
        
        # Fit whitener on dev set
        mu, W = fit_whitener(E_all[mask_dev])
        Z_all = apply_whitener(E_all, mu, W)
        
        classes = sorted(list(set(labels)))
        
        return Precomp(
            manifest=manifest_ordered, 
            labels=labels, 
            classes=classes, 
            E_all=E_all, 
            Z_all=Z_all, 
            mu=mu, 
            W=W, 
            mask_dev=mask_dev, 
            mask_test=mask_test
        )
