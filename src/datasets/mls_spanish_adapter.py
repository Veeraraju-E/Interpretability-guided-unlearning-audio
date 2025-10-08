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
        raw = []
        # MLS Spanish structure: /root/speaker_id/book_id/*.flac
        pattern = os.path.join(root, "*", "*", "*.flac")
        for p in glob.glob(pattern):
            parts = Path(p).parts
            # Extract speaker ID from path: root/speaker_id/book_id/file.flac
            spk = parts[-3] if len(parts) >= 3 else "unknown"
            raw.append({"path": p, "label": spk})
        return raw

    def build_manifest(self, root_or_manifest: str, embeddings_path: Optional[str], dev_frac: float, seed: int) -> List[Dict]:
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
            raw = self._read_raw(root_or_manifest)
        by = {}
        for i, m in enumerate(raw):
            by.setdefault(m['label'], []).append(i)
        rng = random.Random(seed)
        manifest: List[Dict] = []
        for label, indices in by.items():
            rng.shuffle(indices)
            n_dev = max(1, int(len(indices) * dev_frac))
            if len(indices) > 1 and n_dev >= len(indices):
                n_dev = len(indices) - 1
            for idx in indices[:n_dev]:
                e = raw[idx].copy(); e['split'] = 'dev'; manifest.append(e)
            for idx in indices[n_dev:]:
                e = raw[idx].copy(); e['split'] = 'test'; manifest.append(e)
        if len(manifest) == 0:
            raise RuntimeError("No entries found in manifest")
        return manifest

    def precompute(self, root_or_manifest: str, embeddings_path: Optional[str], dev_frac: float, seed: int, encoder_cfg: Dict, device: str) -> Precomp:
        entries = self.build_manifest(root_or_manifest, embeddings_path, dev_frac, seed)
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
