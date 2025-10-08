from typing import List, Dict, Optional
import random
from pathlib import Path
import numpy as np
from .adapter_base import DatasetAdapter, Precomp
from src.core.encoders.registry import extract_embeddings
from src.core.preprocess.whitening import fit_whitener, apply_whitener

class AudioMNISTAdapter(DatasetAdapter):
    name = "audiomnist"
    label_key = "label"

    def build_manifest(self, data_root: str, embeddings_path: Optional[str], dev_frac: float, seed: int) -> List[Dict]:
        if embeddings_path is not None:
            data = np.load(embeddings_path, allow_pickle=True)
            labels = data['labels'].tolist() if 'labels' in data else (data['y'].tolist() if 'y' in data else None)
            ids = data['segment_ids'].tolist() if 'segment_ids' in data else (data['ids'].tolist() if 'ids' in data else None)
            if labels is None:
                raise RuntimeError("embeddings .npz must contain 'labels' or 'y' array when embeddings_path is used")
            raw_entries = [{'path': f'dummy_{i}', 'label': str(labels[i]), 'segment_id': ids[i] if ids is not None else None} for i in range(len(labels))]
        else:
            wavs = sorted(Path(data_root).glob("*.wav"))
            if not wavs:
                raise RuntimeError(f"No wav files found in {data_root}")
            raw_entries: List[Dict] = []
            for p in wavs:
                fname = p.stem
                parts = fname.split("_")
                if len(parts) < 1:
                    continue
                digit = parts[0]
                raw_entries.append({"path": str(p), "label": str(digit), "segment_id": str(p)})
        by_label = {}
        for i, e in enumerate(raw_entries):
            by_label.setdefault(e['label'], []).append(i)
        rng = random.Random(seed)
        manifest: List[Dict] = []
        for label, indices in by_label.items():
            rng.shuffle(indices)
            n_dev = max(1, int(len(indices) * dev_frac))
            if len(indices) > 1 and n_dev >= len(indices):
                n_dev = len(indices) - 1
            dev_idx = indices[:n_dev]; test_idx = indices[n_dev:]
            for idx in dev_idx:
                e = raw_entries[idx].copy(); e['split'] = 'dev'; manifest.append(e)
            for idx in test_idx:
                e = raw_entries[idx].copy(); e['split'] = 'test'; manifest.append(e)
        if len(manifest) == 0:
            raise RuntimeError("No entries found in manifest")
        return manifest

    def precompute(self, data_root: str, embeddings_path: Optional[str], dev_frac: float, seed: int, encoder_cfg: Dict, device: str) -> Precomp:
        entries = self.build_manifest(data_root, embeddings_path, dev_frac, seed)
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
                    sid = str(entry.get('segment_id', ''))
                    sid_b = os.path.basename(sid)
                    if sid in id_to_idx:
                        E_all[i] = embs[id_to_idx[sid]]
                    elif sid_b in id_to_idx:
                        E_all[i] = embs[id_to_idx[sid_b]]
                    else:
                        raise RuntimeError(f"segment_id {sid} not found in embeddings file keys")
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


