from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class Precomp:
    manifest: List[Dict]
    labels: List[str]
    classes: List[str]
    E_all: np.ndarray
    Z_all: np.ndarray
    mu: np.ndarray
    W: np.ndarray
    mask_dev: np.ndarray
    mask_test: np.ndarray

class DatasetAdapter:
    name: str = "base"
    label_key: str = "label"

    def build_manifest(self, root_or_manifest: str, embeddings_path: Optional[str], dev_frac: float, seed: int) -> List[Dict]:
        raise NotImplementedError

    def precompute(self, root_or_manifest: str, embeddings_path: Optional[str], dev_frac: float, seed: int, encoder_cfg: Dict, device: str) -> Precomp:
        raise NotImplementedError


