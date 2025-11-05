from typing import List, Dict, Tuple
import numpy as np
from itertools import combinations
from sklearn.metrics import roc_curve

def compute_scores(embeddings: np.ndarray, trials: List[Tuple[int, int]]):
    return np.array([float(np.dot(embeddings[i], embeddings[j])) for i, j in trials])

def eer_from_scores(scores: np.ndarray, labels: np.ndarray, return_threshold: bool = False):
    if len(scores) == 0 or len(labels) == 0 or np.all(labels == 0) or np.all(labels == 1):
        return (float('nan'), float('nan')) if return_threshold else float('nan')
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    abs_diffs = np.abs(fnr - fpr)
    idx = np.argmin(abs_diffs)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    thr = thresholds[idx]
    return (eer, thr) if return_threshold else eer

def build_verification_trials(labels: List[str], forget_classes: List[str], split_mask: np.ndarray, seed: int = 42, max_pairs_per_class: int = 100) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    rng = np.random.RandomState(seed)
    indices = [i for i, m in enumerate(split_mask) if m]
    trials, trial_labels = [], []
    forget_idx = [i for i in indices if labels[i] in forget_classes]
    retain_idx = [i for i in indices if labels[i] not in forget_classes]

    # Positive pairs: all combinations within forget classes
    pos_pairs = []
    if len(forget_idx) >= 2:
        all_pos_pairs = list(combinations(forget_idx, 2))
        if max_pairs_per_class is None or max_pairs_per_class <= 0:
            pos_pairs = all_pos_pairs
        else:
            sel = rng.choice(len(all_pos_pairs), size=min(len(all_pos_pairs), max_pairs_per_class), replace=False)
            pos_pairs = [all_pos_pairs[i] for i in sel]

    # Negative pairs: all cross pairs between forget and retain
    neg_pairs = []
    if len(forget_idx) > 0 and len(retain_idx) > 0:
        all_neg_pairs = [(f, r) for f in forget_idx for r in retain_idx]
        if max_pairs_per_class is None or max_pairs_per_class <= 0:
            neg_pairs = all_neg_pairs
        else:
            sel = rng.choice(len(all_neg_pairs), size=min(len(all_neg_pairs), max_pairs_per_class), replace=False)
            neg_pairs = [all_neg_pairs[i] for i in sel]

    trials = pos_pairs + neg_pairs
    trial_labels = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs), dtype=int)
    return list(trials), trial_labels

def per_class_eers(emb_before: np.ndarray, emb_after: np.ndarray, labels: List[str],  mask: np.ndarray, rng_seed: int = 1337) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(rng_seed)
    idx_in_mask = np.where(mask)[0].tolist()
    classes = sorted({labels[i] for i in idx_in_mask})
    per_class = {}
    for c in classes:
        pos_idx = [i for i in idx_in_mask if labels[i] == c]
        other_idx = [i for i in idx_in_mask if labels[i] != c]
        if len(pos_idx) < 2 or len(other_idx) < 1:
            continue
        pos_pairs_all = list(combinations(pos_idx, 2))
        pos_pairs_sel = pos_pairs_all
        n_neg = len(pos_pairs_sel)
        pos_side_choices = rng.choice(pos_idx, size=n_neg, replace=True)
        other_side_choices = rng.choice(other_idx, size=n_neg, replace=True)
        neg_pairs_sel = [(int(a), int(b)) for a, b in zip(pos_side_choices, other_side_choices)]
        trials = pos_pairs_sel + neg_pairs_sel
        trial_labels = np.array([1] * len(pos_pairs_sel) + [0] * len(neg_pairs_sel), dtype=int)
        scores_before = compute_scores(emb_before, trials)
        scores_after = compute_scores(emb_after, trials)
        try:
            eer_b, _ = eer_from_scores(scores_before, trial_labels, return_threshold=True)
        except Exception:
            eer_b = float('nan')
        try:
            eer_a, _ = eer_from_scores(scores_after, trial_labels, return_threshold=True)
        except Exception:
            eer_a = float('nan')
        per_class[c] = (float(eer_b), float(eer_a))
    return per_class

def avg_eers(per_class_dict: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    if not per_class_dict:
        return float('nan'), float('nan')
    be = [v[0] for v in per_class_dict.values() if not np.isnan(v[0])]
    ae = [v[1] for v in per_class_dict.values() if not np.isnan(v[1])]
    avg_be = float(np.mean(be)) if len(be) > 0 else float('nan')
    avg_ae = float(np.mean(ae)) if len(ae) > 0 else float('nan')
    return avg_be, avg_ae