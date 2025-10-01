from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from .classification import SimpleClassifier, train_classifier

def train_shadow_models(X_all: np.ndarray, labels: List[str], target_speaker: str, num_shadow_models: int = 5, shadow_train_ratio: float = 0.7, device: str = 'cuda', epochs: int = 15, seed: int = 42) -> List[SimpleClassifier]:
    rng = np.random.RandomState(seed)
    all_speakers = list(set(labels))
    non_target_speakers = [s for s in all_speakers if s != target_speaker]
    if len(non_target_speakers) < 2:
        return []
    shadow_models = []
    for _ in range(num_shadow_models):
        num_shadow_speakers = max(2, int(len(non_target_speakers) * shadow_train_ratio))
        shadow_speakers = rng.choice(non_target_speakers, size=num_shadow_speakers, replace=False)
        shadow_indices = [j for j, label in enumerate(labels) if label in shadow_speakers]
        if len(shadow_indices) < 10:
            continue
        X_shadow = X_all[shadow_indices]
        y_shadow_str = [labels[j] for j in shadow_indices]
        speaker_to_id = {spk: idx for idx, spk in enumerate(sorted(set(y_shadow_str)))}
        y_shadow = [speaker_to_id[spk] for spk in y_shadow_str]
        perm = rng.permutation(len(X_shadow))
        split_idx = int(0.8 * len(perm))
        train_idx, val_idx = perm[:split_idx], perm[split_idx:]
        shadow_model = train_classifier(X_shadow[train_idx], [y_shadow[j] for j in train_idx], X_shadow[val_idx], [y_shadow[j] for j in val_idx], num_classes=len(speaker_to_id), device=device, epochs=epochs, batch_size=256, lr=1e-3)
        shadow_models.append(shadow_model)
    return shadow_models

def extract_mia_features(embeddings: np.ndarray, labels: List[str], target_speaker: str, shadow_models: List[SimpleClassifier], device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
    if len(shadow_models) == 0:
        return np.array([]), np.array([])
    N = embeddings.shape[0]
    num_features = len(shadow_models) * 2
    features = np.zeros((N, num_features))
    membership_labels = np.array([1 if label == target_speaker else 0 for label in labels])
    X_tensor = torch.from_numpy(embeddings).float().to(device)
    for i, shadow_model in enumerate(shadow_models):
        shadow_model.eval().to(device)
        with torch.no_grad():
            logits = shadow_model(X_tensor)
            probs = F.softmax(logits, dim=1)
            max_confidence = probs.max(dim=1)[0].cpu().numpy()
            features[:, i*2] = max_confidence
            entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1).cpu().numpy()
            features[:, i*2 + 1] = entropy
    return features, membership_labels

def train_mia_classifier(mia_features: np.ndarray, membership_labels: np.ndarray, test_size: float = 0.3, seed: int = 42) -> LogisticRegression | None:
    if len(mia_features) == 0 or len(set(membership_labels)) < 2:
        return None
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(mia_features, membership_labels, test_size=test_size, random_state=seed, stratify=membership_labels)
    mia_clf = LogisticRegression(max_iter=1000, random_state=seed)
    mia_clf.fit(X_train, y_train)
    return mia_clf

def compute_mia_score(embeddings: np.ndarray, labels: List[str], target_speaker: str, mask: np.ndarray, num_shadow_models: int = 5, device: str = 'cuda', seed: int = 42) -> float:
    try:
        X_masked = embeddings[mask]
        labels_masked = [labels[i] for i in range(len(labels)) if mask[i]]
        if len(X_masked) == 0 or target_speaker not in labels_masked:
            return float('nan')
        shadow_models = train_shadow_models(X_masked, labels_masked, target_speaker, num_shadow_models=num_shadow_models, device=device, epochs=10, seed=seed)
        if len(shadow_models) == 0:
            return float('nan')
        mia_features, membership_labels = extract_mia_features(X_masked, labels_masked, target_speaker, shadow_models, device=device)
        if len(mia_features) == 0 or len(set(membership_labels)) < 2:
            return float('nan')
        mia_clf = train_mia_classifier(mia_features, membership_labels, seed=seed)
        if mia_clf is None:
            return float('nan')
        mia_probs = mia_clf.predict_proba(mia_features)[:, 1]
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(membership_labels, mia_probs))
    except Exception:
        return float('nan')


