from typing import Tuple, List, Dict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

def _prepare_probe_data(Z_all: np.ndarray, labels: List[str], U: np.ndarray, split_mask: Dict[str, np.ndarray], forget_classes: List[str]):
    Zu = Z_all @ U if U is not None else Z_all
    train_mask = split_mask['dev']
    test_mask  = split_mask['test']
    y_train = np.array([int(l in forget_classes) for i,l in enumerate(labels) if train_mask[i]])
    y_test  = np.array([int(l in forget_classes) for i,l in enumerate(labels) if test_mask[i]])
    X_train = Zu[train_mask]
    X_test  = Zu[test_mask]
    X_train = np.ascontiguousarray(X_train, dtype=np.float64)
    X_test  = np.ascontiguousarray(X_test, dtype=np.float64)
    return X_train, y_train, X_test, y_test

def safe_linear_probe(Z_all: np.ndarray, labels: List[str], U: np.ndarray, split_mask: Dict[str, np.ndarray], forget_classes: List[str]) -> Tuple[float, float]:
    try:
        if U is None or U.shape[1] == 0:
            return float('nan'), float('nan')
        X_train, y_train, X_test, y_test = _prepare_probe_data(Z_all, labels, U, split_mask, forget_classes)
        if (len(X_train) == 0 or len(X_test) == 0 or len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2 or np.any(~np.isfinite(X_train)) or np.any(~np.isfinite(X_test))):
            return float('nan'), float('nan')
        clf = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
        clf.fit(X_train, y_train)
        s = clf.decision_function(X_test) if hasattr(clf, 'decision_function') else clf.predict_proba(X_test)[:,1]
        acc = accuracy_score(y_test, (s > 0).astype(int))
        auc_val = roc_auc_score(y_test, s) if len(set(y_test)) == 2 else float('nan')
        return float(acc), float(auc_val)
    except Exception:
        return float('nan'), float('nan')


def linear_probe_class_identity(
    E_all: np.ndarray,
    labels: List[str],
    mask_dev: np.ndarray,
    mask_test: np.ndarray,
    max_iter: int = 1000,
    random_state: int = 42,
) -> float:
    """
    Train a linear probe for multi-class class identity on E_all; return test accuracy.
    Uses LogisticRegression with multi_class='ovr'.
    """
    try:
        classes = sorted(list(set(labels)))
        if len(classes) < 2:
            return float('nan')
        cls2id = {c: i for i, c in enumerate(classes)}
        X_train = np.ascontiguousarray(E_all[mask_dev], dtype=np.float64)
        X_test = np.ascontiguousarray(E_all[mask_test], dtype=np.float64)
        y_train = np.array([cls2id[labels[i]] for i in range(len(labels)) if mask_dev[i]], dtype=np.int64)
        y_test = np.array([cls2id[labels[i]] for i in range(len(labels)) if mask_test[i]], dtype=np.int64)
        if len(X_train) == 0 or len(X_test) == 0:
            return float('nan')
        if len(np.unique(y_train)) < 2 or np.any(~np.isfinite(X_train)) or np.any(~np.isfinite(X_test)):
            return float('nan')
        clf = LogisticRegression(multi_class='ovr', max_iter=max_iter, random_state=random_state, solver='liblinear')
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        return float(acc)
    except Exception:
        return float('nan')