"""
Trait probe fitting and evaluation module.
Implements Steps 4-5 of the interpretation protocol.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score


def fit_continuous_probe(X_train: np.ndarray, y_train: np.ndarray, lambda_reg: float = 1e-6) -> Tuple[np.ndarray, float]:
    """Fit ridge regressor for continuous trait."""
    clf = Ridge(alpha=lambda_reg, fit_intercept=True)
    clf.fit(X_train, y_train.ravel())
    return clf.coef_.flatten(), clf.intercept_


def fit_discrete_probe(X_train: np.ndarray, y_train: np.ndarray, lambda_reg: float = 1e-6) -> Tuple[np.ndarray, float]:
    """Fit logistic regressor for discrete trait."""
    clf = LogisticRegression(C=1.0/lambda_reg, fit_intercept=True, max_iter=1000, solver='liblinear', random_state=42)
    clf.fit(X_train, y_train.ravel())
    return clf.coef_.flatten(), clf.intercept_[0]


def predict_continuous(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Predict continuous trait values."""
    return X @ w + b


def predict_discrete_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Predict discrete trait probabilities using sigmoid."""
    scores = X @ w + b
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -500, 500)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination R²."""
    if len(y_true) == 0:
        return 0.0
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def compute_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute AUC for binary classification."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_proba)


def fit_trait_probes(
    embeddings_pre: np.ndarray,
    embeddings_post: np.ndarray,
    traits: np.ndarray,
    train_mask: np.ndarray,
    lambda_reg: float = 1e-6,
    trait_types: Optional[List[str]] = None
) -> Dict:
    """
    Fit probes for all traits on pre and post embeddings.
    
    Args:
        embeddings_pre: (n_samples, d) pre-edit embeddings
        embeddings_post: (n_samples, d) post-edit embeddings
        traits: (n_samples, n_traits) trait matrix
        train_mask: (n_samples,) boolean mask for training samples
        lambda_reg: Ridge regularization parameter
        trait_types: List of 'continuous' or 'discrete' for each trait (default: all continuous)
    
    Returns:
        Dictionary with probe weights and biases for pre/post
    """
    n_traits = traits.shape[1]
    if trait_types is None:
        trait_types = ['continuous'] * n_traits
    
    X_train_pre = embeddings_pre[train_mask]
    X_train_post = embeddings_post[train_mask]
    
    probes_pre = []
    probes_post = []
    
    for k in range(n_traits):
        y_train = traits[train_mask, k]
        
        if trait_types[k] == 'continuous':
            w_pre, b_pre = fit_continuous_probe(X_train_pre, y_train, lambda_reg)
            w_post, b_post = fit_continuous_probe(X_train_post, y_train, lambda_reg)
        else:
            w_pre, b_pre = fit_discrete_probe(X_train_pre, y_train, lambda_reg)
            w_post, b_post = fit_discrete_probe(X_train_post, y_train, lambda_reg)
        
        probes_pre.append({'w': w_pre, 'b': b_pre, 'type': trait_types[k]})
        probes_post.append({'w': w_post, 'b': b_post, 'type': trait_types[k]})
    
    return {
        'probes_pre': probes_pre,
        'probes_post': probes_post,
        'trait_types': trait_types
    }


def evaluate_probes(
    probes: Dict,
    embeddings: np.ndarray,
    traits: np.ndarray,
    eval_mask: np.ndarray,
    which: str = 'pre'
) -> Dict[str, np.ndarray]:
    """
    Evaluate probes on evaluation set.
    
    Args:
        probes: Dictionary from fit_trait_probes
        embeddings: (n_samples, d) embeddings to evaluate
        traits: (n_samples, n_traits) ground truth traits
        eval_mask: (n_samples,) boolean mask for evaluation samples
        which: 'pre' or 'post' to select which probes to use
    
    Returns:
        Dictionary with metrics per trait
    """
    probe_list = probes[f'probes_{which}']
    n_traits = len(probe_list)
    
    X_eval = embeddings[eval_mask]
    y_eval = traits[eval_mask]
    
    r2_scores = np.zeros(n_traits)
    auc_scores = np.zeros(n_traits)
    
    for k in range(n_traits):
        probe = probe_list[k]
        w, b = probe['w'], probe['b']
        y_true = y_eval[:, k]
        
        if probe['type'] == 'continuous':
            y_pred = predict_continuous(X_eval, w, b)
            r2_scores[k] = compute_r2(y_true, y_pred)
            auc_scores[k] = np.nan
        else:
            y_proba = predict_discrete_proba(X_eval, w, b)
            auc_scores[k] = compute_auc(y_true, y_proba)
            r2_scores[k] = np.nan
    
    return {
        'r2': r2_scores,
        'auc': auc_scores
    }


def compute_leakage_ratios(
    metrics_pre: Dict[str, np.ndarray],
    metrics_post: Dict[str, np.ndarray],
    trait_types: List[str],
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Compute leakage ratios for forget speaker (Step 5).
    
    Args:
        metrics_pre: Metrics from evaluate_probes on pre-edit embeddings
        metrics_post: Metrics from evaluate_probes on post-edit embeddings
        trait_types: List of 'continuous' or 'discrete' for each trait
        epsilon: Small constant to avoid division by zero
    
    Returns:
        Array of leakage ratios (one per trait)
    """
    n_traits = len(trait_types)
    leak_ratios = np.zeros(n_traits)
    
    for k in range(n_traits):
        if trait_types[k] == 'continuous':
            r2_pre = metrics_pre['r2'][k]
            r2_post = metrics_post['r2'][k]
            denominator = max(abs(r2_pre), epsilon)
            leak_ratios[k] = r2_post / denominator
        else:
            auc_pre = metrics_pre['auc'][k]
            auc_post = metrics_post['auc'][k]
            numerator = abs(auc_post - 0.5)
            denominator = max(abs(auc_pre - 0.5), epsilon)
            leak_ratios[k] = numerator / denominator
    
    return leak_ratios
