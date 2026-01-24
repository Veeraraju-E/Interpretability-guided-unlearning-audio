import math
import numpy as np
from typing import Dict   
from rich.console import Console
from sklearn.metrics import roc_auc_score
try:
    from statsmodels.stats.multitest import multipletests as _multipletests
    multipletests = _multipletests
except Exception:
    multipletests = None


def cohens_d(x_pos: np.ndarray, x_neg: np.ndarray) -> float:
    n1, n2 = len(x_pos), len(x_neg)
    if n1 == 0 or n2 == 0:
        raise ValueError("Cannot compute Cohen's d: positive samples array is empty. This suggests a data filtering or sampling issue.")
    m1, m2 = x_pos.mean(), x_neg.mean()
    s1, s2 = x_pos.var(ddof=1), x_neg.var(ddof=1)
    s_pooled = math.sqrt(((n1-1)*s1 + (n2-1)*s2) / max(1, n1+n2-2))
    
    if s_pooled == 0:
        print(f"  Positive samples (n={n1}): mean={m1:.6f}, var={s1:.6f}, std={math.sqrt(s1):.6f}")
        print(f"  Negative samples (n={n2}): mean={m2:.6f}, var={s2:.6f}, std={math.sqrt(s2):.6f}")
        print(f"  Positive sample values: {x_pos[:min(10, len(x_pos))]}")
        print(f"  Negative sample values: {x_neg[:min(10, len(x_neg))]}")
        print(f"  All positive values unique: {len(np.unique(x_pos)) == 1}")
        print(f"  All negative values unique: {len(np.unique(x_neg)) == 1}")
        raise ValueError("Cannot compute Cohen's d: pooled standard deviation is zero. This indicates no variance in the data.")
    return float((m1 - m2) / s_pooled)

def mann_whitney_u_p(x_pos: np.ndarray, x_neg: np.ndarray) -> tuple[float,float]:
    n1, n2 = len(x_pos), len(x_neg)
    if n1 == 0 or n2 == 0:
        raise ValueError("Cannot compute Mann-Whitney U: samples array is empty. This suggests a data filtering or sampling issue.")

    x = np.concatenate([x_pos, x_neg])
    labels = np.concatenate([np.ones(len(x_pos)), np.zeros(len(x_neg))])
    ranks = np.argsort(np.argsort(x)) + 1
    r1 = ranks[labels==1].sum()
    U1 = r1 - n1*(n1+1)/2
    auc = U1 / (n1*n2 + 1e-12)
    mu = n1*n2/2
    sigma = math.sqrt(n1*n2*(n1+n2+1)/12)
    z = (U1 - mu) / (sigma + 1e-12)
    from math import erf, sqrt
    p = 2*(1 - 0.5*(1+erf(abs(z)/sqrt(2))))
    return auc, p


def feature_stats(A_pos: np.ndarray, A_neg: np.ndarray) -> Dict:
    # Validate input arrays
    if A_pos.shape[0] == 0 or A_neg.shape[0] == 0:
        raise ValueError("Cannot compute feature statistics: positive samples array is empty. This suggests a data filtering or sampling issue.")
    m = A_pos.shape[1]
    pos_activity = (A_pos > 1e-6).sum(axis=0)
    neg_activity = (A_neg > 1e-6).sum(axis=0)
    dead_latents = (pos_activity == 0) & (neg_activity == 0)    
    
    
    stats = {"cohens_d": np.zeros(m), "auc": np.zeros(m), "pval": np.ones(m)}
    
    # Only compute statistics for active features
    for j in range(m):
        if dead_latents[j]:
            stats["cohens_d"][j] = 0.0
            stats["auc"][j] = 0.5
            stats["pval"][j] = 1.0 
            continue
            
        x_pos = A_pos[:, j]; x_neg = A_neg[:, j]
        
        # Check for zero variance before computing Cohen's d
        pos_var = x_pos.var(ddof=1)
        neg_var = x_neg.var(ddof=1)
        if pos_var == 0 and neg_var == 0:
            stats["cohens_d"][j] = 0.0
            stats["auc"][j] = 0.5
            stats["pval"][j] = 1.0
            continue
        
        try:
            stats["cohens_d"][j] = cohens_d(x_pos, x_neg)
        except ValueError as e:
            stats["cohens_d"][j] = 0.0
        
        # ROC AUC computation
        try:
            y_true = np.concatenate([np.ones(len(x_pos)), np.zeros(len(x_neg))])
            y_scores = np.concatenate([x_pos, x_neg])
            stats["auc"][j] = float(roc_auc_score(y_true, y_scores))
        except Exception:
            stats["auc"][j] = 0.5
            
        try:
            _, p = mann_whitney_u_p(x_pos, x_neg)
            stats["pval"][j] = p
        except Exception:
            stats["pval"][j] = 1.0
    
    return stats

def select_features_from_stats(stats: dict, top_k: int = 200, fdr: float = 0.05):
    pvals = stats["pval"]
    
    # Filter out features with AUC = 0.5 (random performance)
    auc_scores = np.abs(stats["auc"] - 0.5)
    valid_features = auc_scores > 1e-6
    
    console = Console()
    console.print(f"Features with discriminative power (AUC != 0.5): {valid_features.sum()} out of {len(valid_features)}", style="yellow")
    
    if valid_features.sum() == 0:
        console.print("WARNING: No features with discriminative power found!", style="red")
        return []
    
    if multipletests is not None:
        # Only apply FDR correction to valid features
        valid_pvals = pvals[valid_features]
        print(f"[INFO]: Valid pvals: {valid_pvals}")
        rej, _, _, _ = multipletests(valid_pvals, alpha=fdr, method='fdr_bh')
        
        # Map back to original indices
        valid_indices = np.where(valid_features)[0]
        significant_indices = valid_indices[rej]
        
        if len(significant_indices) == 0:
            # Fallback: select top features by AUC
            idx = np.argsort(-auc_scores)[:top_k]
            idx = idx[valid_features[idx]]
        else:
            # Sort significant features by AUC
            idx = sorted(significant_indices, key=lambda j: -auc_scores[j])
            idx = idx[:top_k]
    else:
        # Select top features by AUC, but only from valid features
        idx = np.argsort(-auc_scores)[:top_k]
        idx = idx[valid_features[idx]]
    
    return list(idx)


def compute_feature_stats(A_pos: np.ndarray, A_neg: np.ndarray, method: str = "cohens_d") -> Dict:
    """
    Compute feature statistics using Cohen's d method.
    
    Args:
        A_pos: Positive class activations (N_pos x m)
        A_neg: Negative class activations (N_neg x m)
        method: Statistical method to use (only "cohens_d" supported)
    
    Returns:
        Dictionary with computed statistics
    """
    if method == "cohens_d":
        return feature_stats(A_pos, A_neg)
    else:
        raise ValueError(f"Unknown statistical method: {method}. Only 'cohens_d' is supported.")

def select_features(stats: dict, method: str = "cohens_d", top_k: int = 200, fdr: float = 0.05):
    """
    Select features using Cohen's d method.
    
    Args:
        stats: Dictionary with computed statistics
        method: Statistical method to use (only "cohens_d" supported)
        top_k: Maximum number of features to select
        fdr: False Discovery Rate threshold
    
    Returns:
        List of selected feature indices
    """
    if method == "cohens_d":
        return select_features_from_stats(stats, top_k=top_k, fdr=fdr)
    else:
        raise ValueError(f"Unknown statistical method: {method}. Only 'cohens_d' is supported.")