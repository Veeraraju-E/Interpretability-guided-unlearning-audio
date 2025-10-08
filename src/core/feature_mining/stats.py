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
    
    # Fail fast for empty arrays - this indicates a data issue
    if n1 == 0:
        raise ValueError("Cannot compute Cohen's d: positive samples array is empty. This suggests a data filtering or sampling issue.")
    if n2 == 0:
        raise ValueError("Cannot compute Cohen's d: negative samples array is empty. This suggests a data filtering or sampling issue.")
    
    m1, m2 = x_pos.mean(), x_neg.mean()
    s1, s2 = x_pos.var(ddof=1), x_neg.var(ddof=1)
    s_pooled = math.sqrt(((n1-1)*s1 + (n2-1)*s2) / max(1, n1+n2-2))
    
    # Debug: Print detailed statistics when variance is zero
    if s_pooled == 0:
        print(f"DEBUG: Cohen's d zero variance detected!")
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
    
    # Fail fast for empty arrays - this indicates a data issue
    if n1 == 0:
        raise ValueError("Cannot compute Mann-Whitney U: positive samples array is empty. This suggests a data filtering or sampling issue.")
    if n2 == 0:
        raise ValueError("Cannot compute Mann-Whitney U: negative samples array is empty. This suggests a data filtering or sampling issue.")
    
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

def feature_stats(A_pos: np.ndarray, A_neg: np.ndarray, verbose: bool = False) -> Dict:
    # Validate input arrays
    if A_pos.shape[0] == 0:
        raise ValueError("Cannot compute feature statistics: positive samples array is empty. This suggests a data filtering or sampling issue.")
    if A_neg.shape[0] == 0:
        raise ValueError("Cannot compute feature statistics: negative samples array is empty. This suggests a data filtering or sampling issue.")
    
    if verbose:
        print(f"DEBUG: feature_stats called with A_pos shape: {A_pos.shape}, A_neg shape: {A_neg.shape}")
        print(f"DEBUG: A_pos stats - mean: {A_pos.mean():.6f}, std: {A_pos.std():.6f}, min: {A_pos.min():.6f}, max: {A_pos.max():.6f}")
        print(f"DEBUG: A_neg stats - mean: {A_neg.mean():.6f}, std: {A_neg.std():.6f}, min: {A_neg.min():.6f}, max: {A_neg.max():.6f}")
    
    m = A_pos.shape[1]
    
    # Identify dead features (never activated)
    pos_activity = (A_pos > 1e-6).sum(axis=0)
    neg_activity = (A_neg > 1e-6).sum(axis=0)
    dead_features = (pos_activity == 0) & (neg_activity == 0)
    active_features = ~dead_features
    
    if verbose:
        print(f"DEBUG: Dead features (never activated): {dead_features.sum()} out of {m} total features")
        print(f"DEBUG: Active features: {active_features.sum()} out of {m} total features")
    
    stats = {"cohens_d": np.zeros(m), "auc": np.zeros(m), "pval": np.ones(m)}
    
    # Only compute statistics for active features
    for j in range(m):
        if dead_features[j]:
            # Skip dead features - they have no discriminative power
            stats["cohens_d"][j] = 0.0
            stats["auc"][j] = 0.5  # Random performance
            stats["pval"][j] = 1.0  # Non-significant
            continue
            
        x_pos = A_pos[:, j]; x_neg = A_neg[:, j]
        
        # Check for zero variance before computing Cohen's d
        pos_var = x_pos.var(ddof=1)
        neg_var = x_neg.var(ddof=1)
        if pos_var == 0 and neg_var == 0:
            if verbose:
                print(f"DEBUG: Active feature {j} has zero variance in both groups")
            stats["cohens_d"][j] = 0.0
            stats["auc"][j] = 0.5
            stats["pval"][j] = 1.0
            continue
        
        try:
            stats["cohens_d"][j] = cohens_d(x_pos, x_neg)
        except ValueError as e:
            if verbose:
                print(f"DEBUG: Cohen's d failed for feature {j}: {e}")
            stats["cohens_d"][j] = 0.0
        
        # ROC AUC computation
        try:
            y_true = np.concatenate([np.ones(len(x_pos)), np.zeros(len(x_neg))])
            y_scores = np.concatenate([x_pos, x_neg])
            stats["auc"][j] = float(roc_auc_score(y_true, y_scores))
        except Exception as e:
            if verbose:
                print(f"DEBUG: ROC AUC failed for feature {j}: {e}")
            stats["auc"][j] = 0.5
            
        try:
            _, p = mann_whitney_u_p(x_pos, x_neg)
            stats["pval"][j] = p
        except Exception as e:
            if verbose:
                print(f"DEBUG: Mann-Whitney U failed for feature {j}: {e}")
            stats["pval"][j] = 1.0
    
    return stats

def select_features_from_stats(stats: dict, top_k: int = 200, fdr: float = 0.05, verbose: bool = False):
    pvals = stats["pval"]
    
    # Filter out features with AUC = 0.5 (random performance, likely dead features)
    auc_scores = np.abs(stats["auc"] - 0.5)
    valid_features = auc_scores > 1e-6  # Features with some discriminative power
    
    console = Console()
    console.print(f"Features with discriminative power (AUC != 0.5): {valid_features.sum()} out of {len(valid_features)}", style="yellow")
    
    if valid_features.sum() == 0:
        console.print("WARNING: No features with discriminative power found!", style="red")
        return []
    
    if multipletests is not None:
        # Only apply FDR correction to valid features
        valid_pvals = pvals[valid_features]
        rej, _, _, _ = multipletests(valid_pvals, alpha=fdr, method='fdr_bh')
        
        # Map back to original indices
        valid_indices = np.where(valid_features)[0]
        significant_indices = valid_indices[rej]
        
        if len(significant_indices) == 0:
            # Fallback: select top features by AUC
            idx = np.argsort(-auc_scores)[:top_k]
            idx = idx[valid_features[idx]]  # Only keep valid features
        else:
            # Sort significant features by AUC
            idx = sorted(significant_indices, key=lambda j: -auc_scores[j])
            idx = idx[:top_k]
    else:
        # Select top features by AUC, but only from valid features
        idx = np.argsort(-auc_scores)[:top_k]
        idx = idx[valid_features[idx]]  # Only keep valid features
    
    if verbose:
        print(f"DEBUG: Selected {len(idx)} features for editing")
    return list(idx)


