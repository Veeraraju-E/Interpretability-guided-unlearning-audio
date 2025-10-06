import math
import numpy as np
from typing import Dict
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
    if s_pooled == 0: 
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

def feature_stats(A_pos: np.ndarray, A_neg: np.ndarray) -> Dict:
    # Validate input arrays
    if A_pos.shape[0] == 0:
        raise ValueError("Cannot compute feature statistics: positive samples array is empty. This suggests a data filtering or sampling issue.")
    if A_neg.shape[0] == 0:
        raise ValueError("Cannot compute feature statistics: negative samples array is empty. This suggests a data filtering or sampling issue.")
    
    m = A_pos.shape[1]
    stats = {"cohens_d": np.zeros(m), "auc": np.zeros(m), "pval": np.ones(m)}
    for j in range(m):
        x_pos = A_pos[:, j]; x_neg = A_neg[:, j]
        stats["cohens_d"][j] = cohens_d(x_pos, x_neg)
        
        # ROC AUC computation - let it fail if there are issues
        try:
            y_true = np.concatenate([np.ones(len(x_pos)), np.zeros(len(x_neg))])
            y_scores = np.concatenate([x_pos, x_neg])
            stats["auc"][j] = float(roc_auc_score(y_true, y_scores))
        except Exception as e:
            raise ValueError(f"Cannot compute ROC AUC for feature {j}: {e}. This suggests data quality issues.")
            
        _, p = mann_whitney_u_p(x_pos, x_neg)
        stats["pval"][j] = p
    return stats

def select_features_from_stats(stats: dict, top_k: int = 200, fdr: float = 0.05):
    pvals = stats["pval"]
    if multipletests is not None:
        rej, _, _, _ = multipletests(pvals, alpha=fdr, method='fdr_bh')
        idx = np.where(rej)[0]
        if len(idx) == 0:
            idx = np.argsort(-np.abs(stats["auc"] - 0.5))[:top_k]
        else:
            idx = sorted(list(idx), key=lambda j: -abs(stats["auc"][j] - 0.5))
            idx = idx[:top_k]
    else:
        idx = np.argsort(-np.abs(stats["auc"] - 0.5))[:top_k]
    return list(idx)


