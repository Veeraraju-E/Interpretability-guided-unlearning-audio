import math
import numpy as np
from typing import Dict, List, Optional
from rich.console import Console
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def cohens_d(x_pos: np.ndarray, x_neg: np.ndarray) -> float:
    """Cohen's d effect size for two independent samples."""
    n1, n2 = len(x_pos), len(x_neg)
    m1, m2 = x_pos.mean(), x_neg.mean()
    s1, s2 = x_pos.var(ddof=1), x_neg.var(ddof=1)
    s_pooled = math.sqrt(((n1-1)*s1 + (n2-1)*s2) / max(1, n1+n2-2))
    if s_pooled == 0:
        return 0.0
    return float((m1 - m2) / s_pooled)


def hedges_g(x_pos: np.ndarray, x_neg: np.ndarray) -> float:
    """Hedges' g: bias-corrected Cohen's d for small samples (Hedges, 1981)."""
    d = cohens_d(x_pos, x_neg)
    n1, n2 = len(x_pos), len(x_neg)
    df = n1 + n2 - 2
    if df <= 0:
        return d
    # Hedges correction factor J
    J = 1 - 3 / (4 * df - 1)
    return d * J


def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation (equivalent to Cliff's delta) from Mann-Whitney U."""
    # r_rb = 1 - 2U / (n1 * n2)
    denom = n1 * n2
    if denom == 0:
        return 0.0
    return 1 - 2 * u_stat / denom


def bootstrap_ci(x_pos: np.ndarray, x_neg: np.ndarray, stat_fn, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple:
    """Bootstrap confidence interval for a two-sample statistic."""
    rng = np.random.default_rng(seed)
    n1, n2 = len(x_pos), len(x_neg)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        idx1 = rng.choice(n1, size=n1, replace=True)
        idx2 = rng.choice(n2, size=n2, replace=True)
        boot_stats[i] = stat_fn(x_pos[idx1], x_neg[idx2])
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def feature_stats(A_pos: np.ndarray, A_neg: np.ndarray) -> Dict:
    """
    Compute per-latent statistics for ICLR-level reporting.
    
    Returns dict with arrays of length m (number of latents):
      - n_pos, n_neg: sample counts
      - mean_pos, mean_neg, std_pos, std_neg: group summaries
      - u_stat: Mann-Whitney U statistic
      - pval: two-sided p-value from MWU (with tie correction)
      - cohens_d, hedges_g: parametric effect sizes
      - rank_biserial: nonparametric effect size (Cliff's delta)
      - auc: ROC-AUC (equivalent to U / (n1*n2))
    """
    n_pos, n_neg = A_pos.shape[0], A_neg.shape[0]
    m = A_pos.shape[1]
    
    stats = {
        "n_pos": np.full(m, n_pos, dtype=int),
        "n_neg": np.full(m, n_neg, dtype=int),
        "mean_pos": np.zeros(m),
        "mean_neg": np.zeros(m),
        "std_pos": np.zeros(m),
        "std_neg": np.zeros(m),
        "u_stat": np.zeros(m),
        "pval": np.ones(m),
        "cohens_d": np.zeros(m),
        "hedges_g": np.zeros(m),
        "rank_biserial": np.zeros(m),
        "auc": np.full(m, 0.5),
    }
    
    # Identify dead latents (never activated in either group)
    pos_activity = (A_pos > 1e-6).sum(axis=0)
    neg_activity = (A_neg > 1e-6).sum(axis=0)
    dead = (pos_activity == 0) & (neg_activity == 0)
    
    for j in range(m):
        if dead[j]:
            continue
        x_pos, x_neg = A_pos[:, j], A_neg[:, j]
        
        stats["mean_pos"][j] = x_pos.mean()
        stats["mean_neg"][j] = x_neg.mean()
        stats["std_pos"][j] = x_pos.std(ddof=1) if n_pos > 1 else 0.0
        stats["std_neg"][j] = x_neg.std(ddof=1) if n_neg > 1 else 0.0
        
        # Skip if zero variance in both
        if stats["std_pos"][j] == 0 and stats["std_neg"][j] == 0:
            continue
        
        # Mann-Whitney U (scipy, with tie correction)
        U, p = mannwhitneyu(x_pos, x_neg, alternative='two-sided')
        stats["u_stat"][j] = U
        stats["pval"][j] = p
        
        # Effect sizes
        stats["cohens_d"][j] = cohens_d(x_pos, x_neg)
        stats["hedges_g"][j] = hedges_g(x_pos, x_neg)
        stats["rank_biserial"][j] = rank_biserial(U, n_pos, n_neg)
        
        # AUC (from sklearn for consistency; equivalent to U/(n1*n2) for positive class)
        y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        y_scores = np.concatenate([x_pos, x_neg])
        if len(np.unique(y_scores)) > 1:
            stats["auc"][j] = roc_auc_score(y_true, y_scores)
    
    return stats


def apply_fdr(stats: Dict, fdr: float = 0.05) -> Dict:
    """Apply Benjamini-Hochberg FDR correction, add qval to stats dict."""
    pvals = stats["pval"]
    _, qvals, _, _ = multipletests(pvals, alpha=fdr, method='fdr_bh')
    stats["qval"] = qvals
    stats["fdr_alpha"] = fdr
    return stats


def select_features_from_stats(stats: Dict, top_k: int = 200, fdr: float = 0.05) -> List[int]:
    """
    Select features passing FDR threshold, ranked by discriminability |AUC - 0.5|.
    Falls back to top-k by AUC if none pass FDR.
    """
    console = Console()
    
    # Ensure qval is computed
    if "qval" not in stats:
        apply_fdr(stats, fdr)
    
    auc_disc = np.abs(stats["auc"] - 0.5)
    valid = auc_disc > 1e-6
    console.print(f"Features with discriminative power (AUC != 0.5): {valid.sum()} / {len(valid)}", style="yellow")
    
    if valid.sum() == 0:
        console.print("WARNING: No features with discriminative power found!", style="red")
        return []
    
    # Significant features (q <= fdr and valid)
    sig = (stats["qval"] <= fdr) & valid
    sig_idx = np.where(sig)[0]
    
    if len(sig_idx) == 0:
        console.print(f"No features pass FDR q<={fdr}, falling back to top-{top_k} by AUC", style="yellow")
        idx = np.argsort(-auc_disc)
        idx = [i for i in idx if valid[i]][:top_k]
    else:
        # Sort significant by AUC discriminability
        idx = sorted(sig_idx, key=lambda j: -auc_disc[j])[:top_k]
    
    return list(idx)


def compute_bootstrap_cis(A_pos: np.ndarray, A_neg: np.ndarray, feat_idx: List[int],
                          n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> Dict:
    """
    Compute bootstrap 95% CIs for selected features only (for paper reporting).
    Returns dict with arrays indexed by position in feat_idx.
    """
    n = len(feat_idx)
    cis = {
        "feat_idx": np.array(feat_idx),
        "cohens_d_ci_lo": np.zeros(n),
        "cohens_d_ci_hi": np.zeros(n),
        "hedges_g_ci_lo": np.zeros(n),
        "hedges_g_ci_hi": np.zeros(n),
        "auc_ci_lo": np.zeros(n),
        "auc_ci_hi": np.zeros(n),
    }
    
    def auc_fn(xp, xn):
        y = np.concatenate([np.ones(len(xp)), np.zeros(len(xn))])
        s = np.concatenate([xp, xn])
        if len(np.unique(s)) < 2:
            return 0.5
        return roc_auc_score(y, s)
    
    for i, j in enumerate(feat_idx):
        x_pos, x_neg = A_pos[:, j], A_neg[:, j]
        cis["cohens_d_ci_lo"][i], cis["cohens_d_ci_hi"][i] = bootstrap_ci(x_pos, x_neg, cohens_d, n_boot, alpha, seed)
        cis["hedges_g_ci_lo"][i], cis["hedges_g_ci_hi"][i] = bootstrap_ci(x_pos, x_neg, hedges_g, n_boot, alpha, seed)
        cis["auc_ci_lo"][i], cis["auc_ci_hi"][i] = bootstrap_ci(x_pos, x_neg, auc_fn, n_boot, alpha, seed)
    
    return cis

def compute_feature_stats(A_pos: np.ndarray, A_neg: np.ndarray, method: str = "cohens_d") -> Dict:
    """Compute feature statistics. Returns dict with per-latent arrays."""
    return feature_stats(A_pos, A_neg)


def select_features(stats: Dict, method: str = "cohens_d", top_k: int = 200, fdr: float = 0.05) -> List[int]:
    """Select discriminative features using FDR correction and AUC ranking."""
    return select_features_from_stats(stats, top_k=top_k, fdr=fdr)