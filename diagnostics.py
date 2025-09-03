# diagnostics.py
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score
from numpy.linalg import norm, svd

def summary_stats(arr: np.ndarray) -> Dict:
    return dict(mean=float(np.mean(arr)), std=float(np.std(arr)),
                p5=float(np.percentile(arr,5)), p95=float(np.percentile(arr,95)))

# ---------- A. Whitening ----------
def check_whitening(Z_retain_dev: np.ndarray, tol_mean=5e-2, tol_cov=5e-2) -> Dict:
    mu = Z_retain_dev.mean(axis=0, keepdims=True)
    C = np.cov(Z_retain_dev - mu, rowvar=False)
    d = C.shape[0]
    off = C - np.eye(d)
    np.fill_diagonal(off, 0.0)
    out = {
        "mean_l2": float(norm(mu)),
        "mean_ok": bool(norm(mu) < tol_mean),
        "offdiag_abs_mean": float(np.mean(np.abs(off))),
        "offdiag_ok": bool(np.mean(np.abs(off)) < tol_cov),
        "evals_desc": np.sort(np.linalg.eigvalsh(C))[::-1].tolist(),
    }
    return out

# ---------- B. Decoder & Superposition ----------
def decoder_metrics(D: np.ndarray) -> Dict:
    # D: [d, m] decoder atoms (columns)
    Dn = D / (norm(D, axis=0, keepdims=True) + 1e-12)
    G = Dn.T @ Dn
    np.fill_diagonal(G, 0.0)
    mutual_coh = float(np.max(np.abs(G)))
    svals = svd(Dn, full_matrices=False)[1]
    stable_rank = float((svals**2).sum() / (svals.max()**2 + 1e-12))
    return {"mutual_coherence": mutual_coh, "stable_rank": stable_rank}

def activation_sparsity_metrics(A: np.ndarray, k_expected: Optional[int]=None) -> Dict:
    l0 = (A != 0).sum(axis=1)
    out = {"l0_summary": summary_stats(l0)}
    if k_expected is not None:
        out["topk_compliance_rate"] = float(np.mean(l0 == k_expected))
    return out

def feature_discriminability(A_forget: np.ndarray, A_retain: np.ndarray) -> Dict:
    # per-feature AUC of 'forget' vs 'retain'
    y = np.concatenate([np.ones(len(A_forget)), np.zeros(len(A_retain))])
    A_all = np.vstack([A_forget, A_retain])
    aucs = []
    for j in range(A_all.shape[1]):
        try:
            auc = roc_auc_score(y, A_all[:, j])
        except ValueError:
            auc = 0.5
        aucs.append(auc)
    return {"feature_auc": aucs,
            "feature_auc_summary": summary_stats(np.array(aucs))}

# ---------- C. Subspace U ----------
def check_U(U: np.ndarray, D_selected: Optional[np.ndarray]=None) -> Dict:
    I = U.T @ U
    ortho_err = float(norm(I - np.eye(I.shape[0])))
    out = {"U_orthonormality_err": ortho_err}
    if D_selected is not None:
        # principal angles between span(D_selected) and span(U)
        # via canonical correlations of orthonormal bases
        Ud, _s, _vt = svd(D_selected, full_matrices=False)
        Uu = U
        C = Ud.T @ Uu
        s = svd(C, full_matrices=False)[1]
        # angles = arccos(s); we report cosines (closer to 1 is better)
        out["principal_cosines"] = s.tolist()
    return out

def energy_share_in_U(Z: np.ndarray, U: np.ndarray) -> float:
    ZU = Z @ U
    return float((ZU**2).sum() / max((Z**2).sum(), 1e-12))

# ---------- D. LEACE identities in U ----------
def leace_invariants(Z_all: np.ndarray, sets: List[str], U: np.ndarray,
                     Y_dev: np.ndarray, P_r: np.ndarray, ZU_mean: np.ndarray,
                     split_suffix=":dev") -> Dict:
    mask_dev = np.array([s.endswith(split_suffix) for s in sets], bool)
    Z_dev = Z_all[mask_dev]
    ZU = Z_dev @ U
    # center
    ZUc = ZU - ZU_mean
    Yc = Y_dev - Y_dev.mean(0, keepdims=True)
    # empirical cross-cov before
    CZY = (ZUc.T @ Yc) / max(len(ZUc)-1,1)
    # edit once
    ZU_edit = (ZUc @ P_r.T) + ZU_mean
    ZUc2 = ZU_edit - ZU_mean
    CZY_after = (ZUc2.T @ Yc) / max(len(ZUc2)-1,1)
    # idempotence
    ZU_edit2 = ((ZU_edit - ZU_mean) @ P_r.T) + ZU_mean
    idem_err = float(norm(ZU_edit2 - ZU_edit) / (norm(ZU_edit)+1e-12))
    return {
        "cross_cov_before_norm": float(norm(CZY)),
        "cross_cov_after_norm": float(norm(CZY_after)),
        "cross_cov_suppression_ratio": float((norm(CZY_after)+1e-12)/(norm(CZY)+1e-12)),
        "idempotence_rel_err": idem_err,
        "mean_preserved": float(norm(ZU_edit.mean(0) - ZU_mean))  # ~0
    }

# ---------- E. Probe-lens (predictivity through stages) ----------
def probe_lens_auc(stages: Dict[str, np.ndarray], y_bin: np.ndarray) -> Dict:
    # stages: name -> features (N x d)
    out = {}
    for name, X in stages.items():
        # simple L2-regularized logistic is better, but AUC on linear score works as quick lens
        # score with least-squares solution
        Xc = X - X.mean(0, keepdims=True)
        w, *_ = np.linalg.lstsq(Xc, y_bin - y_bin.mean(), rcond=None)
        s = Xc @ w
        try:
            auc = roc_auc_score(y_bin, s)
        except ValueError:
            auc = 0.5
        out[name] = float(auc)
    return out

# ---------- F. CKA (linear, centered) ----------
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    num = norm(Xc.T @ Yc, 'fro')**2
    den = norm(Xc.T @ Xc, 'fro') * norm(Yc.T @ Yc, 'fro') + 1e-12
    return float(num / den)

def cka_matrix(reps: Dict[str, np.ndarray]) -> Dict[Tuple[str,str], float]:
    keys = list(reps.keys())
    out = {}
    for i in range(len(keys)):
        for j in range(i, len(keys)):
            out[(keys[i], keys[j])] = linear_cka(reps[keys[i]], reps[keys[j]])
    return out

# ---------- G. Polysemanticity ----------
def polysemanticity_index(A: np.ndarray, labels: List[str], top_q=0.1) -> Dict:
    # For each atom, how many classes occupy top-q fraction of its total activation mass?
    labels_u = np.array(labels)
    classes = np.unique(labels_u)
    m = A.shape[1]
    counts = []
    for j in range(m):
        totals = []
        col = A[:, j]
        for c in classes:
            totals.append(col[labels_u==c].sum())
        totals = np.array(totals)
        pos = np.abs(totals)
        order = np.argsort(pos)[::-1]
        cum = np.cumsum(pos[order])
        need = pos.sum() * top_q
        k = int(np.searchsorted(cum, need) + 1)
        counts.append(k)
    return {"polysemanticity_counts": counts,
            "summary": summary_stats(np.array(counts))}
