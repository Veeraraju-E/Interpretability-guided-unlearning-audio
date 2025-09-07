from __future__ import annotations
import numpy as np, json, os, time
from numpy.linalg import svd, norm
from scipy.stats import entropy
from common import sae_encode_batch, invert_whitener, l2_normalize
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import numpy as np
import pandas as pd

plt.style.use('default')
sns.set_palette("husl")

def save_fig(save_dir: str, filename: str, dpi: int = 300, bbox_inches: str = 'tight'):
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=dpi, bbox_inches=bbox_inches)
    plt.close()

def get_binary_labels(labels: List[str], forget_classes: List[str]) -> List[str]:
    return ['forget' if label in forget_classes else 'retain' for label in labels]

def plot_tsne_comparison(Z_before: np.ndarray, Z_after: np.ndarray, labels: List[str], forget_classes: List[str], save_dir: str, n_components: int = 2, perplexity: int = 30, random_state: int = 42):
    
    Z_combined = np.vstack([Z_before, Z_after])
    n_samples = Z_combined.shape[0]
    perplexity_eff = max(5, min(perplexity, max(2, (n_samples - 1) // 3)))
    X = np.ascontiguousarray(Z_combined, dtype=np.float64)
    tsne = TSNE(n_components=n_components, perplexity=perplexity_eff, random_state=random_state, init='random')
    Z_tsne_combined = tsne.fit_transform(X)
    
    n_before = Z_before.shape[0]
    Z_tsne_before = Z_tsne_combined[:n_before]
    Z_tsne_after = Z_tsne_combined[n_before:]
    
    binary_labels = get_binary_labels(labels, forget_classes)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for bin_label in ['retain', 'forget']:
        mask = np.array(binary_labels) == bin_label
        color = '#4ecdc4' if bin_label == 'retain' else '#ff6b6b'
        ax1.scatter(Z_tsne_before[mask, 0], Z_tsne_before[mask, 1], c=color, label=bin_label, alpha=0.7, s=20)
    
    ax1.set_title('Before Edit', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for bin_label in ['retain', 'forget']:
        mask = np.array(binary_labels) == bin_label
        color = '#4ecdc4' if bin_label == 'retain' else '#ff6b6b'
        ax2.scatter(Z_tsne_after[mask, 0], Z_tsne_after[mask, 1], c=color, label=bin_label, alpha=0.7, s=20)
    
    ax2.set_title('After Edit', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig(save_dir, 'tsne_comparison.png')
    
    return Z_tsne_before, Z_tsne_after

def plot_umap_comparison(Z_before: np.ndarray, Z_after: np.ndarray, labels: List[str], forget_classes: List[str], save_dir: str, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42):
    
    Z_combined = np.vstack([Z_before, Z_after])
    n_samples = Z_combined.shape[0]
    n_neighbors_eff = max(2, min(n_neighbors, max(2, n_samples - 1)))
    try:
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors_eff, min_dist=min_dist, random_state=random_state)
        Z_umap_combined = reducer.fit_transform(Z_combined)
    except Exception:
        pca = PCA(n_components=min(n_components, max(1, Z_combined.shape[1])))
        Z_umap_combined = pca.fit_transform(Z_combined)
    
    n_before = Z_before.shape[0]
    Z_umap_before = Z_umap_combined[:n_before]
    Z_umap_after = Z_umap_combined[n_before:]
    
    binary_labels = get_binary_labels(labels, forget_classes)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for bin_label in ['retain', 'forget']:
        mask = np.array(binary_labels) == bin_label
        color = '#4ecdc4' if bin_label == 'retain' else '#ff6b6b'
        ax1.scatter(Z_umap_before[mask, 0], Z_umap_before[mask, 1],  c=color, label=bin_label, alpha=0.7, s=20)
    
    ax1.set_title('Before Edit', fontsize=14, fontweight='bold')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for bin_label in ['retain', 'forget']:
        mask = np.array(binary_labels) == bin_label
        color = '#4ecdc4' if bin_label == 'retain' else '#ff6b6b'
        ax2.scatter(Z_umap_after[mask, 0], Z_umap_after[mask, 1], c=color, label=bin_label, alpha=0.7, s=20)
    
    ax2.set_title('After Edit', fontsize=14, fontweight='bold')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig(save_dir, 'umap_comparison.png')
    
    return Z_umap_before, Z_umap_after

def plot_subspace_scatter(ZU_before: np.ndarray, ZU_after: np.ndarray, labels: List[str], forget_classes: List[str], save_dir: str, n_components: int = 2):
    
    ZU_combined = np.vstack([ZU_before, ZU_after])
    pca = PCA(n_components=min(n_components, ZU_combined.shape[1]))
    ZU_pca_combined = pca.fit_transform(ZU_combined)
    
    n_before = ZU_before.shape[0]
    ZU_pca_before = ZU_pca_combined[:n_before]
    ZU_pca_after = ZU_pca_combined[n_before:]
    
    binary_labels = get_binary_labels(labels, forget_classes)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for bin_label in ['retain', 'forget']:
        mask = np.array(binary_labels) == bin_label
        color = '#4ecdc4' if bin_label == 'retain' else '#ff6b6b'
        ax1.scatter(ZU_pca_before[mask, 0], ZU_pca_before[mask, 1], c=color, label=bin_label, alpha=0.7, s=20)
    
    ax1.set_title('Before Edit (U subspace)', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for bin_label in ['retain', 'forget']:
        mask = np.array(binary_labels) == bin_label
        color = '#4ecdc4' if bin_label == 'retain' else '#ff6b6b'
        ax2.scatter(ZU_pca_after[mask, 0], ZU_pca_after[mask, 1], c=color, label=bin_label, alpha=0.7, s=20)
    
    ax2.set_title('After Edit (U subspace)', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig(save_dir, 'subspace_scatter.png')
    
    return ZU_pca_before, ZU_pca_after

def compute_intra_inter_similarities(Z_before: np.ndarray, Z_after: np.ndarray, labels: List[str], forget_classes: List[str]):
    from itertools import combinations
    
    binary_labels = get_binary_labels(labels, forget_classes)
    
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    
    intra_sim_before = {'retain': [], 'forget': []}
    intra_sim_after = {'retain': [], 'forget': []}
    
    for class_type in ['retain', 'forget']:
        indices = [i for i, label in enumerate(binary_labels) if label == class_type]
        if len(indices) >= 2:
            for i, j in combinations(indices, 2):
                intra_sim_before[class_type].append(cosine_sim(Z_before[i], Z_before[j]))
                intra_sim_after[class_type].append(cosine_sim(Z_after[i], Z_after[j]))
    
    retain_indices = [i for i, label in enumerate(binary_labels) if label == 'retain']
    forget_indices = [i for i, label in enumerate(binary_labels) if label == 'forget']
    
    inter_sim_before = []
    inter_sim_after = []
    
    for r_idx in retain_indices:
        for f_idx in forget_indices:
            inter_sim_before.append(cosine_sim(Z_before[r_idx], Z_before[f_idx]))
            inter_sim_after.append(cosine_sim(Z_after[r_idx], Z_after[f_idx]))
    
    return (intra_sim_before, intra_sim_after, 
            {'inter': inter_sim_before}, {'inter': inter_sim_after})

def plot_projection_distributions(ZU_before: np.ndarray, ZU_after: np.ndarray, labels: List[str], forget_classes: List[str], save_dir: str, top_directions: int = 5):
    
    var_before = np.var(ZU_before, axis=0)
    top_indices = np.argsort(var_before)[-top_directions:]
    
    binary_labels = get_binary_labels(labels, forget_classes)
    
    fig, axes = plt.subplots(2, top_directions, figsize=(4*top_directions, 8))
    if top_directions == 1:
        axes = axes.reshape(2, 1)
    
    for i, idx in enumerate(top_indices):
        ax = axes[0, i]
        for bin_label in ['retain', 'forget']:
            mask = np.array(binary_labels) == bin_label
            color = '#4ecdc4' if bin_label == 'retain' else '#ff6b6b'
            ax.hist(ZU_before[mask, idx], alpha=0.7, bins=20, color=color, label=bin_label, density=True)
        ax.set_title(f'Direction {idx+1} (Before)')
        ax.set_xlabel('Projection')
        ax.set_ylabel('Density')
        if i == 0:
            ax.legend()
        
        ax = axes[1, i]
        for bin_label in ['retain', 'forget']:
            mask = np.array(binary_labels) == bin_label
            color = '#4ecdc4' if bin_label == 'retain' else '#ff6b6b'
            ax.hist(ZU_after[mask, idx], alpha=0.7, bins=20, color=color,  label=bin_label, density=True)
        ax.set_title(f'Direction {idx+1} (After)')
        ax.set_xlabel('Projection')
        ax.set_ylabel('Density')
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    save_fig(save_dir, 'projection_distributions.png')

def plot_similarity_distributions(intra_sim_before: Dict, intra_sim_after: Dict, inter_sim_before: Dict, inter_sim_after: Dict, save_dir: str):
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Intra-class similarities - retain
    ax = axes[0, 0]
    if 'retain' in intra_sim_before and len(intra_sim_before['retain']) > 0:
        ax.hist(intra_sim_before['retain'], alpha=0.7, bins=20, color='#4ecdc4',  label='Before', density=True)
    if 'retain' in intra_sim_after and len(intra_sim_after['retain']) > 0:
        ax.hist(intra_sim_after['retain'], alpha=0.7, bins=20, color='#45b7d1', label='After', density=True)
    ax.set_title('Intra-class Similarities - Retain Classes')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.legend()
    
    # Intra-class similarities - forget
    ax = axes[0, 1]
    if 'forget' in intra_sim_before and len(intra_sim_before['forget']) > 0:
        ax.hist(intra_sim_before['forget'], alpha=0.7, bins=20, color='#ff6b6b', label='Before', density=True)
    if 'forget' in intra_sim_after and len(intra_sim_after['forget']) > 0:
        ax.hist(intra_sim_after['forget'], alpha=0.7, bins=20, color='#ff8e8e', label='After', density=True)
    ax.set_title('Intra-class Similarities - Forget Classes')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.legend()
    
    # Inter-class similarities
    ax = axes[1, 0]
    if 'inter' in inter_sim_before and len(inter_sim_before['inter']) > 0:
        ax.hist(inter_sim_before['inter'], alpha=0.7, bins=20, color='#96ceb4', label='Before', density=True)
    if 'inter' in inter_sim_after and len(inter_sim_after['inter']) > 0:
        ax.hist(inter_sim_after['inter'], alpha=0.7, bins=20, color='#feca57', label='After', density=True)
    ax.set_title('Inter-class Similarities (Retain vs Forget)')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.legend()
    
    # Remove the empty subplot
    ax = axes[1, 1]
    ax.axis('off')
    
    plt.tight_layout()
    save_fig(save_dir, 'similarity_distributions.png')

def plot_sae_activations(A_retain: np.ndarray, A_forget: np.ndarray, save_dir: str):
    
    avg_retain = np.mean(A_retain, axis=0)
    avg_forget = np.mean(A_forget, axis=0)
    
    plt.figure(figsize=(15, 8))
    
    neurons = np.arange(len(avg_retain))
    plt.plot(neurons, avg_retain, color='#4ecdc4', alpha=0.7, label='Retain Average', linewidth=1.5)
    plt.plot(neurons, avg_forget, color='#ff6b6b', alpha=0.7, label='Forget Average', linewidth=1.5)
    
    plt.xlabel('Neuron Index')
    plt.ylabel('Average Activation')
    plt.title('SAE Latent Space Activations per Neuron')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if len(avg_retain) > 50:
        ax_inset = plt.axes([0.55, 0.55, 0.35, 0.35])
        top_k = 50
        top_indices = np.argsort(avg_retain + avg_forget)[-top_k:]
        ax_inset.plot(top_indices, avg_retain[top_indices], 'o-', color='#4ecdc4', markersize=3, linewidth=1, label='Retain')
        ax_inset.plot(top_indices, avg_forget[top_indices], 'o-', color='#ff6b6b', markersize=3, linewidth=1, label='Forget')
        ax_inset.set_title(f'Top {top_k} Active Neurons')
        ax_inset.grid(True, alpha=0.3)
        ax_inset.legend()
    
    plt.tight_layout()
    save_fig(save_dir, 'sae_activations.png')

def create_all_plots(Z_before: np.ndarray, Z_after: np.ndarray,
                     ZU_before: np.ndarray, ZU_after: np.ndarray,
                     labels: List[str], forget_classes: List[str],
                     decoder_weights: np.ndarray, feat_idx: List[int],
                     A_retain: np.ndarray, A_forget: np.ndarray,
                     save_dir: str, random_state: int = 42):
    
    print(f"Creating diagnostic plots in {save_dir}...")
    
    print("  - Creating t-SNE plots...")
    plot_tsne_comparison(Z_before, Z_after, labels, forget_classes, save_dir, random_state=random_state)
    
    print("  - Creating UMAP plots...")
    plot_umap_comparison(Z_before, Z_after, labels, forget_classes, save_dir, random_state=random_state)
    
    print("  - Creating subspace scatter plots...")
    plot_subspace_scatter(ZU_before, ZU_after, labels, forget_classes, save_dir)
    
    print("  - Creating projection distributions plots...")
    plot_projection_distributions(ZU_before, ZU_after, labels, forget_classes, save_dir)
    
    print("  - Computing similarity distributions...")
    intra_sim_before, intra_sim_after, inter_sim_before, inter_sim_after = compute_intra_inter_similarities(Z_before, Z_after, labels, forget_classes)
    
    print("  - Creating similarity distributions plots...")
    plot_similarity_distributions(intra_sim_before, intra_sim_after, inter_sim_before, inter_sim_after, save_dir)
    
    print("  - Creating SAE activations plots...")
    plot_sae_activations(A_retain, A_forget, save_dir)
    
    print("All diagnostic plots created successfully!")

def plot_probe_lens(probe_auc: dict, save_dir: str, filename="probe_lens.png"):
    names = list(probe_auc.keys())
    vals = [probe_auc[k] for k in names]
    plt.figure(figsize=(8,4))
    plt.plot(range(len(names)), vals, marker='o')
    plt.xticks(range(len(names)), names, rotation=30, ha='right')
    plt.ylabel('AUC (forget vs retain)')
    plt.title('Probe-Lens Predictivity Across Stages')
    plt.grid(True, alpha=0.3)
    save_fig(save_dir, filename)

def plot_principal_cosines(cosines: list, save_dir: str, filename="principal_cosines.png"):
    plt.figure(figsize=(6,4))
    plt.plot(sorted(cosines, reverse=True), marker='o')
    plt.xlabel('Index (sorted)')
    plt.ylabel('cos(θ)')
    plt.title('Principal Cosines: span(D_F) vs span(U)')
    plt.grid(True, alpha=0.3)
    save_fig(save_dir, filename)

def plot_cka_heatmap(cka_dict: dict, save_dir: str, filename="cka_heatmap.png"):
    keys = sorted(set(k for pair in cka_dict for k in pair))
    M = np.zeros((len(keys), len(keys)))
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            M[i, j] = cka_dict[(a, b)] if (a,b) in cka_dict else cka_dict[(b,a)]
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(M)
    ax.set_xticks(range(len(keys))); ax.set_yticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=30, ha='right'); ax.set_yticklabels(keys)
    plt.colorbar(im, ax=ax, label='Linear CKA')
    plt.title('Representation Similarity (CKA)')
    save_fig(save_dir, filename)

def plot_polysemanticity(counts: list, save_dir: str, filename="polysemanticity.png"):
    plt.figure(figsize=(6,4))
    plt.hist(counts, bins=20, density=True)
    plt.xlabel('#classes to reach top-q mass')
    plt.ylabel('Density')
    plt.title('Polysemanticity Index per Atom')
    plt.grid(True, alpha=0.3)
    save_fig(save_dir, filename)

def summary_stats(arr: np.ndarray) -> Dict:
    return dict(mean=float(np.mean(arr)), std=float(np.std(arr)),
                p5=float(np.percentile(arr,5)), p95=float(np.percentile(arr,95)))

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

def decoder_metrics(D: np.ndarray) -> Dict:
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

def check_U(U: np.ndarray, D_selected: Optional[np.ndarray]=None) -> Dict:
    I = U.T @ U
    ortho_err = float(norm(I - np.eye(I.shape[0])))
    out = {"U_orthonormality_err": ortho_err}
    if D_selected is not None:
        Ud, _s, _vt = svd(D_selected, full_matrices=False)
        Uu = U
        C = Ud.T @ Uu
        s = svd(C, full_matrices=False)[1]
        out["principal_cosines"] = s.tolist()
    return out

def energy_share_in_U(Z: np.ndarray, U: np.ndarray) -> float:
    ZU = Z @ U
    return float((ZU**2).sum() / max((Z**2).sum(), 1e-12))

def leace_invariants(Z_all: np.ndarray, sets: List[str], U: np.ndarray, Y_dev: np.ndarray, P_r: np.ndarray, ZU_mean: np.ndarray, split_suffix=":dev") -> Dict:
    mask_dev = np.array([s.endswith(split_suffix) for s in sets], bool)
    Z_dev = Z_all[mask_dev]
    ZU = Z_dev @ U
    ZUc = ZU - ZU_mean
    Yc = Y_dev - Y_dev.mean(0, keepdims=True)
    CZY = (ZUc.T @ Yc) / max(len(ZUc)-1,1)
    ZU_edit = (ZUc @ P_r.T) + ZU_mean
    ZUc2 = ZU_edit - ZU_mean
    CZY_after = (ZUc2.T @ Yc) / max(len(ZUc2)-1,1)
    ZU_edit2 = ((ZU_edit - ZU_mean) @ P_r.T) + ZU_mean
    idem_err = float(norm(ZU_edit2 - ZU_edit) / (norm(ZU_edit)+1e-12))
    return {
        "cross_cov_before_norm": float(norm(CZY)),
        "cross_cov_after_norm": float(norm(CZY_after)),
        "cross_cov_suppression_ratio": float((norm(CZY_after)+1e-12)/(norm(CZY)+1e-12)),
        "idempotence_rel_err": idem_err,
        "mean_preserved": float(norm(ZU_edit.mean(0) - ZU_mean))
    }

def probe_lens_auc(stages: Dict[str, np.ndarray], y_bin: np.ndarray) -> Dict:
    out = {}
    for name, X in stages.items():
        Xc = X - X.mean(0, keepdims=True)
        w, *_ = np.linalg.lstsq(Xc, y_bin - y_bin.mean(), rcond=None)
        s = Xc @ w
        try:
            auc = roc_auc_score(y_bin, s)
        except ValueError:
            auc = 0.5
        out[name] = float(auc)
    return out

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

def polysemanticity_index(A: np.ndarray, labels: List[str], top_q=0.1) -> Dict:
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

def check_projector_properties(P_r, tol=1e-6):
    idemp_err = norm(P_r @ P_r - P_r) / (norm(P_r) + 1e-12)
    sym_err = norm(P_r - P_r.T) / (norm(P_r) + 1e-12)
    eigs = np.linalg.eigvalsh(P_r) if P_r.size else np.array([])
    return {"idempotence_err": float(idemp_err), "symmetry_err": float(sym_err),
            "eigs_min": float(eigs.min()) if eigs.size else None,
            "eigs_max": float(eigs.max()) if eigs.size else None}

def atom_class_entropy(A_dev, labels_dev):
    labels_u = np.array(labels_dev)
    classes = np.unique(labels_u)
    m = A_dev.shape[1]
    entropies = np.zeros(m)
    for j in range(m):
        totals = np.array([np.abs(A_dev[labels_u==c, j]).sum() for c in classes])
        p = totals / (totals.sum() + 1e-12)
        entropies[j] = entropy(p, base=2)
    return entropies

def run_validation_checks(save_dir, Z_all, Z_edit, E_all_norm, E_edit_norm,
                          ZU_mean, U, P_r, mu_r, dec_w, sae,
                          feat_idx, sets, labels, mask_dev, mask_test,
                          y_dev_bin, leace_ridge, mu, W, device='cpu'):
    os.makedirs(save_dir, exist_ok=True)
    out = {}
    t0 = time.time()

    # A) Whitening
    try:
        Z_ret_dev = Z_all[np.array([s == 'retain:dev' for s in sets])]
        whiten_diag = check_whitening(Z_ret_dev)
        out['whiten_diag'] = whiten_diag
    except Exception as e:
        out['whiten_diag_error'] = str(e)

    # B) SAE/decoder diagnostics
    dec_d = decoder_metrics(dec_w)
    out['decoder_metrics'] = dec_d
    try:
        mask_dev_bool = np.array([s.endswith(':dev') for s in sets])
        A_dev = sae_encode_batch(sae, Z_all[mask_dev_bool], device=device, batch=8192)
        spars = activation_sparsity_metrics(A_dev, k_expected=getattr(sae,'k',None))
        out['sparsity'] = spars
    except Exception as e:
        out['sparsity_error'] = str(e)

    # C) Feature discriminability
    try:
        A_ret = sae_encode_batch(sae, Z_all[np.array([s == 'retain:dev' for s in sets])], device=device, batch=8192)
        A_fgt = sae_encode_batch(sae, Z_all[np.array([s == 'forget:dev' for s in sets])], device=device, batch=8192)
        feat_disc = feature_discriminability(A_fgt, A_ret)
        out['feature_discriminability_summary'] = feat_disc.get('feature_auc_summary', {})
        feat_aucs = np.array(feat_disc['feature_auc'])
        if len(feat_aucs) >= max(feat_idx)+1 if feat_idx else 0:
            out['selected_feat_aucs'] = feat_aucs[feat_idx].tolist() if feat_idx else []
    except Exception as e:
        out['feature_disc_error'] = str(e)

    # D) U checks
    try:
        D_selected = dec_w[:, feat_idx] if len(feat_idx) > 0 else dec_w[:,:0]
        diagU = check_U(U, D_selected if D_selected.shape[1] > 0 else None)
        out['U_diag'] = diagU
        Eshare = energy_share_in_U(Z_all, U)
        out['energy_share_in_U'] = float(Eshare)
    except Exception as e:
        out['U_diag_error'] = str(e)

    # E) LEACE invariants
    try:
        leace_diag = leace_invariants(Z_all, sets, U, y_dev_bin, P_r, ZU_mean)
        out['leace_diag'] = leace_diag
        proj_props = check_projector_properties(P_r)
        out['P_r_props'] = proj_props
    except Exception as e:
        out['leace_diag_error'] = str(e)

    # F) Probe lens
    try:
        stages = {
            "E": invert_whitener(Z_all[mask_dev], mu, W),
            "Z": Z_all[mask_dev],
            "ZU": Z_all[mask_dev] @ U,
            "ZU_center": (Z_all[mask_dev] @ U) - ZU_mean,
            "ZU_edit": ((Z_all[mask_dev] @ U) - ZU_mean) @ P_r.T + ZU_mean,
        }
        lens = probe_lens_auc(stages, y_dev_bin)
        out['probe_lens'] = lens
        plot_probe_lens(lens, save_dir)
    except Exception as e:
        out['probe_lens_error'] = str(e)

    # G) CKA
    try:
        Z_dev = Z_all[mask_dev]; Z_edit_dev = Z_edit[mask_dev]
        reps = {
            "Z(dev)": Z_dev,
            "Z_edit(dev)": Z_edit_dev,
            "ZU(dev)": Z_dev @ U,
            "ZU_edit(dev)": Z_edit_dev @ U,
        }
        cka = cka_matrix(reps)
        out['cka'] = {f"{a}|{b}": v for (a,b), v in cka.items()}
        plot_cka_heatmap(cka, save_dir)
    except Exception as e:
        out['cka_error'] = str(e)

    # H) Polysemanticity
    try:
        A_dev_full = sae_encode_batch(sae, Z_all[mask_dev], device=device, batch=8192)
        binary_labels = get_binary_labels([labels[i] for i in np.where(mask_dev)[0]], 
                                        [l for l in set(labels) if 'forget' in l])
        poly = polysemanticity_index(A_dev_full, binary_labels, top_q=0.1)
        out['polysemanticity'] = poly
        ents = atom_class_entropy(A_dev_full, binary_labels)
        out['atom_entropies_summary'] = {"mean": float(np.mean(ents)), "std": float(np.std(ents))}
        plot_polysemanticity(list(ents), save_dir)
    except Exception as e:
        out['polysemanticity_error'] = str(e)

    print(f"Validation checks completed in {time.time() - t0:.2f}s")
    return out

# --- Interpretability helpers (copied from analysis/validation_checks.py) ---
def build_feature_cards(A_all, label_binary, dec_w, feat_idx, metadata, save_dir):
    import numpy as np
    rows = []
    X = A_all
    y = np.array(label_binary)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.feature_selection import mutual_info_classif
    except Exception:
        LogisticRegression = None
        roc_auc_score = None
        mutual_info_classif = None
    for f in feat_idx:
        x_f = X[:, f]
        usage_rate = float((x_f > 0).mean())
        auc = float('nan')
        mi = float('nan')
        try:
            if LogisticRegression is not None and roc_auc_score is not None and mutual_info_classif is not None:
                if len(np.unique(y)) > 1:
                    clf = LogisticRegression(max_iter=500)
                    clf.fit(x_f.reshape(-1, 1), y)
                    if hasattr(clf, 'predict_proba'):
                        probs = clf.predict_proba(x_f.reshape(-1, 1))[:, 1]
                    else:
                        probs = clf.decision_function(x_f.reshape(-1, 1))
                    auc = float(roc_auc_score(y, probs))
                    mi = float(mutual_info_classif(x_f.reshape(-1, 1), y, discrete_features=False)[0])
        except Exception:
            pass
        top_idx = np.argsort(-x_f)[:24]
        prototypes = [int(i) for i in top_idx.tolist()]
        try:
            if dec_w.ndim == 2 and f < dec_w.shape[0]:
                atom = dec_w[f]
            else:
                atom = np.zeros(X.shape[1])
        except Exception:
            atom = np.zeros(X.shape[1])
        pos_mean = X[y == 1].mean(axis=0) if np.any(y == 1) else np.zeros(X.shape[1])
        neg_mean = X[y == 0].mean(axis=0) if np.any(y == 0) else np.zeros(X.shape[1])
        diff_vec = pos_mean - neg_mean
        cos_dir = float(np.dot(atom, diff_vec) / (np.linalg.norm(atom) * np.linalg.norm(diff_vec) + 1e-12))
        rows.append({
            'feature': int(f),
            'usage_rate': usage_rate,
            'auc_for_forget': auc,
            'mutual_info_for_forget': mi,
            'cos_with_diff': cos_dir,
            'top_prototypes': prototypes
        })
    import pandas as pd
    def save_csv(rows, path, header):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(rows, columns=header).to_csv(path, index=False)
    def save_json(obj, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    save_csv([[r['feature'], r['usage_rate'], r['auc_for_forget'], r['mutual_info_for_forget'], r['cos_with_diff'], r['top_prototypes']] for r in rows],
             os.path.join(save_dir, 'feature_cards.csv'),
             header=['feature', 'usage_rate', 'auc_for_forget', 'mi_for_forget', 'cos_with_diff', 'top_prototypes'])
    save_json(rows, os.path.join(save_dir, 'feature_cards.json'))

def polysemanticity_checks(A_all, labels, feat_idx, save_dir):
    import numpy as np
    rows = []
    labels = np.array(labels)
    for f in feat_idx:
        x_f = A_all[:, f]
        ent = 0.0
        for c in np.unique(labels):
            mask = labels == c
            if mask.sum() == 0:
                continue
            p = float((x_f[mask] > 0).mean())
            if p > 0 and p < 1:
                ent += -(p * np.log(p) + (1 - p) * np.log(1 - p))
        rows.append([int(f), float(ent)])
    import pandas as pd
    def save_csv(rows, path, header):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(rows, columns=header).to_csv(path, index=False)
    save_csv(rows, os.path.join(save_dir, 'polysemanticity.csv'), header=['feature', 'entropy'])

def causal_tests(sae, Z_all, mu, W, trials, labels, feat_idx, save_dir, device, max_feats=50):
    import numpy as np
    rows = []
    A_all = sae_encode_batch(sae, Z_all, device=device, batch=8192)
    dec_w = sae.dec.weight.detach().cpu().numpy()
    sel_feats = list(feat_idx)[:min(len(feat_idx), max_feats)]
    for f in sel_feats:
        A_ko = A_all.copy()
        A_ko[:, f] = 0.0
        try:
            Z_ko = A_ko @ dec_w
        except Exception:
            Z_ko = A_ko @ dec_w.T
        E_ko = invert_whitener(Z_ko, mu, W)
        E_ko_norm = l2_normalize(E_ko)
        from common import compute_scores, eer_from_scores
        scores_ko = compute_scores(E_ko_norm, trials)
        eer_ko = eer_from_scores(scores_ko, labels)
        delta = 0.1
        A_ki = A_all.copy()
        A_ki[:, f] += delta
        try:
            Z_ki = A_ki @ dec_w
        except Exception:
            Z_ki = A_ki @ dec_w.T
        E_ki = invert_whitener(Z_ki, mu, W)
        E_ki_norm = l2_normalize(E_ki)
        scores_ki = compute_scores(E_ki_norm, trials)
        eer_ki = eer_from_scores(scores_ki, labels)
        rows.append([int(f), float(eer_ko), float(eer_ki)])
    save_csv(rows, os.path.join(save_dir, 'causal_effects.csv'), header=['feature', 'eer_knockout', 'eer_knockin'])

def save_csv(rows, path, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows, columns=header).to_csv(path, index=False)

def activation_patching(sae, Z_all, mu, W, trials, labels, forget_indices, retain_indices, save_dir, device, topm=8):
    import numpy as np
    rows = []
    A_all = sae_encode_batch(sae, Z_all, device=device, batch=8192)
    dec_w = sae.dec.weight.detach().cpu().numpy()
    for i in range(min(10, len(forget_indices))):
        for j in range(min(10, len(retain_indices))):
            f_idx = forget_indices[i]
            r_idx = retain_indices[j]
            top_feats = np.argsort(-A_all[f_idx])[:topm]
            A_patched = A_all.copy()
            A_patched[r_idx, top_feats] = A_all[f_idx, top_feats]
            try:
                Z_patched = A_patched @ dec_w
            except Exception:
                Z_patched = A_patched @ dec_w.T
            E_patched = invert_whitener(Z_patched, mu, W)
            E_patched_norm = l2_normalize(E_patched)
            from common import compute_scores, eer_from_scores
            scores_p = compute_scores(E_patched_norm, trials)
            eer_p = eer_from_scores(scores_p, labels)
            rows.append([int(r_idx), int(f_idx), float(eer_p)])
    import pandas as pd
    def save_csv(rows, path, header):
        os.makedirs(os.makedirs(os.path.dirname(path), exist_ok=True))
        pd.DataFrame(rows, columns=header).to_csv(path, index=False)
    save_csv(rows, os.path.join(save_dir, 'patching_results.csv'), header=['retain_idx', 'forget_idx', 'eer_after_patch'])

def top_neurons_analysis(A_all, labels, manifest, save_dir, label_field_name: str = 'label', top_neurons=32, top_samples_per_neuron=20, extra_manifest_keys: Optional[List[str]] = None):

    print(f"Analyzing top {top_neurons} neurons and top {top_samples_per_neuron} samples per neuron...")
    mean_activations = A_all.mean(axis=0)
    top_neuron_indices = np.argsort(-mean_activations)[:top_neurons]
    neuron_analysis = []
    if extra_manifest_keys is None:
        extra_manifest_keys = []
    for neuron_idx in top_neuron_indices:
        neuron_activations = A_all[:, neuron_idx]
        top_sample_indices = np.argsort(-neuron_activations)[:top_samples_per_neuron]
        top_samples_info = []
        for sample_idx in top_sample_indices:
            info = {
                'sample_index': int(sample_idx),
                'activation_value': float(neuron_activations[sample_idx]),
                'label': labels[sample_idx]
            }
            if label_field_name and label_field_name != 'label':
                info[label_field_name] = labels[sample_idx]
            for k in ['set', 'split', 'path'] + list(extra_manifest_keys):
                if isinstance(manifest[sample_idx], dict) and k in manifest[sample_idx]:
                    info[k] = manifest[sample_idx][k]
            top_samples_info.append(info)
        neuron_info = {
            'neuron_index': int(neuron_idx),
            'mean_activation': float(mean_activations[neuron_idx]),
            'max_activation': float(neuron_activations.max()),
            'usage_rate': float((neuron_activations > 0).mean()),
            'top_samples': top_samples_info
        }
        neuron_analysis.append(neuron_info)
    def save_json(obj, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    save_json(neuron_analysis, os.path.join(save_dir, 'top_neurons_analysis.json'))
    neuron_summary_rows = []
    for neuron_info in neuron_analysis:
        neuron_summary_rows.append([
            neuron_info['neuron_index'],
            neuron_info['mean_activation'],
            neuron_info['max_activation'],
            neuron_info['usage_rate'],
            len(neuron_info['top_samples'])
        ])
    def save_csv(rows, path, header):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(rows, columns=header).to_csv(path, index=False)
    save_csv(neuron_summary_rows, os.path.join(save_dir, 'top_neurons_summary.csv'), header=['neuron_index', 'mean_activation', 'max_activation', 'usage_rate', 'num_top_samples'])
    sample_rows = []
    for neuron_info in neuron_analysis:
        for sample_info in neuron_info['top_samples']:
            row = [
                neuron_info['neuron_index'],
                sample_info.get('sample_index'),
                sample_info.get('activation_value'),
                sample_info.get('label')
            ]
            row.extend([sample_info.get('set'), sample_info.get('split'), sample_info.get('path')])
            for k in extra_manifest_keys:
                row.append(sample_info.get(k))
            sample_rows.append(row)
    header = ['neuron_index', 'sample_index', 'activation_value', 'label', 'set', 'split', 'path'] + list(extra_manifest_keys)
    save_csv(sample_rows, os.path.join(save_dir, 'top_neurons_samples.csv'), header=header)
    print(f"Top neurons analysis completed. Results saved to {save_dir}")
    return neuron_analysis