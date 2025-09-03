from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from typing import Dict, List
import os

# Set style
plt.style.use('default')
sns.set_palette("husl")

# -------------------- Utility functions --------------------
def save_fig(save_dir: str, filename: str, dpi: int = 300, bbox_inches: str = 'tight'):
    """Save figure with consistent formatting"""
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=dpi, bbox_inches=bbox_inches)
    plt.close()

def get_color_map(labels: List[str], forget_digits: List[str]) -> Dict[str, str]:
    """Create consistent color mapping for forget vs retain classes"""
    colors = {}
    for label in labels:
        if label in forget_digits:
            colors[label] = '#ff6b6b'  # Red for forget
        else:
            colors[label] = '#4ecdc4'  # Teal for retain
    return colors

# -------------------- Dimensionality reduction plots --------------------
def plot_tsne_comparison(Z_before: np.ndarray, Z_after: np.ndarray, 
                         labels: List[str], forget_digits: List[str],
                         save_dir: str, n_components: int = 2, 
                         perplexity: int = 30, random_state: int = 42):
    """t-SNE plots comparing before vs after edits"""
    
    # Fit t-SNE on combined data for fair comparison
    Z_combined = np.vstack([Z_before, Z_after])
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                random_state=random_state, n_jobs=-1)
    Z_tsne_combined = tsne.fit_transform(Z_combined)
    
    # Split back
    n_before = Z_before.shape[0]
    Z_tsne_before = Z_tsne_combined[:n_before]
    Z_tsne_after = Z_tsne_combined[n_before:]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Before edit
    colors = get_color_map(labels, forget_digits)
    for label in set(labels):
        mask = np.array(labels) == label
        color = colors[label]
        ax1.scatter(Z_tsne_before[mask, 0], Z_tsne_before[mask, 1], 
                   c=color, label=label, alpha=0.7, s=20)
    
    ax1.set_title('Before Edit', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # After edit
    for label in set(labels):
        mask = np.array(labels) == label
        color = colors[label]
        ax2.scatter(Z_tsne_after[mask, 0], Z_tsne_after[mask, 1], 
                   c=color, label=label, alpha=0.7, s=20)
    
    ax2.set_title('After Edit', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig(save_dir, 'tsne_comparison.png')
    
    return Z_tsne_before, Z_tsne_after

def plot_umap_comparison(Z_before: np.ndarray, Z_after: np.ndarray,
                         labels: List[str], forget_digits: List[str],
                         save_dir: str, n_components: int = 2,
                         n_neighbors: int = 15, min_dist: float = 0.1,
                         random_state: int = 42):
    """UMAP plots comparing before vs after edits"""
    
    # Fit UMAP on combined data
    Z_combined = np.vstack([Z_before, Z_after])
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                       min_dist=min_dist, random_state=random_state)
    Z_umap_combined = reducer.fit_transform(Z_combined)
    
    # Split back
    n_before = Z_before.shape[0]
    Z_umap_before = Z_umap_combined[:n_before]
    Z_umap_after = Z_umap_combined[n_before:]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Before edit
    colors = get_color_map(labels, forget_digits)
    for label in set(labels):
        mask = np.array(labels) == label
        color = colors[label]
        ax1.scatter(Z_umap_before[mask, 0], Z_umap_before[mask, 1], 
                   c=color, label=label, alpha=0.7, s=20)
    
    ax1.set_title('Before Edit', fontsize=14, fontweight='bold')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # After edit
    for label in set(labels):
        mask = np.array(labels) == label
        color = colors[label]
        ax2.scatter(Z_umap_after[mask, 0], Z_umap_after[mask, 1], 
                   c=color, label=label, alpha=0.7, s=20)
    
    ax2.set_title('After Edit', fontsize=14, fontweight='bold')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig(save_dir, 'umap_comparison.png')
    
    return Z_umap_before, Z_umap_after

# -------------------- Subspace U analysis --------------------
def plot_subspace_scatter(ZU_before: np.ndarray, ZU_after: np.ndarray,
                          labels: List[str], forget_digits: List[str],
                          save_dir: str, n_components: int = 2):
    """Scatter plot in U subspace (first 2 principal components)"""
    
    # PCA in U subspace
    ZU_combined = np.vstack([ZU_before, ZU_after])
    pca = PCA(n_components=min(n_components, ZU_combined.shape[1]))
    ZU_pca_combined = pca.fit_transform(ZU_combined)
    
    # Split back
    n_before = ZU_before.shape[0]
    ZU_pca_before = ZU_pca_combined[:n_before]
    ZU_pca_after = ZU_pca_combined[n_before:]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Before edit
    colors = get_color_map(labels, forget_digits)
    for label in set(labels):
        mask = np.array(labels) == label
        color = colors[label]
        ax1.scatter(ZU_pca_before[mask, 0], ZU_pca_before[mask, 1], 
                   c=color, label=label, alpha=0.7, s=20)
    
    ax1.set_title('Before Edit (U subspace)', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # After edit
    for label in set(labels):
        mask = np.array(labels) == label
        color = colors[label]
        ax2.scatter(ZU_pca_after[mask, 0], ZU_pca_after[mask, 1], 
                   c=color, label=label, alpha=0.7, s=20)
    
    ax2.set_title('After Edit (U subspace)', fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig(save_dir, 'subspace_scatter.png')
    
    return ZU_pca_before, ZU_pca_after

# -------------------- Histograms and distributions --------------------
def plot_projection_distributions(ZU_before: np.ndarray, ZU_after: np.ndarray,
                                 labels: List[str], forget_digits: List[str],
                                 save_dir: str, top_directions: int = 5):
    """Plot distributions of projections onto top LEACE directions"""
    
    # Get top directions (highest variance)
    var_before = np.var(ZU_before, axis=0)
    top_indices = np.argsort(var_before)[-top_directions:]
    
    fig, axes = plt.subplots(2, top_directions, figsize=(4*top_directions, 8))
    if top_directions == 1:
        axes = axes.reshape(2, 1)
    
    for i, idx in enumerate(top_indices):
        # Before edit
        ax = axes[0, i]
        for label in set(labels):
            mask = np.array(labels) == label
            color = '#ff6b6b' if label in forget_digits else '#4ecdc4'
            ax.hist(ZU_before[mask, idx], alpha=0.7, bins=20, color=color, 
                   label=label, density=True)
        ax.set_title(f'Direction {idx+1} (Before)')
        ax.set_xlabel('Projection')
        ax.set_ylabel('Density')
        if i == 0:
            ax.legend()
        
        # After edit
        ax = axes[1, i]
        for label in set(labels):
            mask = np.array(labels) == label
            color = '#ff6b6b' if label in forget_digits else '#4ecdc4'
            ax.hist(ZU_after[mask, idx], alpha=0.7, bins=20, color=color, 
                   label=label, density=True)
        ax.set_title(f'Direction {idx+1} (After)')
        ax.set_xlabel('Projection')
        ax.set_ylabel('Density')
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    save_fig(save_dir, 'projection_distributions.png')

def plot_similarity_distributions(intra_sim_before: Dict, intra_sim_after: Dict,
                                inter_sim_before: Dict, inter_sim_after: Dict,
                                forget_digits: List[str], save_dir: str):
    """Plot distributions of intra-class and inter-class similarities"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Intra-class similarities
    ax = axes[0, 0]
    retain_before = [v for k, v in intra_sim_before.items() if k not in forget_digits]
    retain_after = [v for k, v in intra_sim_after.items() if k not in forget_digits]
    forget_before = [v for k, v in intra_sim_before.items() if k in forget_digits]
    forget_after = [v for k, v in intra_sim_after.items() if k in forget_digits]
    
    ax.hist(retain_before, alpha=0.7, bins=20, color='#4ecdc4', label='Retain (Before)', density=True)
    ax.hist(retain_after, alpha=0.7, bins=20, color='#45b7d1', label='Retain (After)', density=True)
    ax.set_title('Intra-class Similarities - Retain Classes')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.legend()
    
    ax = axes[0, 1]
    ax.hist(forget_before, alpha=0.7, bins=20, color='#ff6b6b', label='Forget (Before)', density=True)
    ax.hist(forget_after, alpha=0.7, bins=20, color='#ff8e8e', label='Forget (After)', density=True)
    ax.set_title('Intra-class Similarities - Forget Classes')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.legend()
    
    # Inter-class similarities
    ax = axes[1, 0]
    inter_before = list(inter_sim_before.values())
    inter_after = list(inter_sim_after.values())
    
    ax.hist(inter_before, alpha=0.7, bins=20, color='#96ceb4', label='Before', density=True)
    ax.hist(inter_after, alpha=0.7, bins=20, color='#feca57', label='After', density=True)
    ax.set_title('Inter-class Similarities')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.legend()
    
    # Similarity change
    ax = axes[1, 1]
    retain_change = [after - before for before, after in zip(retain_before, retain_after)]
    forget_change = [after - before for before, after in zip(forget_before, forget_after)]
    
    ax.hist(retain_change, alpha=0.7, bins=20, color='#4ecdc4', label='Retain', density=True)
    ax.hist(forget_change, alpha=0.7, bins=20, color='#ff6b6b', label='Forget', density=True)
    ax.set_title('Similarity Change (After - Before)')
    ax.set_xlabel('Change in Similarity')
    ax.set_ylabel('Density')
    ax.legend()
    
    plt.tight_layout()
    save_fig(save_dir, 'similarity_distributions.png')

# -------------------- Heatmaps --------------------
def plot_decoder_overlap_matrix(decoder_weights: np.ndarray, feat_idx: List[int],
                               save_dir: str, max_atoms: int = 50):
    """Plot cosine similarity matrix of decoder atoms"""
    
    # Select top atoms if too many
    if len(feat_idx) > max_atoms:
        feat_idx = feat_idx[:max_atoms]
    
    # Get selected decoder columns
    D_selected = decoder_weights[:, feat_idx]  # [d, r]
    
    # Compute cosine similarities
    D_norm = D_selected / (np.linalg.norm(D_selected, axis=0, keepdims=True) + 1e-12)
    similarity_matrix = D_norm.T @ D_norm  # [r, r]
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='RdBu_r', center=0, 
                xticklabels=False, yticklabels=False, cbar_kws={'label': 'Cosine Similarity'})
    plt.title('Decoder Atom Overlap Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Atom Index')
    plt.ylabel('Atom Index')
    
    save_fig(save_dir, 'decoder_overlap_matrix.png')
    
    return similarity_matrix

def plot_label_direction_energy(decoder_weights: np.ndarray, feat_idx: List[int],
                               label_direction: np.ndarray, save_dir: str):
    """Plot energy distribution of label direction across decoder atoms"""
    
    # Project label direction onto decoder atoms
    D_selected = decoder_weights[:, feat_idx]  # [d, r]
    D_norm = D_selected / (np.linalg.norm(D_selected, axis=0, keepdims=True) + 1e-12)
    
    # Compute projection coefficients
    proj_coeffs = label_direction @ D_norm  # [r]
    energy_distribution = proj_coeffs ** 2
    
    # Sort by energy
    sorted_indices = np.argsort(energy_distribution)[::-1]
    sorted_energy = energy_distribution[sorted_indices]
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Bar plot
    plt.subplot(1, 2, 1)
    plt.bar(range(len(sorted_energy)), sorted_energy, color='#4ecdc4', alpha=0.7)
    plt.title('Label Direction Energy Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Atom Index (sorted by energy)')
    plt.ylabel('Energy')
    plt.yscale('log')
    
    # Cumulative plot
    plt.subplot(1, 2, 2)
    cumulative_energy = np.cumsum(sorted_energy)
    plt.plot(cumulative_energy, color='#ff6b6b', linewidth=2)
    plt.title('Cumulative Energy', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Atoms')
    plt.ylabel('Cumulative Energy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig(save_dir, 'label_direction_energy.png')
    
    return energy_distribution, sorted_indices

# -------------------- Bar plots --------------------
def plot_atom_contribution_scores(decoder_weights: np.ndarray, feat_idx: List[int],
                                 label_direction: np.ndarray, save_dir: str,
                                 top_k: int = 20):
    """Plot atom contribution scores for label direction"""
    
    # Project label direction onto decoder atoms
    D_selected = decoder_weights[:, feat_idx]  # [d, r]
    D_norm = D_selected / (np.linalg.norm(D_selected, axis=0, keepdims=True) + 1e-12)
    
    # Compute projection coefficients
    proj_coeffs = label_direction @ D_norm  # [r]
    
    # Sort by absolute contribution
    sorted_indices = np.argsort(np.abs(proj_coeffs))[::-1]
    sorted_coeffs = proj_coeffs[sorted_indices]
    
    # Plot top-k
    top_k = min(top_k, len(sorted_coeffs))
    plt.figure(figsize=(12, 6))
    
    colors = ['#ff6b6b' if c < 0 else '#4ecdc4' for c in sorted_coeffs[:top_k]]
    plt.bar(range(top_k), sorted_coeffs[:top_k], color=colors, alpha=0.7)
    plt.title(f'Top-{top_k} Atom Contribution Scores', fontsize=14, fontweight='bold')
    plt.xlabel('Atom Index (sorted by |contribution|)')
    plt.ylabel('Contribution Score')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    save_fig(save_dir, 'atom_contribution_scores.png')
    
    return proj_coeffs, sorted_indices

# -------------------- Singular value spectrum --------------------
def plot_singular_value_spectrum(matrix: np.ndarray, save_dir: str, 
                                title: str = "Singular Value Spectrum",
                                filename: str = "singular_value_spectrum.png"):
    """Plot singular value spectrum of a matrix"""
    
    # Compute SVD
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(s, 'o-', color='#4ecdc4', linewidth=2, markersize=6)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.grid(True, alpha=0.3)
    
    # Add cumulative energy line
    cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
    ax2 = plt.twinx()
    ax2.plot(cumulative_energy, 'r--', alpha=0.7, label='Cumulative Energy')
    ax2.set_ylabel('Cumulative Energy Fraction', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.tight_layout()
    save_fig(save_dir, filename)
    
    return s, cumulative_energy

# -------------------- Whitening diagnostics --------------------
def plot_whitening_diagnostics(Z: np.ndarray, save_dir: str):
    """Plot whitening diagnostics including eigenvalue spectrum and variance"""
    
    # Compute covariance matrix
    Z_centered = Z - Z.mean(axis=0, keepdims=True)
    C = (Z_centered.T @ Z_centered) / max(1, Z_centered.shape[0] - 1)
    
    # Eigenvalue spectrum
    evals, _ = np.linalg.eigh(C)
    evals = np.sort(evals)[::-1]  # Sort descending
    
    # Total variance
    total_variance = np.trace(C)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Eigenvalue spectrum
    ax1.semilogy(evals, 'o-', color='#4ecdc4', linewidth=2, markersize=4)
    ax1.set_title('Eigenvalue Spectrum After Whitening', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance
    cumulative_var = np.cumsum(evals) / total_variance
    ax2.plot(cumulative_var, 'o-', color='#ff6b6b', linewidth=2, markersize=4)
    ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Fraction of Total Variance')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
    ax2.legend()
    
    plt.tight_layout()
    save_fig(save_dir, 'whitening_diagnostics.png')
    
    return evals, total_variance, cumulative_var

# -------------------- Main plotting function --------------------
def create_all_plots(Z_before: np.ndarray, Z_after: np.ndarray,
                     ZU_before: np.ndarray, ZU_after: np.ndarray,
                     labels: List[str], forget_digits: List[str],
                     decoder_weights: np.ndarray, feat_idx: List[int],
                     intra_sim_before: Dict, intra_sim_after: Dict,
                     inter_sim_before: Dict, inter_sim_after: Dict,
                     save_dir: str, random_state: int = 42):
    """Create all diagnostic plots"""
    
    print(f"Creating diagnostic plots in {save_dir}...")
    
    # 1. Dimensionality reduction plots
    print("  - Creating t-SNE plots...")
    plot_tsne_comparison(Z_before, Z_after, labels, forget_digits, save_dir, random_state=random_state)
    
    print("  - Creating UMAP plots...")
    plot_umap_comparison(Z_before, Z_after, labels, forget_digits, save_dir, random_state=random_state)
    
    # 2. Subspace analysis
    if ZU_before.shape[1] > 0:
        print("  - Creating subspace scatter plots...")
        plot_subspace_scatter(ZU_before, ZU_after, labels, forget_digits, save_dir)
        
        print("  - Creating projection distributions...")
        plot_projection_distributions(ZU_before, ZU_after, labels, forget_digits, save_dir)
    
    # 3. Similarity analysis
    print("  - Creating similarity distributions...")
    plot_similarity_distributions(intra_sim_before, intra_sim_after,
                                inter_sim_before, inter_sim_after,
                                forget_digits, save_dir)
    
    # 4. Decoder analysis
    if len(feat_idx) > 0:
        print("  - Creating decoder overlap matrix...")
        plot_decoder_overlap_matrix(decoder_weights, feat_idx, save_dir)
        
        # Create dummy label direction for demonstration (you can modify this)
        label_direction = np.random.randn(decoder_weights.shape[0])
        label_direction = label_direction / (np.linalg.norm(label_direction) + 1e-12)
        
        print("  - Creating label direction energy plots...")
        plot_label_direction_energy(decoder_weights, feat_idx, label_direction, save_dir)
        
        print("  - Creating atom contribution scores...")
        plot_atom_contribution_scores(decoder_weights, feat_idx, label_direction, save_dir)
    
    # 5. Whitening diagnostics
    print("  - Creating whitening diagnostics...")
    plot_whitening_diagnostics(Z_before, save_dir)
    
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