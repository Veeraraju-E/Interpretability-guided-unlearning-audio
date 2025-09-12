# interp.py  (full patched file)
import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from itertools import combinations
from typing import List, Tuple, Dict, Any
from scipy.stats import ks_2samp, pearsonr, gaussian_kde
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# audio tools
try:
    import parselmouth
except Exception:
    parselmouth = None

from common import SAE, sae_encode_batch, l2_normalize, compute_scores, eer_from_scores, build_verification_trials, invert_whitener

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.3, rc={"lines.linewidth": 2.0})

#  color palettes
MAIN_PALETTE = {
    'forget': '#E74C3C',     # Vibrant red
    'retain': '#3498DB',     # Professional blue
    'neutral': '#95A5A6',    # Sophisticated gray
    'accent': '#F39C12',     # Gold accent
    'success': '#27AE60',    # Green for positive changes
    'warning': '#E67E22'     # Orange for warnings
}

CATEGORICAL_COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#34495E', '#E67E22']

# Advanced matplotlib configuration
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': ['DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.fancybox': True,
    'legend.shadow': False
})


def ensure_dir(path: str) -> str:
    """Creates a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
    return path


def save_fig(fig: plt.Figure, basepath: str):
    """Saves the current figure in both PNG and PDF formats with optimization."""
    ensure_dir(os.path.dirname(basepath))
    fig.savefig(f"{basepath}.png", bbox_inches="tight", dpi=300, facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close(fig)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates Cohen's d for independent samples."""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    s_pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2) + 1e-12)
    return float((np.mean(a) - np.mean(b)) / s_pooled)


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 500, seed: int = 1337) -> Tuple[float, Tuple[float, float]]:
    """
    Computes the 95% confidence interval for the difference in means via vectorized bootstrapping.
    Reduced default n_boot to 500 for speed; increase if you need tighter CI.
    """
    rng = np.random.RandomState(seed)
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return 0.0, (0.0, 0.0)

    # Vectorized resampling: create indices matrices and compute means along axis
    ia = rng.randint(0, len(a), size=(n_boot, len(a)))
    ib = rng.randint(0, len(b), size=(n_boot, len(b)))
    means_a = a[ia].mean(axis=1)
    means_b = b[ib].mean(axis=1)
    diffs = means_a - means_b
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), (float(lo), float(hi))


def add_statistical_annotation(ax, x1, x2, y, height, pvalue, test_name=""):
    """Add statistical significance annotation between two points."""
    if pvalue < 0.001:
        sig_text = "***"
    elif pvalue < 0.01:
        sig_text = "**"
    elif pvalue < 0.05:
        sig_text = "*"
    else:
        sig_text = "n.s."

    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], 'k-', linewidth=1)
    ax.text((x1 + x2) / 2, y + height, f'{sig_text}\np={pvalue:.1e}', ha='center', va='bottom', fontsize=9)


# ---------------------------
# Performance-critical helpers
# ---------------------------

def sample_pairwise_similarities(Z: np.ndarray, idxA: np.ndarray, idxB: np.ndarray = None, max_pairs: int = 50000, rng: np.random.RandomState = None) -> np.ndarray:
    """
    Efficiently sample up to max_pairs cosine similarities between vectors in Z for:
      - intra set (idxB is None): sample random unique pairs from idxA
      - inter set (idxB provided): sample random pairs (i in idxA, j in idxB)
    Returns a numpy array of cosine similarities.
    This avoids building all combinations in memory and is vectorized.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    idxA = np.asarray(idxA)
    nA = len(idxA)
    if nA == 0:
        return np.array([])

    if idxB is None:
        # Number of possible pairs is nA*(nA-1)//2
        total_pairs = nA * (nA - 1) // 2
        sample_n = min(max_pairs, total_pairs)
        if sample_n <= 0:
            return np.array([])

        # If sample_n is close to total_pairs, compute all pairs via vectorized dot
        if sample_n == total_pairs:
            # compute full similarity matrix upper triangle
            M = Z[idxA]  # shape (nA, dim)
            norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
            Mn = M / norms
            S = Mn @ Mn.T
            iu = np.triu_indices(nA, k=1)
            return S[iu]
        else:
            # sample random pairs with replacement (but avoid trivial i==j)
            i = rng.randint(0, nA, size=sample_n)
            j = rng.randint(0, nA - 1, size=sample_n)
            # fix j so it's not equal to i
            j = (j + (j >= i).astype(int))
            ia = idxA[i]
            ja = idxA[j]
    else:
        idxB = np.asarray(idxB)
        nB = len(idxB)
        if nB == 0:
            return np.array([])
        total_pairs = nA * nB
        sample_n = min(max_pairs, total_pairs)
        if sample_n <= 0:
            return np.array([])
        if sample_n == total_pairs:
            A = Z[idxA]
            B = Z[idxB]
            An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
            Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
            S = An @ Bn.T
            return S.flatten()
        else:
            i = rng.randint(0, nA, size=sample_n)
            j = rng.randint(0, nB, size=sample_n)
            ia = idxA[i]
            ja = idxB[j]

    # compute cosine similarities vectorized
    A = Z[ia]
    B = Z[ja]
    num = np.sum(A * B, axis=1)
    denom = (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)) + 1e-9
    sims = num / denom
    return sims


def decode_subset(sae: SAE, A_all: np.ndarray, indices: np.ndarray, device: str = "cpu", batch_size: int = 4096) -> np.ndarray:
    """
    Decode only rows in 'indices' from activation space A_all using sae.dec.
    Returns decoded Z_subset with same ordering as indices.
    """
    indices = np.asarray(indices)
    chunks = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        A_batch = A_all[batch_idx].astype(np.float32)
        with torch.no_grad():
            Z_batch = sae.dec(torch.from_numpy(A_batch).to(device)).cpu().numpy()
        chunks.append(Z_batch)
    if len(chunks) == 0:
        return np.zeros((0, sae.input_dim), dtype=np.float32)
    return np.vstack(chunks)


# ---------------------------
# Full analysis functions
# ---------------------------

def analyze_feature_discriminability(A_all: np.ndarray, sets: np.ndarray, forget_spks: List[str], labels: np.ndarray, out_dir: str, top_k: int = 50) -> pd.DataFrame:
    """
    Analyze which SAE features discriminate 'forget' speakers from others.
    Returns a DataFrame sorted by KS-statistic and effect size.

    Modified: Figure 1 now plots neuron index on x-axis and activations (mean difference) on y-axis.
    """
    print("\n[1/7] Analyzing feature discriminability with  visualizations...")
    outdir = ensure_dir(os.path.join(out_dir, "1_feature_discriminability"))

    mask_dev = np.array([s.endswith(":dev") for s in sets])
    A_dev = A_all[mask_dev]
    labels_dev = labels[mask_dev]

    forget_mask = np.isin(labels_dev, forget_spks)
    retain_mask = ~forget_mask

    n_features = A_dev.shape[1]
    stats = []
    for f in range(n_features):
        col = A_dev[:, f]
        col_f = col[forget_mask]
        col_r = col[retain_mask]
        if len(col_f) < 2 or len(col_r) < 2:
            ks_p = 1.0
            p_mean = 0.0
            mean_diff = 0.0
            effect = 0.0
        else:
            _, ks_p = ks_2samp(col_f, col_r)
            mean_diff = np.mean(col_f) - np.mean(col_r)
            effect = cohen_d(col_f, col_r)
            p_mean = pearsonr(col_f, col_r[:len(col_f)])[1] if (len(col_f) == len(col_r) and len(col_f) > 1) else np.nan
        stats.append((f, mean_diff, effect, ks_p, np.nanmean(col_f), np.nanmean(col_r)))

    df = pd.DataFrame(stats, columns=["feature_index", "mean_diff", "cohen_d", "ks_p", "mean_forget", "mean_retain"])
    df["abs_mean_diff"] = np.abs(df["mean_diff"])
    df = df.sort_values(["abs_mean_diff", "cohen_d"], ascending=False).reset_index(drop=True)

    # plotting top features (vertical bars) -- neuron index on x-axis, activation y-axis
    top_df = df.head(top_k)
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    x = top_df["feature_index"].astype(int).values
    yvals = top_df["mean_diff"].values
    colors = [MAIN_PALETTE['forget'] if v > 0 else MAIN_PALETTE['retain'] for v in yvals]
    bars = ax.bar(np.arange(len(top_df)), top_df["abs_mean_diff"], color=[MAIN_PALETTE['forget'] if v > 0 else MAIN_PALETTE['retain'] for v in top_df["mean_diff"]], alpha=0.9, edgecolor='k')

    ax.set_xticks(np.arange(len(top_df)))
    ax.set_xticklabels(top_df["feature_index"].astype(int), rotation=90)
    ax.set_ylabel("Total Difference (Forget − Retain)")
    ax.set_title("Top Features by Absolute Mean Difference")

    top5_idx = np.argsort(top_df["abs_mean_diff"].values)[-5:]
    for i in top5_idx:
        ax.text(i, top_df["abs_mean_diff"].iloc[i] + 0.01, f"{top_df['abs_mean_diff'].iloc[i]:.2f}", ha='center', va='bottom', fontsize=9)
    save_fig(fig, os.path.join(outdir, "figure1_top_feature_diffs"))

    # distribution plot for top 6 features (keep as Figure 2)
    top6 = df.head(6)["feature_index"].values
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, hspace=0.35)
    for i, feat in enumerate(top6):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        fvals = A_dev[:, feat]
        sns.kdeplot(fvals[forget_mask], ax=ax, label='forget', fill=True, alpha=0.5, linewidth=1.5)
        sns.kdeplot(fvals[retain_mask], ax=ax, label='retain', fill=True, alpha=0.5, linewidth=1.5)
        ax.set_title(f"Feature {feat} (meanΔ={df.loc[df.feature_index == feat, 'mean_diff'].values[0]:+.3f})")
        ax.legend()
        ax.set_xlabel("Activation")
        ax.set_ylabel("Density")
    save_fig(fig, os.path.join(outdir, "figure2_top6_feature_kdes"))

    # Save stats
    df.to_csv(os.path.join(outdir, "feature_discriminability_stats.csv"), index=False)
    print(f"  -> Saved Feature discriminability stats and figures to {outdir}")
    return df


def analyze_latent_space_activations(A_all: np.ndarray, sets: np.ndarray, labels: np.ndarray, manifest: np.ndarray, forget_spks: List[str], out_dir: str):
    """
    Visualize latent activation magnitudes (not vectors).

    - For the dev set, use the forget-speaker samples only.
    - For each neuron index (x-axis) we plot jittered scatter of |activation| across samples (y-axis).
    - Vertical line at the mean magnitude per neuron.
    - Annotate top-20 neurons by mean magnitude (forget set).
    - Save CSV listing top-20 samples (global indices and manifest path if available) per annotated neuron.
    """
    print("\n[2/7] Visualizing latent activation magnitudes (forget set only)...")
    outdir = ensure_dir(os.path.join(out_dir, "2_latent_activations"))

    mask_dev = np.array([s.endswith(":dev") for s in sets])
    dev_indices = np.where(mask_dev)[0]
    if len(dev_indices) == 0:
        print("  -> No dev samples found; skipping latent activation visualization.")
        return

    labels_dev = labels[mask_dev]
    A_dev = A_all[mask_dev]

    # identify forget samples within dev
    forget_mask_dev = np.isin(labels_dev, forget_spks)
    forget_dev_indices = dev_indices[forget_mask_dev]  # global indices in A_all
    if len(forget_dev_indices) == 0:
        print("  -> No forget samples in dev set; skipping latent activation visualization.")
        return

    A_forget = A_all[forget_dev_indices]  # shape (n_forget_samples, n_neurons)
    n_samples, n_neurons = A_forget.shape

    # Compute mean magnitudes per neuron
    mean_mags = np.mean(np.abs(A_forget), axis=0)
    top20_neurons = np.argsort(mean_mags)[-20:][::-1]  # descending

    # Average activations per neuron
    avg_acts = A_dev.mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(avg_acts))
    ax.plot(x, avg_acts, color=MAIN_PALETTE['neutral'], linewidth=1.5)

    # Highlight top-10 neurons
    top10_idx = np.argsort(avg_acts)[-10:]
    for idx in top10_idx:
        ax.axvline(idx, color=MAIN_PALETTE['accent'], linestyle="--", alpha=0.7)
        ax.text(idx, avg_acts[idx] + 0.01, str(idx),
                ha="center", va="bottom", fontsize=9, rotation=90)

    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("Average Activation Magnitude")
    ax.set_title("Latent Activations (Average Magnitudes, dev set)")

    save_fig(fig, os.path.join(outdir, "figure3_latent_avg_magnitudes"))


    # Save CSV with top-20 samples per top neuron
    rows = []
    # Build a normalized manifest path list aligned with A_all (if manifest is provided)
    manifest_list = []
    if isinstance(manifest, (list, np.ndarray)):
        manifest_list = list(manifest)
    else:
        try:
            manifest_list = list(manifest)
        except Exception:
            manifest_list = [""] * A_all.shape[0]

    for neuron_idx in top20_neurons:
        neuron_acts = np.abs(A_forget[:, neuron_idx])
        # get indices in forget_dev_indices that correspond to top 20 sample activations for this neuron
        order = np.argsort(neuron_acts)[-20:][::-1]
        for rank, local_pos in enumerate(order, start=1):
            global_idx = int(forget_dev_indices[local_pos])  # original global index in A_all
            act_val = float(np.abs(A_all[global_idx, neuron_idx]))
            path = ""
            try:
                ent = manifest_list[global_idx]
                if isinstance(ent, dict):
                    path = ent.get("path", "")
                elif isinstance(ent, (list, tuple)) and len(ent) > 0:
                    path = ent[0]
                else:
                    path = str(ent)
            except Exception:
                path = ""
            rows.append({"neuron": int(neuron_idx), "rank": int(rank), "global_index": int(global_idx), "activation": act_val, "manifest_path": path})
    csv_path = os.path.join(outdir, "latent_top20_forget_samples.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  -> Saved latent activation plot and top-20 sample CSV to {outdir}")


def visualize_subspace_edit(Z_all: np.ndarray, Z_edit: np.ndarray, U: np.ndarray, sets: np.ndarray, labels: np.ndarray, forget_spks: List[str], out_dir: str):
    """
    Visualize Z_all vs Z_edit in a low-dim projection and show per-speaker centroids.
    """
    print("\n[4/7] Visualizing subspace edit effects...")
    outdir = ensure_dir(os.path.join(out_dir, "4_subspace_edit"))

    mask_dev = np.array([s.endswith(":dev") for s in sets])
    Z_dev = Z_all[mask_dev]
    Z_edit_dev = Z_edit[mask_dev]
    labels_dev = labels[mask_dev]

    # PCA on concatenated to get shared projection
    concat = np.vstack([Z_dev, Z_edit_dev])
    pca = PCA(n_components=2)
    proj = pca.fit_transform(concat)
    proj_before = proj[:Z_dev.shape[0], :]
    proj_after = proj[Z_dev.shape[0]:, :]

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    # color by forget vs retain
    forget_mask = np.isin(labels_dev, forget_spks)
    ax.scatter(proj_before[~forget_mask, 0], proj_before[~forget_mask, 1], s=12, alpha=0.6, label='retain (before)', c=MAIN_PALETTE['retain'])
    ax.scatter(proj_before[forget_mask, 0], proj_before[forget_mask, 1], s=20, alpha=0.9, label='forget (before)', c=MAIN_PALETTE['forget'], marker='x')

    ax.scatter(proj_after[~forget_mask, 0], proj_after[~forget_mask, 1], s=12, alpha=0.5, label='retain (after)', edgecolor='k', facecolor='none')
    ax.scatter(proj_after[forget_mask, 0], proj_after[forget_mask, 1], s=40, alpha=0.9, label='forget (after)', marker='P', facecolor=MAIN_PALETTE['accent'])

    # draw arrows for a random subset to illustrate shifts
    rng = np.random.RandomState(0)
    n_arrows = min(80, proj_before.shape[0])
    idxs = rng.choice(np.arange(proj_before.shape[0]), size=n_arrows, replace=False)
    for i in idxs:
        ax.arrow(proj_before[i, 0], proj_before[i, 1], proj_after[i, 0] - proj_before[i, 0], proj_after[i, 1] - proj_before[i, 1], alpha=0.15, width=0.0005, head_width=0.01, color='gray')

    ax.set_title("Subspace Edit: Before vs After (PCA projection)")
    ax.legend()
    save_fig(fig, os.path.join(outdir, "figure6_subspace_edit_pca"))

    # per-speaker centroid shift visualization (top speakers by shift magnitude)
    unique_speakers = np.unique(labels_dev)
    centroids_before = []
    centroids_after = []
    speaker_ids = []
    for spk in unique_speakers:
        idx = np.where(labels_dev == spk)[0]
        if len(idx) < 2:
            continue
        cent_before = proj_before[idx].mean(axis=0)
        cent_after = proj_after[idx].mean(axis=0)
        centroids_before.append(cent_before)
        centroids_after.append(cent_after)
        speaker_ids.append(spk)
    centroids_before = np.array(centroids_before)
    centroids_after = np.array(centroids_after)
    shifts = np.linalg.norm(centroids_after - centroids_before, axis=1)
    top_speaker_idx = np.argsort(shifts)[-12:][::-1]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    for i in top_speaker_idx:
        a = centroids_before[i]
        b = centroids_after[i]
        ax.plot([a[0], b[0]], [a[1], b[1]], '-o', alpha=0.8, linewidth=1.8)
        ax.text(b[0], b[1], str(speaker_ids[i]), fontsize=9)
    ax.set_title("Top Speaker Centroid Shifts (post-edit)")
    save_fig(fig, os.path.join(outdir, "figure7_speaker_centroid_shifts"))


def quantify_geometric_impact(Z_all: np.ndarray, Z_edit: np.ndarray, sets: np.ndarray, labels: np.ndarray, forget_spks: List[str], out_dir: str, max_pairs: int = 50000):
    """ geometric impact analysis with more efficient pair sampling."""
    print("\n[5/7]  geometric impact quantification...")
    outdir = ensure_dir(os.path.join(out_dir, "5_geometric_impact"))

    mask_dev = np.array([s.endswith(":dev") for s in sets])
    Z_dev, Z_edit_dev = Z_all[mask_dev], Z_edit[mask_dev]
    labels_dev = labels[mask_dev]
    idx_f = np.where(np.isin(labels_dev, forget_spks))[0]
    idx_r = np.where(~np.isin(labels_dev, forget_spks))[0]

    rng = np.random.RandomState(1337)

    # Use sampled similarities instead of enumerating all pairs
    similarity_types = {
        "Intra-Forget": (sample_pairwise_similarities(Z_dev, idx_f, None, max_pairs=max_pairs, rng=rng),
                         sample_pairwise_similarities(Z_edit_dev, idx_f, None, max_pairs=max_pairs, rng=rng)),
        "Inter-Class": (sample_pairwise_similarities(Z_dev, idx_f, idx_r, max_pairs=max_pairs, rng=rng),
                        sample_pairwise_similarities(Z_edit_dev, idx_f, idx_r, max_pairs=max_pairs, rng=rng)),
        "Intra-Retain": (sample_pairwise_similarities(Z_dev, idx_r, None, max_pairs=max_pairs, rng=rng),
                         sample_pairwise_similarities(Z_edit_dev, idx_r, None, max_pairs=max_pairs, rng=rng))
    }

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, height_ratios=[1.5, 1, 1], width_ratios=[1, 1, 1, 0.8], hspace=0.4, wspace=0.3)
    colors = [MAIN_PALETTE['forget'], MAIN_PALETTE['neutral'], MAIN_PALETTE['retain']]

    for i, (title, (before, after)) in enumerate(similarity_types.items()):
        ax = fig.add_subplot(gs[0, i])
        if len(before) == 0 or len(after) == 0:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            continue

        x_range = np.linspace(min(before.min(), after.min()), max(before.max(), after.max()), 200)
        if len(before) > 10 and len(after) > 10:
            kde_before = gaussian_kde(before)
            kde_after = gaussian_kde(after)
            density_before = kde_before(x_range)
            density_after = kde_after(x_range)
            ax.fill_between(x_range, density_before, alpha=0.4, color='gray', label='Before')
            ax.fill_between(x_range, density_after, alpha=0.6, color=colors[i], label='After')
            ax.plot(x_range, density_before, color='gray', linewidth=2)
            ax.plot(x_range, density_after, color=colors[i], linewidth=2)

        ks_stat, p_val = ks_2samp(before, after)
        effect_size = cohen_d(after, before)
        ax.text(0.05, 0.95, f'KS: p = {p_val:.1e}\nCohen\'s d = {effect_size:.3f}', transform=ax.transAxes, va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_title(f"{title} Similarities", fontweight='bold', pad=15)
        ax.set_xlabel("Cosine Similarity", fontweight='bold')
        ax.set_ylabel("Density", fontweight='bold')
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.axvline(before.mean(), color='gray', linestyle='--', alpha=0.8, linewidth=2)
        ax.axvline(after.mean(), color=colors[i], linestyle='--', alpha=0.8, linewidth=2)

    # Summary panel
    ax_summary = fig.add_subplot(gs[0, 3])
    ax_summary.axis('off')
    summary_text = "Statistical Summary\n\n"
    for i, (title, (before, after)) in enumerate(similarity_types.items()):
        if len(before) == 0 or len(after) == 0:
            summary_text += f"{title}: Insufficient data\n\n"
            continue
        mean_before, mean_after = before.mean(), after.mean()
        std_before, std_after = before.std(), after.std()
        _, p_val = ks_2samp(before, after)
        effect_size = cohen_d(after, before)
        summary_text += f"{title}:\n"
        summary_text += f"  Δμ: {mean_after - mean_before:+.3f}\n"
        summary_text += f"  Δσ: {std_after - std_before:+.3f}\n"
        summary_text += f"  p-val: {p_val:.2e}\n"
        summary_text += f"  Cohen's d: {effect_size:.3f}\n\n"
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, va='top', ha='left', fontsize=10, fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # Middle row: boxplots for each similarity type before/after
    for i, (title, (before, after)) in enumerate(similarity_types.items()):
        ax = fig.add_subplot(gs[1, i])
        data = [before, after]
        ax.boxplot(data, labels=['Before', 'After'], patch_artist=True, boxprops=dict(facecolor='lightgray'))
        ax.set_title(f"{title} Boxplot")
        ax.set_ylabel("Cosine Sim")
        if len(before) > 1 and len(after) > 1:
            ks_stat, p_val = ks_2samp(before, after)
            add_statistical_annotation(ax, 1, 2, max(np.max(before), np.max(after)) * 0.95, 0.02, p_val)

    # Bottom row: change-in-mean bars (delta)
    for i, (title, (before, after)) in enumerate(similarity_types.items()):
        ax = fig.add_subplot(gs[2, i])
        delta = after.mean() - before.mean()
        ax.bar(0, delta, color=MAIN_PALETTE['accent'] if delta > 0 else MAIN_PALETTE['neutral'])
        ax.set_xticks([0])
        ax.set_xticklabels([title])
        ax.set_ylabel("ΔMean (After - Before)")
        ax.axhline(0, color='black', linewidth=1)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Geometric Impact Analysis", fontsize=16, fontweight='bold', y=0.98)
    save_fig(fig, os.path.join(outdir, "figure8_geometric_impact"))
    print("  -> Saved  Figure 8 (Geometric Impact Analysis).")


def run_causal_interventions(sae: SAE, Z_all: np.ndarray, mu: np.ndarray, W: np.ndarray, sets: np.ndarray, labels: np.ndarray, forget_spks: List[str], sorted_features: pd.DataFrame, out_dir: str, top_k: int = 12, device: str = "cpu", trial_bootstrap: int = 200):
    """
     causal intervention analysis with major speedups:
      - only decode embeddings for samples that appear in verification trials (unique indices)
      - smaller default bootstrap (trial_bootstrap) but configurable

    Modified: Figure 10 now presents three uniform subplots (Knockout / Amplify / Suppress)
    with neuron indices on x-axis and percent change relative to baseline on y-axis.
    The bottom table is removed.
    """
    print("\n[6/7] Running  causal interventions...")
    outdir = ensure_dir(os.path.join(out_dir, "6_causal_interventions"))

    mask_test = np.array([s.endswith(":test") for s in sets])
    trials, trial_labels = build_verification_trials(labels, forget_spks, mask_test)
    if len(trials) == 0:
        print("  -> No verification trials found for test set. Skipping.")
        return

    # Encode full A_all once (was already done earlier but keep here for safety)
    A_all = sae_encode_batch(sae, Z_all, device=device, batch=8192)

    # Find unique sample indices used in 'trials' -> only decode those rows for interventions
    unique_trial_indices = np.unique([idx for pair in trials for idx in pair])
    unique_trial_indices = np.asarray(unique_trial_indices, dtype=int)
    # Pre-decode baseline for those trial indices
    with torch.no_grad():
        Z_decoded_for_trials = decode_subset(sae, A_all, unique_trial_indices, device=device, batch_size=4096)
    E_norm_base_subset = l2_normalize(invert_whitener(Z_decoded_for_trials, mu, W))

    # Helper: compute EER given modified A_all but only for trial indices
    def compute_eer_for_modified_A(A_modified):
        # A_modified is full A_all; we'll only decode the rows corresponding to unique_trial_indices
        Z_mod_subset = decode_subset(sae, A_modified, unique_trial_indices, device=device, batch_size=4096)
        E_mod_norm_subset = l2_normalize(invert_whitener(Z_mod_subset, mu, W))
        # Need to remap trials to 0..len(unique_trial_indices)-1
        index_map = {orig: i for i, orig in enumerate(unique_trial_indices)}
        remapped_trials = [(index_map[a], index_map[b]) for (a, b) in trials]
        return eer_from_scores(compute_scores(E_mod_norm_subset, remapped_trials), trial_labels)

    base_eer = compute_eer_for_modified_A(A_all)
    print(f"  -> Baseline EER on test trials: {base_eer:.6f}")

    intervention_results = []
    top_features = sorted_features.head(top_k)["feature_index"].values

    # Knockout: apply per-feature but only decode subset inside compute_eer_for_modified_A
    knockout_results = [{"intervention": "Baseline", "feature": "None", "feature_idx": -1, "eer": base_eer, "delta_eer": 0.0}]
    for feat_idx in top_features:
        A_ko = A_all.copy()
        A_ko[:, feat_idx] = 0.0
        eer_ko = compute_eer_for_modified_A(A_ko)
        knockout_results.append({"intervention": "Knockout", "feature": f"F{feat_idx}", "feature_idx": int(feat_idx), "eer": eer_ko, "delta_eer": eer_ko - base_eer})

    # Amplify (limit to first 6 for speed)
    amplify_results = []
    amplify_factor = 2.0
    for feat_idx in top_features[:6]:
        A_amp = A_all.copy()
        A_amp[:, feat_idx] *= amplify_factor
        eer_amp = compute_eer_for_modified_A(A_amp)
        amplify_results.append({"intervention": "Amplify", "feature": f"F{feat_idx}", "feature_idx": int(feat_idx), "eer": eer_amp, "delta_eer": eer_amp - base_eer})

    # Suppress (limit to first 6 for speed)
    suppress_results = []
    suppress_factor = 0.1
    for feat_idx in top_features[:6]:
        A_sup = A_all.copy()
        A_sup[:, feat_idx] *= suppress_factor
        eer_sup = compute_eer_for_modified_A(A_sup)
        suppress_results.append({"intervention": "Suppress", "feature": f"F{feat_idx}", "feature_idx": int(feat_idx), "eer": eer_sup, "delta_eer": eer_sup - base_eer})

    all_results = knockout_results + amplify_results + suppress_results
    intervention_df = pd.DataFrame(all_results)
    intervention_df.to_csv(os.path.join(outdir, "intervention_eer_results.csv"), index=False)

    # Convert delta_eer to percent relative to baseline safely
    def to_percent(delta_val, baseline):
        if baseline is None or baseline <= 1e-12:
            # baseline too small -> fallback to scaled absolute (x100)
            return delta_val * 100.0
        return (delta_val / baseline) * 100.0

    # Plot unified subplots: Knockout (top_k features), Amplify (first up to 6), Suppress (first up to 6)
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, hspace=0.35, wspace=0.3)

    # Panel A: Knockout (all knockout rows except baseline)
    ax1 = fig.add_subplot(gs[0, 0])
    knockout_df = intervention_df[intervention_df["intervention"] == "Knockout"].copy()
    if len(knockout_df) == 0:
        ax1.text(0.5, 0.5, 'No knockout results', ha='center', va='center')
    else:
        # ensure neuron indices on x-axis (sorted by index)
        knockout_df = knockout_df.sort_values("feature_idx")
        x = knockout_df["feature_idx"].astype(int).values
        deltas = knockout_df["delta_eer"].values
        pct_changes = np.array([to_percent(d, base_eer) for d in deltas])
        colors = [MAIN_PALETTE['success'] if v < 0 else MAIN_PALETTE['forget'] for v in pct_changes]
        bars = ax1.bar(x, pct_changes, color=colors, alpha=0.9, edgecolor='k', linewidth=0.8)
        ax1.set_xticks(x)
        ax1.set_xlabel("Neuron (feature) index", fontweight='bold')
        ax1.set_ylabel("ΔEER (%) relative to baseline", fontweight='bold')
        ax1.set_title(f"Knockout Effects (Top {len(x)} features)", fontweight='bold')
        ax1.axhline(0, color='k', linewidth=1, linestyle='--')
        ax1.grid(True, alpha=0.25)
        for xi, bar, pct in zip(x, bars, pct_changes):
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.02 * np.sign(h if h != 0 else 1), f"{pct:+.2f}%", ha='center', va='bottom', fontsize=9)

    # Panel B: Amplify
    ax2 = fig.add_subplot(gs[0, 1])
    amp_df = intervention_df[intervention_df["intervention"] == "Amplify"].copy()
    if len(amp_df) == 0:
        ax2.text(0.5, 0.5, 'No amplify results', ha='center', va='center')
    else:
        amp_df = amp_df.sort_values("feature_idx")
        x2 = amp_df["feature_idx"].astype(int).values
        deltas2 = amp_df["delta_eer"].values
        pct2 = np.array([to_percent(d, base_eer) for d in deltas2])
        colors2 = [MAIN_PALETTE['accent'] if v > 0 else MAIN_PALETTE['success'] for v in pct2]
        bars2 = ax2.bar(x2, pct2, color=colors2, edgecolor='k')
        ax2.set_xticks(x2)
        ax2.set_xlabel("Neuron (feature) index", fontweight='bold')
        ax2.set_ylabel("ΔEER (%) relative to baseline", fontweight='bold')
        ax2.set_title("Amplify Effects", fontweight='bold')
        ax2.axhline(0, color='k', linewidth=1, linestyle='--')
        ax2.grid(True, alpha=0.25)
        for xi, bar, pct in zip(x2, bars2, pct2):
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.02 * np.sign(h if h != 0 else 1), f"{pct:+.2f}%", ha='center', va='bottom', fontsize=9)

    # Panel C: Suppress
    ax3 = fig.add_subplot(gs[0, 2])
    sup_df = intervention_df[intervention_df["intervention"] == "Suppress"].copy()
    if len(sup_df) == 0:
        ax3.text(0.5, 0.5, 'No suppress results', ha='center', va='center')
    else:
        sup_df = sup_df.sort_values("feature_idx")
        x3 = sup_df["feature_idx"].astype(int).values
        deltas3 = sup_df["delta_eer"].values
        pct3 = np.array([to_percent(d, base_eer) for d in deltas3])
        colors3 = [MAIN_PALETTE['success'] if v < 0 else MAIN_PALETTE['neutral'] for v in pct3]
        bars3 = ax3.bar(x3, pct3, color=colors3, edgecolor='k')
        ax3.set_xticks(x3)
        ax3.set_xlabel("Neuron (feature) index", fontweight='bold')
        ax3.set_ylabel("ΔEER (%) relative to baseline", fontweight='bold')
        ax3.set_title("Suppress Effects", fontweight='bold')
        ax3.axhline(0, color='k', linewidth=1, linestyle='--')
        ax3.grid(True, alpha=0.25)
        for xi, bar, pct in zip(x3, bars3, pct3):
            h = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.02 * np.sign(h if h != 0 else 1), f"{pct:+.2f}%", ha='center', va='bottom', fontsize=9)

    plt.suptitle("Causal Intervention Effects (ΔEER in %)", fontsize=16, fontweight='bold', y=0.98)
    save_fig(fig, os.path.join(outdir, "figure10_causal_interventions"))
    print("  -> Saved Figure 10 (Causal Interventions) with percentage ΔEER and no table.")


def correlate_with_acoustic_features(sorted_features: pd.DataFrame, A_all: np.ndarray, manifest: np.ndarray, sets: np.ndarray, out_dir: str, top_k: int = 20, acoustic_limit: int = 2000):
    """ acoustic correlation analysis with robust handling to avoid spurious zero correlations."""
    if parselmouth is None:
        print("\n[7/7] Skipping acoustic correlation: 'parselmouth' not installed.")
        return
    print("\n[7/7]  acoustic correlation analysis...")
    outdir = ensure_dir(os.path.join(out_dir, "7_acoustic_correlation"))

    mask_dev = np.array([s.endswith(":dev") for s in sets])
    A_dev = A_all[mask_dev]
    manifest_dev = manifest[mask_dev]

    cache_path = os.path.join(outdir, "acoustic_features_cache.csv")
    if os.path.exists(cache_path):
        acoustic_df = pd.read_csv(cache_path)
    else:
        print("  -> Computing acoustic features (this may take a while) ...")
        rows = []
        # limit to first acoustic_limit files for speed/first pass
        for i, entry in enumerate(manifest_dev[:acoustic_limit]):
            if i % 100 == 0:
                print(f"     Processing {i+1}/{min(len(manifest_dev), acoustic_limit)} audio files...")
            path = None
            if isinstance(entry, dict):
                path = entry.get("path", "")
            elif isinstance(entry, (list, tuple)):
                path = entry[0] if len(entry) > 0 else ""
            else:
                path = str(entry)
            if not path or not os.path.exists(path):
                rows.append({"path": path, **{col: np.nan for col in ['f0_mean', 'f0_std', 'f0_range', 'jitter', 'shimmer', 'hnr', 'spectral_centroid', 'spectral_rolloff']}})
                continue
            try:
                snd = parselmouth.Sound(path)
                pitch = snd.to_pitch()
                f0 = pitch.selected_array['frequency']
                f0_valid = f0[f0 != 0]
                point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", pitch.tmin, pitch.tmax)
                harmonicity = snd.to_harmonicity()
                spectrum = snd.to_spectrum()
                row_data = {
                    "path": path,
                    "f0_mean": float(np.nanmean(f0_valid)) if len(f0_valid) > 0 else np.nan,
                    "f0_std": float(np.nanstd(f0_valid)) if len(f0_valid) > 0 else np.nan,
                    "f0_range": float(np.ptp(f0_valid)) if len(f0_valid) > 0 else np.nan,
                    "jitter": float(parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)) if len(f0_valid) > 10 else np.nan,
                    "shimmer": float(parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)) if len(f0_valid) > 10 else np.nan,
                    "hnr": float(np.mean(harmonicity.values[harmonicity.values != -200])) if hasattr(harmonicity, 'values') and len(harmonicity.values[harmonicity.values != -200]) > 0 else np.nan,
                    "spectral_centroid": float(parselmouth.praat.call(spectrum, "Get centre of gravity", 2)) if hasattr(spectrum, 'values') else np.nan,
                    "spectral_rolloff": np.nan
                }
                rows.append(row_data)
            except Exception:
                rows.append({"path": path, **{col: np.nan for col in ['f0_mean', 'f0_std', 'f0_range', 'jitter', 'shimmer', 'hnr', 'spectral_centroid', 'spectral_rolloff']}})
        acoustic_df = pd.DataFrame(rows).fillna(np.nan)
        acoustic_df.to_csv(cache_path, index=False)
        print("  -> Acoustic features computed and cached (limited to acoustic_limit).")

    acoustic_cols = [c for c in acoustic_df.columns if c != "path"]

    # normalize and map manifest paths
    def normalize_path(p):
        try:
            if not isinstance(p, str) or p == "":
                return ""
            return os.path.normpath(os.path.abspath(p))
        except Exception:
            return str(p)

    acoustic_df['path'] = acoustic_df['path'].astype(str).apply(lambda p: normalize_path(p))
    manifest_paths = []
    for entry in manifest_dev:
        if isinstance(entry, dict):
            p = entry.get("path", "")
        elif isinstance(entry, (list, tuple)):
            p = entry[0] if len(entry) > 0 else ""
        else:
            p = str(entry)
        manifest_paths.append(normalize_path(p))

    # intersection
    acoustic_paths_set = set(acoustic_df['path'].tolist())
    manifest_paths_set = set(manifest_paths)
    intersect_paths = list(acoustic_paths_set.intersection(manifest_paths_set))
    if len(intersect_paths) == 0:
        print("  -> No overlap between manifest paths and computed acoustic features; skipping correlation.")
        return

    # preserve manifest order for indices
    path_to_dev_idx = {p: i for i, p in enumerate(manifest_paths)}
    common_paths_sorted = sorted(intersect_paths, key=lambda x: path_to_dev_idx.get(x, 0))
    common_indices = [path_to_dev_idx[p] for p in common_paths_sorted]

    # Subset A_dev to the common_indices
    A_common = A_dev[common_indices]

    # subset acoustic df
    acoustic_df_small = acoustic_df.set_index("path")
    # compute correlations
    top_feats = sorted_features.head(top_k)["feature_index"].astype(int).values
    corr_records = []
    for feat in top_feats:
        if feat < 0 or feat >= A_common.shape[1]:
            # out-of-range feature; skip
            for ac in acoustic_cols:
                corr_records.append({"feature": int(feat), "acoustic": ac, "r": np.nan, "p": np.nan})
            continue
        feat_vals = A_common[:, feat].astype(float)
        # ensure finite values for feature (should be)
        for ac in acoustic_cols:
            try:
                ac_vals = acoustic_df_small.loc[common_paths_sorted, ac].values.astype(float)
            except Exception:
                ac_vals = np.array([np.nan] * len(common_paths_sorted))
            # filter pairs where both are finite
            valid_mask = np.isfinite(feat_vals) & np.isfinite(ac_vals)
            if valid_mask.sum() < 3:
                r, p = np.nan, np.nan
            else:
                try:
                    r, p = pearsonr(feat_vals[valid_mask], ac_vals[valid_mask])
                except Exception:
                    r, p = np.nan, np.nan
            corr_records.append({"feature": int(feat), "acoustic": ac, "r": float(r) if np.isfinite(r) else np.nan, "p": float(p) if np.isfinite(p) else np.nan})

    corr_df = pd.DataFrame(corr_records)
    corr_df.to_csv(os.path.join(outdir, "feature_acoustic_correlations.csv"), index=False)

    # Plot heatmap (features x acoustic) with NaNs masked
    heat = corr_df.pivot(index='feature', columns='acoustic', values='r')
    fig = plt.figure(figsize=(12, max(4, heat.shape[0] * 0.25)))
    ax = fig.add_subplot(111)
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="vlag", center=0, cbar_kws={'label': 'Pearson r'}, mask=heat.isna())
    ax.set_title("Correlation between top SAE features and acoustic attributes")
    save_fig(fig, os.path.join(outdir, "figure11_feature_acoustic_correlation"))
    print("  -> Acoustic correlation plotting complete (see saved figures).")



def main(run_dir: str, device: str):
    """Main function to load data and orchestrate the  analysis pipeline."""
    print(f"===  Interpretability Analysis for Run: {run_dir} ===")
    out_root = ensure_dir(os.path.join(run_dir, "interpretability_analysis"))

    # --- Load Data and Models ---
    print("Loading data and models...")
    try:
        data = np.load(os.path.join(run_dir, "interpretability_data.npz"), allow_pickle=True)
        # Expect keys: Z_all, Z_edit, U, labels, sets, manifest, forget_spks
        Z_all, Z_edit = data['Z_all'], data['Z_edit']
        U, labels, sets = data['U'], data['labels'], data['sets']
        manifest, forget_spks = data['manifest'], data['forget_spks'].tolist()

        whitener = np.load(os.path.join(run_dir, "whitener.npz"))
        mu, W = whitener['mu'], whitener['W']

        # Load SAE model
        sae = SAE(input_dim=Z_all.shape[1], hidden_mult=10.0, k=128)
        sae.load_state_dict(torch.load(os.path.join(run_dir, "sae_state_dict.pt"), map_location=device))
        sae.to(device).eval()

        print(f"  -> Loaded {Z_all.shape[0]} embeddings with {Z_all.shape[1]} dimensions")
        print(f"  -> Forget speakers: {len(forget_spks)}, Total speakers: {len(np.unique(labels))}")

    except FileNotFoundError as e:
        print(f"Error: A required data or model file was not found: {e}")
        print("Please ensure you have run data generation and saved required files (interpretability_data.npz, whitener.npz, sae_state_dict.pt).")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return

    print("Computing SAE activations for all embeddings...")
    A_all = sae_encode_batch(sae, Z_all, device=device, batch=8192)
    print(f"  -> Generated {A_all.shape[0]} x {A_all.shape[1]} activation matrix")

    # --- Run  Analysis Pipeline ---
    print("\n" + "=" * 60)
    print("RUNNING  ANALYSIS PIPELINE")
    print("=" * 60)

    sorted_features = analyze_feature_discriminability(A_all, sets, forget_spks, labels, out_root)
    analyze_latent_space_activations(A_all, sets, labels, manifest, forget_spks, out_root)
    visualize_subspace_edit(Z_all, Z_edit, U, sets, labels, forget_spks, out_root)
    quantify_geometric_impact(Z_all, Z_edit, sets, labels, forget_spks, out_root)
    run_causal_interventions(sae, Z_all, mu, W, sets, labels, forget_spks, sorted_features, out_root, device=device)
    correlate_with_acoustic_features(sorted_features, A_all, manifest, sets, out_root)

    # Generate summary report
    summary_path = os.path.join(out_root, "analysis_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Dataset Statistics:\n")
        f.write(f"  - Embedding dimension: {Z_all.shape[1]}\n")
        f.write(f"  - SAE features: {A_all.shape[1]}\n")
        f.write(f"  - Forget speakers: {len(forget_spks)}\n")
        f.write(f"  - Total speakers: {len(np.unique(labels))}\n\n")
        f.write(f"  - Check folder: {out_root}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="./out_voxceleb", help="Run directory containing interpretability data and model.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    args = parser.parse_args()
    main(args.run_dir, args.device)
