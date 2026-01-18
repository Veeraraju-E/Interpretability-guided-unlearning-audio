"""
Interpretation protocol module (Steps 1-6).
Implements the full protocol from main-interpretation.tex.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from src.core.encoders import extract_embeddings
from src.core.acoustic_features import extract_acoustic_features_batch
from src.core.evals.trait_probes import fit_trait_probes, evaluate_probes, compute_leakage_ratios
from src.core.evals.attribute_directions import (
    compute_attribute_directions, compute_projections, compute_wasserstein_distances
)
from src.core.utils.whitening import invert_whitener
from src.datasets.adapter_base import DatasetAdapter


def load_run_artifacts(run_dir: str) -> Dict:
    """Load artifacts from a previous run directory."""
    run_path = Path(run_dir)
    
    artifacts = {}
    
    interpretability_path = run_path / "interpretability_data.npz"
    if interpretability_path.exists():
        data = np.load(interpretability_path, allow_pickle=True)
        artifacts['Z_all'] = data['Z_all']
        artifacts['Z_edit'] = data['Z_edit']
        artifacts['U'] = data['U']
        artifacts['labels'] = data['labels'].tolist()
        artifacts['sets'] = data['sets'].tolist()
        artifacts['manifest'] = data['manifest'].tolist()
        artifacts['forget_spks'] = data['forget_spks'].tolist()
    
    whitener_path = run_path / "whitener.npz"
    if whitener_path.exists():
        data = np.load(whitener_path)
        artifacts['mu'] = data['mu']
        artifacts['W'] = data['W']
    
    return artifacts


def determine_trait_types(traits: np.ndarray, train_mask: np.ndarray, 
                          discrete_threshold: int = 10) -> List[str]:
    """
    Determine if traits are continuous or discrete based on unique values.
    """
    n_traits = traits.shape[1]
    trait_types = []
    
    for k in range(n_traits):
        unique_vals = np.unique(traits[train_mask, k])
        if len(unique_vals) <= discrete_threshold:
            trait_types.append('discrete')
        else:
            trait_types.append('continuous')
    
    return trait_types


def run_interpretation_protocol(
    run_dir: str,
    adapter: DatasetAdapter,
    encoder_cfg: Dict,
    device: str = 'cuda',
    lambda_reg: float = 1e-6,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Run the full interpretation protocol (Steps 1-6).
    
    Args:
        run_dir: Path to run directory with saved artifacts
        adapter: Dataset adapter for loading audio
        encoder_cfg: Encoder configuration
        device: Device to use
        lambda_reg: Ridge regularization for probes
        output_dir: Directory to save results (default: run_dir/interpretation)
    
    Returns:
        Dictionary with all results
    """
    print(f"[Step 0] Loading artifacts from {run_dir}")
    artifacts = load_run_artifacts(run_dir)
    
    if 'labels' not in artifacts or 'manifest' not in artifacts:
        raise ValueError(f"Missing required artifacts in {run_dir}")
    
    labels = artifacts['labels']
    manifest = artifacts['manifest']
    forget_speakers = artifacts.get('forget_spks', [])
    
    if len(forget_speakers) == 0:
        raise ValueError("No forget speakers found in artifacts")
    
    forget_speaker = str(forget_speakers[0])
    print(f"[Step 0] Forget speaker: {forget_speaker}")
    
    print(f"[Step 1] Extracting acoustic features for {len(manifest)} utterances")
    traits, trait_names = extract_acoustic_features_batch(manifest, target_sr=16000, n_mfcc=13)
    print(f"[Step 1] Extracted {len(trait_names)} features: {trait_names[:5]}...")
    
    print(f"[Step 2] Creating speaker-disjoint splits from existing sets")
    sets = artifacts.get('sets', None)
    if sets is None:
        raise ValueError("'sets' field not found in artifacts. Cannot create speaker-disjoint splits.")
    
    sets = np.array([str(s) for s in sets])
    
    # Reuse existing sets structure: retain:dev -> retain_train, retain:test -> retain_test
    # Forget samples (both dev and test) are used only for evaluation
    forget_mask = np.array([s.startswith('forget:') for s in sets], dtype=bool)
    retain_train_mask = np.array([s == 'retain:dev' for s in sets], dtype=bool)
    retain_test_mask = np.array([s == 'retain:test' for s in sets], dtype=bool)
    
    splits = {
        'forget': forget_mask,
        'retain_train': retain_train_mask,
        'retain_test': retain_test_mask
    }
    
    n_forget = splits['forget'].sum()
    n_retain_train = splits['retain_train'].sum()
    n_retain_test = splits['retain_test'].sum()
    print(f"[Step 2] Forget: {n_forget}, Retain train: {n_retain_train}, Retain test: {n_retain_test}")
    
    if n_forget == 0:
        raise ValueError(f"No samples found for forget speaker '{forget_speaker}'")
    if n_retain_train == 0:
        raise ValueError("No retain training samples found")
    if n_retain_test == 0:
        raise ValueError("No retain test samples found")
    
    print(f"[Step 3] Computing pre- and post-edit embeddings")
    
    if 'Z_all' in artifacts and 'mu' in artifacts and 'W' in artifacts:
        Z_all = artifacts['Z_all']
        mu = artifacts['mu']
        W = artifacts['W']
        embeddings_pre = invert_whitener(Z_all, mu, W)
        embeddings_pre = embeddings_pre / (np.linalg.norm(embeddings_pre, axis=1, keepdims=True) + 1e-12)
    else:
        print("[Step 3] Pre-edit embeddings not found in artifacts, extracting from audio...")
        embeddings_pre, _, _ = extract_embeddings(manifest, device=device, **encoder_cfg)
        embeddings_pre = embeddings_pre / (np.linalg.norm(embeddings_pre, axis=1, keepdims=True) + 1e-12)
    
    if 'Z_edit' in artifacts:
        embeddings_post = artifacts['Z_edit']
        embeddings_post = embeddings_post / (np.linalg.norm(embeddings_post, axis=1, keepdims=True) + 1e-12)
    else:
        raise ValueError("Z_edit not found in artifacts")
    
    print(f"[Step 3] Embeddings shape: pre={embeddings_pre.shape}, post={embeddings_post.shape}")
    
    print(f"[Step 4] Fitting trait probes on retain speakers")
    train_mask = splits['retain_train']
    trait_types = determine_trait_types(traits, train_mask)
    
    probes = fit_trait_probes(
        embeddings_pre, embeddings_post, traits, train_mask,
        lambda_reg=lambda_reg, trait_types=trait_types
    )
    print(f"[Step 4] Fitted {len(probes['probes_pre'])} probes")
    
    print(f"[Step 5] Evaluating probes and computing leakage ratios")
    metrics_retain_test_pre = evaluate_probes(probes, embeddings_pre, traits, splits['retain_test'], 'pre')
    metrics_retain_test_post = evaluate_probes(probes, embeddings_post, traits, splits['retain_test'], 'post')
    
    metrics_forget_pre = evaluate_probes(probes, embeddings_pre, traits, splits['forget'], 'pre')
    metrics_forget_post = evaluate_probes(probes, embeddings_post, traits, splits['forget'], 'post')
    
    leak_ratios = compute_leakage_ratios(metrics_forget_pre, metrics_forget_post, trait_types)
    
    print(f"[Step 5] Leakage ratios computed for {len(leak_ratios)} traits")
    
    print(f"[Step 6] Computing attribute directions and Wasserstein distances")
    directions_pre = compute_attribute_directions(probes, 'pre')
    directions_post = compute_attribute_directions(probes, 'post')
    
    projections_pre = compute_projections(embeddings_pre, directions_pre)
    projections_post = compute_projections(embeddings_post, directions_post)
    
    w1_forget = compute_wasserstein_distances(projections_pre, projections_post, splits['forget'])
    w1_retain_test = compute_wasserstein_distances(projections_pre, projections_post, splits['retain_test'])
    
    print(f"[Step 6] Computed Wasserstein distances for {len(w1_forget)} traits")
    
    results = {
        'trait_names': trait_names,
        'trait_types': trait_types,
        'forget_speaker': forget_speaker,
        'n_samples': {
            'forget': int(n_forget),
            'retain_train': int(n_retain_train),
            'retain_test': int(n_retain_test)
        },
        'retain_test_metrics': {
            'pre': {
                'r2': metrics_retain_test_pre['r2'].tolist(),
                'auc': metrics_retain_test_pre['auc'].tolist()
            },
            'post': {
                'r2': metrics_retain_test_post['r2'].tolist(),
                'auc': metrics_retain_test_post['auc'].tolist()
            }
        },
        'forget_metrics': {
            'pre': {
                'r2': metrics_forget_pre['r2'].tolist(),
                'auc': metrics_forget_pre['auc'].tolist()
            },
            'post': {
                'r2': metrics_forget_post['r2'].tolist(),
                'auc': metrics_forget_post['auc'].tolist()
            }
        },
        'leakage_ratios': leak_ratios.tolist(),
        'wasserstein_distances': {
            'forget': w1_forget.tolist(),
            'retain_test': w1_retain_test.tolist()
        }
    }
    
    if output_dir is None:
        output_dir = os.path.join(run_dir, "interpretation")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "interpretation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    np.savez_compressed(
        os.path.join(output_dir, "interpretation_data.npz"),
        traits=traits,
        embeddings_pre=embeddings_pre,
        embeddings_post=embeddings_post,
        projections_pre=projections_pre,
        projections_post=projections_post,
        directions_pre=directions_pre,
        directions_post=directions_post
    )
    
    print(f"[Done] Results saved to {output_dir}")
    
    return results
