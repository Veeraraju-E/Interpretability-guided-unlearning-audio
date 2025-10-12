#!/usr/bin/env python3
"""
SAE Training Loss Analysis
Creates a unified script for SAE hyperparameter grid search that records validation MSE loss curves
and generates publication-quality visualization with 4 subplots showing loss vs epochs for each alpha value.
"""

import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.datasets.voxceleb_adapter import VoxCelebAdapter
from src.core.utils.seed import seed_all
from src.core.sae.model import SAE
from src.core.sae.train import train_sae

# Set publication-quality style matching the project theme
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 6,
    'axes.titlesize': 7,
    'axes.labelsize': 6,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 5,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
})


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def cache_embeddings(adapter, cfg: Dict, device: str) -> Tuple:
    """
    Cache embeddings to avoid recomputation across hyperparameter combinations.
    Returns (pre, classes) where pre contains all precomputed data.
    """
    base_save_dir = cfg['experiment']['save_dir']
    os.makedirs(base_save_dir, exist_ok=True)
    
    embeddings_cache_path = os.path.join(base_save_dir, "sae_embeddings_cache.pkl")
    
    if os.path.exists(embeddings_cache_path):
        print("Loading cached embeddings...")
        with open(embeddings_cache_path, 'rb') as f:
            cache_data = pickle.load(f)
            pre = cache_data['pre']
            classes = cache_data['classes']
        print(f"Loaded cached embeddings: {pre.E_all.shape[0]} samples, {pre.E_all.shape[1]} dimensions")
    else:
        print("Computing embeddings (this will be cached for future runs)...")
        pre = adapter.precompute(
            root_or_manifest=cfg['dataset']['root_or_manifest'],
            embeddings_path=cfg['encoder'].get('embeddings_path'),
            dev_frac=cfg['dataset']['dev_frac'],
            seed=cfg['experiment']['seed'],
            encoder_cfg=cfg['encoder'],
            device=device,
        )
        classes = pre.classes
        
        # Cache embeddings
        cache_data = {'pre': pre, 'classes': classes}
        with open(embeddings_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Embeddings cached successfully: {pre.E_all.shape[0]} samples, {pre.E_all.shape[1]} dimensions")
    
    return pre, classes


def run_training_analysis(cfg: Dict, verbose: bool = False) -> Dict:
    """
    Run SAE training across all hyperparameter combinations and record loss histories.
    Returns nested dictionary with results.
    """
    device = cfg['experiment']['device']
    seed_all(cfg['experiment']['seed'])
    
    # Define hyperparameter combinations
    expansion_factors = [2, 4, 8, 16]
    topk_values = [1, 2, 4, 8, 16, 32, 64, 128]
    
    print(f"SAE Training Loss Analysis Configuration:")
    print(f"  - Expansion factors (α): {expansion_factors}")
    print(f"  - TopK values (k): {topk_values}")
    print(f"  - Total combinations: {len(expansion_factors) * len(topk_values)}")
    print(f"  - Epochs per training: {cfg['sae']['epochs']}")
    print(f"  - Device: {device}")
    print(f"  - Dataset: {cfg['dataset']['name']}")
    print(f"  - Encoder: {cfg['encoder']['name']}")
    
    # Initialize adapter and cache embeddings
    adapter = VoxCelebAdapter()
    pre, classes = cache_embeddings(adapter, cfg, device)
    
    # Set up data for SAE training
    Z_all = pre.Z_all
    mask_dev = pre.mask_dev
    
    # Split dev data into train/val for SAE training
    Z_train = Z_all[mask_dev]
    perm = np.random.permutation(Z_train.shape[0])
    cut = int(0.8 * len(perm))
    Z_sae_train, Z_sae_val = Z_train[perm[:cut]], Z_train[perm[cut:]]
    
    print(f"SAE training data: {Z_sae_train.shape[0]} train, {Z_sae_val.shape[0]} validation samples")
    
    # Initialize results dictionary
    results = {}
    total_combinations = len(expansion_factors) * len(topk_values)
    
    print(f"\nStarting SAE training analysis...")
    combination_num = 0
    
    for alpha in expansion_factors:
        results[alpha] = {}
        for k in topk_values:
            combination_num += 1
            print(f"\n[{combination_num}/{total_combinations}] Training SAE: α={alpha}, k={k}")
            
            try:
                # Create SAE model
                sae = SAE(
                    input_dim=Z_all.shape[1], 
                    hidden_mult=alpha, 
                    k=k, 
                    variant='topk', 
                    hidden_dim=cfg['sae']['hidden_dim']
                )
                
                # Train SAE with loss history tracking
                sae, loss_history = train_sae(
                    sae, Z_sae_train, Z_sae_val, 
                    epochs=cfg['sae']['epochs'], 
                    batch_size=cfg['sae']['batch_size'],
                    lr=1e-4,
                    device=device, 
                    return_loss_history=True
                )
                
                # Store results
                results[alpha][k] = {
                    'epochs': loss_history['epochs'],
                    'val_mse': loss_history['val_loss']
                }
                
                if verbose:
                    print(f"    Final validation MSE: {loss_history['val_loss'][-1]:.6f}")
                else:
                    print(f"    Final validation MSE: {loss_history['val_loss'][-1]:.6f}")
                    
            except Exception as e:
                print(f"    ERROR: {str(e)}")
                results[alpha][k] = {
                    'epochs': [],
                    'val_mse': [],
                    'error': str(e)
                }
    
    # Save results
    output_dir = cfg['experiment']['save_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, 'sae_training_loss_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nSAE training analysis completed!")
    print(f"Results saved to: {results_path}")
    print(f"Total combinations tested: {total_combinations}")
    
    return results


def create_visualization(cfg: Dict, results: Dict = None) -> None:
    """
    Create publication-quality visualization of SAE training loss curves.
    """
    # Load results if not provided
    if results is None:
        output_dir = cfg['experiment']['save_dir']
        results_path = os.path.join(output_dir, 'sae_training_loss_results.pkl')
        
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found: {results_path}. Run training first.")
        
        print(f"Loading results from: {results_path}")
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
    
    # Define hyperparameter values
    expansion_factors = [2, 4, 8, 16]
    topk_values = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 6.0))
    fig.suptitle('SAE Validation Loss', fontsize=8, fontweight='bold', y=0.98)
    
    # Define colors for each k value (8 distinct professional colors)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    color_map = {k: colors[i] for i, k in enumerate(topk_values)}
    
    # Plot each alpha value
    for idx, alpha in enumerate(expansion_factors):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Plot each k value for current alpha
        for k in topk_values:
            if alpha in results and k in results[alpha]:
                data = results[alpha][k]
                if 'error' not in data and len(data['epochs']) > 0:
                    ax.plot(data['epochs'], data['val_mse'], 
                           marker='D', 
                           linewidth=1.5, 
                           markersize=2,
                           markevery=25,  # Plot every 25th epoch for clarity
                           color=color_map[k],
                           markerfacecolor=color_map[k],
                           markeredgecolor=color_map[k],
                           markeredgewidth=0.5,
                           alpha=0.8,
                           label=f'k = {k}')
        
        # Format subplot
        ax.set_title(f'α = {alpha}', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation MSE')
        ax.grid(True, alpha=0.3)
        
        # Set axis limits
        ax.set_xlim(0, 500)
        
        # Set y-axis to log scale with adaptive range
        # ax.set_yscale('log')
        
        # Add legend to each subplot (compact)
        ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False, 
                 fontsize=4, markerscale=0.8, handlelength=1.5, ncol=2)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.3)
    
    # Save figures
    output_dir = cfg['experiment']['save_dir']
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as PNG
    png_path = output_path / 'sae_training_loss_curves.png'
    plt.savefig(png_path, dpi=600, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved PNG figure: {png_path}")
    
    print("Visualization completed successfully!")


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description='SAE Training Loss Analysis')
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to YAML configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'plot', 'all'], default='all', help='Mode: train (run training only), plot (generate plot only), all (both)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    print(f"SAE Training Loss Analysis - Mode: {args.mode}")
    print(f"Configuration: {args.config}")
    
    results = None
    
    # Run training if requested
    if args.mode in ['train', 'all']:
        print("\n" + "="*60)
        print("RUNNING TRAINING ANALYSIS")
        print("="*60)
        results = run_training_analysis(cfg, verbose=args.verbose)
    
    # Generate visualization if requested
    if args.mode in ['plot', 'all']:
        print("\n" + "="*60)
        print("GENERATING VISUALIZATION")
        print("="*60)
        create_visualization(cfg, results)
    
    print(f"\nAnalysis complete! Check results in: {cfg['experiment']['save_dir']}")


if __name__ == "__main__":
    main()
