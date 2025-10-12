import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import argparse

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.datasets.voxceleb_adapter import VoxCelebAdapter
from src.core.utils.seed import seed_all
from src.core.sae.model import SAE
from src.core.sae.train import train_sae
from src.core.sae.encode import sae_encode_batch
from src.core.utils.stats import compute_feature_stats, select_features
from src.core.utils.subspace import build_U_from_decoder
from src.core.evals.classification import train_classifier, evaluate_classifier
from src.core.evals.probes import safe_linear_probe


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
    
    embeddings_cache_path = os.path.join(base_save_dir, "hyperparameter_analysis_embeddings.pkl")
    
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


def evaluate_hyperparameter_combination(pre, classes, cfg: Dict, expansion_factor: float, topk: int, device: str, baseline_clf, baseline_FA: float, baseline_RA: float, verbose: bool = False) -> Dict:
    """
    Evaluate a single hyperparameter combination (expansion_factor, topk).
    Returns FA and RA metrics.
    """
    if verbose:
        print(f"  Evaluating: expansion_factor={expansion_factor}, topk={topk}")
    
    # Set up data
    Z_all = pre.Z_all
    all_labels = pre.labels  # Use same naming as single_run
    mask_dev = pre.mask_dev
    mask_test = pre.mask_test
    
    # Create forget/retain masks for the first forget label (following single_run pattern)
    forget_label = cfg['dataset']['forget_labels'][0]
    forget_label_str = str(forget_label)
    
    if verbose:
        print(f"    Debug: forget_label = {forget_label_str}")
        print(f"    Debug: all_labels sample = {all_labels[:5]}")
        print(f"    Debug: all_labels types = {[type(lbl) for lbl in all_labels[:5]]}")
    
    # Create sets following single_run pattern
    sets = [f"{'forget' if str(lbl) == forget_label_str else 'retain'}:{'dev' if mask_dev[i] else 'test'}" for i, lbl in enumerate(all_labels)]
    
    M_ret_dev = np.array([s == 'retain:dev' for s in sets], dtype=bool)
    M_fgt_dev = np.array([s == 'forget:dev' for s in sets], dtype=bool)
    M_ret_test = np.array([s == 'retain:test' for s in sets], dtype=bool)
    M_fgt_test = np.array([s == 'forget:test' for s in sets], dtype=bool)
    
    if verbose:
        print(f"    Debug: M_ret_dev samples: {M_ret_dev.sum()}, M_fgt_dev samples: {M_fgt_dev.sum()}")
        print(f"    Debug: M_ret_test samples: {M_ret_test.sum()}, M_fgt_test samples: {M_fgt_test.sum()}")
    
    # Train SAE with current hyperparameters
    Z_train = Z_all[mask_dev]
    perm = np.random.permutation(Z_train.shape[0])
    cut = int(0.8 * len(perm))
    Z_sae_train, Z_sae_val = Z_train[perm[:cut]], Z_train[perm[cut:]]
    
    sae = SAE(
        input_dim=Z_all.shape[1], 
        hidden_mult=expansion_factor, 
        k=topk, 
        variant='topk', 
        hidden_dim=cfg['sae']['hidden_dim']
    )
    
    # Train SAE
    sae = train_sae(
        sae, Z_sae_train, Z_sae_val, 
        epochs=cfg['sae']['epochs'], 
        device=device, 
        return_loss_history=False
    )
    
    # Get SAE activations
    A_all = sae_encode_batch(sae, Z_all, device=device)
    
    # Feature mining (following single_run pattern)
    A_ret = A_all[M_ret_dev]
    A_fgt = A_all[M_fgt_dev]
    
    if verbose:
        print(f"    Debug: A_ret shape: {A_ret.shape}, A_fgt shape: {A_fgt.shape}")
        print(f"    Debug: A_ret activation stats - mean: {A_ret.mean():.6f}, std: {A_ret.std():.6f}")
        print(f"    Debug: A_fgt activation stats - mean: {A_fgt.mean():.6f}, std: {A_fgt.std():.6f}")
    
    # Check if we have enough samples
    if M_fgt_dev.sum() < 5 or M_ret_dev.sum() < 5:
        print(f"    WARNING: Insufficient samples for feature selection (forget: {M_fgt_dev.sum()}, retain: {M_ret_dev.sum()})")
        return {
            'expansion_factor': expansion_factor,
            'topk': topk,
            'FA_before': float('nan'),
            'FA_after': float('nan'),
            'RA_before': float('nan'),
            'RA_after': float('nan'),
            'linear_probe_auc_before': float('nan'),
            'linear_probe_auc_after': float('nan'),
            'forget_label': forget_label,
            'error': 'Insufficient samples for feature selection'
        }
    
    # Get statistical method from config, default to cohens_d
    method = cfg['feature_mining'].get('statistical_method', 'cohens_d')
    stats = compute_feature_stats(A_fgt, A_ret, method=method, verbose=verbose)
    feat_idx = select_features(stats, method=method, top_k=cfg['feature_mining']['build_U_top'], fdr=0.05, verbose=verbose)
    
    if len(feat_idx) == 0:
        print(f"    ERROR: No discriminative features found for forget class '{forget_label_str}'. Cannot proceed with editing.")
        return {
            'expansion_factor': expansion_factor,
            'topk': topk,
            'FA_before': float('nan'),
            'FA_after': float('nan'),
            'RA_before': float('nan'),
            'RA_after': float('nan'),
            'linear_probe_auc_before': float('nan'),
            'linear_probe_auc_after': float('nan'),
            'forget_label': forget_label,
            'error': 'No discriminative features found'
        }
    
    # Build subspace and apply edits (following single_run pattern)
    dec_w = sae.dec.weight.detach().cpu().numpy()
    from src.core.utils.subspace import prune_and_normalize_decoder
    prune_and_normalize_decoder(dec_w)
    U = build_U_from_decoder(dec_w, feat_idx)
    
    # LEACE params - create labels for the full dataset (following single_run pattern)
    Y_all = np.array([1 if all_labels[i] == forget_label else 0 for i in range(len(all_labels))], dtype=int).reshape(-1, 1)
    
    if verbose:
        print(f"    Debug: Y_all shape: {Y_all.shape}, Y_all unique values: {np.unique(Y_all)}")
        print(f"    Debug: forget_label type: {type(forget_label)}, all_labels[0] type: {type(all_labels[0])}")
    
    # Apply edits using the same function as single_run
    from src.core.edits.apply import apply_edits
    E_edit, info = apply_edits(
        pre.Z_all, sae, feat_idx, U, pre.mu, pre.W,
        beta=cfg['edits']['beta'], do_leace=cfg['edits']['leace'], 
        leace_params={'Y_dev': Y_all, 'ridge': cfg['edits']['leace_ridge']}, 
        noise_sigma=cfg['edits']['noise_sigma'], device=device
    )
    E_all_edited = E_edit
    E_all_norm = pre.E_all / (np.linalg.norm(pre.E_all, axis=1, keepdims=True) + 1e-12)
    E_all_norm2 = E_all_edited / (np.linalg.norm(E_all_edited, axis=1, keepdims=True) + 1e-12)
    
    # Use the baseline classifier (no need to retrain)
    cls2id = {c: i for i, c in enumerate(classes)}
    
    # Evaluate classification accuracy using baseline classifier
    E_ret_test = E_all_norm[M_ret_test]
    L_ret_test = [cls2id[d] for d in np.array(all_labels)[M_ret_test]]
    E_fgt_test = E_all_norm[M_fgt_test]
    L_fgt_test = [cls2id[d] for d in np.array(all_labels)[M_fgt_test]]
    E_ret_test_edit = E_all_norm2[M_ret_test]
    E_fgt_test_edit = E_all_norm2[M_fgt_test]
    
    # Calculate FA and RA using baseline classifier
    acc_ret_before = baseline_RA  # Use precomputed baseline
    acc_fgt_before = baseline_FA  # Use precomputed baseline
    acc_ret_after = evaluate_classifier(baseline_clf, E_ret_test_edit, L_ret_test, device=device)
    acc_fgt_after = evaluate_classifier(baseline_clf, E_fgt_test_edit, L_fgt_test, device=device)
    
    # Additional metrics for completeness
    split_mask_dict = {'dev': mask_dev, 'test': mask_test}
    _, auc_before = safe_linear_probe(E_all_norm, all_labels, U, split_mask_dict, [forget_label])
    _, auc_after = safe_linear_probe(E_all_norm2, all_labels, U, split_mask_dict, [forget_label])
    
    return {
        'expansion_factor': expansion_factor,
        'topk': topk,
        'FA_before': acc_fgt_before,
        'FA_after': acc_fgt_after,
        'RA_before': acc_ret_before,
        'RA_after': acc_ret_after,
        'linear_probe_auc_before': auc_before,
        'linear_probe_auc_after': auc_after,
        'forget_label': forget_label
    }


def run_hyperparameter_analysis(cfg: Dict, verbose: bool = False) -> pd.DataFrame:
    """
    Run hyperparameter analysis for the specified expansion factors and topk values.
    """
    device = cfg['experiment']['device']
    seed_all(cfg['experiment']['seed'])
    
    # Define hyperparameter combinations
    expansion_factors = [2, 4, 8, 16]
    topk_values = [1, 2, 4, 8, 16, 32, 64, 128]
    
    print(f"Hyperparameter Analysis Configuration:")
    print(f"  - Expansion factors: {expansion_factors}")
    print(f"  - TopK values: {topk_values}")
    print(f"  - Total combinations: {len(expansion_factors) * len(topk_values)}")
    print(f"  - Device: {device}")
    print(f"  - Dataset: {cfg['dataset']['name']}")
    print(f"  - Encoder: {cfg['encoder']['name']}")
    print(f"  - Forget label: {cfg['dataset']['forget_labels'][0]}")
    
    # Initialize adapter and cache embeddings
    adapter = VoxCelebAdapter()
    pre, classes = cache_embeddings(adapter, cfg, device)
    
    # Train classifier ONCE at the beginning (this ensures consistent FA/RA baseline)
    print(f"\nTraining baseline classifier...")
    E_all_norm = pre.E_all / (np.linalg.norm(pre.E_all, axis=1, keepdims=True) + 1e-12)
    all_labels = pre.labels
    cls2id = {c: i for i, c in enumerate(classes)}
    
    X_cls_train = E_all_norm[pre.mask_dev]
    y_cls_train = [cls2id[l] for l in np.array(all_labels)[pre.mask_dev]]
    X_cls_val = E_all_norm[pre.mask_test]
    y_cls_val = [cls2id[l] for l in np.array(all_labels)[pre.mask_test]]
    
    baseline_clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, 
                                   num_classes=len(classes), device=device, 
                                   epochs=cfg['orchestration']['classifier_epochs'], 
                                   batch_size=512, lr=1e-3)
    
    # Compute baseline FA and RA (these should be consistent across all combinations)
    forget_label = cfg['dataset']['forget_labels'][0]
    forget_label_str = str(forget_label)
    sets = [f"{'forget' if str(lbl) == forget_label_str else 'retain'}:{'dev' if pre.mask_dev[i] else 'test'}" for i, lbl in enumerate(all_labels)]
    M_ret_test = np.array([s == 'retain:test' for s in sets], dtype=bool)
    M_fgt_test = np.array([s == 'forget:test' for s in sets], dtype=bool)
    
    E_ret_test = E_all_norm[M_ret_test]
    L_ret_test = [cls2id[d] for d in np.array(all_labels)[M_ret_test]]
    E_fgt_test = E_all_norm[M_fgt_test]
    L_fgt_test = [cls2id[d] for d in np.array(all_labels)[M_fgt_test]]
    
    baseline_FA = evaluate_classifier(baseline_clf, E_fgt_test, L_fgt_test, device=device)
    baseline_RA = evaluate_classifier(baseline_clf, E_ret_test, L_ret_test, device=device)
    
    print(f"Baseline FA: {baseline_FA:.4f}, Baseline RA: {baseline_RA:.4f}")
    
    # Run hyperparameter analysis
    results = []
    total_combinations = len(expansion_factors) * len(topk_values)
    
    print(f"\nStarting hyperparameter analysis...")
    for i, expansion_factor in enumerate(expansion_factors):
        for j, topk in enumerate(topk_values):
            combination_num = i * len(topk_values) + j + 1
            print(f"\n[{combination_num}/{total_combinations}] Testing expansion_factor={expansion_factor}, topk={topk}")
            
            try:
                result = evaluate_hyperparameter_combination(
                    pre, classes, cfg, expansion_factor, topk, device, baseline_clf, 
                    baseline_FA, baseline_RA, verbose
                )
                results.append(result)
                
                if verbose:
                    print(f"    FA_before: {result['FA_before']:.4f}, FA_after: {result['FA_after']:.4f}")
                    print(f"    RA_before: {result['RA_before']:.4f}, RA_after: {result['RA_after']:.4f}")
                else:
                    print(f"    FA: {result['FA_before']:.4f} → {result['FA_after']:.4f}, RA: {result['RA_before']:.4f} → {result['RA_after']:.4f}")
                    
            except Exception as e:
                print(f"    ERROR: {str(e)}")
                results.append({
                    'expansion_factor': expansion_factor,
                    'topk': topk,
                    'FA_before': baseline_FA,
                    'FA_after': float('nan'),
                    'RA_before': baseline_RA,
                    'RA_after': float('nan'),
                    'linear_probe_auc_before': float('nan'),
                    'linear_probe_auc_after': float('nan'),
                    'forget_label': cfg['dataset']['forget_labels'][0],
                    'error': str(e)
                })
    
    # Save results
    results_df = pd.DataFrame(results)
    output_dir = cfg['experiment']['save_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'hyperparameter_analysis_results.csv')
    results_df.to_csv(output_path, index=False)
    
    print(f"\nHyperparameter analysis completed!")
    print(f"Results saved to: {output_path}")
    print(f"Total combinations tested: {len(results_df)}")
    
    # Print summary
    print(f"\nSummary of results:")
    print(results_df[['expansion_factor', 'topk', 'FA_after', 'RA_after']].to_string(index=False))
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Analysis for TopK SAE Module')
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to YAML configuration file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    cfg = load_config(args.config)    
    results_df = run_hyperparameter_analysis(cfg, verbose=args.verbose)
    
    print(f"\nAnalysis complete! Check the results in: {cfg['experiment']['save_dir']}/hyperparameter_analysis_results.csv")


if __name__ == "__main__":
    main()
