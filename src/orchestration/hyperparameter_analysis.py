import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.datasets.voxceleb_adapter import VoxCelebAdapter
from src.core.utils.seed import seed_all
from src.core.sae import SAE, train_sae, sae_encode_batch
from src.core.utils.stats import compute_feature_stats, select_features
from src.core.utils.subspace import build_U_from_decoder
from src.core.evals.classification import train_classifier, evaluate_classifier
from src.core.evals.probes import safe_linear_probe
from src.core.edits import apply_edits
from src.core.visualization.plot_hyperparameter_analysis import create_hyperparameter_plot


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f: return yaml.safe_load(f)

def cache_embeddings(adapter, cfg: Dict, device: str) -> Tuple:
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
        print("Computing embeddings (will be cached)...")
        data_path_param = cfg['dataset']['root_or_manifest']
        if hasattr(adapter, 'name'):
            if adapter.name in ['audiomnist', 'speech_commands']:
                pre = adapter.precompute(
                    data_root=data_path_param,
                    embeddings_path=cfg['encoder'].get('embeddings_path'),
                    dev_frac=cfg['dataset']['dev_frac'],
                    seed=cfg['experiment']['seed'],
                    encoder_cfg=cfg['encoder'],
                    device=device,
                )
            elif adapter.name == 'librispeech':
                pre = adapter.precompute(
                    root=data_path_param,
                    embeddings_path=cfg['encoder'].get('embeddings_path'),
                    dev_frac=cfg['dataset']['dev_frac'],
                    seed=cfg['experiment']['seed'],
                    encoder_cfg=cfg['encoder'],
                    device=device,
                )
            elif adapter.name == 'lahaja':
                pre = adapter.precompute(
                    split=data_path_param,
                    embeddings_path=cfg['encoder'].get('embeddings_path'),
                    dev_frac=cfg['dataset']['dev_frac'],
                    seed=cfg['experiment']['seed'],
                    encoder_cfg=cfg['encoder'],
                    device=device,
                )
            elif adapter.name == 'mls_spanish':
                pre = adapter.precompute(
                    root_or_manifest=data_path_param,
                    embeddings_path=cfg['encoder'].get('embeddings_path'),
                    dev_frac=cfg['dataset']['dev_frac'],
                    seed=cfg['experiment']['seed'],
                    encoder_cfg=cfg['encoder'],
                    device=device,
                )
            else:  # voxceleb and others
                pre = adapter.precompute(
                    root_or_manifest=data_path_param,
                    embeddings_path=cfg['encoder'].get('embeddings_path'),
                    dev_frac=cfg['dataset']['dev_frac'],
                    seed=cfg['experiment']['seed'],
                    encoder_cfg=cfg['encoder'],
                    device=device,
                )
        else:
            raise NotImplementedError(f"Adapter {adapter.name} not implemented for {cfg['dataset']['name']}.")
        classes = pre.classes
        
        # Cache embeddings
        cache_data = {'pre': pre, 'classes': classes}
        with open(embeddings_cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Embeddings cached successfully: {pre.E_all.shape[0]} samples, {pre.E_all.shape[1]} dimensions")
    
    return pre, classes


def evaluate_hyperparameter_combination(pre, classes, cfg: Dict, expansion_factor: float, topk: int, device: str, baseline_clf, baseline_FA: float, baseline_RA: float) -> Dict:
    """
    Evaluate a single hyperparameter combination (expansion_factor, topk).
    Returns FA and RA metrics.
    """
    
    Z_all = pre.Z_all
    all_labels = pre.labels
    mask_dev = pre.mask_dev
    mask_test = pre.mask_test
    
    forget_label = cfg['dataset']['forget_labels'][0]
    forget_label_str = str(forget_label)

    sets = [f"{'forget' if str(lbl) == forget_label_str else 'retain'}:{'dev' if mask_dev[i] else 'test'}" for i, lbl in enumerate(all_labels)]
    
    M_ret_dev = np.array([s == 'retain:dev' for s in sets], dtype=bool)
    M_fgt_dev = np.array([s == 'forget:dev' for s in sets], dtype=bool)
    M_ret_test = np.array([s == 'retain:test' for s in sets], dtype=bool)
    M_fgt_test = np.array([s == 'forget:test' for s in sets], dtype=bool)
    
    Z_train = Z_all[mask_dev]; perm = np.random.permutation(Z_train.shape[0]); cut = int(0.8 * len(perm)); Z_sae_train, Z_sae_val = Z_train[perm[:cut]], Z_train[perm[cut:]]
    
    sae = SAE(input_dim=Z_all.shape[1], hidden_mult=expansion_factor, k=topk, variant='topk', hidden_dim=cfg['sae']['hidden_dim'])
    
    sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=cfg['sae']['epochs'], device=device, return_loss_history=False)
    A_all = sae_encode_batch(sae, Z_all, device=device)
    
    # Feature mining (following single_run pattern)
    A_ret = A_all[M_ret_dev]
    A_fgt = A_all[M_fgt_dev]
    
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
            'dead_latents_pct': float('nan'),
            'error': 'Insufficient samples for feature selection'
        }
    
    # Get statistical method from config, default to cohens_d
    method = cfg['feature_mining'].get('statistical_method', 'cohens_d')
    stats = compute_feature_stats(A_fgt, A_ret, method=method)
    
    # Compute percentage of dead latents (AUC = 0.5 or discriminability = 0)
    auc_scores = stats['auc']
    cohens_d_scores = stats['cohens_d']
    dead_latents = (np.abs(auc_scores - 0.5) < 1e-6) | (np.abs(cohens_d_scores) < 1e-6)
    dead_latents_pct = 100.0 * dead_latents.sum() / len(dead_latents)
    
    feat_idx = select_features(stats, method=method, top_k=cfg['feature_mining']['build_U_top'], fdr=0.05)
    
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
            'dead_latents_pct': dead_latents_pct,
            'error': 'No discriminative features found'
        }
    
    # Build subspace and apply edits
    dec_w = sae.dec.weight.detach().cpu().numpy()
    from src.core.utils.subspace import prune_and_normalize_decoder
    prune_and_normalize_decoder(dec_w)
    U = build_U_from_decoder(dec_w, feat_idx)
    
    # LEACE params - create labels for the full dataset
    Y_all = np.array([1 if str(all_labels[i]) == forget_label_str else 0 for i in range(len(all_labels))], dtype=int).reshape(-1, 1)
    
    # Apply edits
    E_edit, _ = apply_edits(
        pre.Z_all, sae, feat_idx, U, pre.mu, pre.W,
        cfg['edits']['beta'], cfg['edits']['leace'], 
        {'Y_dev': Y_all, 'ridge': cfg['edits']['leace_ridge']}, 
        cfg['edits']['noise_sigma']
    )
    E_all_edited = E_edit
    E_all_norm = pre.E_all / (np.linalg.norm(pre.E_all, axis=1, keepdims=True) + 1e-12)
    E_all_norm2 = E_all_edited / (np.linalg.norm(E_all_edited, axis=1, keepdims=True) + 1e-12)
    
    cls2id = {c: i for i, c in enumerate(classes)}
    
    E_ret_test = E_all_norm[M_ret_test]
    L_ret_test = [cls2id[d] for d in np.array(all_labels)[M_ret_test]]
    E_fgt_test = E_all_norm[M_fgt_test]
    L_fgt_test = [cls2id[d] for d in np.array(all_labels)[M_fgt_test]]
    E_ret_test_edit = E_all_norm2[M_ret_test]
    E_fgt_test_edit = E_all_norm2[M_fgt_test]
    
    # Calculate FA and RA using baseline classifier
    acc_ret_before = baseline_RA
    acc_fgt_before = baseline_FA
    acc_ret_after = evaluate_classifier(baseline_clf, E_ret_test_edit, L_ret_test, device=device)
    acc_fgt_after = evaluate_classifier(baseline_clf, E_fgt_test_edit, L_fgt_test, device=device)
    split_mask_dict = {'dev': mask_dev, 'test': mask_test}
    _, auc_before = safe_linear_probe(E_all_norm, all_labels, U, split_mask_dict, [forget_label_str])
    _, auc_after = safe_linear_probe(E_all_norm2, all_labels, U, split_mask_dict, [forget_label_str])
    
    return {
        'expansion_factor': expansion_factor,
        'topk': topk,
        'FA_before': acc_fgt_before,
        'FA_after': acc_fgt_after,
        'RA_before': acc_ret_before,
        'RA_after': acc_ret_after,
        'linear_probe_auc_before': auc_before,
        'linear_probe_auc_after': auc_after,
        'forget_label': forget_label,
        'dead_latents_pct': dead_latents_pct
    }


def run_hyperparameter_analysis(cfg: Dict) -> None:
    device = cfg['experiment']['device']
    seed_all(cfg['experiment']['seed'])
    
    # Define combinations
    expansion_factors = [2, 4, 8, 16]
    topk_values = [1, 2, 4, 8, 16, 32, 64, 128]
      
    # Initialize adapter and cache embeddings
    adapter = VoxCelebAdapter()
    pre, classes = cache_embeddings(adapter, cfg, device)
    
    # Train classifier ONCE at the beginning
    print(f"\nTraining baseline classifier...")
    E_all_norm = pre.E_all / (np.linalg.norm(pre.E_all, axis=1, keepdims=True) + 1e-12)
    all_labels = pre.labels
    cls2id = {c: i for i, c in enumerate(classes)}
    
    X_cls_train = E_all_norm[pre.mask_dev]
    y_cls_train = [cls2id[l] for l in np.array(all_labels)[pre.mask_dev]]
    X_cls_val = E_all_norm[pre.mask_test]
    y_cls_val = [cls2id[l] for l in np.array(all_labels)[pre.mask_test]]
    
    baseline_clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(classes), device=device, epochs=cfg['orchestration']['classifier_epochs'], batch_size=512, lr=1e-3)
    
    # Compute baseline FA and RA
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
                result = evaluate_hyperparameter_combination(pre, classes, cfg, expansion_factor, topk, device, baseline_clf, baseline_FA, baseline_RA)
                results.append(result)
                
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
                    'dead_latents_pct': float('nan'),
                    'error': str(e)
                })
    
    # Save
    results_df = pd.DataFrame(results)
    output_dir = cfg['experiment']['save_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'hyperparameter_analysis_results.csv')
    results_df.to_csv(output_path, index=False)
    
    print(f"\nHyperparameter analysis completed, results are saved to: {output_path}")

def create_visualization(cfg: Dict) -> None:
    output_dir = cfg['experiment']['save_dir']
    results_path = os.path.join(output_dir, 'hyperparameter_analysis_results.csv')
    
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}. Run analysis first.")
    
    create_hyperparameter_plot(results_path, output_dir)


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Analysis for TopK SAE Module')
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to YAML configuration file')
    parser.add_argument('--mode', type=str, choices=['analyze', 'plot', 'all'], default='all', help='Mode: analyze (run analysis only), plot (generate plot only), all (both)')
    
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    
    # Run analysis if requested
    if args.mode in ['analyze', 'all']:
        print("RUN mode")
        run_hyperparameter_analysis(cfg)
    
    # Generate visualization if requested
    if args.mode in ['plot', 'all']:
        print("PLOT mode")
        create_visualization(cfg)


if __name__ == "__main__":
    main()