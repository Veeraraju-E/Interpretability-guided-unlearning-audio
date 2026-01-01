import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
import csv
from typing import Dict, Tuple

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.datasets.voxceleb_adapter import VoxCelebAdapter
from src.datasets.speech_commands_adapter import SpeechCommandsAdapter
from src.datasets.audiomnist_adapter import AudioMNISTAdapter
from src.datasets.librispeech_adapter import LibriSpeechAdapter
from src.datasets.lahaja_adapter import LahajaAdapter
from src.datasets.mls_spanish_adapter import MLSSpanishAdapter
from src.core.utils.seed import seed_all
from src.core.sae import SAE, train_sae, sae_encode_batch
from src.core.utils.stats import compute_feature_stats, select_features
from src.core.utils.subspace import prune_and_normalize_decoder, build_U_from_decoder
from src.core.evals.classification import train_classifier, evaluate_classifier
from src.core.edits import apply_edits


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f: return yaml.safe_load(f)


def calculate_erb_score(fa_after: float, ra_after: float) -> float:
    """Calculate ERB score: 2*(100-FA)*RA / (100-FA+RA)"""
    fa_percent = fa_after * 100 if fa_after <= 1.0 else fa_after
    ra_percent = ra_after * 100 if ra_after <= 1.0 else ra_after
    Nr = 2 * (100 - fa_percent) * ra_percent
    Dr = (100 - fa_percent) + ra_percent
    return Nr / Dr if Dr else 0.0

def get_encoder_dimensions() -> Dict[str, int]:
    return {'ecapa': 192, 'wav2vec2': 768, 'hubert': 768}


def calculate_build_u_top_values(encoder: str, multiplier: float = 8.0) -> Dict[str, int]:
    base_dim = get_encoder_dimensions()[encoder]
    sae_latent_dim = int(base_dim * multiplier)
    
    percentages = [1, 2, 5, 10, 25]
    return {f"{p}%": int(sae_latent_dim * p / 100) for p in percentages}


def cache_embeddings(adapter, encoder: str, cfg: Dict, device: str, cache_dir: str) -> Tuple:
    # Create cache key (w model name)
    encoder_cfg = cfg['encoder'].copy()
    if encoder == 'wav2vec2':
        model_name = encoder_cfg.get('wav2vec2_model', 'facebook/wav2vec2-base').replace('/', '_').replace('-', '_')
        cache_key = f"embeddings_{encoder}_{model_name}"
    elif encoder == 'hubert':
        model_name = encoder_cfg.get('hubert_model', 'facebook/hubert-base-ls960').replace('/', '_').replace('-', '_')
        cache_key = f"embeddings_{encoder}_{model_name}"
    else:  # ecapa doesn't have model parameter
        cache_key = f"embeddings_{encoder}"
    
    embeddings_cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if os.path.exists(embeddings_cache_path):
        print(f"Loading cached embeddings for {encoder}...")
        with open(embeddings_cache_path, 'rb') as f:
            cache_data = pickle.load(f)
            pre = cache_data['pre']
            classes = cache_data['classes']
        print(f"Loaded cached embeddings: {pre.E_all.shape[0]} samples, {pre.E_all.shape[1]} dimensions")
    else:
        print(f"Computing embeddings for {encoder} (this will be cached)...")
        
        # Update encoder config
        encoder_cfg = cfg['encoder'].copy()
        encoder_cfg['name'] = encoder
        
        # Use appropriate parameter name based on adapter type
        data_path_param = cfg['dataset']['root_or_manifest']
        if hasattr(adapter, 'name'):
            if adapter.name in ['audiomnist', 'speech_commands']:
                pre = adapter.precompute(
                    data_root=data_path_param,
                    embeddings_path=cfg['encoder'].get('embeddings_path'),
                    dev_frac=cfg['dataset']['dev_frac'],
                    seed=cfg['experiment']['seed'],
                    encoder_cfg=encoder_cfg,
                    device=device,
                )
            elif adapter.name == 'librispeech':
                pre = adapter.precompute(
                    root=data_path_param,
                    embeddings_path=cfg['encoder'].get('embeddings_path'),
                    dev_frac=cfg['dataset']['dev_frac'],
                    seed=cfg['experiment']['seed'],
                    encoder_cfg=encoder_cfg,
                    device=device,
                )
            elif adapter.name == 'lahaja':
                pre = adapter.precompute(
                    split=data_path_param,
                    embeddings_path=cfg['encoder'].get('embeddings_path'),
                    dev_frac=cfg['dataset']['dev_frac'],
                    seed=cfg['experiment']['seed'],
                    encoder_cfg=encoder_cfg,
                    device=device,
                )
            elif adapter.name == 'mls_spanish':
                pre = adapter.precompute(
                    root_or_manifest=data_path_param,
                    embeddings_path=cfg['encoder'].get('embeddings_path'),
                    dev_frac=cfg['dataset']['dev_frac'],
                    seed=cfg['experiment']['seed'],
                    encoder_cfg=encoder_cfg,
                    device=device,
                )
            else:  # voxceleb and others
                pre = adapter.precompute(
                    root_or_manifest=data_path_param,
                    embeddings_path=cfg['encoder'].get('embeddings_path'),
                    dev_frac=cfg['dataset']['dev_frac'],
                    seed=cfg['experiment']['seed'],
                    encoder_cfg=encoder_cfg,
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


def cache_sae_and_activations(pre, encoder: str, cfg: Dict, device: str, cache_dir: str) -> Tuple:
    # Create cache key (with model name)
    encoder_cfg = cfg['encoder'].copy()
    if encoder == 'wav2vec2':
        model_name = encoder_cfg.get('wav2vec2_model', 'facebook/wav2vec2-base').replace('/', '_').replace('-', '_')
        cache_key = f"{encoder}_{model_name}"
    elif encoder == 'hubert':
        model_name = encoder_cfg.get('hubert_model', 'facebook/hubert-base-ls960').replace('/', '_').replace('-', '_')
        cache_key = f"{encoder}_{model_name}"
    else:  # ecapa doesn't have model parameter
        cache_key = encoder
    
    sae_cache_path = os.path.join(cache_dir, f"sae_{cache_key}.pkl")
    activations_cache_path = os.path.join(cache_dir, f"activations_{cache_key}.pkl")
    
    if os.path.exists(sae_cache_path) and os.path.exists(activations_cache_path):
        print(f"Loading cached SAE and activations for {encoder}...")
        with open(sae_cache_path, 'rb') as f:
            sae_state = pickle.load(f)
        with open(activations_cache_path, 'rb') as f:
            cache_data = pickle.load(f)
            A_all = cache_data['A_all']
            dec_w = cache_data['dec_w']
        
        # Reconstruct SAE model
        sae = SAE(input_dim=pre.Z_all.shape[1], hidden_mult=cfg['sae']['multiplier'], k=cfg['sae']['topk'], variant=cfg['sae']['variant'], hidden_dim=cfg['sae']['hidden_dim'])
        sae.load_state_dict(sae_state)
        
        print(f"SAE and activations loaded for {encoder}")
    else:
        print(f"Training SAE for {encoder} (this will be cached)...")
        seed_all(cfg['experiment']['seed'])
        
        Z_train = pre.Z_all[pre.mask_dev]
        perm = np.random.permutation(Z_train.shape[0])
        cut = int(0.8 * len(perm)) if len(perm) > 1 else int(0.5 * len(perm))
        Z_sae_train, Z_sae_val = (Z_train, Z_train) if cut <= 0 else (Z_train[perm[:cut]], Z_train[perm[cut:]])
        
        sae = SAE(input_dim=pre.Z_all.shape[1], hidden_mult=cfg['sae']['multiplier'], k=cfg['sae']['topk'], variant=cfg['sae']['variant'], hidden_dim=cfg['sae']['hidden_dim'])
        sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=cfg['sae']['epochs'], batch_size=cfg['sae']['batch_size'], lr=1e-3, device=device, lambda_sparsity=cfg['sae']['lambda_sparsity'])
        A_all = sae_encode_batch(sae, pre.Z_all, device=device, batch=256)
        dec_w = sae.dec.weight.detach().cpu().numpy()
        
        # Cache SAE and activations
        sae_state = sae.state_dict()
        with open(sae_cache_path, 'wb') as f: pickle.dump(sae_state, f)
        with open(activations_cache_path, 'wb') as f: pickle.dump({'A_all': A_all, 'dec_w': dec_w}, f)
        print(f"SAE and activations cached for {encoder}")
    
    return sae, A_all, dec_w


def cache_classifier(pre, classes, encoder: str, cfg: Dict, device: str, cache_dir: str):
    """
    Cache trained classifier to avoid retraining.
    Returns trained classifier.
    """
    # Create cache key that includes model name for consistency
    encoder_cfg = cfg['encoder'].copy()
    if encoder == 'wav2vec2':
        model_name = encoder_cfg.get('wav2vec2_model', 'facebook/wav2vec2-base').replace('/', '_').replace('-', '_')
        cache_key = f"{encoder}_{model_name}"
    elif encoder == 'hubert':
        model_name = encoder_cfg.get('hubert_model', 'facebook/hubert-base-ls960').replace('/', '_').replace('-', '_')
        cache_key = f"{encoder}_{model_name}"
    else:  # ecapa doesn't have model parameter
        cache_key = encoder
    
    classifier_cache_path = os.path.join(cache_dir, f"classifier_{cache_key}.pkl")
    
    if os.path.exists(classifier_cache_path):
        print(f"Loading cached classifier for {encoder}...")
        with open(classifier_cache_path, 'rb') as f: clf = pickle.load(f)
        print(f"Classifier loaded for {encoder}")
    else:
        print(f"Training classifier for {encoder} (this will be cached)...")
        
        E_all_norm = pre.E_all / (np.linalg.norm(pre.E_all, axis=1, keepdims=True) + 1e-12)
        all_labels = pre.labels
        cls2id = {c: i for i, c in enumerate(classes)}
        
        X_cls_train = E_all_norm[pre.mask_dev]
        y_cls_train = [cls2id[s] for i, s in enumerate(all_labels) if pre.mask_dev[i]]
        X_cls_val = E_all_norm[pre.mask_test]
        y_cls_val = [cls2id[s] for i, s in enumerate(all_labels) if pre.mask_test[i]]
        
        clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(classes), device=device, epochs=cfg['orchestration']['classifier_epochs'], batch_size=512, lr=1e-3)
        
        # Cache classifier
        with open(classifier_cache_path, 'wb') as f: pickle.dump(clf, f)
        print(f"Classifier cached for {encoder}")
    
    return clf


def run_single_build_u_top_analysis(pre, A_all, dec_w, clf, sae, encoder: str, build_u_top_pct: str, build_u_top_value: int, cfg: Dict, device: str) -> Dict:
    print(f"  Analyzing {encoder} with {build_u_top_pct} ({build_u_top_value} features)...")
    
    # Setup data
    all_labels = pre.labels
    forget_label = cfg['dataset']['forget_labels'][0]
    forget_label_str = str(forget_label)
    
    # Create forget/retain masks
    sets = [f"{'forget' if str(lbl) == forget_label_str else 'retain'}:{'dev' if pre.mask_dev[i] else 'test'}" for i, lbl in enumerate(all_labels)]
    
    M_ret_dev = np.array([s == 'retain:dev' for s in sets], dtype=bool)
    M_fgt_dev = np.array([s == 'forget:dev' for s in sets], dtype=bool)
    M_ret_test = np.array([s == 'retain:test' for s in sets], dtype=bool)
    M_fgt_test = np.array([s == 'forget:test' for s in sets], dtype=bool)
    
    # Check minimum sample requirement
    forget_count = M_fgt_dev.sum()
    if forget_count < 5:
        print(f"[ERROR]: Forget class has only {forget_count} samples, minimum 5 required. Skipping.")
        return {
            'encoder': encoder,
            'build_U_top_pct': build_u_top_pct,
            'build_U_top_value': build_u_top_value,
            'FA_before': float('nan'),
            'FA_after': float('nan'),
            'RA_before': float('nan'),
            'RA_after': float('nan'),
            'ERB_score': float('nan'),
            'error': 'Insufficient samples'
        }
    
    # Feature mining
    A_ret = A_all[M_ret_dev]
    A_fgt = A_all[M_fgt_dev]
    
    # Get statistical method from config, default to cohens_d
    method = cfg['feature_mining'].get('statistical_method', 'cohens_d')
    stats = compute_feature_stats(A_fgt, A_ret, method=method)
    feat_idx = select_features(stats, method=method, top_k=build_u_top_value, fdr=0.05)
    
    if len(feat_idx) == 0:
        print(f"[ERROR] No discriminative features found. Cannot proceed with editing.")
        return {
            'encoder': encoder,
            'build_U_top_pct': build_u_top_pct,
            'build_U_top_value': build_u_top_value,
            'FA_before': float('nan'),
            'FA_after': float('nan'),
            'RA_before': float('nan'),
            'RA_after': float('nan'),
            'ERB_score': float('nan'),
            'error': 'No discriminative features found'
        }
    
    # Build subspace and apply edits
    prune_and_normalize_decoder(dec_w)
    U = build_U_from_decoder(dec_w, feat_idx)
    
    # LEACE params - create labels for the full dataset
    # Use string comparison to match the label matching logic used elsewhere
    Y_all = np.array([1 if str(all_labels[i]) == forget_label_str else 0 for i in range(len(all_labels))], dtype=int).reshape(-1, 1)
    
    E_edit, _ = apply_edits(
        pre.Z_all, sae, feat_idx, U, pre.mu, pre.W,
        cfg['edits']['beta'], 
        cfg['edits']['leace'], 
        {'Y_dev': Y_all, 'ridge': cfg['edits']['leace_ridge']}, 
        cfg['edits']['noise_sigma']
    )
    E_all_edited = E_edit
    E_all_norm = pre.E_all / (np.linalg.norm(pre.E_all, axis=1, keepdims=True) + 1e-12)
    E_all_norm2 = E_all_edited / (np.linalg.norm(E_all_edited, axis=1, keepdims=True) + 1e-12)
    
    # Classification evaluation
    classes = pre.classes
    cls2id = {c: i for i, c in enumerate(classes)}
    
    # Ensure label types match - convert all_labels to strings for consistent lookup
    all_labels_str = [str(lbl) for lbl in all_labels]
    E_ret_test, E_fgt_test = E_all_norm[M_ret_test], E_all_norm[M_fgt_test]
    L_ret_test, L_fgt_test = [cls2id[str(d)] for d in np.array(all_labels)[M_ret_test]], [cls2id[str(d)] for d in np.array(all_labels)[M_fgt_test]]
    E_ret_test_edit, E_fgt_test_edit = E_all_norm2[M_ret_test], E_all_norm2[M_fgt_test]
    
    acc_ret_before, acc_fgt_before = evaluate_classifier(clf, E_ret_test, L_ret_test, device=device), evaluate_classifier(clf, E_fgt_test, L_fgt_test, device=device)
    acc_ret_after, acc_fgt_after = evaluate_classifier(clf, E_ret_test_edit, L_ret_test, device=device), evaluate_classifier(clf, E_fgt_test_edit, L_fgt_test, device=device)
    
    # Calculate ERB score
    erb_score = calculate_erb_score(acc_fgt_after, acc_ret_after)
    
    print(f"[INFO] FA: {acc_fgt_before:.4f} → {acc_fgt_after:.4f}, RA: {acc_ret_before:.4f} → {acc_ret_after:.4f}, ERB: {erb_score:.2f}")
    
    return {
        'encoder': encoder,
        'build_U_top_pct': build_u_top_pct,
        'build_U_top_value': build_u_top_value,
        'FA_before': acc_fgt_before,
        'FA_after': acc_fgt_after,
        'RA_before': acc_ret_before,
        'RA_after': acc_ret_after,
        'ERB_score': erb_score
    }


def run_build_u_top_analysis(cfg: Dict) -> pd.DataFrame:
    device = cfg['experiment']['device']
    seed_all(cfg['experiment']['seed'])
    
    # Define encoders based on dataset (for non-English datasets, only use wav2vec2 with specialized models)
    dataset_name = cfg['dataset']['name']
    if dataset_name in ['lahaja', 'mls_spanish']:
        encoders = ['wav2vec2']
    else:
        encoders = ['ecapa', 'wav2vec2', 'hubert']
    
    build_u_top_percentages = ['1%', '2%', '5%', '10%', '25%']
    
    print(f"[INFO] Build_U_top Analysis Configuration:")
    print(f"  - Encoders: {encoders}")
    print(f"  - Build_U_top percentages: {build_u_top_percentages}")
    print(f"  - Total combinations: {len(encoders) * len(build_u_top_percentages)}")
    print(f"  - Device: {device}")
    print(f"  - Dataset: {cfg['dataset']['name']}")
    print(f"  - Forget label: {cfg['dataset']['forget_labels'][0]}")
    
    # Setup cache directory
    base_save_dir = cfg['experiment']['save_dir']
    cache_dir = os.path.join(base_save_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Setup results CSV
    results_path = os.path.join(base_save_dir, "build_u_top_analysis_results.csv")
    fieldnames = ["encoder", "build_U_top_pct", "build_U_top_value", "FA_before", "FA_after", "RA_before", "RA_after", "ERB_score", "error"]
    write_header = not os.path.exists(results_path)
    
    # Initialize adapter based on dataset configuration
    dataset_name = cfg['dataset']['name']
    adapters = {
        'voxceleb': VoxCelebAdapter(),
        'speech_commands': SpeechCommandsAdapter(),
        'audiomnist': AudioMNISTAdapter(),
        'librispeech': LibriSpeechAdapter(),
        'lahaja': LahajaAdapter(),
        'mls_spanish': MLSSpanishAdapter(),
    }
    
    if dataset_name not in adapters:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(adapters.keys())}")
    
    adapter = adapters[dataset_name]
    print(f"Using adapter for dataset: {dataset_name}")
    
    results = []
    
    with open(results_path, "a", newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        
        for encoder in encoders:
            print(f"\n[INFO] Processing encoder: {encoder.upper()}")            
            try:
                # Cache embeddings, SAE, and classifier for this encoder
                pre, classes = cache_embeddings(adapter, encoder, cfg, device, cache_dir)
                sae, A_all, dec_w = cache_sae_and_activations(pre, encoder, cfg, device, cache_dir)
                clf = cache_classifier(pre, classes, encoder, cfg, device, cache_dir)
                
                # Calculate build_U_top values for this encoder
                build_u_top_values = calculate_build_u_top_values(encoder, cfg['sae']['multiplier'])
                print(f"[INFO] Build_U_top values for {encoder}: {build_u_top_values}")
                
                # Run analysis for each build_U_top percentage
                for build_u_top_pct in build_u_top_percentages:
                    build_u_top_value = build_u_top_values[build_u_top_pct]
                    
                    try:
                        result = run_single_build_u_top_analysis(
                            pre, A_all, dec_w, clf, sae, encoder, build_u_top_pct, 
                            build_u_top_value, cfg, device
                        )
                        results.append(result)
                        writer.writerow(result)
                        fh.flush()
                        
                    except Exception as e:
                        print(f"[ERROR] {str(e)}")
                        error_result = {
                            'encoder': encoder,
                            'build_U_top_pct': build_u_top_pct,
                            'build_U_top_value': build_u_top_values[build_u_top_pct],
                            'FA_before': float('nan'),
                            'FA_after': float('nan'),
                            'RA_before': float('nan'),
                            'RA_after': float('nan'),
                            'ERB_score': float('nan'),
                            'error': str(e)
                        }
                        results.append(error_result)
                        writer.writerow(error_result)
                        fh.flush()
                        
            except Exception as e:
                print(f"[ERROR] Encoder-level failure for {encoder}: {e}")
                for build_u_top_pct in build_u_top_percentages:
                    error_result = {
                        'encoder': encoder,
                        'build_U_top_pct': build_u_top_pct,
                        'build_U_top_value': 0,
                        'FA_before': float('nan'),
                        'FA_after': float('nan'),
                        'RA_before': float('nan'),
                        'RA_after': float('nan'),
                        'ERB_score': float('nan'),
                        'error': f"encoder-level failure: {str(e)}"
                    }
                    results.append(error_result)
                    writer.writerow(error_result)
                fh.flush()
    
    # Create DataFrame and save summary
    results_df = pd.DataFrame(results)
    summary_path = os.path.join(base_save_dir, "build_u_top_analysis_summary.csv")
    results_df.to_csv(summary_path, index=False)
    
    print(f"\n[INFO] Analysis completed!")
    print(f"Results saved to: {results_path}")
    print(f"[INFO] Summary saved to: {summary_path}")
    print(f"[INFO] Total combinations tested: {len(results_df)}")
    
    # Print summary
    print(f"\n[INFO] Summary of results:")
    valid_results = results_df.dropna(subset=['ERB_score'])
    if len(valid_results) > 0:
        print(f"[INFO] {valid_results[['encoder', 'build_U_top_pct', 'ERB_score']].to_string(index=False)}")
    else:
        print(f"[ERROR] No valid results found.")
    
    return results_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Build_U_top Percentage Analysis')
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to YAML configuration file')
    
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg['experiment']['save_dir'] = os.path.join(cfg['experiment']['save_dir'], 'build_u_top_analysis')
    run_build_u_top_analysis(cfg)
    
    print(f"\n[INFO] Analysis complete! Check the results in: {cfg['experiment']['save_dir']}")


if __name__ == "__main__":
    main()
