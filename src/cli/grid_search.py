"""
Grid search implementation for SAE hyperparameter optimization.
This module contains the actual grid search logic that was previously in grid.py.
"""

import os
import yaml
import numpy as np
import pandas as pd
from src.datasets.voxceleb_adapter import build_manifest
from src.core.utils.seed import seed_all
from src.core.encoders.registry import extract_embeddings
from src.core.preprocess.whitening import fit_whitener, apply_whitener, invert_whitener
from src.core.sae.model import SAE
from src.core.sae.train import train_sae
from src.core.sae.encode import sae_encode_batch
from src.core.feature_mining.stats import feature_stats, select_features_from_stats
from src.core.feature_mining.subspace import build_U_from_decoder
from src.core.edits.leace import leace_projector_in_U
from src.core.edits.noise import inject_noise_in_subspace
from src.core.evals.probes import safe_linear_probe
from src.core.evals.verification import per_class_eers, avg_eers

def load_config(path: str):
    """Load YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_grid_search(cfg: dict):
    """
    Run grid search over SAE parameters.
    
    Args:
        cfg: Configuration dictionary
    """
    device = cfg['experiment']['device']
    seed_all(cfg['experiment']['seed'])

    # Build a minimal grid from config overrides
    multipliers = cfg.get('grid_search', {}).get('sae_multipliers', [2.0, 4.0, 8.0, 16.0, 32.0])
    topks = cfg.get('grid_search', {}).get('sae_topks', [32, 64, 128, 256, 512])
    sweep_epochs = cfg.get('grid_search', {}).get('sweep_epochs', 100)

    # Precompute on VoxCeleb-style manifest for consistency
    entries = build_manifest(
        cfg['dataset']['root_or_manifest'], 
        embeddings_path=cfg['encoder'].get('embeddings_path'), 
        dev_frac_per_spk=cfg['dataset']['dev_frac'], 
        seed=cfg['experiment']['seed']
    )
    sets = [f"retain:{m['split']}" for m in entries]  # placeholder set labels
    labels = [m['speaker'] if 'speaker' in m else (m['label'] if 'label' in m else 'NA') for m in entries]
    E_all, _labels, _ = extract_embeddings(
        entries, 
        encoder=cfg['encoder']['name'], 
        device=device, 
        wav2vec2_model=cfg['encoder'].get('wav2vec2_model', 'facebook/wav2vec2-base'), 
        hubert_model=cfg['encoder'].get('hubert_model', 'facebook/hubert-base-ls960'), 
        conformer_model=cfg['encoder'].get('conformer_model', 'facebook/wav2vec2-conformer-rel-pos-large'), 
        label_key=('speaker' if 'speaker' in entries[0] else 'label')
    )
    mask_dev = np.array([m['split'] == 'dev' for m in entries], dtype=bool)
    mask_test = np.array([m['split'] == 'test' for m in entries], dtype=bool)
    mu, W = fit_whitener(E_all[mask_dev])
    Z_all = apply_whitener(E_all, mu, W)

    results = []
    total_combinations = len(multipliers) * len(topks)
    current_combination = 0
    
    for mult in multipliers:
        for k in topks:
            current_combination += 1
            print(f"Running combination {current_combination}/{total_combinations}: mult={mult}, k={k}")
            
            Z_train = Z_all[mask_dev]
            perm = np.random.permutation(Z_train.shape[0])
            cut = int(0.8 * len(perm))
            Z_sae_train, Z_sae_val = Z_train[perm[:cut]], Z_train[perm[cut:]]
            sae = SAE(
                input_dim=Z_all.shape[1], 
                hidden_mult=mult, 
                k=k, 
                variant='topk', 
                hidden_dim=cfg['sae']['hidden_dim']
            )
            sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=sweep_epochs, device=device, return_loss_history=False)
            A_all = sae_encode_batch(sae, Z_all, device=device)
            stats = feature_stats(A_all[mask_dev], A_all[mask_dev])
            feat_idx = select_features_from_stats(stats, top_k=cfg['feature_mining']['build_U_top'])
            dec_w = sae.dec.weight.detach().cpu().numpy()
            U = build_U_from_decoder(dec_w, feat_idx)
            
            # LEACE on dev split for a placeholder Y_dev (binary zeros)
            Y_dev = np.zeros((mask_dev.sum(), 1), dtype=int)
            P_r, mu_r = leace_projector_in_U(Z_all[mask_dev], U, Y_dev, ridge=cfg['edits']['leace_ridge'])
            Z_edit = Z_all.copy()
            ZU = Z_edit @ U
            ZU_prime = (ZU - mu_r) @ P_r.T + mu_r
            Z_edit = Z_edit - (ZU @ U.T) + (ZU_prime @ U.T)
            if cfg['edits']['noise_sigma'] > 0:
                Z_edit = inject_noise_in_subspace(Z_edit, U, sigma=cfg['edits']['noise_sigma'])
            
            # metrics
            split_mask = {"dev": mask_dev, "test": mask_test}
            acc_before, auc_before = safe_linear_probe(Z_all, labels, U, split_mask, forget_classes=cfg['dataset']['forget_labels'])
            acc_after, auc_after = safe_linear_probe(Z_edit, labels, U, split_mask, forget_classes=cfg['dataset']['forget_labels'])
            per_class = per_class_eers(
                E_all/(np.linalg.norm(E_all, axis=1, keepdims=True) + 1e-12), 
                invert_whitener(Z_edit, mu, W)/(np.linalg.norm(invert_whitener(Z_edit, mu, W), axis=1, keepdims=True) + 1e-12), 
                labels, 
                mask=(mask_test), 
                rng_seed=cfg['experiment']['seed']
            )
            ret_b, ret_a = avg_eers({c: v for c, v in per_class.items() if c not in cfg['dataset']['forget_labels']})
            fgt_b, fgt_a = avg_eers({c: v for c, v in per_class.items() if c in cfg['dataset']['forget_labels']})
            results.append({
                "mult": mult, 
                "topk": k, 
                "probe_auc_before": auc_before, 
                "probe_auc_after": auc_after, 
                "retain_eer_before": ret_b, 
                "retain_eer_after": ret_a, 
                "forget_eer_before": fgt_b, 
                "forget_eer_after": fgt_a
            })

    out_dir = cfg['experiment']['save_dir']
    os.makedirs(out_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_dir, 'grid_search_results.csv'), index=False)
    
    print(f"Grid search completed. Results saved to: {os.path.join(out_dir, 'grid_search_results.csv')}")
    return results_df
