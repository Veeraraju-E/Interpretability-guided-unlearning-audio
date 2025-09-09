# grid_search.py
import os
import time
import json
import itertools
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Any
from collections import Counter

# Import necessary functions from your existing files
from common import (
    seed_all, human_time, extract_embeddings, fit_whitener, apply_whitener,
    invert_whitener, l2_normalize, SAE, train_sae, sae_encode_batch,
    prune_and_normalize_decoder, feature_stats, select_features_from_stats,
    build_U_from_decoder, leace_projector_in_U, apply_subspace_leace,
    apply_sauce_gating, inject_noise_in_subspace, compute_scores,
    eer_from_scores, build_verification_trials, ResNetClassifier, train_classifier,
    evaluate_classifier, per_class_eers, avg_eers, safe_linear_probe,
    calculate_monosemanticity_score, compute_monosemanticity_statistics
)
from voxceleb import build_manifest

# --- 1. CONFIGURATION ---
# Define the paths and base parameters
MANIFEST_PATH = "/data/veer/testing/new/datasets/voxceleb2/manifest.csv"
SAVE_DIR = "./grid_search_runs"
CACHE_DIR = os.path.join(SAVE_DIR, "cache")
RESULTS_CSV_PATH = os.path.join(SAVE_DIR, "grid_search_results.csv")
DEVICE = "cuda"
SEED = 1337
FORGET_SPKS = ["id00017"] # Fixed for all runs as requested
RETAIN_FRAC = 0.7 # Fixed for all runs

# --- 2. HYPERPARAMETER GRID ---
# Define the values to search over for each hyperparameter.
HPARAM_GRID = {
    'encoder': ['ecapa', 'wav2vec2', 'hubert', 'conformer'],
    'dev_frac_per_spk': [0.7],
    'sae_multiplier': [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    'sae_variant': ['topk', 'standard'],
    # 'sae_epochs': [100, 250, 500],
    # 'sae_lr': [1e-3],
    'sae_topk': [1, 4, 16, 64, 256, 1024],              # Only for 'topk' variant
    'lambda_sparsity': [1, 1e-1, 1e-2, 1e-3, 1e-4],     # Only for 'standard' variant
    'build_U_top': [1, 10, 100, 200, 500],
    'noise_sigma': [0.0, 0.01, 0.1, 1],
    'leace_ridge': [1e-6, 1e-5],
}

def generate_hparam_combinations(grid: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generates a list of all valid hyperparameter combinations."""
    combinations = []
    # Create a list of key-value tuples for itertools.product
    param_lists = []
    for key, values in grid.items():
        if key not in ['sae_topk', 'lambda_sparsity']:
            param_lists.append([(key, v) for v in values])

    # Generate product of all non-conditional params
    base_products = list(itertools.product(*param_lists))

    for prod in base_products:
        params = dict(prod)
        if params['sae_variant'] == 'topk':
            for topk_val in grid['sae_topk']:
                combo = params.copy()
                combo['sae_topk'] = topk_val
                combo['lambda_sparsity'] = 0.0 # Not applicable
                combinations.append(combo)
        elif params['sae_variant'] == 'standard':
            for lambda_val in grid['lambda_sparsity']:
                combo = params.copy()
                combo['lambda_sparsity'] = lambda_val
                combo['sae_topk'] = 0 # Not applicable
                combinations.append(combo)
    return combinations

def get_cached_data(encoder: str, dev_frac: float) -> Dict:
    """Loads level 1 cache (Embeddings, Splits, Whitened data)."""
    cache_filename = f"level1_cache_enc-{encoder}_devfrac-{dev_frac}.npz"
    cache_path = os.path.join(CACHE_DIR, cache_filename)

    if os.path.exists(cache_path):
        print(f"--> Found level 1 cache, loading from: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    return None

def set_cached_data(encoder: str, dev_frac: float, data: Dict):
    """Saves level 1 cache."""
    cache_filename = f"level1_cache_enc-{encoder}_devfrac-{dev_frac}.npz"
    cache_path = os.path.join(CACHE_DIR, cache_filename)
    print(f"--> Saving level 1 cache to: {cache_path}")
    np.savez_compressed(cache_path, **data)

def get_cached_models(encoder: str, dev_frac: float) -> Dict:
    """Loads level 2 cache (classifier model)."""
    # Note: Trials are no longer loaded from cache for EER, but the function is kept for the classifier
    clf_path = os.path.join(CACHE_DIR, f"level2_clf_enc-{encoder}_devfrac-{dev_frac}.pt")

    if os.path.exists(clf_path):
        print("--> Found level 2 cache, loading classifier.")
        clf_state = torch.load(clf_path)
        return {'clf_state': clf_state}
    return None

def set_cached_models(encoder: str, dev_frac: float, data: Dict):
    """Saves level 2 cache (classifier model)."""
    clf_path = os.path.join(CACHE_DIR, f"level2_clf_enc-{encoder}_devfrac-{dev_frac}.pt")
    
    print("--> Saving level 2 cache (classifier).")
    torch.save(data['clf_state'], clf_path)


def main():
    """Main function to run the grid search."""
    seed_all(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Generate all hyperparameter combinations
    hparam_combinations = generate_hparam_combinations(HPARAM_GRID)
    total_runs = len(hparam_combinations)
    print(f"--- Starting Grid Search ---")
    print(f"Total experiments to run: {total_runs}")

    results_list = []
    
    # --- Major Outer Loop for Cacheable Parts ---
    # Group by cacheable parameters
    cache_groups = itertools.groupby(sorted(hparam_combinations, key=lambda x: (x['encoder'], x['dev_frac_per_spk'])), key=lambda x: (x['encoder'], x['dev_frac_per_spk']))
    
    run_count = 0
    t0_all = time.time()

    for (encoder, dev_frac), group in cache_groups:
        
        # --- 3. LEVEL 1 CACHE: DATA PREPARATION, EMBEDDINGS, WHITENING ---
        cached_data = get_cached_data(encoder, dev_frac)

        if cached_data is None:
            print(f"\n[CACHE MISS] Pre-computing for encoder='{encoder}', dev_frac_per_spk={dev_frac}")
            # 1) Build manifest and splits
            entries = build_manifest(MANIFEST_PATH, dev_frac_per_spk=dev_frac, seed=SEED)
            
            # This logic is simplified from run_pipeline to get splits
            spks = sorted(list(set(e['speaker'] for e in entries)))
            retain_spks = [s for s in spks if s not in FORGET_SPKS]
            
            manifest = []
            for entry in entries:
                entry_copy = entry.copy()
                entry_copy['set'] = 'forget' if entry['speaker'] in FORGET_SPKS else 'retain'
                manifest.append(entry_copy)
            
            # 2) Extract embeddings
            E_all, L_all, _ = extract_embeddings(manifest, encoder=encoder, device=DEVICE)

            # Create masks
            sets = [f"{m['set']}:{m['split']}" for m in manifest]
            M_ret_dev = np.array([s == 'retain:dev' for s in sets], dtype=bool)
            M_fgt_dev = np.array([s == 'forget:dev' for s in sets], dtype=bool)
            M_ret_test = np.array([s == 'retain:test' for s in sets], dtype=bool)
            M_fgt_test = np.array([s == 'forget:test' for s in sets], dtype=bool)

            # 3) Whitening on retain-dev
            E_ret_train = E_all[M_ret_dev]
            mu, W = fit_whitener(E_ret_train)
            Z_all = apply_whitener(E_all, mu, W)
            
            cached_data = {
                'manifest': np.array(manifest, dtype=object),
                'E_all': E_all, 'L_all': np.array(L_all, dtype=object), 'Z_all': Z_all,
                'mu': mu, 'W': W, 'retain_spks': np.array(retain_spks, dtype=object),
                'M_ret_dev': M_ret_dev, 'M_fgt_dev': M_fgt_dev,
                'M_ret_test': M_ret_test, 'M_fgt_test': M_fgt_test
            }
            set_cached_data(encoder, dev_frac, cached_data)
        
        # Unpack data from cache
        manifest = cached_data['manifest'].tolist()
        E_all, L_all, Z_all = cached_data['E_all'], cached_data['L_all'].tolist(), cached_data['Z_all']
        mu, W, retain_spks = cached_data['mu'], cached_data['W'], cached_data['retain_spks'].tolist()
        M_ret_dev, M_fgt_dev = cached_data['M_ret_dev'], cached_data['M_fgt_dev']
        M_ret_test, M_fgt_test = cached_data['M_ret_test'], cached_data['M_fgt_test']
        
        E_all_norm = l2_normalize(E_all)

        # --- 4. LEVEL 2 CACHE: CLASSIFIER ---
        cached_models = get_cached_models(encoder, dev_frac)
        if cached_models is None:
            print(f"[CACHE MISS] Pre-computing classifier for encoder='{encoder}', dev_frac_per_spk={dev_frac}")
            # Train the ResNet classifier on the original embeddings
            all_spks = sorted(list(set(L_all)))
            spk2id = {s: i for i, s in enumerate(all_spks)}
            X_cls_train = E_all_norm[M_ret_dev | M_fgt_dev]
            y_cls_train = [spk2id[s] for i, s in enumerate(L_all) if (M_ret_dev | M_fgt_dev)[i]]
            X_cls_val = E_all_norm[M_ret_test | M_fgt_test]
            y_cls_val = [spk2id[s] for i, s in enumerate(L_all) if (M_ret_test | M_fgt_test)[i]]
            clf_model = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(all_spks), device=DEVICE)
            
            cached_models = {
                'clf_state': {k: v.cpu() for k,v in clf_model.state_dict().items()}
            }
            set_cached_models(encoder, dev_frac, cached_models)
        
        clf_state = cached_models['clf_state']

        # --- 5. INNER LOOP: SAE, LEACE, and EVALUATION ---
        hparams_for_this_group = list(group)
        for params in hparams_for_this_group:
            run_count += 1
            t0_run = time.time()
            print(f"\n--- Running Experiment {run_count}/{total_runs} ---")
            print(json.dumps(params, indent=2))
            
            current_results = params.copy()
            
            # 4) SAE training
            Z_train = Z_all[M_ret_dev | M_fgt_dev]
            perm = np.random.permutation(Z_train.shape[0])
            cut = int(0.8 * len(perm))
            Z_sae_train, Z_sae_val = Z_train[perm[:cut]], Z_train[perm[cut:]]
            sae = SAE(input_dim=Z_all.shape[1], hidden_mult=params['sae_multiplier'], k=params['sae_topk'], variant=params['sae_variant'])
            sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=300, lr=1e-3, device=DEVICE, lambda_sparsity=params['lambda_sparsity'])

            # 5) SAE encodes for feature mining
            Z_ret_dev = Z_all[M_ret_dev]
            Z_fgt_dev = Z_all[M_fgt_dev]
            A_ret, A_fgt = sae_encode_batch(sae, Z_ret_dev, device=DEVICE), sae_encode_batch(sae, Z_fgt_dev, device=DEVICE)
            
            # Monosemanticity for SAE activations
            A_all = sae_encode_batch(sae, Z_all, device=DEVICE)
            ms_sae = calculate_monosemanticity_score(A_all, E_all_norm)
            ms_sae_stats = compute_monosemanticity_statistics(ms_sae)
            current_results.update({
                "ms_sae_mean": ms_sae_stats["mean"],
                "ms_sae_min": ms_sae_stats["min"],
                "ms_sae_max": ms_sae_stats["max"]
            })

            # 6) Feature stats & selection
            stats = feature_stats(A_fgt, A_ret)
            feat_idx = select_features_from_stats(stats, top_k=params['build_U_top'])

            # 7) Build U subspace
            dec_w = sae.dec.weight.detach().cpu().numpy()
            U = build_U_from_decoder(dec_w, feat_idx)

            # 8) Compute LEACE in U
            mask_dev = M_ret_dev | M_fgt_dev
            dev_indices = np.where(mask_dev)[0]
            Y_dev = np.array([1 if manifest[i]['speaker'] in FORGET_SPKS else 0 for i in dev_indices], dtype=int).reshape(-1, 1)
            P_r, mu_r = leace_projector_in_U(Z_all[mask_dev], U, Y_dev, ridge=params['leace_ridge'])

            # 9) Apply SAUCE and LEACE edit
            Z_edit = apply_subspace_leace(Z_all, U, P_r, mu_r)
            if params['noise_sigma'] > 0:
                Z_edit = inject_noise_in_subspace(Z_edit, U, sigma=params['noise_sigma'])

            E_edit = invert_whitener(Z_edit, mu, W)
            E_edit_norm = l2_normalize(E_edit)
            
            # --- 6. EVALUATION METRICS ---

            # A) ROBUST PER-CLASS EER CALCULATION
            print("  Calculating per-class EERs...")
            test_mask = M_ret_test | M_fgt_test
            
            # Diagnostic: Count samples per speaker to understand eligibility
            test_labels_list = np.array(L_all)[test_mask]
            test_speaker_counts = Counter(test_labels_list)
            eligible_speakers = {spk for spk, count in test_speaker_counts.items() if count >= 2}
            print(f"  Found {len(eligible_speakers)} speakers with >= 2 test samples (eligible for EER calculation).")
            print(f"  Total speakers in test set: {len(test_speaker_counts)}.")
            
            # Calculate EERs only for speakers where it's possible
            per_class_test = per_class_eers(E_all_norm, E_edit_norm, L_all, mask=test_mask, rng_seed=SEED)

            # Robustly calculate and average RETAIN EER
            per_class_ret = {c: vals for c, vals in per_class_test.items() if c in retain_spks}
            if not per_class_ret:
                print("  Warning: No retain speakers were eligible for EER calculation (all have < 2 test samples). Retain EER will be NaN.")
                retain_avg_before, retain_avg_after = float('nan'), float('nan')
            else:
                print(f"  Calculating average Retain EER from {len(per_class_ret)} eligible retain speakers.")
                retain_avg_before, retain_avg_after = avg_eers(per_class_ret)

            # Robustly calculate and average FORGET EER
            per_class_fgt = {c: vals for c, vals in per_class_test.items() if c in FORGET_SPKS}
            if not per_class_fgt:
                print(f"  Warning: No forget speakers were eligible for EER calculation (all have < 2 test samples). Forget EER will be NaN.")
                forget_avg_before, forget_avg_after = float('nan'), float('nan')
            else:
                print(f"  Calculating average Forget EER from {len(per_class_fgt)} eligible forget speakers.")
                forget_avg_before, forget_avg_after = avg_eers(per_class_fgt)
            
            current_results.update({
                "retain_EER_before_avg": retain_avg_before, "retain_EER_after_avg": retain_avg_after,
                "forget_EER_before_avg": forget_avg_before, "forget_EER_after_avg": forget_avg_after,
            })

            # B) LINEAR PROBING
            print("  Calculating linear probe performance...")
            split_mask = {"dev": M_ret_dev | M_fgt_dev, "test": M_ret_test | M_fgt_test}
            acc_before, auc_before = safe_linear_probe(Z_all, L_all, U, split_mask, forget_classes=FORGET_SPKS)
            acc_after, auc_after = safe_linear_probe(Z_edit, L_all, U, split_mask, forget_classes=FORGET_SPKS)
            current_results.update({
                "linear_probe_acc_before": acc_before, "linear_probe_acc_after": acc_after,
                "linear_probe_auc_before": auc_before, "linear_probe_auc_after": auc_after,
            })

            # C) CLASSIFIER ACCURACY (Utility Metric)
            print("  Calculating classifier accuracy...")
            all_spks = sorted(list(set(L_all)))
            spk2id = {s: i for i, s in enumerate(all_spks)}
            clf = ResNetClassifier(E_all_norm.shape[1], len(all_spks))
            clf.load_state_dict(clf_state)

            E_ret_test = E_all_norm[M_ret_test]
            L_ret_test = [spk2id[d] for d in np.array(L_all)[M_ret_test]]
            E_fgt_test = E_all_norm[M_fgt_test]
            L_fgt_test = [spk2id[d] for d in np.array(L_all)[M_fgt_test]]
            E_ret_test_edit = E_edit_norm[M_ret_test]
            E_fgt_test_edit = E_edit_norm[M_fgt_test]
            
            acc_ret_before = evaluate_classifier(clf, E_ret_test, L_ret_test, device=DEVICE)
            acc_fgt_before = evaluate_classifier(clf, E_fgt_test, L_fgt_test, device=DEVICE)
            acc_ret_after = evaluate_classifier(clf, E_ret_test_edit, L_ret_test, device=DEVICE)
            acc_fgt_after = evaluate_classifier(clf, E_fgt_test_edit, L_fgt_test, device=DEVICE)
            current_results.update({
                "resnet_acc_retain_before": acc_ret_before, "resnet_acc_retain_after": acc_ret_after,
                "resnet_acc_forget_before": acc_fgt_before, "resnet_acc_forget_after": acc_fgt_after,
            })
            
            print(f"  Run complete in {human_time(time.time() - t0_run)}")
            print(f"  Metrics: Retain EER {retain_avg_after:.4f} | Forget EER {forget_avg_after:.4f} | Forget Acc {acc_fgt_after*100:.2f}%")
            
            # --- 7. LOGGING ---
            results_list.append(current_results)
            
            # Save results to CSV after each run
            df = pd.DataFrame(results_list)
            df.to_csv(RESULTS_CSV_PATH, index=False)

    print(f"\n--- Grid Search Finished ---")
    print(f"Total time: {human_time(time.time() - t0_all)}")
    print(f"Results saved to {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()