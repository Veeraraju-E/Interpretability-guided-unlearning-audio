from typing import Dict
import os, csv, pickle
import numpy as np
from src.core.utils.seed import seed_all
from src.core.sae import SAE, train_sae, sae_encode_batch
from src.core.utils.stats import compute_feature_stats, select_features
from src.core.utils.subspace import prune_and_normalize_decoder, build_U_from_decoder
from src.core.evals.classification import train_classifier, evaluate_classifier
from src.core.evals.probes import safe_linear_probe
from src.core.evals.verification import build_verification_trials, compute_scores, per_class_eers, avg_eers
from src.core.edits import apply_edits

def run_multi(adapter, cfg: Dict):
    base_save_dir = cfg['experiment']['save_dir']
    os.makedirs(base_save_dir, exist_ok=True)
    seeds = cfg['orchestration']['seeds']
    device = cfg['experiment']['device']
    
    # Cache paths
    embeddings_cache_path = os.path.join(base_save_dir, "cached_embeddings.pkl")
    sae_cache_dir = os.path.join(base_save_dir, "sae_checkpoints")
    os.makedirs(sae_cache_dir, exist_ok=True)
    
    # Step 1: Compute embeddings once (if not cached)
    if os.path.exists(embeddings_cache_path):
        print("Loading cached embeddings...")
        with open(embeddings_cache_path, 'rb') as f:
            cache_data = pickle.load(f)
            pre = cache_data['pre']
            classes = cache_data['classes']
    else:
        print("Computing embeddings (this will be cached)...")
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
        with open(embeddings_cache_path, 'wb') as f: pickle.dump(cache_data, f)
        print("Embeddings cached successfully!")
    
    # Step 2: Train SAE and compute activations once per seed (if not cached)
    csv_path = os.path.join(base_save_dir, "multi_run_results.csv")
    fieldnames = ["seed", "forget_label", "FA_before", "FA_after", "RA_before", "RA_after", "forget_EER_before", "forget_EER_after", "retain_EER_before", "retain_EER_after", "linear_probe_auc_before", "linear_probe_auc_after", "error"]
    write_header = not os.path.exists(csv_path)
    
    with open(csv_path, "a", newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            
        for seed in seeds:
            sae_cache_path = os.path.join(sae_cache_dir, f"sae_seed_{seed}.pkl")
            activations_cache_path = os.path.join(sae_cache_dir, f"activations_seed_{seed}.pkl")
            
            try:
                # Load or train SAE
                if os.path.exists(sae_cache_path) and os.path.exists(activations_cache_path):
                    print(f"Loading cached SAE and activations for seed {seed}...")
                    with open(sae_cache_path, 'rb') as f:
                        sae_state = pickle.load(f)
                    with open(activations_cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                        A_all = cache_data['A_all']
                        dec_w = cache_data['dec_w']
                else:
                    print(f"Training SAE for seed {seed} (this will be cached)...")
                    seed_all(seed)
                    
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
                    print(f"SAE and activations cached for seed {seed}!")
                
                # Step 3: Run unlearning for each class (reusing cached data)
                for forget in classes:
                    try:
                        print(f"Processing seed {seed}, forget class: {forget}")
                        res = run_single_class_unlearning(pre, A_all, dec_w, forget, classes, cfg, device)
                        row = {"seed": seed, "forget_label": forget, **res, "error": ""}
                        writer.writerow(row)
                        fh.flush()
                    except Exception as e:
                        print(f"Error processing seed {seed}, class {forget}: {e}")
                        writer.writerow({"seed": seed, "forget_label": forget, "error": str(e)})
                        fh.flush()
                        
            except Exception as e:
                print(f"Seed-level failure for seed {seed}: {e}")
                for forget in classes:
                    writer.writerow({"seed": seed, "forget_label": forget, "error": f"seed-level failure: {str(e)}"})
                fh.flush()


def run_single_class_unlearning(pre, A_all, dec_w, forget_label, classes, cfg, device):
    from rich.console import Console
    console = Console()
    
    forget_label_str = str(forget_label)
    all_labels = pre.labels
    
    forget_count = sum(1 for lbl in all_labels if str(lbl) == forget_label_str)
    if forget_count < 5:
        console.print(f"WARNING: Forget class '{forget_label_str}' has only {forget_count} samples, minimum 5 required. Skipping this class.", style="red")
        return {"error": f"Insufficient samples: {forget_count}"}
    
    sets = [f"{'forget' if str(lbl) == forget_label_str else 'retain'}:{'dev' if pre.mask_dev[i] else 'test'}" for i, lbl in enumerate(all_labels)]
    
    M_ret_dev = np.array([s == 'retain:dev' for s in sets], dtype=bool)
    M_fgt_dev = np.array([s == 'forget:dev' for s in sets], dtype=bool)
    M_ret_test = np.array([s == 'retain:test' for s in sets], dtype=bool)
    M_fgt_test = np.array([s == 'forget:test' for s in sets], dtype=bool)
    
    # Train multi-class classifier
    E_all_norm = pre.E_all / (np.linalg.norm(pre.E_all, axis=1, keepdims=True) + 1e-12)
    
    # Create class mapping
    unique_labels = sorted(list(set(all_labels)))
    cls2id = {d: i for i, d in enumerate(unique_labels)}
    
    # Training data
    X_train = E_all_norm[pre.mask_dev]
    y_train = [cls2id[d] for d in np.array(all_labels)[pre.mask_dev]]
    
    # Validation data 
    X_val = E_all_norm[pre.mask_test]
    y_val = [cls2id[d] for d in np.array(all_labels)[pre.mask_test]]
    
    # Train multi-class classifier
    clf = train_classifier(X_train, y_train, X_val, y_val, num_classes=len(unique_labels), device=device, epochs=cfg['orchestration']['classifier_epochs'], batch_size=512, lr=1e-3)
    
    # Feature mining
    A_ret = A_all[M_ret_dev]
    A_fgt = A_all[M_fgt_dev]
    
    # Get statistical method from config, default to cohens_d
    method = cfg['feature_mining'].get('statistical_method', 'cohens_d')
    stats = compute_feature_stats(A_fgt, A_ret, method=method)
    feat_idx = select_features(stats, method=method, top_k=cfg['feature_mining']['build_U_top'], fdr=0.05)
    
    if len(feat_idx) == 0:
        console.print(f"ERROR: No discriminative features found for forget class '{forget_label_str}'. Cannot proceed with editing.", style="red")
        return {"error": "No discriminative features found"}
    
    # Build subspace and apply edits
    prune_and_normalize_decoder(dec_w)
    U = build_U_from_decoder(dec_w, feat_idx)
    
    # LEACE params - create labels for the full dataset
    Y_all = np.array([1 if str(all_labels[i]) == forget_label_str else 0 for i in range(len(all_labels))], dtype=int).reshape(-1, 1)
    E_edit, info = apply_edits(
        pre.Z_all, None, feat_idx, U, pre.mu, pre.W,
        cfg['edits']['beta'], cfg['edits']['leace'], {'Y_dev': Y_all, 'ridge': cfg['edits']['leace_ridge']}, cfg['edits']['noise_sigma']
    )
    E_all_edited = E_edit
    E_all_norm2 = E_all_edited / (np.linalg.norm(E_all_edited, axis=1, keepdims=True) + 1e-12)
    
    # Verification trials
    test_mask_for_trials = M_ret_test | M_fgt_test
    trials, tlabels = build_verification_trials(pre.labels, [forget_label_str], test_mask_for_trials, seed=cfg['experiment']['seed'])
    
    # Normalize embeddings for verification
    E_all_norm = pre.E_all / (np.linalg.norm(pre.E_all, axis=1, keepdims=True) + 1e-12)
    E_all_norm2 = E_all_edited / (np.linalg.norm(E_all_edited, axis=1, keepdims=True) + 1e-12)
    
    # Compute scores before and after editing (using full normalized embeddings)
    scores_before = compute_scores(E_all_norm, trials)
    scores_after = compute_scores(E_all_norm2, trials)
    
    # Compute EERs using embeddings
    per_class = per_class_eers(E_all_norm, E_all_norm2, pre.labels, mask=test_mask_for_trials, rng_seed=cfg['experiment']['seed'])
    
    # Extract EERs for retain and forget classes
    per_class_ret = {c: vals for c, vals in per_class.items() if str(c) != forget_label_str}
    per_class_fgt = {c: vals for c, vals in per_class.items() if str(c) == forget_label_str}
    
    # Get average EERs for retain classes
    retain_avg_before, retain_avg_after = avg_eers(per_class_ret)
    
    # Get EERs for forget class
    forget_eer_before, forget_eer_after = per_class_fgt.get(forget_label_str, (0.0, 0.0))
    
    # Classification evaluation (same as single_run)
    E_ret_test = E_all_norm[M_ret_test]
    L_ret_test = [cls2id[d] for d in np.array(all_labels)[M_ret_test]]
    E_fgt_test = E_all_norm[M_fgt_test]
    L_fgt_test = [cls2id[d] for d in np.array(all_labels)[M_fgt_test]]
    E_ret_test_edit = E_all_norm2[M_ret_test]
    E_fgt_test_edit = E_all_norm2[M_fgt_test]
    acc_ret_before = evaluate_classifier(clf, E_ret_test, L_ret_test, device=device)
    acc_fgt_before = evaluate_classifier(clf, E_fgt_test, L_fgt_test, device=device)
    acc_ret_after = evaluate_classifier(clf, E_ret_test_edit, L_ret_test, device=device)
    acc_fgt_after = evaluate_classifier(clf, E_fgt_test_edit, L_fgt_test, device=device)
    
    # Linear probe AUC
    split_mask_dict = {'dev': pre.mask_dev, 'test': pre.mask_test}
    _, auc_before = safe_linear_probe(E_all_norm, all_labels, U, split_mask_dict, [forget_label_str])
    _, auc_after = safe_linear_probe(E_all_norm2, all_labels, U, split_mask_dict, [forget_label_str])
    
    # Return results
    return {
        "FA_before": acc_fgt_before,
        "FA_after": acc_fgt_after,
        "RA_before": acc_ret_before,
        "RA_after": acc_ret_after,
        "forget_EER_before": forget_eer_before,
        "forget_EER_after": forget_eer_after,
        "retain_EER_before": retain_avg_before,
        "retain_EER_after": retain_avg_after,
        "linear_probe_auc_before": auc_before,
        "linear_probe_auc_after": auc_after
    }