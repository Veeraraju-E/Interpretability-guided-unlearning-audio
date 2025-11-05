from typing import Dict
import os
import json
import numpy as np
from src.core.utils.seed import seed_all
from src.core.sae import SAE, train_sae, sae_encode_batch
from src.core.utils.stats import compute_feature_stats, select_features
from src.core.utils.subspace import prune_and_normalize_decoder, build_U_from_decoder
from src.core.evals.classification import train_classifier, evaluate_classifier
from src.core.evals.probes import safe_linear_probe
from src.core.evals.verification import build_verification_trials, compute_scores, eer_from_scores, per_class_eers, avg_eers
from src.core.edits import apply_edits
from src.core.visualization import create_embedding_visualizations, create_embedding_summary_stats
 

def run_single(adapter, cfg: Dict):
    seed, device, save_dir = cfg['experiment']['seed'], cfg['experiment']['device'], cfg['experiment']['save_dir']
    seed_all(seed)

    # Different adapters use different parameter names for the data path
    data_path_param = cfg['dataset']['root_or_manifest']
    if hasattr(adapter, 'name'):
        if adapter.name in ['audiomnist', 'speech_commands']:
            pre = adapter.precompute(
                data_root=data_path_param,
                embeddings_path=cfg['encoder'].get('embeddings_path'),
                dev_frac=cfg['dataset']['dev_frac'],
                seed=seed,
                encoder_cfg=cfg['encoder'],
                device=device,
            )
        elif adapter.name == 'librispeech':
            pre = adapter.precompute(
                root=data_path_param,
                embeddings_path=cfg['encoder'].get('embeddings_path'),
                dev_frac=cfg['dataset']['dev_frac'],
                seed=seed,
                encoder_cfg=cfg['encoder'],
                device=device,
            )
        elif adapter.name == 'lahaja':
            pre = adapter.precompute(
                split=data_path_param,
                embeddings_path=cfg['encoder'].get('embeddings_path'),
                dev_frac=cfg['dataset']['dev_frac'],
                seed=seed,
                encoder_cfg=cfg['encoder'],
                device=device,
            )
        elif adapter.name == 'mls_spanish':
            pre = adapter.precompute(
                root_or_manifest=data_path_param,
                embeddings_path=cfg['encoder'].get('embeddings_path'),
                dev_frac=cfg['dataset']['dev_frac'],
                seed=seed,
                encoder_cfg=cfg['encoder'],
                device=device,
            )
        else:  # voxceleb
            pre = adapter.precompute(
                root_or_manifest=data_path_param,
                embeddings_path=cfg['encoder'].get('embeddings_path'),
                dev_frac=cfg['dataset']['dev_frac'],
                seed=seed,
                encoder_cfg=cfg['encoder'],
                device=device,
            )
    else:
        raise NotImplementedError(f"Adapter {adapter.name} not implemented for {cfg['dataset']['name']}.")

    # SAE train once per seed
    Z_train = pre.Z_all[pre.mask_dev]
    perm = np.random.permutation(Z_train.shape[0])
    cut = int(0.8 * len(perm)) if len(perm) > 1 else int(0.5 * len(perm))
    Z_sae_train, Z_sae_val = (Z_train, Z_train) if cut <= 0 else (Z_train[perm[:cut]], Z_train[perm[cut:]])
    # print(cfg)
    sae = SAE(input_dim=pre.Z_all.shape[1], hidden_mult=cfg['sae']['multiplier'], k=cfg['sae']['topk'], variant=cfg['sae']['variant'], hidden_dim=cfg['sae']['hidden_dim'])
    sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=cfg['sae']['epochs'], batch_size=cfg['sae']['batch_size'], lr=1e-3, device=device, lambda_sparsity=cfg['sae']['lambda_sparsity'])
    A_all = sae_encode_batch(sae, pre.Z_all, device=device, batch=256)
    dec_w = sae.dec.weight.detach().cpu().numpy()

    # Classifier once
    E_all_norm = pre.E_all / (np.linalg.norm(pre.E_all, axis=1, keepdims=True) + 1e-12)
    all_labels = pre.labels
    classes = pre.classes
    cls2id = {c: i for i, c in enumerate(classes)}
    X_cls_train = E_all_norm[pre.mask_dev]
    y_cls_train = [cls2id[s] for i, s in enumerate(all_labels) if pre.mask_dev[i]]
    X_cls_val = E_all_norm[pre.mask_test]
    y_cls_val = [cls2id[s] for i, s in enumerate(all_labels) if pre.mask_test[i]]
    clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(classes), device=device, epochs=cfg['orchestration']['classifier_epochs'], batch_size=512, lr=1e-3)

    results_all = {}
    forget_list = cfg['dataset']['forget_labels']
    os.makedirs(save_dir, exist_ok=True)
    
    # Always show key information in yellow
    from rich.console import Console
    console = Console()
    console.print(f"Forget list: {forget_list}", style="yellow")
    
    # Debug: Print label types and values
    # Debug: Count samples per class
    from collections import Counter
    label_counts = Counter(all_labels)
    
    # Filter out classes with less than 5 samples
    valid_labels = {label for label, count in label_counts.items() if count >= 5}
    
    # Show unique labels (first 20 only) in yellow
    unique_labels_list = sorted(list(set(all_labels)))
    labels_to_show = unique_labels_list[:20]
    labels_display = labels_to_show + (["..."] if len(unique_labels_list) > 20 else [])
    console.print(f"Unique labels (first 20): {labels_display}", style="yellow")
    
    # Check if forget labels are valid
    invalid_forget_labels = [label for label in forget_list if str(label) not in valid_labels]
    if invalid_forget_labels:
        console.print(f"WARNING: Some forget labels have <5 samples: {invalid_forget_labels}", style="red")
        console.print(f"Available valid labels: {sorted(valid_labels)}", style="yellow")
    
    for forget_label in forget_list:
        # Convert forget_label to string for comparison
        forget_label_str = str(forget_label)
        sets = [f"{'forget' if str(lbl) == forget_label_str else 'retain'}:{'dev' if pre.mask_dev[i] else 'test'}" for i, lbl in enumerate(all_labels)]
        
        # Debug: Count forget/retain samples
        forget_count = sum(1 for s in sets if s.startswith('forget'))
        retain_count = sum(1 for s in sets if s.startswith('retain'))
        # Check minimum sample requirement
        if forget_count < 5:
            console.print(f"WARNING: Forget class '{forget_label_str}' has only {forget_count} samples, minimum 5 required. Skipping this class.", style="red")
            continue
            
        M_ret_dev = np.array([s == 'retain:dev' for s in sets], dtype=bool)
        M_fgt_dev = np.array([s == 'forget:dev' for s in sets], dtype=bool)
        M_ret_test = np.array([s == 'retain:test' for s in sets], dtype=bool)
        M_fgt_test = np.array([s == 'forget:test' for s in sets], dtype=bool)

        # Feature mining
        A_ret = A_all[M_ret_dev]
        A_fgt = A_all[M_fgt_dev]
        
        # Check for dead features (never activated)
        ret_active_features = (A_ret > 1e-6).sum(axis=0)
        fgt_active_features = (A_fgt > 1e-6).sum(axis=0)
        dead_features = (ret_active_features == 0) & (fgt_active_features == 0)
        # Get statistical method from config, default to cohens_d
        method = cfg['feature_mining'].get('statistical_method', 'cohens_d')
        balance_negs = cfg['feature_mining'].get('balance_negatives', False)
        if balance_negs and A_ret.shape[0] > 0 and A_fgt.shape[0] > 0 and A_ret.shape[0] != A_fgt.shape[0]:
            rng = np.random.default_rng(seed)
            sel = rng.choice(A_ret.shape[0], size=min(A_ret.shape[0], A_fgt.shape[0]), replace=False)
            A_ret_bal = A_ret[sel]
        else:
            A_ret_bal = A_ret
        stats = compute_feature_stats(A_fgt, A_ret_bal, method=method)
        fdr_thresh = cfg['feature_mining'].get('fdr', 0.05)
        feat_idx = select_features(stats, method=method, top_k=cfg['feature_mining']['build_U_top'], fdr=fdr_thresh)
        
        if len(feat_idx) == 0:
            print(f"ERROR: No discriminative features found for forget class '{forget_label_str}'. Cannot proceed with editing.")
            continue
            
        prune_and_normalize_decoder(dec_w)
        U = build_U_from_decoder(dec_w, feat_idx)

        # LEACE params - create labels for the full dataset
        Y_all = np.array([1 if str(all_labels[i]) == forget_label_str else 0 for i in range(len(all_labels))], dtype=int).reshape(-1, 1)
        E_edit, info = apply_edits(
            pre.Z_all, sae, feat_idx, U, pre.mu, pre.W,
            cfg['edits']['beta'], cfg['edits']['leace'], {'Y_dev': Y_all, 'ridge': cfg['edits']['leace_ridge']}, cfg['edits']['noise_sigma']
        )
        E_all_edited = E_edit
        E_all_norm2 = E_all_edited / (np.linalg.norm(E_all_edited, axis=1, keepdims=True) + 1e-12)

        # EERs
        test_mask_for_trials = M_ret_test | M_fgt_test
        # Use all pos/neg pairs by default (set to 0 or None to mean "no cap")
        max_pairs = cfg['orchestration'].get('max_pairs_per_class', 0)
        trials, tlabels = build_verification_trials(all_labels, [forget_label_str], test_mask_for_trials, seed=seed, max_pairs_per_class=max_pairs)
        pos_count = sum(1 for t in trials if all_labels[t[0]] == all_labels[t[1]] == forget_label_str)
        neg_count = len(trials) - pos_count
        scores_before = compute_scores(E_all_norm, trials)
        scores_after = compute_scores(E_all_norm2, trials)
        eer_before = eer_from_scores(scores_before, tlabels)
        eer_after = eer_from_scores(scores_after, tlabels)
        per_class = per_class_eers(E_all_norm, E_all_norm2, all_labels, mask=test_mask_for_trials, rng_seed=seed)
        per_class_ret = {c: vals for c, vals in per_class.items() if str(c) != forget_label_str}
        per_class_fgt = {c: vals for c, vals in per_class.items() if str(c) == forget_label_str}
        retain_avg_before, retain_avg_after = avg_eers(per_class_ret)
        forget_avg_before, forget_avg_after = avg_eers(per_class_fgt)

        # Classifier acc
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

        # Probe AUC
        split_mask_dict = {'dev': pre.mask_dev, 'test': pre.mask_test}
        _, auc_before = safe_linear_probe(E_all_norm, all_labels, U, split_mask_dict, [forget_label_str])
        _, auc_after = safe_linear_probe(E_all_norm2, all_labels, U, split_mask_dict, [forget_label_str])

        res = {
            "FA_before": acc_fgt_before, "FA_after": acc_fgt_after,
            "RA_before": acc_ret_before, "RA_after": acc_ret_after,
            "forget_EER_before": forget_avg_before, "forget_EER_after": forget_avg_after,
            "retain_EER_before": retain_avg_before, "retain_EER_after": retain_avg_after,
            "linear_probe_auc_before": auc_before, "linear_probe_auc_after": auc_after,
            "EER_before_raw": eer_before, "EER_after_raw": eer_after,
        }
        results_all[forget_label] = res

        # save full artifacts for interpretability and downstream analyses
        run_save_dir = os.path.join(save_dir, f"seed{seed}_forget_{str(forget_label)}")
        os.makedirs(run_save_dir, exist_ok=True)
        with open(os.path.join(run_save_dir, "verification_eer_summary.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        # interpretability dump: Z_all, Z_edit, U, labels, sets, manifest, forget_labels
        sets_np = np.array(sets)
        np.savez_compressed(
            os.path.join(run_save_dir, "interpretability_data.npz"),
            Z_all=pre.Z_all, Z_edit=E_all_edited, U=U,
            labels=np.array(all_labels), sets=sets_np, manifest=np.array(pre.manifest, dtype=object), forget_spks=np.array([forget_label])
        )
        # SAE and whitener
        import torch
        torch.save(sae.state_dict(), os.path.join(run_save_dir, "sae_state_dict.pt"))
        np.savez_compressed(os.path.join(run_save_dir, "whitener.npz"), mu=pre.mu, W=pre.W)
        
        # Create embedding visualizations
        print(f"[VIZ] Creating embedding visualizations for forget label: {forget_label}")
        try:
            viz_dir = os.path.join(run_save_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Filter to only use test data for visualizations
            test_mask = pre.mask_test
            E_test_before = E_all_norm[test_mask]
            E_test_after = E_all_norm2[test_mask]
            test_labels = [all_labels[i] for i in range(len(all_labels)) if test_mask[i]]
            
            print(f"[VIZ] Using {len(E_test_before)} test samples for visualizations")
            
            # Create visualizations using test-only normalized embeddings
            plot_paths = create_embedding_visualizations(embeddings_before=E_test_before, embeddings_after=E_test_after, labels=test_labels, forget_labels=[forget_label], save_dir=viz_dir, title_prefix=f"Class {forget_label}")
            
            # Create summary statistics using test-only data
            stats = create_embedding_summary_stats(embeddings_before=E_test_before, embeddings_after=E_test_after, labels=test_labels, forget_labels=[forget_label], save_dir=viz_dir)
            
            # Add visualization paths to results
            res["visualization_paths"] = plot_paths
            res["embedding_statistics"] = stats
            
            print(f"[VIZ] Visualizations saved to: {viz_dir}")
            for plot_name, plot_path in plot_paths.items():
                print(f"  - {plot_name}: {plot_path}")
            
            # Print summary information
            if 'summary' in plot_paths:
                print(f"[VIZ] Summary page: {plot_paths['summary']}")
            print(f"[VIZ] Statistics: {stats}")
                
        except Exception as e:
            print(f"[VIZ] Warning: Failed to create visualizations: {e}")
            res["visualization_error"] = str(e)

    return results_all