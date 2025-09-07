import os
import sys
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# Ensure src/ is on sys.path for local module imports
_THIS_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..', 'src'))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)

from common import (
    seed_all, extract_embeddings, fit_whitener, apply_whitener, invert_whitener,
    SAE, train_sae, sae_encode_batch, prune_and_normalize_decoder,
    feature_stats, select_features_from_stats, build_U_from_decoder,
    leace_projector_in_U, apply_subspace_leace, apply_sauce_gating,
    inject_noise_in_subspace, l2_normalize,
    compute_scores, eer_from_scores, calculate_monosemanticity_score, compute_monosemanticity_statistics,
    train_classifier, evaluate_classifier
)
from validation_checks import (
    run_validation_checks, create_all_plots,
    decoder_metrics, activation_sparsity_metrics,
    check_U, energy_share_in_U,
    leace_invariants, check_projector_properties
)


# --------------------------- Small utility helpers ---------------------------

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_csv(rows, path, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows, columns=header).to_csv(path, index=False)


# --------------------------- Interpretability helpers ---------------------------

def build_feature_cards(A_all, label_binary, dec_w, feat_idx, metadata, save_dir):
    rows = []
    X = A_all
    y = np.array(label_binary)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.feature_selection import mutual_info_classif
    except Exception:
        LogisticRegression = None
        roc_auc_score = None
        mutual_info_classif = None

    for f in feat_idx:
        x_f = X[:, f]
        usage_rate = float((x_f > 0).mean())

        auc = float('nan')
        mi = float('nan')
        try:
            if LogisticRegression is not None and roc_auc_score is not None and mutual_info_classif is not None:
                if len(np.unique(y)) > 1:
                    clf = LogisticRegression(max_iter=500)
                    clf.fit(x_f.reshape(-1, 1), y)
                    if hasattr(clf, 'predict_proba'):
                        probs = clf.predict_proba(x_f.reshape(-1, 1))[:, 1]
                    else:
                        probs = clf.decision_function(x_f.reshape(-1, 1))
                    auc = float(roc_auc_score(y, probs))
                    mi = float(mutual_info_classif(x_f.reshape(-1, 1), y, discrete_features=False)[0])
        except Exception:
            pass

        top_idx = np.argsort(-x_f)[:24]
        prototypes = [int(i) for i in top_idx.tolist()]

        try:
            if dec_w.ndim == 2 and f < dec_w.shape[0]:
                atom = dec_w[f]
            else:
                atom = np.zeros(X.shape[1])
        except Exception:
            atom = np.zeros(X.shape[1])

        pos_mean = X[y == 1].mean(axis=0) if np.any(y == 1) else np.zeros(X.shape[1])
        neg_mean = X[y == 0].mean(axis=0) if np.any(y == 0) else np.zeros(X.shape[1])
        diff_vec = pos_mean - neg_mean
        cos_dir = float(np.dot(atom, diff_vec) / (np.linalg.norm(atom) * np.linalg.norm(diff_vec) + 1e-12))

        rows.append({
            'feature': int(f),
            'usage_rate': usage_rate,
            'auc_for_forget': auc,
            'mutual_info_for_forget': mi,
            'cos_with_diff': cos_dir,
            'top_prototypes': prototypes
        })

    save_csv([[r['feature'], r['usage_rate'], r['auc_for_forget'], r['mutual_info_for_forget'], r['cos_with_diff'], r['top_prototypes']] for r in rows],
             os.path.join(save_dir, 'feature_cards.csv'),
             header=['feature', 'usage_rate', 'auc_for_forget', 'mi_for_forget', 'cos_with_diff', 'top_prototypes'])
    save_json(rows, os.path.join(save_dir, 'feature_cards.json'))


def polysemanticity_checks(A_all, labels, feat_idx, save_dir):
    rows = []
    labels = np.array(labels)
    for f in feat_idx:
        x_f = A_all[:, f]
        ent = 0.0
        for c in np.unique(labels):
            mask = labels == c
            if mask.sum() == 0:
                continue
            p = float((x_f[mask] > 0).mean())
            if p > 0 and p < 1:
                ent += -(p * np.log(p) + (1 - p) * np.log(1 - p))
        rows.append([int(f), float(ent)])
    save_csv(rows, os.path.join(save_dir, 'polysemanticity.csv'), header=['feature', 'entropy'])


def causal_tests(sae, Z_all, mu, W, trials, labels, feat_idx, save_dir, device, max_feats=50):
    rows = []
    A_all = sae_encode_batch(sae, Z_all, device=device, batch=8192)
    dec_w = sae.dec.weight.detach().cpu().numpy()

    sel_feats = list(feat_idx)[:min(len(feat_idx), max_feats)]

    for f in sel_feats:
        A_ko = A_all.copy()
        A_ko[:, f] = 0.0
        try:
            Z_ko = A_ko @ dec_w
        except Exception:
            Z_ko = A_ko @ dec_w.T
        E_ko = invert_whitener(Z_ko, mu, W)
        E_ko_norm = l2_normalize(E_ko)
        scores_ko = compute_scores(E_ko_norm, trials)
        eer_ko = eer_from_scores(scores_ko, labels)

        delta = 0.1
        A_ki = A_all.copy()
        A_ki[:, f] += delta
        try:
            Z_ki = A_ki @ dec_w
        except Exception:
            Z_ki = A_ki @ dec_w.T
        E_ki = invert_whitener(Z_ki, mu, W)
        E_ki_norm = l2_normalize(E_ki)
        scores_ki = compute_scores(E_ki_norm, trials)
        eer_ki = eer_from_scores(scores_ki, labels)

        rows.append([int(f), float(eer_ko), float(eer_ki)])

    save_csv(rows, os.path.join(save_dir, 'causal_effects.csv'), header=['feature', 'eer_knockout', 'eer_knockin'])


def activation_patching(sae, Z_all, mu, W, trials, labels, forget_indices, retain_indices, save_dir, device, topm=8):
    rows = []
    A_all = sae_encode_batch(sae, Z_all, device=device, batch=8192)
    dec_w = sae.dec.weight.detach().cpu().numpy()

    for i in range(min(10, len(forget_indices))):
        for j in range(min(10, len(retain_indices))):
            f_idx = forget_indices[i]
            r_idx = retain_indices[j]
            top_feats = np.argsort(-A_all[f_idx])[:topm]
            A_patched = A_all.copy()
            A_patched[r_idx, top_feats] = A_all[f_idx, top_feats]
            try:
                Z_patched = A_patched @ dec_w
            except Exception:
                Z_patched = A_patched @ dec_w.T
            E_patched = invert_whitener(Z_patched, mu, W)
            E_patched_norm = l2_normalize(E_patched)
            scores_p = compute_scores(E_patched_norm, trials)
            eer_p = eer_from_scores(scores_p, labels)
            rows.append([int(r_idx), int(f_idx), float(eer_p)])

    save_csv(rows, os.path.join(save_dir, 'patching_results.csv'), header=['retain_idx', 'forget_idx', 'eer_after_patch'])


def top_neurons_analysis(A_all, labels, manifest, save_dir, label_field_name: str = 'label', top_neurons=32, top_samples_per_neuron=20, extra_manifest_keys: Optional[List[str]] = None):
    print(f"Analyzing top {top_neurons} neurons and top {top_samples_per_neuron} samples per neuron...")

    mean_activations = A_all.mean(axis=0)
    top_neuron_indices = np.argsort(-mean_activations)[:top_neurons]

    neuron_analysis = []
    if extra_manifest_keys is None:
        extra_manifest_keys = []

    for neuron_idx in top_neuron_indices:
        neuron_activations = A_all[:, neuron_idx]
        top_sample_indices = np.argsort(-neuron_activations)[:top_samples_per_neuron]
        top_samples_info = []
        for sample_idx in top_sample_indices:
            info = {
                'sample_index': int(sample_idx),
                'activation_value': float(neuron_activations[sample_idx]),
                'label': labels[sample_idx]
            }
            # Keep backward-compatible alias (speaker_id/command_id/digit_id) when requested
            if label_field_name and label_field_name != 'label':
                info[label_field_name] = labels[sample_idx]
            for k in ['set', 'split', 'path'] + list(extra_manifest_keys):
                if isinstance(manifest[sample_idx], dict) and k in manifest[sample_idx]:
                    info[k] = manifest[sample_idx][k]
            top_samples_info.append(info)

        neuron_info = {
            'neuron_index': int(neuron_idx),
            'mean_activation': float(mean_activations[neuron_idx]),
            'max_activation': float(neuron_activations.max()),
            'usage_rate': float((neuron_activations > 0).mean()),
            'top_samples': top_samples_info
        }
        neuron_analysis.append(neuron_info)

    save_json(neuron_analysis, os.path.join(save_dir, 'top_neurons_analysis.json'))

    neuron_summary_rows = []
    for neuron_info in neuron_analysis:
        neuron_summary_rows.append([
            neuron_info['neuron_index'],
            neuron_info['mean_activation'],
            neuron_info['max_activation'],
            neuron_info['usage_rate'],
            len(neuron_info['top_samples'])
        ])
    save_csv(neuron_summary_rows, os.path.join(save_dir, 'top_neurons_summary.csv'), header=['neuron_index', 'mean_activation', 'max_activation', 'usage_rate', 'num_top_samples'])

    sample_rows = []
    for neuron_info in neuron_analysis:
        for sample_info in neuron_info['top_samples']:
            row = [
                neuron_info['neuron_index'],
                sample_info.get('sample_index'),
                sample_info.get('activation_value'),
                sample_info.get('label')
            ]
            row.extend([sample_info.get('set'), sample_info.get('split'), sample_info.get('path')])
            for k in extra_manifest_keys:
                row.append(sample_info.get(k))
            sample_rows.append(row)
    header = ['neuron_index', 'sample_index', 'activation_value', 'label', 'set', 'split', 'path'] + list(extra_manifest_keys)
    save_csv(sample_rows, os.path.join(save_dir, 'top_neurons_samples.csv'), header=header)

    print(f"Top neurons analysis completed. Results saved to {save_dir}")
    return neuron_analysis


# --------------------------- Three-stage LEACE testing ---------------------------

def run_three_stage_leace_pipeline(entries: List[dict], cfg: Dict, dataset_api):
    """
    Run LEACE testing across three stages:
    1. Original embeddings (baseline)
    2. LEACE applied directly on original embeddings (bypassing SAE)
    3. LEACE applied in SAE space (current implementation)
    
    Returns comprehensive comparison results across all three stages.
    """
    seed_all(cfg['seed'])
    save_dir = cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    t0_all = time.time()

    label_key = cfg['label_key']

    # Decide forget/retain sets
    if cfg.get('forget_classes') is None:
        forget_classes = _choose_forget_classes(entries, label_key, cfg['retain_frac'], cfg['seed'])
    else:
        forget_classes = list(cfg['forget_classes'])

    manifest = []
    for entry in entries:
        e2 = entry.copy()
        e2['set'] = 'forget' if entry[label_key] in forget_classes else 'retain'
        manifest.append(e2)

    dev_manifest = [m for m in manifest if m['split'] == 'dev']
    test_manifest = [m for m in manifest if m['split'] == 'test']

    # Extract embeddings
    if cfg.get('embeddings_path'):
        data = np.load(cfg['embeddings_path'], allow_pickle=True)
        embs = data['embeddings']
        seg_ids = data.get('segment_ids', None)
        if seg_ids is not None:
            id_to_idx = {str(sid): i for i, sid in enumerate(seg_ids)}
            E_all = np.zeros((len(manifest), embs.shape[1]), dtype=np.float32)
            for i, entry in enumerate(manifest):
                sid = entry.get('segment_id', None)
                if sid is None:
                    raise RuntimeError('Embeddings provided but manifest entries lack segment_id')
                if str(sid) not in id_to_idx:
                    raise RuntimeError(f'segment_id {sid} not found')
                E_all[i] = embs[id_to_idx[str(sid)]]
            L_all = [m[label_key] for m in manifest]
        else:
            if embs.shape[0] != len(manifest):
                raise RuntimeError('Embeddings length mismatch')
            E_all = embs.astype(np.float32)
            L_all = [m[label_key] for m in manifest]
    else:
        all_manifest = dev_manifest + test_manifest
        E_all, L_all, _ = extract_embeddings(
            all_manifest,
            encoder=cfg['encoder'], device=cfg['device'],
            wav2vec2_model=cfg.get('wav2vec2_model'),
            hubert_model=cfg.get('hubert_model'),
            conformer_model=cfg.get('conformer_model'),
            label_key=label_key
        )
        manifest = list(dev_manifest) + list(test_manifest)

    labels = L_all
    sets = [f"{m['set']}:{m['split']}" for m in manifest]
    M_ret_dev = np.array([s == 'retain:dev' for s in sets], dtype=bool)
    M_fgt_dev = np.array([s == 'forget:dev' for s in sets], dtype=bool)
    M_ret_test = np.array([s == 'retain:test' for s in sets], dtype=bool)
    M_fgt_test = np.array([s == 'forget:test' for s in sets], dtype=bool)

    # Prepare data for all three stages
    E_all_norm = l2_normalize(E_all)
    
    # Stage 1: Original embeddings (baseline)
    print("=== STAGE 1: Original Embeddings (Baseline) ===")
    stage1_results = evaluate_stage(E_all_norm, E_all_norm, labels, forget_classes, M_ret_test | M_fgt_test, dataset_api, cfg['seed'], cfg['device'])
    
    # Stage 2: Direct LEACE on original embeddings
    print("=== STAGE 2: Direct LEACE on Original Embeddings ===")
    E_direct_leace = apply_direct_leace_on_embeddings(E_all, labels, forget_classes, M_ret_dev | M_fgt_dev, cfg)
    E_direct_leace_norm = l2_normalize(E_direct_leace)
    stage2_results = evaluate_stage(E_all_norm, E_direct_leace_norm, labels, forget_classes, M_ret_test | M_fgt_test, dataset_api, cfg['seed'], cfg['device'])
    
    # Stage 3: SAE-based LEACE (current implementation)
    print("=== STAGE 3: SAE-based LEACE (Current Implementation) ===")
    E_sae_leace, ms_stats = apply_sae_based_leace(E_all, labels, forget_classes, 
                                        M_ret_dev, M_fgt_dev, M_ret_test, M_fgt_test, cfg)
    E_sae_leace_norm = l2_normalize(E_sae_leace)
    stage3_results = evaluate_stage(E_all_norm, E_sae_leace_norm, labels, forget_classes, M_ret_test | M_fgt_test, dataset_api, cfg['seed'], cfg['device'])
    
    # Add monosemanticity stats to stage3 results
    stage3_results.update({
        'ms_original_mean': ms_stats['ms_original_stats']['mean'],
        'ms_original_std': ms_stats['ms_original_stats']['std'],
        'ms_original_min': ms_stats['ms_original_stats']['min'],
        'ms_original_max': ms_stats['ms_original_stats']['max'],
        'ms_sae_mean': ms_stats['ms_sae_stats']['mean'],
        'ms_sae_std': ms_stats['ms_sae_stats']['std'],
        'ms_sae_min': ms_stats['ms_sae_stats']['min'],
        'ms_sae_max': ms_stats['ms_sae_stats']['max'],
        'ms_final_mean': ms_stats['ms_final_stats']['mean'],
        'ms_final_std': ms_stats['ms_final_stats']['std'],
        'ms_final_min': ms_stats['ms_final_stats']['min'],
        'ms_final_max': ms_stats['ms_final_stats']['max'],
    })
    
    # Compile comprehensive results
    comprehensive_results = {
        'stage1_original': stage1_results,
        'stage2_direct_leace': stage2_results,
        'stage3_sae_leace': stage3_results,
        'forget_classes': forget_classes,
        'config': cfg
    }
    
    # Save results
    save_json(comprehensive_results, os.path.join(save_dir, 'three_stage_results.json'))
    
    # Create comparison summary
    comparison_summary = create_comparison_summary(stage1_results, stage2_results, stage3_results)
    save_json(comparison_summary, os.path.join(save_dir, 'comparison_summary.json'))
    
    # Create comprehensive metrics table
    metrics_table = create_metrics_comparison_table(stage1_results, stage2_results, stage3_results)
    save_csv(metrics_table, os.path.join(save_dir, 'metrics_comparison.csv'), header=['Metric', 'Stage1_Original', 'Stage2_Direct_LEACE', 'Stage3_SAE_LEACE'])
    
    # Create visualization plots
    try:
        create_comparison_plots(stage1_results, stage2_results, stage3_results, save_dir)
    except Exception as e:
        print(f"Failed to create comparison plots: {e}")
    
    # Full validation and interpretability suite for SAE+LEACE stage
    try:
        print("Running full validation and interpretability suite for SAE+LEACE stage...")
        validation_dir = os.path.join(save_dir, 'validation_stage3')
        os.makedirs(validation_dir, exist_ok=True)

        # Re-run SAE-based LEACE to get all intermediate variables
        # (copy the logic from apply_sae_based_leace, but keep all intermediates)
        # --- Whitening ---
        E_ret_train = E_all[M_ret_dev]
        mu, W = fit_whitener(E_ret_train)
        Z_all_whitened = apply_whitener(E_all, mu, W)
        # --- SAE ---
        Z_train = Z_all_whitened[M_ret_dev | M_fgt_dev]
        perm = np.random.permutation(Z_train.shape[0])
        cut = int(0.8 * len(perm))
        Z_sae_train, Z_sae_val = Z_train[perm[:cut]], Z_train[perm[cut:]]
        sae = SAE(input_dim=Z_all_whitened.shape[1], hidden_mult=cfg['sae_multiplier'], k=cfg['sae_topk'], variant=cfg.get('sae_variant', 'topk'))
        sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=cfg['sae_epochs'], batch_size=64, lr=1e-3, device=cfg['device'], lambda_sparsity=cfg.get('lambda_sparsity', 0.0))
        dec_w = sae.dec.weight.detach().cpu().numpy()
        # --- SAE encodes ---
        Z_ret_dev = Z_all_whitened[M_ret_dev]
        Z_fgt_dev = Z_all_whitened[M_fgt_dev]
        A_ret = sae_encode_batch(sae, Z_ret_dev, device=cfg['device'], batch=8192)
        A_fgt = sae_encode_batch(sae, Z_fgt_dev, device=cfg['device'], batch=8192)
        A_all = sae_encode_batch(sae, Z_all_whitened, device=cfg['device'], batch=8192)
        # --- Feature selection ---
        stats = feature_stats(A_fgt, A_ret)
        feat_idx = select_features_from_stats(stats, top_k=cfg['build_U_top'], fdr=0.05)
        # --- Build U ---
        U = build_U_from_decoder(dec_w, feat_idx)
        # --- LEACE ---
        mask_dev = M_ret_dev | M_fgt_dev
        dev_indices = np.where(mask_dev)[0]
        Y_dev = np.array([1 if labels[i] in forget_classes else 0 for i in dev_indices], dtype=int).reshape(-1, 1)
        if cfg.get('leace', True):
            try:
                P_r, mu_r = leace_projector_in_U(Z_all_whitened[mask_dev], U, Y_dev, ridge=cfg.get('leace_ridge', 1e-6))
            except Exception as e:
                print(f"  Warning: LEACE computation failed ({e}), falling back to identity projector")
                P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
                mu_r = np.zeros((1, U.shape[1]))
        else:
            P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
            mu_r = np.zeros((1, U.shape[1]))
        # --- Apply edits ---
        Z_edit = Z_all_whitened.copy()
        if cfg.get('beta', 0.0) > 0.0:
            Z_edit = apply_sauce_gating(sae, Z_all_whitened, feat_idx, beta=cfg['beta'], device=cfg['device'], batch=8192)
        if cfg.get('leace', True):
            Z_edit = apply_subspace_leace(Z_edit, U, P_r, mu_r)
        if cfg.get('noise_sigma', 0.0) > 0:
            Z_edit = inject_noise_in_subspace(Z_edit, U, sigma=cfg['noise_sigma'])
        E_edit = invert_whitener(Z_edit, mu, W)
        E_all_norm = l2_normalize(E_all)
        E_edit_norm = l2_normalize(E_edit)
        sets = [f"{m['set']}:{m['split']}" for m in manifest]
        mask_dev = np.array([s.endswith(':dev') for s in sets], dtype=bool)
        mask_test = M_ret_test | M_fgt_test
        # --- Validation checks ---
        run_validation_checks(
            save_dir=validation_dir,
            Z_all=Z_all_whitened, Z_edit=Z_edit,
            E_all_norm=E_all_norm, E_edit_norm=E_edit_norm,
            ZU_mean=(Z_all_whitened[mask_dev] @ U).mean(0, keepdims=True) if U.shape[1] > 0 else None,
            U=U, P_r=P_r, mu_r=mu_r, dec_w=dec_w, sae=sae,
            feat_idx=feat_idx, sets=sets, labels=labels,
            mask_dev=mask_dev, mask_test=mask_test,
            y_dev_bin=Y_dev, leace_ridge=cfg['leace_ridge'],
            mu=mu, W=W, device=cfg['device']
        )
        # --- All plots ---
        Z_before = Z_all_whitened[mask_dev]
        Z_after = Z_edit[mask_dev]
        ZU_before = Z_before @ U if U.shape[1] > 0 else Z_before[:, :0]
        ZU_after = Z_after @ U if U.shape[1] > 0 else Z_after[:, :0]
        create_all_plots(
            Z_before, Z_after, ZU_before, ZU_after,
            labels=[labels[i] for i in np.where(mask_dev)[0]],
            forget_classes=forget_classes,
            decoder_weights=dec_w, feat_idx=feat_idx,
            A_retain=A_ret, A_forget=A_fgt,
            save_dir=validation_dir, random_state=cfg['seed']
        )
        # --- Interpretability suite ---
        interp_dir = os.path.join(validation_dir, 'interpretability')
        os.makedirs(interp_dir, exist_ok=True)
        label_binary = np.array([1 if c in forget_classes else 0 for c in labels])
        try:
            build_feature_cards(A_all, label_binary, dec_w, feat_idx, None, interp_dir)
        except Exception as e:
            print('feature card build failed', e)
        try:
            polysemanticity_checks(A_all, np.array(labels), feat_idx, interp_dir)
        except Exception as e:
            print('polysemanticity failed', e)
        try:
            test_trials, test_labels = dataset_api.build_verification_trials(labels, forget_classes, mask_test)
            causal_tests(sae, Z_all_whitened, mu, W, test_trials, test_labels, feat_idx, interp_dir, cfg['device'], max_feats=100)
        except Exception as e:
            print('causal tests failed', e)
        try:
            forget_indices = np.where(M_fgt_test)[0].tolist()
            retain_indices = np.where(M_ret_test)[0].tolist()
            test_trials, test_labels = dataset_api.build_verification_trials(labels, forget_classes, mask_test)
            activation_patching(sae, Z_all_whitened, mu, W, test_trials, test_labels, forget_indices, retain_indices, interp_dir, cfg['device'], topm=8)
        except Exception as e:
            print('patching failed', e)
        try:
            top_neurons_analysis(A_all, labels, manifest, interp_dir)
        except Exception as e:
            print('top neurons analysis failed', e)
    except Exception as e:
        print(f"Failed to run full validation suite for SAE+LEACE stage: {e}")
    
    print(f"Three-stage LEACE pipeline finished in {time.time() - t0_all:.1f}s")
    print(f"Results saved to {save_dir}")
    
    return comprehensive_results


def apply_direct_leace_on_embeddings(E_all, labels, forget_classes, dev_mask, cfg):
    """
    Apply LEACE directly on original embeddings using PCA for dimensionality reduction.
    This bypasses the SAE entirely.
    """
    print("Applying LEACE directly on original embeddings...")
    
    # Use PCA to reduce dimensionality for LEACE computation
    from sklearn.decomposition import PCA
    
    # Get retain dev indices (simpler approach)
    dev_indices = np.where(dev_mask)[0]
    retain_dev_indices = [i for i in dev_indices if labels[i] not in forget_classes]
    
    if len(retain_dev_indices) == 0:
        print("Warning: No retain dev data found, using all dev data for PCA")
        retain_dev_indices = dev_indices
    
    E_retain_dev = E_all[retain_dev_indices]
    
    # Determine PCA components (use same dimensionality as SAE hidden layer)
    pca_components = min(cfg.get('sae_multiplier', 4.0) * E_all.shape[1], E_retain_dev.shape[0] - 1, E_retain_dev.shape[1])
    pca_components = max(1, int(pca_components))
    
    print(f"Using PCA with {pca_components} components")
    pca = PCA(n_components=pca_components)
    pca.fit(E_retain_dev)
    
    # Transform all data to PCA space
    Z_pca = pca.transform(E_all)
    
    # Create U matrix from PCA components
    U_pca = pca.components_.T  # [original_dim, pca_components]
    
    # Prepare labels for LEACE
    dev_indices = np.where(dev_mask)[0]
    Y_dev = np.array([1 if labels[i] in forget_classes else 0 for i in dev_indices], dtype=int).reshape(-1, 1)
    
    # Compute LEACE projector in PCA space
    try:
        P_r, mu_r = leace_projector_in_U(Z_pca[dev_mask], U_pca, Y_dev, ridge=cfg.get('leace_ridge', 1e-6))
        print(f"LEACE projector computed successfully in PCA space")
    except Exception as e:
        print(f"LEACE computation failed: {e}, using identity projector")
        P_r = np.eye(U_pca.shape[1])
        mu_r = np.zeros((1, U_pca.shape[1]))
    
    # Apply LEACE in PCA space
    Z_pca_edit = apply_subspace_leace(Z_pca, U_pca, P_r, mu_r)
    
    # Transform back to original embedding space
    E_edit = pca.inverse_transform(Z_pca_edit)
    
    return E_edit


def apply_sae_based_leace(E_all, labels, forget_classes, M_ret_dev, M_fgt_dev, M_ret_test, M_fgt_test, cfg):
    """
    Apply the current SAE-based LEACE implementation.
    This matches the voxceleb.py pipeline structure exactly.
    """
    print("Applying SAE-based LEACE...")
    
    # Whitening on retain-dev (matches voxceleb.py step 3)
    print("Fitting whitener on Train-retain (retain-dev)...")
    E_ret_train = E_all[M_ret_dev]
    if E_ret_train.shape[0] < 2: 
        raise RuntimeError("Not enough retain-dev samples to fit whitener")
    mu, W = fit_whitener(E_ret_train)
    Z_all = apply_whitener(E_all, mu, W)
    
    # SAE training on dev (matches voxceleb.py step 4)
    print("Training SAE on Z_train (dev union: retain+forget)...")
    Z_train = Z_all[M_ret_dev | M_fgt_dev]
    perm = np.random.permutation(Z_train.shape[0])
    cut = int(0.8 * len(perm))
    Z_sae_train, Z_sae_val = Z_train[perm[:cut]], Z_train[perm[cut:]]
    sae = SAE(input_dim=Z_all.shape[1], hidden_mult=cfg['sae_multiplier'], k=cfg['sae_topk'], variant=cfg.get('sae_variant', 'topk'))
    sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=cfg['sae_epochs'], batch_size=4096, lr=1e-3, device=cfg['device'], lambda_sparsity=cfg.get('lambda_sparsity', 0.0))
    
    # SAE encodes for feature mining (matches voxceleb.py step 5)
    print("Encoding SAE activations for train-retain and train-forget...")
    Z_ret_dev = Z_all[M_ret_dev]
    Z_fgt_dev = Z_all[M_fgt_dev]
    A_ret = sae_encode_batch(sae, Z_ret_dev, device=cfg['device'], batch=8192)
    A_fgt = sae_encode_batch(sae, Z_fgt_dev, device=cfg['device'], batch=8192)
    
    # Calculate Monosemanticity Score for original embeddings (matches voxceleb.py step 5.1)
    print("Computing Monosemanticity Score for original embeddings...")
    E_all_norm = l2_normalize(E_all)
    ms_original = calculate_monosemanticity_score(E_all_norm, E_all_norm, top_m=20, min_activations=5)
    ms_original_stats = compute_monosemanticity_statistics(ms_original)
    print(f"  Original embeddings MS: mean={ms_original_stats['mean']:.4f}, std={ms_original_stats['std']:.4f}, "
          f"min={ms_original_stats['min']:.4f}, max={ms_original_stats['max']:.4f}")
    
    # Calculate Monosemanticity Score for SAE activations (matches voxceleb.py step 5.2)
    print("Computing Monosemanticity Score for SAE activations...")
    A_all = sae_encode_batch(sae, Z_all, device=cfg['device'], batch=8192)
    ms_sae = calculate_monosemanticity_score(A_all, E_all_norm, top_m=20, min_activations=5)
    ms_sae_stats = compute_monosemanticity_statistics(ms_sae)
    print(f"  SAE activations MS: mean={ms_sae_stats['mean']:.4f}, std={ms_sae_stats['std']:.4f}, "
          f"min={ms_sae_stats['min']:.4f}, max={ms_sae_stats['max']:.4f}")
    
    # Feature stats & selection (matches voxceleb.py step 6)
    print("Computing feature statistics and selecting features...")
    stats = feature_stats(A_fgt, A_ret)
    feat_idx = select_features_from_stats(stats, top_k=cfg['build_U_top'], fdr=0.05)
    
    # Build U subspace (matches voxceleb.py step 7)
    print("Building U subspace...")
    dec_w = sae.dec.weight.detach().cpu().numpy()
    dec_pruned = prune_and_normalize_decoder(dec_w, prune_threshold=1e-6)
    U = build_U_from_decoder(dec_w, feat_idx)
    print(f"  U shape: {U.shape}")
    
    # Compute LEACE in U (matches voxceleb.py step 8)
    P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
    mu_r = None
    if cfg.get('leace', True):
        print("Computing LEACE projector in U...")
        mask_dev = M_ret_dev | M_fgt_dev
        dev_indices = np.where(mask_dev)[0]
        Y_dev = np.array([1 if labels[i] in forget_classes else 0 for i in dev_indices], dtype=int).reshape(-1, 1)
        
        if Y_dev.shape[0] != Z_all[mask_dev].shape[0]: 
            raise RuntimeError("Mismatch between Y_dev and Z_all[mask_dev] lengths")
        try:
            P_r, mu_r = leace_projector_in_U(Z_all[mask_dev], U, Y_dev, ridge=cfg.get('leace_ridge', 1e-6))
        except Exception as e:
            print(f"  Warning: LEACE computation failed ({e}), falling back to identity projector")
            P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
            mu_r = np.zeros((1, U.shape[1]))
    else:
        print("Skipping LEACE (identity)")
        mask_dev = M_ret_dev | M_fgt_dev
        dev_indices = np.where(mask_dev)[0]
        Y_dev = np.array([1 if labels[i] in forget_classes else 0 for i in dev_indices], dtype=int).reshape(-1, 1)
    
    # Apply SAUCE gating and LEACE edit (matches voxceleb.py step 9)
    print("Applying SAUCE gating (beta) and LEACE edit...")
    Z_edit = Z_all.copy()
    if cfg.get('beta', 0.0) > 0.0:
        print(" applying SAUCE gating with beta", cfg['beta'])
        Z_edit = apply_sauce_gating(sae, Z_all, feat_idx, beta=cfg['beta'], device=cfg['device'], batch=8192)
    if cfg.get('leace', True):
        Z_edit = apply_subspace_leace(Z_edit, U, P_r, mu_r)
    if cfg.get('noise_sigma', 0.0) > 0:
        Z_edit = inject_noise_in_subspace(Z_edit, U, sigma=cfg['noise_sigma'])
    
    E_edit = invert_whitener(Z_edit, mu, W)
    E_edit_norm = l2_normalize(E_edit)
    
    # Calculate Monosemanticity Score for final edited embeddings (matches voxceleb.py step 9.1)
    print("Computing Monosemanticity Score for final edited embeddings...")
    ms_final = calculate_monosemanticity_score(E_edit_norm, E_edit_norm, top_m=20, min_activations=5)
    ms_final_stats = compute_monosemanticity_statistics(ms_final)
    print(f"  Final edited embeddings MS: mean={ms_final_stats['mean']:.4f}, std={ms_final_stats['std']:.4f}, "
          f"min={ms_final_stats['min']:.4f}, max={ms_final_stats['max']:.4f}")
    
    return E_edit, {
        'ms_original_stats': ms_original_stats,
        'ms_sae_stats': ms_sae_stats, 
        'ms_final_stats': ms_final_stats
    }


def evaluate_stage(E_before, E_after, labels, forget_classes, test_mask, dataset_api, seed, device='cuda'):
    """
    Evaluate a single stage with comprehensive metrics.
    """
    results = {}
    
    # Verification EERs
    test_trials, test_labels = dataset_api.build_verification_trials(labels, forget_classes, test_mask)
    scores_before = compute_scores(E_before, test_trials)
    scores_after = compute_scores(E_after, test_trials)
    eer_before = eer_from_scores(scores_before, test_labels)
    eer_after = eer_from_scores(scores_after, test_labels)
    
    results['EER_before'] = eer_before
    results['EER_after'] = eer_after
    
    # Calculate separate EERs for retain and forget classes
    retain_classes = [c for c in set(labels) if c not in set(forget_classes)]
    
    # For retain EER: build trials with retain classes as positive, forget classes as negative
    if len(retain_classes) > 0 and len(forget_classes) > 0:
        # Create a modified forget_classes list that includes retain classes as "forget" for this calculation
        # This way, retain classes become positive trials, forget classes become negative trials
        retain_trials, retain_trial_labels = dataset_api.build_verification_trials(labels, retain_classes, test_mask)
        if len(retain_trials) > 0:
            retain_scores_before = compute_scores(E_before, retain_trials)
            retain_scores_after = compute_scores(E_after, retain_trials)
            retain_eer_before = eer_from_scores(retain_scores_before, retain_trial_labels)
            retain_eer_after = eer_from_scores(retain_scores_after, retain_trial_labels)
        else:
            retain_eer_before = retain_eer_after = 0.0
    else:
        retain_eer_before = retain_eer_after = 0.0
    
    # For forget EER: build trials with forget classes as positive, retain classes as negative
    if len(forget_classes) > 0 and len(retain_classes) > 0:
        forget_trials, forget_trial_labels = dataset_api.build_verification_trials(labels, forget_classes, test_mask)
        if len(forget_trials) > 0:
            forget_scores_before = compute_scores(E_before, forget_trials)
            forget_scores_after = compute_scores(E_after, forget_trials)
            forget_eer_before = eer_from_scores(forget_scores_before, forget_trial_labels)
            forget_eer_after = eer_from_scores(forget_scores_after, forget_trial_labels)
        else:
            forget_eer_before = forget_eer_after = 0.0
    else:
        forget_eer_before = forget_eer_after = 0.0
    
    results.update({
        'retain_EER_before': retain_eer_before,
        'retain_EER_after': retain_eer_after,
        'forget_EER_before': forget_eer_before,
        'forget_EER_after': forget_eer_after,
    })
    
    # Linear probe accuracy - simplified approach
    try:
        # Create binary labels for forget vs retain
        binary_labels = np.array([1 if l in forget_classes else 0 for l in labels])
        test_indices = np.where(test_mask)[0]
        
        if len(test_indices) > 0 and len(np.unique(binary_labels[test_indices])) > 1:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            
            # Before editing
            X_before = E_before[test_indices]
            y = binary_labels[test_indices]
            
            clf_before = LogisticRegression(max_iter=1000, random_state=seed)
            clf_before.fit(X_before, y)
            acc_before = clf_before.score(X_before, y)
            
            if hasattr(clf_before, 'predict_proba'):
                prob_before = clf_before.predict_proba(X_before)[:, 1]
            else:
                prob_before = clf_before.decision_function(X_before)
            auc_before = roc_auc_score(y, prob_before)
            
            # After editing
            X_after = E_after[test_indices]
            clf_after = LogisticRegression(max_iter=1000, random_state=seed)
            clf_after.fit(X_after, y)
            acc_after = clf_after.score(X_after, y)
            
            if hasattr(clf_after, 'predict_proba'):
                prob_after = clf_after.predict_proba(X_after)[:, 1]
            else:
                prob_after = clf_after.decision_function(X_after)
            auc_after = roc_auc_score(y, prob_after)
        else:
            acc_before = acc_after = 0.5
            auc_before = auc_after = 0.5
            
    except Exception as e:
        print(f"Linear probe failed: {e}")
        acc_before = acc_after = 0.5
        auc_before = auc_after = 0.5
    
    results.update({
        'linear_probe_acc_before': acc_before,
        'linear_probe_acc_after': acc_after,
        'linear_probe_auc_before': auc_before,
        'linear_probe_auc_after': auc_after,
    })
    
    # ResNet Classification (matches voxceleb.py step 12)
    print("Speaker classification accuracy (ResNet)...")
    all_classes = sorted(list(set(labels)))
    class2id = {c: i for i, c in enumerate(all_classes)}
    
    # Create train/test masks for classification
    train_mask = ~test_mask  # Use dev data for training
    test_indices = np.where(test_mask)[0]
    train_indices = np.where(train_mask)[0]
    
    if len(train_indices) > 0 and len(test_indices) > 0:
        # Train classifier on original embeddings (before editing)
        X_cls_train = E_before[train_indices]
        y_cls_train = [class2id[labels[i]] for i in train_indices]
        X_cls_test = E_before[test_indices]
        y_cls_test = [class2id[labels[i]] for i in test_indices]
        
        # Train classifier
        clf = train_classifier(X_cls_train, y_cls_train, X_cls_test, y_cls_test, 
                              num_classes=len(all_classes), device=device, epochs=20, batch_size=512, lr=1e-3)
        
        # Evaluate on retain/forget sets before and after editing
        retain_test_indices = [i for i in test_indices if labels[i] in retain_classes]
        forget_test_indices = [i for i in test_indices if labels[i] in forget_classes]
        
        if len(retain_test_indices) > 0:
            E_ret_test_before = E_before[retain_test_indices]
            L_ret_test = [class2id[labels[i]] for i in retain_test_indices]
            acc_ret_before = evaluate_classifier(clf, E_ret_test_before, L_ret_test, device=device)
            
            E_ret_test_after = E_after[retain_test_indices]
            acc_ret_after = evaluate_classifier(clf, E_ret_test_after, L_ret_test, device=device)
        else:
            acc_ret_before = acc_ret_after = 0.0
            
        if len(forget_test_indices) > 0:
            E_fgt_test_before = E_before[forget_test_indices]
            L_fgt_test = [class2id[labels[i]] for i in forget_test_indices]
            acc_fgt_before = evaluate_classifier(clf, E_fgt_test_before, L_fgt_test, device=device)
            
            E_fgt_test_after = E_after[forget_test_indices]
            acc_fgt_after = evaluate_classifier(clf, E_fgt_test_after, L_fgt_test, device=device)
        else:
            acc_fgt_before = acc_fgt_after = 0.0
            
        print(f"  Classifier Retain: before {acc_ret_before*100:.2f}% after {acc_ret_after*100:.2f}%")
        print(f"  Classifier Forget: before {acc_fgt_before*100:.2f}% after {acc_fgt_after*100:.2f}%")
        
        results.update({
            'resnet_acc_retain_before': acc_ret_before,
            'resnet_acc_retain_after': acc_ret_after,
            'resnet_acc_forget_before': acc_fgt_before,
            'resnet_acc_forget_after': acc_fgt_after,
        })
    else:
        print("  Warning: Not enough data for ResNet classification")
        results.update({
            'resnet_acc_retain_before': 0.0,
            'resnet_acc_retain_after': 0.0,
            'resnet_acc_forget_before': 0.0,
            'resnet_acc_forget_after': 0.0,
        })
    
    return results


def create_comparison_summary(stage1_results, stage2_results, stage3_results):
    """
    Create a summary comparing all three stages.
    """
    summary = {
        'stage1_original': {
            'EER_after': stage1_results['EER_after'],
            'retain_EER_after': stage1_results['retain_EER_after'],
            'forget_EER_after': stage1_results['forget_EER_after'],
            'linear_probe_acc_after': stage1_results['linear_probe_acc_after'],
            'linear_probe_auc_after': stage1_results['linear_probe_auc_after'],
            'resnet_acc_retain_after': stage1_results['resnet_acc_retain_after'],
            'resnet_acc_forget_after': stage1_results['resnet_acc_forget_after']
        },
        'stage2_direct_leace': {
            'EER_after': stage2_results['EER_after'],
            'retain_EER_after': stage2_results['retain_EER_after'],
            'forget_EER_after': stage2_results['forget_EER_after'],
            'linear_probe_acc_after': stage2_results['linear_probe_acc_after'],
            'linear_probe_auc_after': stage2_results['linear_probe_auc_after'],
            'resnet_acc_retain_after': stage2_results['resnet_acc_retain_after'],
            'resnet_acc_forget_after': stage2_results['resnet_acc_forget_after']
        },
        'stage3_sae_leace': {
            'EER_after': stage3_results['EER_after'],
            'retain_EER_after': stage3_results['retain_EER_after'],
            'forget_EER_after': stage3_results['forget_EER_after'],
            'linear_probe_acc_after': stage3_results['linear_probe_acc_after'],
            'linear_probe_auc_after': stage3_results['linear_probe_auc_after'],
            'resnet_acc_retain_after': stage3_results['resnet_acc_retain_after'],
            'resnet_acc_forget_after': stage3_results['resnet_acc_forget_after']
        }
    }
    
    return summary


def create_metrics_comparison_table(stage1_results, stage2_results, stage3_results):
    """
    Create a comprehensive metrics comparison table.
    """
    metrics = [
        ('EER_After', stage1_results['EER_after'], stage2_results['EER_after'], stage3_results['EER_after']),
        ('Retain_EER_After', stage1_results['retain_EER_after'], stage2_results['retain_EER_after'], stage3_results['retain_EER_after']),
        ('Forget_EER_After', stage1_results['forget_EER_after'], stage2_results['forget_EER_after'], stage3_results['forget_EER_after']),
        ('Linear_Probe_Acc_After', stage1_results['linear_probe_acc_after'], stage2_results['linear_probe_acc_after'], stage3_results['linear_probe_acc_after']),
        ('Linear_Probe_AUC_After', stage1_results['linear_probe_auc_after'], stage2_results['linear_probe_auc_after'], stage3_results['linear_probe_auc_after']),
        ('ResNet_Acc_Retain_After', stage1_results['resnet_acc_retain_after'], stage2_results['resnet_acc_retain_after'], stage3_results['resnet_acc_retain_after']),
        ('ResNet_Acc_Forget_After', stage1_results['resnet_acc_forget_after'], stage2_results['resnet_acc_forget_after'], stage3_results['resnet_acc_forget_after']),
    ]
    
    table_rows = []
    for metric_name, s1_val, s2_val, s3_val in metrics:
        table_rows.append([metric_name, f"{s1_val:.4f}", f"{s2_val:.4f}", f"{s3_val:.4f}"])
    
    return table_rows


def create_comparison_plots(stage1_results, stage2_results, stage3_results, save_dir):
    """
    Create comprehensive comparison plots using matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping plots")
        return
    
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    stages = ['Original', 'Direct LEACE', 'SAE LEACE']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    # Detailed Performance Breakdown
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 18))
    
    # Overall EER
    overall_eers = [stage1_results['EER_after'], stage2_results['EER_after'], stage3_results['EER_after']]
    bars1 = ax1.bar(stages, overall_eers, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_title('Final Overall EER', fontweight='bold')
    ax1.set_ylabel('EER')
    ax1.grid(True, alpha=0.3)
    for bar, value in zip(bars1, overall_eers):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    # Retain EER
    retain_eers = [stage1_results['retain_EER_after'], stage2_results['retain_EER_after'], stage3_results['retain_EER_after']]
    bars2 = ax2.bar(stages, retain_eers, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax2.set_title('Final Retain EER', fontweight='bold')
    ax2.set_ylabel('EER')
    ax2.grid(True, alpha=0.3)
    for bar, value in zip(bars2, retain_eers):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    # Forget EER
    forget_eers = [stage1_results['forget_EER_after'], stage2_results['forget_EER_after'], stage3_results['forget_EER_after']]
    bars3 = ax3.bar(stages, forget_eers, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax3.set_title('Final Forget EER', fontweight='bold')
    ax3.set_ylabel('EER')
    ax3.grid(True, alpha=0.3)
    for bar, value in zip(bars3, forget_eers):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    # Linear Probe Accuracy
    accuracies = [stage1_results['linear_probe_acc_after'], stage2_results['linear_probe_acc_after'], stage3_results['linear_probe_acc_after']]
    bars4 = ax4.bar(stages, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax4.set_title('Final Linear Probe Accuracy', fontweight='bold')
    ax4.set_ylabel('Accuracy')
    ax4.grid(True, alpha=0.3)
    for bar, value in zip(bars4, accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    # ResNet Retain Accuracy
    resnet_retain_accs = [stage1_results['resnet_acc_retain_after'], stage2_results['resnet_acc_retain_after'], stage3_results['resnet_acc_retain_after']]
    bars5 = ax5.bar(stages, resnet_retain_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax5.set_title('Final ResNet Retain Accuracy', fontweight='bold')
    ax5.set_ylabel('Accuracy')
    ax5.grid(True, alpha=0.3)
    for bar, value in zip(bars5, resnet_retain_accs):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    # ResNet Forget Accuracy
    resnet_forget_accs = [stage1_results['resnet_acc_forget_after'], stage2_results['resnet_acc_forget_after'], stage3_results['resnet_acc_forget_after']]
    bars6 = ax6.bar(stages, resnet_forget_accs, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax6.set_title('Final ResNet Forget Accuracy', fontweight='bold')
    ax6.set_ylabel('Accuracy')
    ax6.grid(True, alpha=0.3)
    for bar, value in zip(bars6, resnet_forget_accs):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.suptitle('Performance Comparison Across Stages', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {plots_dir}")


# --------------------------- General pipeline ---------------------------

def _choose_forget_classes(entries: List[dict], label_key: str, retain_frac: float, seed: int) -> List[str]:
    by = {}
    for e in entries:
        by.setdefault(e[label_key], []).append(e)
    classes = sorted(by.keys())
    n_total = len(classes)
    n_retain = max(1, int(round(retain_frac * n_total)))
    n_forget = max(1, n_total - n_retain)
    rng = np.random.RandomState(seed)
    eligible = [c for c in classes if sum(1 for x in by[c] if x.get('split') == 'test') > 0 and len(by[c]) > 1]
    if len(eligible) >= n_forget:
        forget = list(rng.choice(eligible, size=n_forget, replace=False))
    else:
        forget = list(rng.choice(classes, size=n_forget, replace=False))
    return forget


def run_pipeline(entries: List[dict], cfg: Dict, dataset_api):
    seed_all(cfg['seed'])
    save_dir = cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    t0_all = time.time()

    label_key = cfg['label_key']

    # Decide forget/retain sets
    if cfg.get('forget_classes') is None:
        forget_classes = _choose_forget_classes(entries, label_key, cfg['retain_frac'], cfg['seed'])
    else:
        forget_classes = list(cfg['forget_classes'])

    manifest = []
    for entry in entries:
        e2 = entry.copy()
        e2['set'] = 'forget' if entry[label_key] in forget_classes else 'retain'
        manifest.append(e2)

    dev_manifest = [m for m in manifest if m['split'] == 'dev']
    test_manifest = [m for m in manifest if m['split'] == 'test']

    # Embeddings
    if cfg.get('embeddings_path'):
        data = np.load(cfg['embeddings_path'], allow_pickle=True)
        embs = data['embeddings']
        seg_ids = data.get('segment_ids', None)
        if seg_ids is not None:
            id_to_idx = {str(sid): i for i, sid in enumerate(seg_ids)}
            E_all = np.zeros((len(manifest), embs.shape[1]), dtype=np.float32)
            for i, entry in enumerate(manifest):
                sid = entry.get('segment_id', None)
                if sid is None:
                    raise RuntimeError('Embeddings provided but manifest entries lack segment_id')
                if str(sid) not in id_to_idx:
                    raise RuntimeError(f'segment_id {sid} not found')
                E_all[i] = embs[id_to_idx[str(sid)]]
            L_all = [m[label_key] for m in manifest]
        else:
            if embs.shape[0] != len(manifest):
                raise RuntimeError('Embeddings length mismatch')
            E_all = embs.astype(np.float32)
            L_all = [m[label_key] for m in manifest]
    else:
        all_manifest = dev_manifest + test_manifest
        E_all, L_all, _ = extract_embeddings(
            all_manifest,
            encoder=cfg['encoder'], device=cfg['device'],
            wav2vec2_model=cfg.get('wav2vec2_model'),
            hubert_model=cfg.get('hubert_model'),
            conformer_model=cfg.get('conformer_model'),
            label_key=label_key
        )
        manifest = list(dev_manifest) + list(test_manifest)

    labels = L_all
    sets = [f"{m['set']}:{m['split']}" for m in manifest]
    M_ret_dev = np.array([s == 'retain:dev' for s in sets], dtype=bool)
    M_fgt_dev = np.array([s == 'forget:dev' for s in sets], dtype=bool)
    M_ret_test = np.array([s == 'retain:test' for s in sets], dtype=bool)
    M_fgt_test = np.array([s == 'forget:test' for s in sets], dtype=bool)

    # Whitening
    mu, W = fit_whitener(E_all[M_ret_dev])
    Z_all = apply_whitener(E_all, mu, W)

    # SAE
    Z_train = Z_all[M_ret_dev | M_fgt_dev]
    perm = np.random.permutation(Z_train.shape[0])
    cut = int(0.8 * len(perm))
    Z_sae_train, Z_sae_val = Z_train[perm[:cut]], Z_train[perm[cut:]]
    sae = SAE(input_dim=Z_all.shape[1], hidden_mult=cfg['sae_multiplier'], k=cfg['sae_topk'], variant=cfg.get('sae_variant', 'topk'))
    sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=cfg['sae_epochs'], batch_size=4096, lr=1e-3, device=cfg['device'], lambda_sparsity=cfg.get('lambda_sparsity', 0.0))

    dec_w = sae.dec.weight.detach().cpu().numpy()
    dec_diag = decoder_metrics(dec_w)

    try:
        A_dev = sae_encode_batch(sae, Z_all[M_ret_dev | M_fgt_dev], device=cfg['device'], batch=8192)
        sparsity_diag = activation_sparsity_metrics(A_dev, k_expected=getattr(sae, 'k', None))
    except Exception:
        A_dev = None
        sparsity_diag = {}

    Z_ret_dev = Z_all[M_ret_dev]
    Z_fgt_dev = Z_all[M_fgt_dev]
    A_ret = sae_encode_batch(sae, Z_ret_dev, device=cfg['device'], batch=8192)
    A_fgt = sae_encode_batch(sae, Z_fgt_dev, device=cfg['device'], batch=8192)
    A_all = sae_encode_batch(sae, Z_all, device=cfg['device'], batch=8192)

    # Feature stats & selection
    stats = feature_stats(A_fgt, A_ret)
    feat_idx_global = select_features_from_stats(stats, top_k=cfg['build_U_top'], fdr=0.05)
    feat_idx = feat_idx_global

    # Build U
    _ = prune_and_normalize_decoder(dec_w, prune_threshold=1e-6)
    U = build_U_from_decoder(dec_w, feat_idx)

    try:
        D_selected = dec_w[:, feat_idx] if len(feat_idx) > 0 else dec_w[:, :0]
        U_diag = check_U(U, D_selected if D_selected.shape[1] > 0 else None)
        Eshare = energy_share_in_U(Z_all, U)
    except Exception:
        U_diag = {}
        Eshare = {}

    # LEACE
    P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
    mu_r = None
    mask_dev = np.array([s.endswith(':dev') for s in sets], dtype=bool)
    dev_indices = np.where(mask_dev)[0]
    Y_dev = np.array([1 if manifest[i][label_key] in forget_classes else 0 for i in dev_indices], dtype=int).reshape(-1, 1)

    if cfg.get('leace'):
        try:
            P_r, mu_r = leace_projector_in_U(Z_all[mask_dev], U, Y_dev, ridge=cfg['leace_ridge'])
            _ = leace_invariants(Z_all, sets, U, Y_dev, P_r, (Z_all[mask_dev] @ U).mean(0, keepdims=True))
            proj_props = check_projector_properties(P_r)
        except Exception as e:
            print('LEACE failed:', e)
            P_r = np.eye(U.shape[1]) if U.shape[1] > 0 else np.eye(0)
            mu_r = np.zeros((1, U.shape[1]))

    # Apply edits
    Z_edit = Z_all.copy()
    if cfg['beta'] > 0.0:
        Z_edit = apply_sauce_gating(sae, Z_all, feat_idx, beta=cfg['beta'], device=cfg['device'], batch=8192)
    if cfg.get('leace'):
        Z_edit = apply_subspace_leace(Z_edit, U, P_r, mu_r)
    if cfg.get('noise_sigma') and cfg['noise_sigma'] > 0:
        Z_edit = inject_noise_in_subspace(Z_edit, U, sigma=cfg['noise_sigma'])

    E_edit = invert_whitener(Z_edit, mu, W)
    E_all_norm, E_edit_norm = l2_normalize(E_all), l2_normalize(E_edit)

    # Calculate Monosemanticity Score for original embeddings
    print("Computing Monosemanticity Score for original embeddings...")
    ms_original = calculate_monosemanticity_score(E_all_norm, E_all_norm, top_m=20, min_activations=5)
    ms_original_stats = compute_monosemanticity_statistics(ms_original)
    print(f"  Original embeddings MS: mean={ms_original_stats['mean']:.4f}, std={ms_original_stats['std']:.4f}, "
          f"min={ms_original_stats['min']:.4f}, max={ms_original_stats['max']:.4f}")
    
    # Calculate Monosemanticity Score for SAE activations
    print("Computing Monosemanticity Score for SAE activations...")
    ms_sae = calculate_monosemanticity_score(A_all, E_all_norm, top_m=20, min_activations=5)
    ms_sae_stats = compute_monosemanticity_statistics(ms_sae)
    print(f"  SAE activations MS: mean={ms_sae_stats['mean']:.4f}, std={ms_sae_stats['std']:.4f}, "
          f"min={ms_sae_stats['min']:.4f}, max={ms_sae_stats['max']:.4f}")

    # Calculate Monosemanticity Score for final edited embeddings
    print("Computing Monosemanticity Score for final edited embeddings...")
    ms_final = calculate_monosemanticity_score(E_edit_norm, E_edit_norm, top_m=20, min_activations=5)
    ms_final_stats = compute_monosemanticity_statistics(ms_final)
    print(f"  Final edited embeddings MS: mean={ms_final_stats['mean']:.4f}, std={ms_final_stats['std']:.4f}, "
          f"min={ms_final_stats['min']:.4f}, max={ms_final_stats['max']:.4f}")

    # Validation checks
    run_validation_checks(
        save_dir=os.path.join(save_dir, 'validation'),
        Z_all=Z_all, Z_edit=Z_edit,
        E_all_norm=E_all_norm, E_edit_norm=E_edit_norm,
        ZU_mean=(Z_all[mask_dev] @ U).mean(0, keepdims=True) if U.shape[1] > 0 else None,
        U=U, P_r=P_r, mu_r=mu_r, dec_w=dec_w, sae=sae,
        feat_idx=feat_idx, sets=sets, labels=labels,
        mask_dev=mask_dev, mask_test=(M_ret_test | M_fgt_test),
        y_dev_bin=Y_dev, leace_ridge=cfg['leace_ridge'],
        mu=mu, W=W, device=cfg['device']
    )

    # Plots
    Z_before = Z_all[mask_dev]
    Z_after = Z_edit[mask_dev]
    ZU_before = Z_before @ U if U.shape[1] > 0 else Z_before[:, :0]
    ZU_after = Z_after @ U if U.shape[1] > 0 else Z_after[:, :0]

    create_all_plots(
        Z_before, Z_after, ZU_before, ZU_after,
        labels=[labels[i] for i in np.where(mask_dev)[0]],
        forget_classes=forget_classes,
        decoder_weights=dec_w, feat_idx=feat_idx,
        A_retain=A_ret, A_forget=A_fgt,
        save_dir=os.path.join(save_dir, 'validation'), random_state=cfg['seed']
    )

    # Verification EERs
    results = {}
    test_trials, test_labels = dataset_api.build_verification_trials(labels, forget_classes, M_ret_test | M_fgt_test)
    scores_test_before, scores_test_after = compute_scores(E_all_norm, test_trials), compute_scores(E_edit_norm, test_trials)
    eer_before, eer_after = eer_from_scores(scores_test_before, test_labels), eer_from_scores(scores_test_after, test_labels)
    results['EER_before_raw'] = eer_before
    results['EER_after_raw'] = eer_after

    per_class_test = dataset_api.per_class_eers(E_all_norm, E_edit_norm, labels, mask=(M_ret_test | M_fgt_test), rng_seed=cfg['seed'])
    retain_classes = [c for c in set(labels) if c not in set(forget_classes)]
    per_class_ret = {c: vals for c, vals in per_class_test.items() if c in retain_classes}
    per_class_fgt = {c: vals for c, vals in per_class_test.items() if c in forget_classes}
    retain_avg_before, retain_avg_after = dataset_api.avg_eers(per_class_ret)
    forget_avg_before, forget_avg_after = dataset_api.avg_eers(per_class_fgt)
    results.update({
        'retain_EER_before_avg': retain_avg_before,
        'retain_EER_after_avg': retain_avg_after,
        'forget_EER_before_avg': forget_avg_before,
        'forget_EER_after_avg': forget_avg_after,
    })

    # Linear probe
    split_mask = {'dev': M_ret_dev | M_fgt_dev, 'test': M_ret_test | M_fgt_test}
    acc_before, auc_before = dataset_api.safe_linear_probe(Z_all, labels, U, split_mask, forget_classes)
    acc_after, auc_after = dataset_api.safe_linear_probe(Z_edit, labels, U, split_mask, forget_classes)
    results.update({
        'linear_probe_acc_before': acc_before,
        'linear_probe_acc_after': acc_after,
        'linear_probe_auc_before': auc_before,
        'linear_probe_auc_after': auc_after,
    })

    # Add Monosemanticity Score results
    results.update({
        'ms_original_mean': ms_original_stats['mean'],
        'ms_original_std': ms_original_stats['std'],
        'ms_original_min': ms_original_stats['min'],
        'ms_original_max': ms_original_stats['max'],
        'ms_sae_mean': ms_sae_stats['mean'],
        'ms_sae_std': ms_sae_stats['std'],
        'ms_sae_min': ms_sae_stats['min'],
        'ms_sae_max': ms_sae_stats['max'],
        'ms_final_mean': ms_final_stats['mean'],
        'ms_final_std': ms_final_stats['std'],
        'ms_final_min': ms_final_stats['min'],
        'ms_final_max': ms_final_stats['max'],
    })

    # Save summaries
    save_json({
        'decoder_metrics': dec_diag,
        'sparsity_diag': sparsity_diag,
        'U_diag': U_diag,
        'energy_share_in_U': Eshare,
        'leace_proj_props': locals().get('proj_props', {}),
        'results_summary': results
    }, os.path.join(save_dir, 'summary.json'))

    # Interpretability suite
    interp_dir = os.path.join(save_dir, 'interpretability')
    os.makedirs(interp_dir, exist_ok=True)

    label_binary = np.array([1 if c in forget_classes else 0 for c in labels])
    try:
        build_feature_cards(A_all, label_binary, dec_w, feat_idx, None, interp_dir)
    except Exception as e:
        print('feature card build failed', e)
    try:
        polysemanticity_checks(A_all, np.array(labels), feat_idx, interp_dir)
    except Exception as e:
        print('polysemanticity failed', e)

    try:
        causal_tests(sae, Z_all, mu, W, test_trials, test_labels, feat_idx, interp_dir, cfg['device'], max_feats=100)
    except Exception as e:
        print('causal tests failed', e)

    try:
        forget_indices = np.where(M_fgt_test)[0].tolist()
        retain_indices = np.where(M_ret_test)[0].tolist()
        activation_patching(sae, Z_all, mu, W, test_trials, test_labels, forget_indices, retain_indices, interp_dir, cfg['device'], topm=8)
    except Exception as e:
        print('patching failed', e)

    try:
        # Choose a sensible alias for label in per-dataset runners by passing later
        top_neurons_analysis(A_all, labels, manifest, interp_dir)
    except Exception as e:
        print('top neurons analysis failed', e)

    save_json(results, os.path.join(save_dir, 'verification_results.json'))

    print(f"Pipeline finished in {time.time() - t0_all:.1f}s. Results saved to {save_dir}")
    return results


