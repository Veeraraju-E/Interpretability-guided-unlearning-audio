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
    TopKSAE, train_sae, sae_encode_batch, prune_and_normalize_decoder,
    feature_stats, select_features_from_stats, build_U_from_decoder,
    leace_projector_in_U, apply_subspace_leace, apply_sauce_gating,
    inject_noise_in_subspace, l2_normalize,
    compute_scores, eer_from_scores
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
    sae = TopKSAE(input_dim=Z_all.shape[1], hidden_mult=cfg['sae_hidden_mult'], k=cfg['sae_topk'])
    sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=cfg['sae_epochs'], batch_size=4096, lr=1e-3, device=cfg['device'])

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


