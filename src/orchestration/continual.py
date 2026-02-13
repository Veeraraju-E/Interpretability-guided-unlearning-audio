from typing import Dict, List, Optional
import os, csv
import numpy as np
import sys
import json
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.evals.classification import train_classifier, evaluate_classifier
from src.core.evals.probes import safe_linear_probe, linear_probe_class_identity
from src.core.evals.verification import *
from src.core.edits import apply_edits
from src.core.sae import SAE, train_sae, sae_encode_batch
from src.core.utils.seed import seed_all
from src.core.utils.stats import compute_feature_stats, select_features
from src.core.utils.subspace import prune_and_normalize_decoder, build_U_from_decoder
from src.core.visualization.latent_selection import plot_discriminability_landscape
from src.core.visualization import create_visual_abstract
from src.datasets.adapter_base import Precomp
from src.orchestration.linear_probe_experiment import should_run_linear_probe, append_linear_probe_result

def filter_precomp_by_forgotten(pre: Precomp, forgotten: List[str]) -> Precomp:
    """
    Filter Precomp data to exclude forgotten speakers.
    """
    forgotten_set = {str(f) for f in forgotten}
    active_mask = np.array([str(label) not in forgotten_set for label in pre.labels], dtype=bool)
    
    # Filter all arrays and lists
    filtered_manifest = [pre.manifest[i] for i in range(len(pre.manifest)) if active_mask[i]]
    filtered_labels = [pre.labels[i] for i in range(len(pre.labels)) if active_mask[i]]
    filtered_classes = sorted(list(set(filtered_labels)))
    
    filtered_E_all = pre.E_all[active_mask]
    filtered_Z_all = pre.Z_all[active_mask]
    filtered_mask_dev = pre.mask_dev[active_mask]
    filtered_mask_test = pre.mask_test[active_mask]
    
    return Precomp(
        manifest=filtered_manifest,
        labels=filtered_labels,
        classes=filtered_classes,
        E_all=filtered_E_all,
        Z_all=filtered_Z_all,
        mu=pre.mu,
        W=pre.W,
        mask_dev=filtered_mask_dev,
        mask_test=filtered_mask_test
    )

def run_single_continual_step(pre: Precomp, sae: Optional[SAE], forget_label: str, cfg: Dict, device: str, seed: int, save_dir: str) -> Dict:
    """
    Perform a single unlearning step in continual unlearning.
    When sae is None (SAE disabled), LEACE is applied directly on whitened embeddings in full space.
    """
    from rich.console import Console
    console = Console()

    forget_label_str = str(forget_label)
    all_labels = pre.labels

    # Check minimum sample requirement
    forget_count = sum(1 for lbl in all_labels if str(lbl) == forget_label_str)
    if forget_count < 5:
        console.print(f"[ERROR] Forget class '{forget_label_str}' has only {forget_count} samples, minimum 5 required. Skipping this class.", style="red")
        return {"error": f"Insufficient samples: {forget_count} for forget class '{forget_label_str}'"}

    # Create forget/retain masks
    sets = [f"{'forget' if str(lbl) == forget_label_str else 'retain'}:{'dev' if pre.mask_dev[i] else 'test'}" for i, lbl in enumerate(all_labels)]

    M_ret_dev, M_fgt_dev = np.array([s == 'retain:dev' for s in sets], dtype=bool), np.array([s == 'forget:dev' for s in sets], dtype=bool)
    M_ret_test, M_fgt_test = np.array([s == 'retain:test' for s in sets], dtype=bool), np.array([s == 'forget:test' for s in sets], dtype=bool)

    run_save_dir = os.path.join(save_dir, f"seed{seed}_forget_{str(forget_label)}")
    os.makedirs(run_save_dir, exist_ok=True)
    viz_dir = os.path.join(run_save_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    skip_sae = (sae is None) or not cfg['sae'].get('enabled', True)
    if skip_sae:
        U = np.eye(pre.Z_all.shape[1], dtype=pre.Z_all.dtype)
        feat_idx = []
        feat_stats = None
        disc_png_path = None
        A_all = None
        dec_w = None
    else:
        A_all = sae_encode_batch(sae, pre.Z_all, device=device, batch=256)
        dec_w = sae.dec.weight.detach().cpu().numpy()
        A_ret = A_all[M_ret_dev]
        A_fgt = A_all[M_fgt_dev]
        method = cfg['feature_mining'].get('statistical_method', 'cohens_d')
        balance_negs = cfg['feature_mining'].get('balance_negatives', False)
        if balance_negs and A_ret.shape[0] > 0 and A_fgt.shape[0] > 0 and A_ret.shape[0] != A_fgt.shape[0]:
            rng = np.random.default_rng(seed)
            sel = rng.choice(A_ret.shape[0], size=min(A_ret.shape[0], A_fgt.shape[0]), replace=False)
            A_ret_bal = A_ret[sel]
        else:
            A_ret_bal = A_ret
        feat_stats = compute_feature_stats(A_fgt, A_ret_bal, method=method)
        fdr_thresh = cfg['feature_mining'].get('fdr', 0.05)
        feat_idx = select_features(feat_stats, method=method, top_k=cfg['feature_mining']['build_U_top'], fdr=fdr_thresh)
        if len(feat_idx) == 0:
            print(f"[ERROR] No discriminative features found for forget class '{forget_label_str}'. Cannot proceed with editing.")
            return {"error": "No discriminative features found"}
        disc_pdf_path = os.path.join(viz_dir, "discriminability_landscape.pdf")
        disc_png_path = os.path.join(viz_dir, "discriminability_landscape.png")
        plot_discriminability_landscape(
            stats=feat_stats,
            selected_idx=feat_idx,
            fdr_alpha=fdr_thresh,
            auc_eps=1e-6,
            out_path_pdf=disc_pdf_path,
            out_path_png=disc_png_path,
            title=f"Discriminability Landscape - Class {forget_label_str}",
            highlight_top_n=20
        )
        print(f"[VIZ] Discriminability landscape plot saved to: {disc_pdf_path}")
        prune_and_normalize_decoder(dec_w)
        U = build_U_from_decoder(dec_w, feat_idx)

    # Train classifier on active classes only
    E_all_norm = pre.E_all / (np.linalg.norm(pre.E_all, axis=1, keepdims=True) + 1e-12)
    classes = pre.classes
    cls2id = {c: i for i, c in enumerate(classes)}
    X_cls_train, y_cls_train = E_all_norm[pre.mask_dev], [cls2id[s] for i, s in enumerate(all_labels) if pre.mask_dev[i]]
    X_cls_val, y_cls_val = E_all_norm[pre.mask_test], [cls2id[s] for i, s in enumerate(all_labels) if pre.mask_test[i]]
    clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(classes), device=device, epochs=cfg['orchestration']['classifier_epochs'], batch_size=512, lr=1e-3)

    # LEACE params - create labels for the filtered dataset
    Y_all = np.array([1 if str(all_labels[i]) == forget_label_str else 0 for i in range(len(all_labels))], dtype=int).reshape(-1, 1)
    E_edit, _ = apply_edits(
        pre.Z_all, sae, feat_idx, U, pre.mu, pre.W,
        cfg['edits']['beta'], cfg['edits']['leace'], {'Y_dev': Y_all, 'ridge': cfg['edits']['leace_ridge']}, cfg['edits']['noise_sigma']
    )
    E_all_edited = E_edit
    E_all_norm2 = E_all_edited / (np.linalg.norm(E_all_edited, axis=1, keepdims=True) + 1e-12)
    
    # EERs
    test_mask_for_trials = M_ret_test | M_fgt_test
    max_pairs = cfg['orchestration'].get('max_pairs_per_class', 0)
    trials, tlabels = build_verification_trials(all_labels, [forget_label_str], test_mask_for_trials, seed=seed, max_pairs_per_class=max_pairs)
    scores_before, scores_after = compute_scores(E_all_norm, trials), compute_scores(E_all_norm2, trials)
    eer_before, eer_after = eer_from_scores(scores_before, tlabels), eer_from_scores(scores_after, tlabels)
    per_class = per_class_eers(E_all_norm, E_all_norm2, all_labels, mask=test_mask_for_trials, rng_seed=seed)
    per_class_ret = {c: vals for c, vals in per_class.items() if str(c) != forget_label_str}
    per_class_fgt = {c: vals for c, vals in per_class.items() if str(c) == forget_label_str}
    retain_avg_before, retain_avg_after = avg_eers(per_class_ret)
    forget_avg_before, forget_avg_after = avg_eers(per_class_fgt)
    
    # Classifier acc
    E_ret_test, E_fgt_test = E_all_norm[M_ret_test], E_all_norm[M_fgt_test]
    L_ret_test, L_fgt_test = [cls2id[d] for d in np.array(all_labels)[M_ret_test]], [cls2id[d] for d in np.array(all_labels)[M_fgt_test]]
    E_ret_test_edit, E_fgt_test_edit = E_all_norm2[M_ret_test], E_all_norm2[M_fgt_test]
    acc_ret_before, acc_fgt_before = evaluate_classifier(clf, E_ret_test, L_ret_test, device=device), evaluate_classifier(clf, E_fgt_test, L_fgt_test, device=device)
    acc_ret_after, acc_fgt_after = evaluate_classifier(clf, E_ret_test_edit, L_ret_test, device=device), evaluate_classifier(clf, E_fgt_test_edit, L_fgt_test, device=device)
    
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
        # "EER_before_raw": eer_before, "EER_after_raw": eer_after,
    }
    
    # Save full artifacts for interpretability and downstream analyses
    # (run_save_dir already created above for visualization)
    with open(os.path.join(run_save_dir, "verification_eer_summary.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    
    # Interpretability dump
    sets_np = np.array(sets)
    np.savez_compressed(
        os.path.join(run_save_dir, "interpretability_data.npz"),
        Z_all=pre.Z_all, Z_edit=E_all_edited, U=U,
        labels=np.array(all_labels), sets=sets_np, manifest=np.array(pre.manifest, dtype=object), forget_spks=np.array([forget_label]),
        feat_idx=np.array(feat_idx, dtype=np.int64)
    )
    if sae is not None:
        torch.save(sae.state_dict(), os.path.join(run_save_dir, "sae_state_dict.pt"))
    np.savez_compressed(os.path.join(run_save_dir, "whitener.npz"), mu=pre.mu, W=pre.W)

    # Visual abstract (SAE-only)
    va_cfg = cfg.get('visual_abstract', {})
    if not skip_sae and va_cfg.get('enabled', True) and feat_stats is not None and dec_w is not None and A_all is not None:
        try:
            _ = create_visual_abstract(
                Z_all=pre.Z_all,
                E_edit=E_all_edited,
                E_before=E_all_norm,
                E_after=E_all_norm2,
                mu=pre.mu,
                W=pre.W,
                U=U,
                labels=all_labels,
                forget_label=forget_label_str,
                stats=feat_stats,
                selected_idx=feat_idx,
                dec_w=dec_w,
                A_all=A_all,
                save_dir=viz_dir,
                ridge=float(cfg['edits'].get('leace_ridge', 1e-6)),
                max_points=int(va_cfg.get('max_points', 1500)),
                k_plot=int(va_cfg.get('k_plot', 5)),
                seed=seed,
                overlay_top_n=int(va_cfg.get('overlay_top_n', 8)),
                hist_top_n=int(va_cfg.get('hist_top_n', 8)),
                ablation=bool(va_cfg.get('ablation', True)),
                overlay_xlim=tuple(va_cfg.get('overlay_xlim')) if va_cfg.get('overlay_xlim') is not None else None,
                overlay_ylim=tuple(va_cfg.get('overlay_ylim')) if va_cfg.get('overlay_ylim') is not None else None,
                sets=sets,
                discriminability_path=disc_png_path
            )
        except Exception:
            pass
    
    return res

# Main
def run_continual(adapter, cfg: Dict, sequence: List[str]):
    base_save_dir = cfg['experiment']['save_dir']
    os.makedirs(base_save_dir, exist_ok=True)
    seed = cfg['experiment']['seed']
    device = cfg['experiment']['device']
    seed_all(seed)
    
    # Extract embeddings once for ALL classes
    print(f"[INFO] Extracting embeddings for all classes")
    data_path_param = cfg['dataset']['root_or_manifest']
    if hasattr(adapter, 'name'):
        if adapter.name in ['audiomnist', 'speech_commands']:
            pre_full = adapter.precompute(
                data_root=data_path_param,
                embeddings_path=cfg['encoder'].get('embeddings_path'),
                dev_frac=cfg['dataset']['dev_frac'],
                seed=seed,
                encoder_cfg=cfg['encoder'],
                device=device,
            )
        elif adapter.name == 'librispeech':
            pre_full = adapter.precompute(
                root=data_path_param,
                embeddings_path=cfg['encoder'].get('embeddings_path'),
                dev_frac=cfg['dataset']['dev_frac'],
                seed=seed,
                encoder_cfg=cfg['encoder'],
                device=device,
            )
        elif adapter.name == 'lahaja':
            pre_full = adapter.precompute(
                split=data_path_param,
                embeddings_path=cfg['encoder'].get('embeddings_path'),
                dev_frac=cfg['dataset']['dev_frac'],
                seed=seed,
                encoder_cfg=cfg['encoder'],
                device=device,
            )
        elif adapter.name == 'mls_spanish':
            pre_full = adapter.precompute(
                root_or_manifest=data_path_param,
                embeddings_path=cfg['encoder'].get('embeddings_path'),
                dev_frac=cfg['dataset']['dev_frac'],
                seed=seed,
                encoder_cfg=cfg['encoder'],
                device=device,
            )
        else:  # voxceleb and others
            pre_full = adapter.precompute(
                root_or_manifest=data_path_param,
                embeddings_path=cfg['encoder'].get('embeddings_path'),
                dev_frac=cfg['dataset']['dev_frac'],
                seed=seed,
                encoder_cfg=cfg['encoder'],
                device=device,
            )
    else:
        raise NotImplementedError(f"Adapter {adapter.name} not implemented for {cfg['dataset']['name']}.")
    
    print(f"Extracted embeddings for {len(pre_full.labels)} samples across {len(pre_full.classes)} classes")

    if should_run_linear_probe(cfg):
        E_all_norm = pre_full.E_all / (np.linalg.norm(pre_full.E_all, axis=1, keepdims=True) + 1e-12)
        lp_accuracy = linear_probe_class_identity(
            E_all_norm, pre_full.labels, pre_full.mask_dev, pre_full.mask_test,
            max_iter=cfg["linear_probe"]["epochs"], random_state=42,
        )
        append_linear_probe_result(cfg, lp_accuracy)

    use_sae = cfg['sae'].get('enabled', True)
    sae = None
    if use_sae:
        print(f"[INFO] Training SAE")
        Z_train = pre_full.Z_all[pre_full.mask_dev]
        perm = np.random.permutation(Z_train.shape[0])
        cut = int(0.8 * len(perm)) if len(perm) > 1 else int(0.5 * len(perm))
        Z_sae_train, Z_sae_val = (Z_train, Z_train) if cut <= 0 else (Z_train[perm[:cut]], Z_train[perm[cut:]])
        sae = SAE(input_dim=pre_full.Z_all.shape[1], hidden_mult=cfg['sae']['multiplier'], k=cfg['sae']['topk'], variant=cfg['sae']['variant'], hidden_dim=cfg['sae']['hidden_dim'])
        sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=cfg['sae']['epochs'], batch_size=cfg['sae']['batch_size'], lr=1e-3, device=device, lambda_sparsity=cfg['sae']['lambda_sparsity'])
        print(f"[INFO] SAE training complete.")
    else:
        print(f"[INFO] SAE disabled; applying LEACE directly on whitened embeddings.")

    # Run continual unlearning steps
    print(f"[INFO] Running continual unlearning")
    csv_path = os.path.join(base_save_dir, f"continual_unlearning_results_seed{seed}.csv")
    fieldnames = ["step", "forgotten_label", "num_active_labels", "num_active_samples", "FA_before", "FA_after", "RA_before", "RA_after", "forget_EER_before", "forget_EER_after", "retain_EER_before", "retain_EER_after", "linear_probe_auc_before", "linear_probe_auc_after", "error"]
    
    with open(csv_path, "w", newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        forgotten: List[str] = []
        
        for i, target in enumerate(sequence):
            step = i + 1
            print(f"\n[INFO] Step {step}: Forgetting '{target}'")
            
            try:
                # Filter out already forgotten speakers from the corpus
                pre_filtered = filter_precomp_by_forgotten(pre_full, forgotten)
                
                # Check if target is still in the filtered data
                if str(target) not in pre_filtered.classes:
                    print(f"[WARNING] Target '{target}' not found in active classes. It may have been already forgotten or doesn't exist.")
                    writer.writerow({"step": step, "forgotten_label": target, "error": f"Target not found in active classes"})
                    fh.flush()
                    forgotten.append(target)
                    continue
                
                labels_active = pre_filtered.labels
                classes_active = pre_filtered.classes
                
                print(f"[INFO] Active classes: {len(classes_active)}, Active samples: {len(labels_active)}")
                
                # Perform unlearning step (using pre-trained SAE, no retraining)
                res = run_single_continual_step(
                    pre=pre_filtered,
                    sae=sae,
                    forget_label=target,
                    cfg=cfg,
                    device=device,
                    seed=seed,
                    save_dir=base_save_dir
                )
                
                if "error" in res:
                    writer.writerow({
                        "step": step,
                        "forgotten_label": target,
                        "num_active_labels": len(classes_active),
                        "num_active_samples": len(labels_active),
                        "error": res["error"]
                    })
                else:
                    writer.writerow({"step": step, "forgotten_label": target, "num_active_labels": len(classes_active), "num_active_samples": len(labels_active), **res, "error": ""})
                fh.flush()
                
                # Remove this forget speaker from future consideration
                forgotten.append(target)
                print(f"[INFO] Speaker '{target}' has been forgotten and removed from corpus.")
                
            except Exception as e:
                import traceback
                error_msg = str(e)
                traceback.print_exc()
                writer.writerow({"step": step, "forgotten_label": target, "error": error_msg})
                fh.flush()
                forgotten.append(target)
    print(f"\n\n[STATUS] Complete! Results saved to: {csv_path}")
