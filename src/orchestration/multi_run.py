from typing import Dict
import os, csv, pickle, torch
import numpy as np
from core.utils.seed import seed_all
from core.sae.model import SAE
from core.sae.train import train_sae
from core.sae.encode import sae_encode_batch
from core.feature_mining.stats import feature_stats, select_features_from_stats
from core.feature_mining.subspace import prune_and_normalize_decoder, build_U_from_decoder
from core.evals.classification import train_classifier, evaluate_classifier
from core.evals.probes import safe_linear_probe
from core.evals.verification import build_verification_trials, compute_scores, eer_from_scores, per_class_eers, avg_eers
from core.edits.apply import apply_edits

def run_multi(adapter, cfg: Dict):
    base_save_dir = cfg['experiment']['save_dir']
    os.makedirs(base_save_dir, exist_ok=True)
    seeds = cfg['orchestration']['seeds']
    # Precompute once per config
    pre = adapter.precompute(
        root_or_manifest=cfg['dataset']['root_or_manifest'],
        embeddings_path=cfg['encoder'].get('embeddings_path'),
        dev_frac=cfg['dataset']['dev_frac'],
        seed=cfg['experiment']['seed'],
        encoder_cfg=cfg['encoder'],
        device=cfg['experiment']['device'],
    )
    classes = pre.classes
    csv_path = os.path.join(base_save_dir, "multi_run_results.csv")
    fieldnames = ["seed", "forget_label", "FA_before", "FA_after", "RA_before", "RA_after", "forget_EER_before", "forget_EER_after", "retain_EER_before", "retain_EER_after", "linear_probe_auc_before", "linear_probe_auc_after", "error"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for seed in seeds:
            try:
                Z_train = pre.Z_all[pre.mask_dev]
                perm = np.random.permutation(Z_train.shape[0])
                cut = int(0.8 * len(perm)) if len(perm) > 1 else int(0.5 * len(perm))
                Z_sae_train, Z_sae_val = (Z_train, Z_train) if cut <= 0 else (Z_train[perm[:cut]], Z_train[perm[cut:]])
                sae = SAE(input_dim=pre.Z_all.shape[1], hidden_mult=cfg['sae']['multiplier'], k=cfg['sae']['topk'], variant=cfg['sae']['variant'], hidden_dim=cfg['sae']['hidden_dim'])
                sae = train_sae(sae, Z_sae_train, Z_sae_val, epochs=cfg['sae']['epochs'], batch_size=cfg['sae']['batch_size'], lr=1e-3, device=cfg['experiment']['device'], lambda_sparsity=cfg['sae']['lambda_sparsity'])
                A_all = sae_encode_batch(sae, pre.Z_all, device=cfg['experiment']['device'], batch=8192)
                dec_w = sae.dec.weight.detach().cpu().numpy()
                # classifier per seed
                E_all_norm = pre.E_all / (np.linalg.norm(pre.E_all, axis=1, keepdims=True) + 1e-12)
                cls2id = {c: i for i, c in enumerate(classes)}
                X_cls_train = E_all_norm[pre.mask_dev]
                y_cls_train = [cls2id[s] for i, s in enumerate(pre.labels) if pre.mask_dev[i]]
                X_cls_val = E_all_norm[pre.mask_test]
                y_cls_val = [cls2id[s] for i, s in enumerate(pre.labels) if pre.mask_test[i]]
                clf = train_classifier(X_cls_train, y_cls_train, X_cls_val, y_cls_val, num_classes=len(classes), device=cfg['experiment']['device'], epochs=cfg['orchestration']['classifier_epochs'], batch_size=512, lr=1e-3)
                # run per class
                for forget in classes:
                    try:
                        cfg_local = dict(cfg)
                        cfg_local['dataset'] = dict(cfg['dataset'])
                        cfg_local['dataset']['forget_labels'] = [forget]
                        res = run_single(adapter, cfg_local)[forget]
                        row = {"seed": seed, "forget_label": forget, **res, "error": ""}
                        writer.writerow(row); fh.flush()
                    except Exception as e:
                        writer.writerow({"seed": seed, "forget_label": forget, "error": str(e)})
                        fh.flush()
            except Exception as e:
                tb = traceback.format_exc()
                for forget in classes:
                    writer.writerow({"seed": seed, "forget_label": forget, "error": f"seed-level failure: {str(e)}"})
                fh.flush()


