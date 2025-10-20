from typing import Dict, List
from copy import deepcopy
import os, csv
import numpy as np
from core.utils.seed import seed_all
from .single_run import run_single

def run_continual(adapter, cfg: Dict, sequence: List[str]):
    base_save_dir = cfg['experiment']['save_dir']
    os.makedirs(base_save_dir, exist_ok=True)
    seed = cfg['experiment']['seed']
    # Precompute once on full set
    pre = adapter.precompute(
        root_or_manifest=cfg['dataset']['root_or_manifest'],
        embeddings_path=cfg['encoder'].get('embeddings_path'),
        dev_frac=cfg['dataset']['dev_frac'],
        seed=seed,
        encoder_cfg=cfg['encoder'],
        device=cfg['experiment']['device'],
    )
    csv_path = os.path.join(base_save_dir, f"continual_unlearning_results_seed{seed}.csv")
    fieldnames = ["step", "forgotten_label", "num_active_labels", "num_active_samples", "FA_before", "FA_after", "RA_before", "RA_after", "forget_EER_before", "forget_EER_after", "retain_EER_before", "retain_EER_after", "linear_probe_auc_before", "linear_probe_auc_after", "error"]
    with open(csv_path, "w", newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames); writer.writeheader()
        forgotten: List[str] = []
        for i, target in enumerate(sequence):
            step = i + 1
            try:
                active_mask = np.array([l not in forgotten for l in pre.labels], dtype=bool)
                labels_active = list(np.array(pre.labels)[active_mask])
                classes_active = sorted(list(set(labels_active)))
                cfg_local = deepcopy(cfg)
                cfg_local['dataset']['forget_labels'] = [target]
                # reuse run_single; adapters will still read from cfg for paths
                res = run_single(adapter, cfg_local)[target]
                writer.writerow({
                    "step": step,
                    "forgotten_label": target,
                    "num_active_labels": len(classes_active),
                    "num_active_samples": int(active_mask.sum()),
                    **res,
                    "error": ""
                }); fh.flush()
                forgotten.append(target)
            except Exception as e:
                writer.writerow({"step": step, "forgotten_label": target, "error": str(e)}); fh.flush(); forgotten.append(target)


