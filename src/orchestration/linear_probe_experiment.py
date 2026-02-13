# Linear probing: train and test linear classifier for class identity on E_all (encoder output).
from typing import Dict
import os
import csv
import numpy as np
from src.core.evals.probes import linear_probe_class_identity
from src.core.utils.seed import seed_all


def should_run_linear_probe(cfg: Dict) -> bool:
    """True if config has linear_probe section with output_csv_dir and epochs (optional add-on in main runs)."""
    lp = cfg.get("linear_probe")
    if not lp or not isinstance(lp, dict):
        return False
    return bool(lp.get("output_csv_dir") and "epochs" in lp)


def append_linear_probe_result(cfg: Dict, accuracy: float) -> None:
    """Append one row to linear_probe_{seed}.csv. No-op if linear_probe config is missing."""
    if not should_run_linear_probe(cfg):
        return
    seed = cfg["experiment"]["seed"]
    output_dir = cfg["linear_probe"]["output_csv_dir"]
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"linear_probe_{seed}.csv")
    fieldnames = [
        "seed", "encoder", "dataset", "sae_enabled", "feature_mining_build_U_top",
        "edits_leace", "linear_probe_epochs", "accuracy",
    ]
    row = {
        "seed": seed,
        "encoder": cfg["encoder"]["name"],
        "dataset": cfg["dataset"]["name"],
        "sae_enabled": cfg["sae"].get("enabled", True),
        "feature_mining_build_U_top": cfg["feature_mining"].get("build_U_top", ""),
        "edits_leace": cfg["edits"].get("leace", True),
        "linear_probe_epochs": cfg["linear_probe"]["epochs"],
        "accuracy": accuracy,
    }
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()


def _precompute(adapter, cfg: Dict):
    """Reuse same precompute dispatch as single_run for all adapters."""
    seed = cfg['experiment']['seed']
    device = cfg['experiment']['device']
    data_path_param = cfg['dataset']['root_or_manifest']
    if hasattr(adapter, 'name'):
        if adapter.name in ['audiomnist', 'speech_commands']:
            return adapter.precompute(
                data_root=data_path_param,
                embeddings_path=cfg['encoder'].get('embeddings_path'),
                dev_frac=cfg['dataset']['dev_frac'],
                seed=seed,
                encoder_cfg=cfg['encoder'],
                device=device,
            )
        if adapter.name == 'librispeech':
            return adapter.precompute(
                root=data_path_param,
                embeddings_path=cfg['encoder'].get('embeddings_path'),
                dev_frac=cfg['dataset']['dev_frac'],
                seed=seed,
                encoder_cfg=cfg['encoder'],
                device=device,
            )
        if adapter.name == 'lahaja':
            return adapter.precompute(
                split=data_path_param,
                embeddings_path=cfg['encoder'].get('embeddings_path'),
                dev_frac=cfg['dataset']['dev_frac'],
                seed=seed,
                encoder_cfg=cfg['encoder'],
                device=device,
            )
        if adapter.name == 'mls_spanish':
            return adapter.precompute(
                root_or_manifest=data_path_param,
                embeddings_path=cfg['encoder'].get('embeddings_path'),
                dev_frac=cfg['dataset']['dev_frac'],
                seed=seed,
                encoder_cfg=cfg['encoder'],
                device=device,
            )
        # voxceleb
        return adapter.precompute(
            root_or_manifest=data_path_param,
            embeddings_path=cfg['encoder'].get('embeddings_path'),
            dev_frac=cfg['dataset']['dev_frac'],
            seed=seed,
            encoder_cfg=cfg['encoder'],
            device=device,
        )
    raise NotImplementedError(f"Adapter {adapter.name} not implemented for {cfg['dataset']['name']}.")


def run_linear_probe_experiment(adapter, cfg: Dict) -> float:
    """
    Run one linear-probe experiment: precompute E_all, train/test linear probe for class identity, append row to CSV.
    Returns test accuracy.
    """
    seed = cfg['experiment']['seed']
    seed_all(seed)

    pre = _precompute(adapter, cfg)
    E_all_norm = pre.E_all / (np.linalg.norm(pre.E_all, axis=1, keepdims=True) + 1e-12)
    max_iter = cfg["linear_probe"]["epochs"]
    accuracy = linear_probe_class_identity(
        E_all_norm,
        pre.labels,
        pre.mask_dev,
        pre.mask_test,
        max_iter=max_iter,
        random_state=42,
    )
    append_linear_probe_result(cfg, accuracy)
    return accuracy
