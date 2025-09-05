import os
import sys
import json
import argparse
import itertools
import numpy as np

from experiment_core import run_pipeline
_THIS_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..', 'src'))
if _SRC_DIR not in sys.path:
    sys.path.append(_SRC_DIR)
from librispeech import build_manifest, build_verification_trials, safe_linear_probe, avg_eers, per_class_eers


class _DatasetAPI:
    def build_verification_trials(self, labels, forget_classes, mask):
        return build_verification_trials(labels, forget_classes, mask)
    def safe_linear_probe(self, Z, labels, U, split_mask, forget_classes):
        return safe_linear_probe(Z, labels, U, split_mask, forget_classes)
    def avg_eers(self, per_class):
        return avg_eers(per_class)
    def per_class_eers(self, E_all_norm, E_edit_norm, labels, mask, rng_seed):
        return per_class_eers(E_all_norm, E_edit_norm, labels, mask=mask, rng_seed=rng_seed)


def run_single_experiment(cfg):
    entries = build_manifest(cfg['data_root'], dev_frac_per_spk=cfg['dev_frac_per_spk'], seed=cfg['seed'])
    cfg2 = dict(cfg)
    cfg2['label_key'] = 'speaker'
    if 'forget_spks' in cfg:
        cfg2['forget_classes'] = cfg.get('forget_spks')
    return run_pipeline(entries, cfg2, _DatasetAPI())


def expand_grid(base_cfg, grid_cfg):
    keys, values = zip(*grid_cfg.items())
    for combo in itertools.product(*values):
        cfg = dict(base_cfg)
        for k, v in zip(keys, combo):
            cfg[k] = v
        yield cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/data/veer/testing/new/datasets/librispeech_data_dev/LibriSpeech")
    parser.add_argument("--save_dir", type=str, default="./out_librispeech")
    parser.add_argument("--encoder", type=str, default="ecapa")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--retain_frac", type=float, default=0.7)
    parser.add_argument("--dev_frac_per_spk", type=float, default=0.7)
    parser.add_argument("--sae_hidden_mult", type=float, default=5.0)
    parser.add_argument("--sae_topk", type=int, default=128)
    parser.add_argument("--sae_epochs", type=int, default=150)
    parser.add_argument("--build_U_top", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--leace", action="store_true")
    parser.add_argument("--leace_ridge", type=float, default=1e-6)
    parser.add_argument("--noise_sigma", type=float, default=0.02)
    parser.add_argument("--forget_spks", nargs="*", default=["174"])
    parser.add_argument("--wav2vec2_model", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--hubert_model", type=str, default="facebook/hubert-base-ls960")
    parser.add_argument("--conformer_model", type=str, default="facebook/wav2vec2-conformer-rel-pos-large")
    parser.add_argument("--grid_config", type=str, default=None)
    args = parser.parse_args()

    base_cfg = vars(args)
    if args.grid_config:
        grid_cfg = json.load(open(args.grid_config))
        for i, cfg_i in enumerate(expand_grid(base_cfg, grid_cfg)):
            run_dir = os.path.join(args.save_dir, f"grid_{i}")
            cfg_i['save_dir'] = run_dir
            run_single_experiment(cfg_i)
    else:
        run_single_experiment(base_cfg)


if __name__ == "__main__":
    main()
