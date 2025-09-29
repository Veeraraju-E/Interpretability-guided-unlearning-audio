import os
import argparse
import yaml
import sys
sys.path.append('/data/veer/testing/new/Interpretability-guided-unlearning-audio/src')
from src.datasets.voxceleb_adapter import VoxCelebAdapter
from src.datasets.speech_commands_adapter import SpeechCommandsAdapter
from src.datasets.audiomnist_adapter import AudioMNISTAdapter
from src.datasets.librispeech_adapter import LibriSpeechAdapter
from src.datasets.lahaja_adapter import LahajaAdapter
from src.orchestration.single_run import run_single
from src.orchestration.multi_run import run_multi
from src.orchestration.continual import run_continual

ADAPTERS = {
    'voxceleb': VoxCelebAdapter(),
    'speech_commands': SpeechCommandsAdapter(),
    'audiomnist': AudioMNISTAdapter(),
    'librispeech': LibriSpeechAdapter(),
    'lahaja': LahajaAdapter(),
}

def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default=os.path.join('configs', 'base.yaml'))
    args = p.parse_args()
    cfg = load_config(args.config)
    adapter_name = cfg['dataset']['name']
    if adapter_name not in ADAPTERS:
        raise ValueError(f"Unknown dataset adapter '{adapter_name}'. Available: {list(ADAPTERS.keys())}")
    mode = cfg['orchestration'].get('mode', 'single')
    if mode == 'single':
        run_single(ADAPTERS[adapter_name], cfg)
    elif mode == 'multi':
        run_multi(ADAPTERS[adapter_name], cfg)
    elif mode == 'continual':
        # default sequence: all classes if not provided
        run_continual(ADAPTERS[adapter_name], cfg, sequence=cfg['dataset'].get('forget_labels', []))
    else:
        raise ValueError(f"Unknown mode '{mode}'")

if __name__ == '__main__':
    main()


