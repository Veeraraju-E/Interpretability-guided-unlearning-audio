import os
import argparse
import yaml
from pathlib import Path

def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', type=str, default=None)
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        if args.config is None:
            raise ValueError("Provide either --run_dir or --config")
        cfg = load_config(args.config)
        run_dir = cfg['experiment']['save_dir']

    # Defer to the original analysis entrypoint which expects artifacts in run_dir
    # This keeps analysis features intact during refactor
    from src import interp as interpret_module
    interpret_module.main(run_dir=run_dir, device=args.device)

if __name__ == '__main__':
    main()


