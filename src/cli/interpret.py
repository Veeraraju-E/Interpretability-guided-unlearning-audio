"""
Legacy interpret.py - now delegates to the new Typer-based CLI.
Maintains backward compatibility while providing modern CLI features.
"""

import sys
import os

# Add src to path for imports
sys.path.append('/data/veer/testing/new/Interpretability-guided-unlearning-audio/src')

def main():
    """Legacy main function that delegates to the new Typer CLI."""
    # Import and run the new Typer app with interpret command
    from src.cli.main import app
    
    # Convert to interpret command format
    args = []
    if '--run-dir' in sys.argv:
        run_dir_idx = sys.argv.index('--run-dir')
        if run_dir_idx + 1 < len(sys.argv):
            run_dir = sys.argv[run_dir_idx + 1]
            args.extend(['--run-dir', run_dir])
    if '--config' in sys.argv:
        config_idx = sys.argv.index('--config')
        if config_idx + 1 < len(sys.argv):
            config_path = sys.argv[config_idx + 1]
            args.extend(['--config', config_path])
    if '--device' in sys.argv:
        device_idx = sys.argv.index('--device')
        if device_idx + 1 < len(sys.argv):
            device = sys.argv[device_idx + 1]
            args.extend(['--device', device])
    
    if not args:
        # No arguments, show help
        sys.argv = ['interpret.py', '--help']
    else:
        sys.argv = ['interpret.py', 'interpret'] + args
    
    # Run the Typer app
    app()

if __name__ == '__main__':
    main()


