"""
Legacy run.py - now delegates to the new Typer-based CLI.
Maintains backward compatibility while providing modern CLI features.
"""

import sys
import os

# Add src to path for imports
sys.path.append('/data/veer/testing/new/Interpretability-guided-unlearning-audio/src')

def main():
    """Legacy main function that delegates to the new Typer CLI."""
    # Import and run the new Typer app with run command
    from src.cli.main import app
    
    # Convert to run command format
    if len(sys.argv) == 1:
        # No arguments, show help
        sys.argv = ['run.py', '--help']
    elif '--config' in sys.argv:
        # Extract config path and convert to new format
        config_idx = sys.argv.index('--config')
        if config_idx + 1 < len(sys.argv):
            config_path = sys.argv[config_idx + 1]
            sys.argv = ['run.py', 'run', '--config', config_path]
        else:
            sys.argv = ['run.py', 'run', '--help']
    else:
        # Default config
        default_config = os.path.join('configs', 'base.yaml')
        sys.argv = ['run.py', 'run', '--config', default_config]
    
    # Run the Typer app
    app()

if __name__ == '__main__':
    main()


