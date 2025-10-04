"""
Legacy grid.py - now delegates to the new Typer-based CLI.
Maintains backward compatibility while providing modern CLI features.
"""

import sys
import os

# Add src to path for imports
sys.path.append('/data/veer/testing/new/Interpretability-guided-unlearning-audio/src')

def main():
    """Legacy main function that delegates to the new Typer CLI."""
    # Import and run the new Typer app with grid command
    from src.cli.main import app
    
    # Convert to grid command format
    if '--config' in sys.argv:
        config_idx = sys.argv.index('--config')
        if config_idx + 1 < len(sys.argv):
            config_path = sys.argv[config_idx + 1]
            sys.argv = ['grid.py', 'grid', '--config', config_path]
        else:
            sys.argv = ['grid.py', 'grid', '--help']
    else:
        # No config provided, show help
        sys.argv = ['grid.py', 'grid', '--help']
    
    # Run the Typer app
    app()

if __name__ == '__main__':
    main()


