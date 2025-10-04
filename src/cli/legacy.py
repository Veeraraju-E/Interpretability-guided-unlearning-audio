"""
Legacy CLI interface for backward compatibility.
Maintains the original argparse-based interface while delegating to the new Typer CLI.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append('/data/veer/testing/new/Interpretability-guided-unlearning-audio/src')

def main():
    """Legacy main function that delegates to the new Typer CLI."""
    # Import the new Typer app
    from src.cli.main import app
    
    # Convert sys.argv to work with Typer
    if len(sys.argv) < 2:
        # No command provided, show help
        sys.argv = ['main.py', '--help']
    elif sys.argv[1] in ['run', 'interpret', 'grid', 'list-configs', 'validate-config']:
        # Command is already in the correct format
        pass
    else:
        # Legacy format: determine command from script name or arguments
        script_name = os.path.basename(sys.argv[0])
        
        if 'run.py' in script_name or 'run' in sys.argv:
            # This is a run command
            if '--config' in sys.argv:
                config_idx = sys.argv.index('--config')
                if config_idx + 1 < len(sys.argv):
                    config_path = sys.argv[config_idx + 1]
                    sys.argv = ['main.py', 'run', '--config', config_path]
                else:
                    sys.argv = ['main.py', 'run', '--help']
            else:
                sys.argv = ['main.py', 'run', '--help']
                
        elif 'interpret.py' in script_name or 'interpret' in sys.argv:
            # This is an interpret command
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
            
            sys.argv = ['main.py', 'interpret'] + args
            
        elif 'grid.py' in script_name or 'grid' in sys.argv:
            # This is a grid command
            if '--config' in sys.argv:
                config_idx = sys.argv.index('--config')
                if config_idx + 1 < len(sys.argv):
                    config_path = sys.argv[config_idx + 1]
                    sys.argv = ['main.py', 'grid', '--config', config_path]
                else:
                    sys.argv = ['main.py', 'grid', '--help']
            else:
                sys.argv = ['main.py', 'grid', '--help']
        else:
            # Unknown command, show help
            sys.argv = ['main.py', '--help']
    
    # Run the Typer app
    app()

if __name__ == '__main__':
    main()
