"""
Main CLI interface for Interpretability-guided-unlearning-audio using Typer.
"""

import sys
import yaml
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

sys.path.append('/data/veer/testing/new/Interpretability-guided-unlearning-audio/src')

from src.datasets.voxceleb_adapter import VoxCelebAdapter
from src.datasets.speech_commands_adapter import SpeechCommandsAdapter
from src.datasets.audiomnist_adapter import AudioMNISTAdapter
from src.datasets.librispeech_adapter import LibriSpeechAdapter
from src.datasets.lahaja_adapter import LahajaAdapter
from src.datasets.mls_spanish_adapter import MLSSpanishAdapter
from src.orchestration.single_run import run_single
from src.orchestration.multi_run import run_multi
from src.orchestration.continual import run_continual
from src.orchestration.linear_probe_experiment import run_linear_probe_experiment

# Initialize Typer app and console
app = typer.Typer(
    name="unlearn-audio",
    help="Interpretability-guided unlearning for audio with SAE-driven feature mining",
    add_completion=False,
    rich_markup_mode="rich"
)
console = Console()

# Dataset adapters registry
ADAPTERS = {
    'voxceleb': VoxCelebAdapter(),
    'speech_commands': SpeechCommandsAdapter(),
    'audiomnist': AudioMNISTAdapter(),
    'librispeech': LibriSpeechAdapter(),
    'lahaja': LahajaAdapter(),
    'mls_spanish': MLSSpanishAdapter(),
}

def load_config(path: str) -> dict:
    try:
        with open(path, 'r') as f: return yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[red]Error: Config file '{path}' not found[/red]")
        raise typer.Exit(1)
    except yaml.YAMLError as e:
        console.print(f"[red]Error: Invalid YAML in config file: {e}[/red]")
        raise typer.Exit(1)

def validate_config_dict(cfg: dict) -> None:
    """Validate configuration parameters."""
    required_sections = ['experiment', 'dataset', 'encoder', 'sae', 'feature_mining', 'edits', 'orchestration']
    for section in required_sections:
        if section not in cfg:
            console.print(f"[red]Error: Missing required config section '{section}'[/red]")
            raise typer.Exit(1)
    
    # Validate dataset adapter
    adapter_name = cfg['dataset']['name']
    if adapter_name not in ADAPTERS:
        console.print(f"[red]Error: Unknown dataset adapter '{adapter_name}'. Available: {list(ADAPTERS.keys())}[/red]")
        raise typer.Exit(1)


def validate_linear_probe_config(cfg: dict) -> None:
    """Validate config for linear-probe experiment (no orchestration required)."""
    required_sections = ['experiment', 'dataset', 'encoder', 'sae', 'feature_mining', 'edits', 'linear_probe']
    for section in required_sections:
        if section not in cfg:
            console.print(f"[red]Error: Missing required config section '{section}'[/red]")
            raise typer.Exit(1)
    lp = cfg['linear_probe']
    if 'output_csv_dir' not in lp:
        console.print("[red]Error: linear_probe.output_csv_dir is required[/red]")
        raise typer.Exit(1)
    if 'epochs' not in lp:
        console.print("[red]Error: linear_probe.epochs is required[/red]")
        raise typer.Exit(1)
    adapter_name = cfg['dataset']['name']
    if adapter_name not in ADAPTERS:
        console.print(f"[red]Error: Unknown dataset adapter '{adapter_name}'. Available: {list(ADAPTERS.keys())}[/red]")
        raise typer.Exit(1)

def print_config_summary(cfg: dict) -> None:
    """Print a summary of the configuration."""
    table = Table(title="Configuration Summary")
    table.add_column("Section", style="cyan")
    table.add_column("Key", style="magenta")
    table.add_column("Value", style="green")
    
    # Experiment settings
    exp = cfg['experiment']
    table.add_row("Experiment", "Name", exp['name'])
    table.add_row("Experiment", "Save Dir", exp['save_dir'])
    table.add_row("Experiment", "Device", exp['device'])
    table.add_row("Experiment", "Seed", str(exp['seed']))
    
    # Dataset settings
    dataset = cfg['dataset']
    table.add_row("Dataset", "Name", dataset['name'])
    table.add_row("Dataset", "Path", dataset['root_or_manifest'])
    table.add_row("Dataset", "Dev Fraction", str(dataset['dev_frac']))
    table.add_row("Dataset", "Forget Labels", str(dataset['forget_labels']))
    
    # Encoder settings
    encoder = cfg['encoder']
    table.add_row("Encoder", "Name", encoder['name'])
    
    # SAE settings
    sae = cfg['sae']
    table.add_row("SAE", "Variant", sae['variant'])
    table.add_row("SAE", "Multiplier", str(sae['multiplier']))
    table.add_row("SAE", "TopK", str(sae['topk']))
    table.add_row("SAE", "Epochs", str(sae['epochs']))
    
    # Orchestration settings
    orch = cfg['orchestration']
    table.add_row("Orchestration", "Mode", orch['mode'])
    
    console.print(table)

@app.command()
def run(
    config: Path = typer.Option(
        ..., 
        "--config", "-c", 
        help="Path to YAML configuration file",
        exists=True
    ),
    show_config: bool = typer.Option(
        False, 
        "--show-config", 
        help="Show configuration summary before running"
    )
):
    """
    Run unlearning experiments based on configuration.
    
    This is the main entry point for running unlearning experiments.
    The mode (single/multi/continual) is determined by the configuration file.
    """
    console.print(Panel.fit(" Starting Unlearning Experiment", style="bold blue"))
    
    # Load and validate config
    cfg = load_config(str(config))
    validate_config_dict(cfg)
    
    if show_config:
        print_config_summary(cfg)
    
    # Get adapter and mode
    adapter_name = cfg['dataset']['name']
    adapter = ADAPTERS[adapter_name]
    mode = cfg['orchestration'].get('mode', 'single')
    
    console.print(f"[green]Using dataset adapter: {adapter_name}[/green]")
    console.print(f"[green]Running in mode: {mode}[/green]")
    
    try:
        if mode == 'single':
            console.print("[yellow]Running single unlearning experiment...[/yellow]")
            results = run_single(adapter, cfg)
            console.print(f"[green] Single unlearning completed. Results: {len(results)} classes processed[/green]")
            
        elif mode == 'multi':
            console.print("[yellow]Running multi-class unlearning experiment...[/yellow]")
            run_multi(adapter, cfg)
            console.print("[green] Multi-class unlearning completed[/green]")
            
        elif mode == 'continual':
            console.print("[yellow]Running continual unlearning experiment...[/yellow]")
            sequence = cfg['dataset'].get('forget_labels', [])
            run_continual(adapter, cfg, sequence)
            console.print(f"[green] Continual unlearning completed. Sequence: {sequence}[/green]")
            
        else:
            console.print(f"[red]Error: Unknown mode '{mode}'. Available: single, multi, continual[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error during execution: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def grid(
    config: Path = typer.Option(
        ..., 
        "--config", "-c", 
        help="Path to YAML configuration file",
        exists=True
    ),
    show_config: bool = typer.Option(
        False, 
        "--show-config", 
        help="Show configuration summary before running"
    )
):
    """
    Run grid search over SAE parameters.
    
    Performs hyperparameter search over SAE multipliers and top-k values.
    Results are saved to a CSV file in the experiment directory.
    """
    console.print(Panel.fit(" Starting Grid Search", style="bold blue"))
    
    # Load and validate config
    cfg = load_config(str(config))
    validate_config_dict(cfg)
    
    if show_config:
        print_config_summary(cfg)
    
    # Get grid search parameters
    multipliers = cfg.get('grid_search', {}).get('sae_multipliers', [2.0, 4.0, 8.0, 16.0, 32.0])
    topks = cfg.get('grid_search', {}).get('sae_topks', [32, 64, 128, 256, 512])
    sweep_epochs = cfg.get('grid_search', {}).get('sweep_epochs', 100)
    
    console.print(f"[green]Grid search parameters:[/green]")
    console.print(f"  - Multipliers: {multipliers}")
    console.print(f"  - TopKs: {topks}")
    console.print(f"  - Sweep epochs: {sweep_epochs}")
    console.print(f"  - Total combinations: {len(multipliers) * len(topks)}")
    
    try:
        # Import and run the grid search functionality
        from src.cli.grid_search import run_grid_search
        
        results_df = run_grid_search(cfg)
        
        console.print("[green] Grid search completed[/green]")
        console.print(f"[green]Results saved to: {cfg['experiment']['save_dir']}/grid_search_results.csv[/green]")
        console.print(f"[green]Total combinations tested: {len(results_df)}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during grid search: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def linear_probe(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to YAML configuration file",
        exists=True
    ),
    show_config: bool = typer.Option(
        False,
        "--show-config",
        help="Show configuration summary before running"
    )
):
    """
    Run linear probing for class identity on encoder representations (E_all).
    Appends one row to linear_probe_<seed>.csv in the configured output_csv_dir.
    """
    console.print(Panel.fit(" Linear Probing", style="bold blue"))

    cfg = load_config(str(config))
    validate_linear_probe_config(cfg)

    if show_config:
        exp = cfg['experiment']
        lp = cfg['linear_probe']
        console.print(f"[cyan]Experiment:[/cyan] {exp.get('name')} (seed={exp.get('seed')})")
        console.print(f"[cyan]Dataset:[/cyan] {cfg['dataset']['name']}")
        console.print(f"[cyan]Encoder:[/cyan] {cfg['encoder']['name']}")
        console.print(f"[cyan]Linear probe:[/cyan] output_csv_dir={lp['output_csv_dir']}, epochs={lp['epochs']}")

    adapter_name = cfg['dataset']['name']
    adapter = ADAPTERS[adapter_name]
    console.print(f"[green]Using dataset adapter: {adapter_name}[/green]")

    try:
        accuracy = run_linear_probe_experiment(adapter, cfg)
        seed = cfg['experiment']['seed']
        csv_path = str(Path(cfg['linear_probe']['output_csv_dir']) / f"linear_probe_{seed}.csv")
        console.print(f"[green]Linear probe completed. Test accuracy: {accuracy:.4f}[/green]")
        console.print(f"[green]Appended to: {csv_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error during linear probing: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_configs():
    console.print(Panel.fit(" Available Configuration Files", style="bold blue"))
    
    configs_dir = Path("configs")
    if not configs_dir.exists():
        console.print("[yellow]No configs directory found[/yellow]")
        return
    
    table = Table(title="Configuration Files")
    table.add_column("File", style="cyan")
    table.add_column("Dataset", style="magenta")
    table.add_column("Description", style="green")
    
    config_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))
    
    for config_file in sorted(config_files):
        try:
            cfg = load_config(str(config_file))
            dataset_name = cfg.get('dataset', {}).get('name', 'unknown')
            exp_name = cfg.get('experiment', {}).get('name', 'unnamed')
            table.add_row(config_file.name, dataset_name, exp_name)
        except:
            table.add_row(config_file.name, "error", "Invalid config")
    
    console.print(table)

@app.command()
def validate_config(
    config: Path = typer.Option(
        ..., 
        "--config", "-c", 
        help="Path to YAML configuration file to validate",
        exists=True
    )
):
    console.print(Panel.fit(" Validating Configuration", style="bold blue"))
    
    try:
        cfg = load_config(str(config))
        validate_config_dict(cfg)
        console.print("[green] Configuration is valid![/green]")
        print_config_summary(cfg)
        
    except Exception as e:
        console.print(f"[red] Configuration validation failed: {e}[/red]")
        raise typer.Exit(1)

@app.callback()
def main(
    version: bool = typer.Option(
        False, 
        "--version", 
        help="Show version information"
    )
):
    """
    Main interface for SAE-guided unlearning for audio.    
    """
    if version:
        console.print("SAE-guided-unlearning-audio")
        raise typer.Exit()

if __name__ == "__main__":
    app()
