"""
CLI for ablation runs: run --config produces results.csv + Pareto frontier plot.
Tuning is done via separate scripts in src/core/ablation/tuning/ (no tune command here).
"""
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from src.orchestration.ablation_run import run_ablation
from src.core.visualization.plot_pareto_frontier import plot_pareto_frontier

app = typer.Typer(
    name="ablation",
    help="Run ablation methods (INLP, SAL, IGBP, L̄EOPARD) and plot Pareto frontier",
    add_completion=False,
)
console = Console()


def load_config(path: str) -> dict:
    import yaml
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[red]Error: Config file '{path}' not found[/red]")
        raise typer.Exit(1)
    except yaml.YAMLError as e:
        console.print(f"[red]Error: Invalid YAML: {e}[/red]")
        raise typer.Exit(1)


def validate_ablation_config(cfg: dict) -> None:
    """Require experiment, dataset, encoder, sae, feature_mining, edits, orchestration, ablation."""
    required = [
        "experiment", "dataset", "encoder", "sae", "feature_mining", "edits", "orchestration", "ablation"
    ]
    for section in required:
        if section not in cfg:
            console.print(f"[red]Error: Missing config section '{section}'[/red]")
            raise typer.Exit(1)
    ab = cfg["ablation"]
    if "output_csv" not in ab:
        console.print("[red]Error: ablation.output_csv is required[/red]")
        raise typer.Exit(1)


@app.command()
def run(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to ablation YAML config",
        exists=True,
    ),
):
    """
    Run ablation with best hyperparams from config; write results.csv and Pareto frontier plot.
    """
    console.print(Panel.fit(" Ablation Run ", style="bold blue"))
    cfg = load_config(str(config))
    validate_ablation_config(cfg)
    ab = cfg["ablation"]
    output_csv = ab["output_csv"]
    try:
        csv_path = run_ablation(cfg)
        console.print(f"[green]Results written to {csv_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error during ablation run: {e}[/red]")
        raise typer.Exit(1)
    plot_cfg = ab.get("plot", {})
    out_png = plot_cfg.get("pareto_png", "pareto_frontier.png")
    out_pdf = plot_cfg.get("pareto_pdf", "pareto_frontier.pdf")
    base_dir = Path(csv_path).parent
    if not Path(out_png).is_absolute():
        out_png = str(base_dir / out_png)
    if not Path(out_pdf).is_absolute():
        out_pdf = str(base_dir / out_pdf)
    try:
        plot_pareto_frontier(csv_path, output_png=out_png, output_pdf=out_pdf)
        console.print(f"[green]Pareto plot saved: {out_png}, {out_pdf}[/green]")
    except Exception as e:
        console.print(f"[yellow]Pareto plot failed: {e}[/yellow]")


if __name__ == "__main__":
    app()
