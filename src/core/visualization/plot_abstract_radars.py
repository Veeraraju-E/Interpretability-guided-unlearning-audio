"""
Radar charts for adversarial robustness table (abstract_table.csv): one figure per
encoder model, two axes — Audio MNIST and Speech Commands. Attack axes ordered with
NIL at the top (compass north), then FGSM, PGD, Auto clockwise.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import apply_publication_style, get_palette, save_plot

_AXIS_LABEL_FONTSIZE = 11
_TICK_FONTSIZE = 10
_LEGEND_FONTSIZE = 9

_ATTACK_LABELS = ("NIL", "FGSM", "PGD", "Auto")


def _default_csv_path() -> Path:
    return Path(__file__).resolve().parents[3] / "abstract_table.csv"


def plot_model_radars(
    df: pd.DataFrame,
    model_name: str,
    *,
    output_png: str | None = None,
    output_pdf: str | None = None,
    figsize: tuple[float, float] = (7.2, 3.6),
    ylabel: str = "ERB (%)",
    rlim: tuple[float, float] = (0.0, 100.0),
    palette_name: str = "primary",
) -> dict[str, Any]:
    """
    Two polar subplots (Audio MNIST | Speech Commands) for one encoder model.
    """
    apply_publication_style()
    sub = df[df["model"].astype(str) == model_name].copy()
    if sub.empty:
        raise ValueError(f"No rows for model={model_name!r}")

    palette = get_palette(palette_name)
    methods = list(sub["method"].astype(str))

    # 4 axes; NIL first index -> drawn at theta=0 -> north after set_theta_zero_location
    n_axes = len(_ATTACK_LABELS)
    theta = np.linspace(0.0, 2.0 * np.pi, n_axes, endpoint=False)
    theta_closed = np.append(theta, theta[0])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        subplot_kw=dict(projection="polar"),
        constrained_layout=True,
    )
    datasets: tuple[tuple[str, str], ...] = (
        ("audio_mnist", "Audio MNIST"),
        ("speech_commands", "Speech Commands"),
    )

    for ax, (prefix, title) in zip(axes, datasets):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        if prefix == "audio_mnist":
            cols = [
                "audio_mnist_nil",
                "audio_mnist_fgsm",
                "audio_mnist_pgd",
                "audio_mnist_auto",
            ]
        else:
            cols = [
                "speech_commands_nil",
                "speech_commands_fgsm",
                "speech_commands_pgd",
                "speech_commands_auto",
            ]

        for mi, method in enumerate(methods):
            row = sub[sub["method"].astype(str) == method].iloc[0]
            r = np.array([float(row[c]) for c in cols], dtype=float)
            r_closed = np.append(r, r[0])
            color = palette[mi % len(palette)]
            ax.plot(theta_closed, r_closed, color=color, linewidth=1.2, label=method)
            ax.fill(theta_closed, r_closed, color=color, alpha=0.12)

        ax.set_xticks(theta)
        ax.set_xticklabels(list(_ATTACK_LABELS), fontsize=_TICK_FONTSIZE)
        ax.set_ylim(rlim[0], rlim[1])
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels([20, 40, 60, 80, 100], fontsize=_TICK_FONTSIZE - 1)
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=_AXIS_LABEL_FONTSIZE, fontweight="bold", pad=12)

    # Single y-label on the figure (radial quantity)
    fig.supylabel(ylabel, fontsize=_AXIS_LABEL_FONTSIZE, fontweight="bold", x=0.02)
    fig.suptitle(model_name, fontsize=_AXIS_LABEL_FONTSIZE + 1, fontweight="bold", y=1.02)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(5, len(methods)),
        frameon=True,
        fancybox=False,
        fontsize=_LEGEND_FONTSIZE,
        bbox_to_anchor=(0.5, -0.02),
    )

    out: dict[str, Any] = {"fig": fig, "axes": axes}
    if output_png:
        os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
        save_plot(fig, output_png)
        out["png"] = output_png
    if output_pdf:
        os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
        save_plot(fig, output_pdf)
        out["pdf"] = output_pdf
    if output_png or output_pdf:
        plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Radar plots from abstract_table.csv (per model).")
    parser.add_argument("--csv", type=str, default="", help="Path to abstract_table.csv")
    parser.add_argument(
        "--out-w2v",
        default="abstract_radar_wav2vec2.png",
        help="Output PNG for Wav2Vec2",
    )
    parser.add_argument(
        "--out-hubert",
        default="abstract_radar_hubert.png",
        help="Output PNG for HuBERT",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve() if args.csv else _default_csv_path()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    plot_model_radars(df, "Wav2Vec2", output_png=str(Path(args.out_w2v).resolve()))
    print(f"Saved {args.out_w2v}")

    plot_model_radars(df, "HuBERT", output_png=str(Path(args.out_hubert).resolve()))
    print(f"Saved {args.out_hubert}")


if __name__ == "__main__":
    main()
