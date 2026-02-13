#!/usr/bin/env python3
"""
Build_U_top Analysis Visualization for SpeechCommands Dataset
Creates a publication-quality 1x3 subplot figure showing the effect of 
build_U_top percentage on ERB scores for ECAPA, Wav2Vec2, and HuBERT encoders.
Adapted for SpeechCommands dataset with command-based labels.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from .style import apply_publication_style, get_palette, create_figure, style_plot, save_plot
import matplotlib.patches as mpatches

# Apply unified styling
apply_publication_style()


def load_and_process_data(csv_path: str) -> pd.DataFrame:
    """Load CSV data and prepare for plotting."""
    df = pd.read_csv(csv_path)
    
    # Remove rows with NaN ERB scores
    df = df.dropna(subset=['ERB_score'])
    
    # Convert build_U_top_pct to numeric for sorting
    df['build_U_top_pct_numeric'] = df['build_U_top_pct'].str.rstrip('%').astype(int)
    
    # Sort by encoder and percentage
    df = df.sort_values(['encoder', 'build_U_top_pct_numeric'])
    
    return df


def create_build_u_top_plot(output_dir: str, dataset_name: str = None):
    """Create plots with different layouts based on dataset."""
   
    encoders = ['ecapa', 'wav2vec2', 'hubert']
    encoder_names = {
        'ecapa': 'ECAPA-TDNN',
        'wav2vec2': 'Wav2Vec2',
        'hubert': 'HuBERT'
    }
    
    # Define colors for each encoder
    colors = ['#E91E63', '#9C27B0', '#3F51B5']  # Pink, Purple, Blue
    
    if dataset_name == 'lahaja' or dataset_name == 'mls_spanish':
        # Single combined grouped-bar plot for Lahaja and MLS Spanish (formal style)
        fig, ax = create_figure((8.0, 4.0), nrows=1, ncols=1)
        
        # Load both datasets
        lahaja_path = '/data/veer/testing/new/Interpretability-guided-unlearning-audio/runs/lahaja/build_u_top_analysis/build_u_top_analysis/build_u_top_analysis_summary.csv'
        mls_spanish_path = '/data/veer/testing/new/Interpretability-guided-unlearning-audio/runs/mls_spanish/build_u_top_analysis/build_u_top_analysis/build_u_top_analysis_summary.csv'
        
        # Use primary palette from shared style (consistent with SAE hyperparameter plots)
        palette = get_palette('primary')
        dataset_colors = [palette[0], palette[1]]
        
        df_lahaja, df_mls = None, None
        if os.path.exists(lahaja_path):
            df_lahaja = load_and_process_data(lahaja_path)
        if os.path.exists(mls_spanish_path):
            df_mls = load_and_process_data(mls_spanish_path)
        
        # Draw grouped bars if both are present; otherwise fall back to single bars
        if df_lahaja is not None and df_mls is not None:
            plot_grouped_bars_two_datasets(ax, df_lahaja, df_mls, dataset_colors, ['Lahaja', 'MLS Spanish'], y_min=0)
        elif df_lahaja is not None:
            plot_single_dataset_bars(df_lahaja, ax, dataset_colors[0], 'Lahaja', 0, show_y_label=True, show_x_label=True)
        elif df_mls is not None:
            plot_single_dataset_bars(df_mls, ax, dataset_colors[1], 'MLS Spanish', 0, show_y_label=True, show_x_label=True)
        
        # Labels (no title, legend suppressed within helpers)
        ax.set_xlabel('Discriminative Latents (%)', fontweight='bold', fontsize=14)
        
        
    elif dataset_name == 'voxceleb' or dataset_name == 'librispeech':
        # Combined plot for VoxCeleb and LibriSpeech
        fig, axes = create_figure((8.0, 4.0), nrows=1, ncols=2)
        
        # Load both datasets
        voxceleb_path = '/data/veer/testing/new/Interpretability-guided-unlearning-audio/runs/voxceleb/build_u_top_analysis/build_u_top_analysis_summary.csv'
        librispeech_path = '/data/veer/testing/new/Interpretability-guided-unlearning-audio/runs/librispeech/build_u_top_analysis/build_u_top_analysis_summary.csv'
        
        # Plot VoxCeleb (left subplot)
        if os.path.exists(voxceleb_path):
            df_vox = load_and_process_data(voxceleb_path)
            plot_single_dataset(df_vox, axes[0], encoders, encoder_names, colors, 50, show_y_label=True, show_x_label=False)
            axes[0].set_title('VoxCeleb', fontweight='bold', fontsize=14)
        
        # Plot LibriSpeech (right subplot)
        if os.path.exists(librispeech_path):
            df_lib = load_and_process_data(librispeech_path)
            plot_single_dataset(df_lib, axes[1], encoders, encoder_names, colors, 0, show_y_label=False, show_x_label=False)
            axes[1].set_title('LibriSpeech', fontweight='bold', fontsize=14)
        
        # Add common x-axis label centered below both subplots with sufficient gap
        fig.text(0.5, -0.05, 'Discriminative Latents (%)', ha='center', va='top', fontweight='bold', fontsize=14)
        
    else:
        # Combined plot for SpeechCommands and AudioMNIST
        fig, axes = create_figure((8.0, 4.0), nrows=1, ncols=2)
        
        # Load both datasets
        speech_commands_path = '/data/veer/testing/new/Interpretability-guided-unlearning-audio/runs/build_u_top_analysis/speech_commands/build_u_top_analysis/build_u_top_analysis_summary.csv'
        audiomnist_path = '/data/veer/testing/new/Interpretability-guided-unlearning-audio/runs/build_u_top_analysis/audiomnist/build_u_top_analysis/build_u_top_analysis_summary.csv'
        
        # Plot SpeechCommands (left subplot)
        if os.path.exists(speech_commands_path):
            df_sc = load_and_process_data(speech_commands_path)
            plot_single_dataset(df_sc, axes[0], encoders, encoder_names, colors, 40, show_y_label=True, show_x_label=False)
            axes[0].set_title('Speech Commands', fontweight='bold', fontsize=14)
        
        # Plot AudioMNIST (right subplot)
        if os.path.exists(audiomnist_path):
            df_am = load_and_process_data(audiomnist_path)
            plot_single_dataset(df_am, axes[1], encoders, encoder_names, colors, 0, show_y_label=False, show_x_label=False)
            axes[1].set_title('AudioMNIST', fontweight='bold', fontsize=14)
        
        # Add common x-axis label centered below both subplots with sufficient gap
        fig.text(0.5, -0.05, 'Discriminative Latents (%)', ha='center', va='top', fontweight='bold', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figures
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine filename based on layout
    if dataset_name == 'lahaja' or dataset_name == 'mls_spanish':
        filename_suffix = 'lahaja_mls_spanish'
    elif dataset_name == 'voxceleb' or dataset_name == 'librispeech':
        filename_suffix = 'voxceleb_librispeech'
    else:
        filename_suffix = 'speech_commands_audiomnist'
    
    # Save as PNG
    png_path = output_path / f'build_u_top_analysis_{filename_suffix}.png'
    save_plot(fig, str(png_path))
    print(f"Saved PNG figure: {png_path}")
    
    # Save as PDF
    pdf_path = output_path / f'build_u_top_analysis_{filename_suffix}.pdf'
    save_plot(fig, str(pdf_path))
    print(f"Saved PDF figure: {pdf_path}")
    
    plt.show()
    
    return fig


def plot_single_dataset_bars(df, ax, color, label, y_min, show_y_label=True, show_x_label=True):
    """Plot a single dataset as bars on the given axes (for datasets with only wav2vec2)."""
    
    # Filter data for wav2vec2 (should be the only encoder)
    encoder_data = df[df['encoder'] == 'wav2vec2'].copy()
    
    if len(encoder_data) == 0:
        return
    
    # Get x and y values
    x_values = encoder_data['build_U_top_pct_numeric'].values
    y_values = encoder_data['ERB_score'].values
    
    # Create bar plot
    # Use evenly spaced positions for bars as well
    x_cats = sorted(encoder_data['build_U_top_pct_numeric'].unique())
    x = np.arange(len(x_cats))
    y = [float(encoder_data[encoder_data['build_U_top_pct_numeric'] == c]['ERB_score'].iloc[0]) for c in x_cats]
    # Adaptive width similar to grouped case but single bar -> use ~0.5 of group
    if len(x_cats) <= 4:
        bar_width = 0.5
    elif len(x_cats) <= 8:
        bar_width = 0.45
    else:
        bar_width = 0.4
    bars = ax.bar(x, y, width=bar_width, color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Format the plot
    if show_x_label:
        ax.set_xlabel('Discriminative Latents (%)', fontweight='bold', fontsize=14)
    if show_y_label:
        ax.set_ylabel('ERB Score', fontweight='bold', fontsize=14)
    
    # Set x-axis properties
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(c)}' for c in x_cats], fontsize=12)
    
    # Set y-axis properties
    y_max = min(100, df['ERB_score'].max() + 5)
    ax.set_ylim(y_min, y_max)
    # Set y-ticks with appropriate spacing
    y_range = y_max - y_min
    if y_range <= 20:
        y_tick_step = 5
    elif y_range <= 50:
        y_tick_step = 10
    else:
        y_tick_step = 20
    y_ticks = np.arange(y_min, y_max + 1, y_tick_step)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{int(y)}' for y in y_ticks], fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    # No legend in single-dataset mode


def plot_grouped_bars_two_datasets(ax, df_a, df_b, colors, labels, y_min = 0):
    """Grouped bars for two datasets (assumes only wav2vec2 in each),
    using evenly spaced x positions and formal styling similar to SAE plots."""
    data_a = df_a[df_a['encoder'] == 'wav2vec2'].copy()
    data_b = df_b[df_b['encoder'] == 'wav2vec2'].copy()
    if len(data_a) == 0 and len(data_b) == 0:
        return
    
    # Ensure aligned x categories (percent integers)
    x_cats = sorted(set(data_a['build_U_top_pct_numeric'].tolist()) | set(data_b['build_U_top_pct_numeric'].tolist()))
    # Build y arrays aligned to x_cats, fill missing with np.nan to leave gaps
    map_a = {int(r['build_U_top_pct_numeric']): float(r['ERB_score']) for _, r in data_a.iterrows()}
    map_b = {int(r['build_U_top_pct_numeric']): float(r['ERB_score']) for _, r in data_b.iterrows()}
    y_a = [map_a.get(x, np.nan) for x in x_cats]
    y_b = [map_b.get(x, np.nan) for x in x_cats]
    
    # Use evenly spaced positions (0..N-1) for grouped bars
    x = np.arange(len(x_cats), dtype=float)
    # Adaptive group width similar to SAE plot
    n_groups = len(x_cats)
    if n_groups <= 4:
        total_group_width = 0.8
    elif n_groups <= 8:
        total_group_width = 0.75
    else:
        total_group_width = 0.7
    # Two bars per group
    bar_width = total_group_width / 2.0
    # Center the pair around each x
    ax.bar(x - bar_width/2, y_a, width=bar_width, color=colors[0], edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.bar(x + bar_width/2, y_b, width=bar_width, color=colors[1], edgecolor='black', linewidth=0.5, alpha=0.85)
    
    # Axes formatting
    ax.set_ylabel('ERB Score', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(v)}' for v in x_cats], fontsize=12)
    
    # Y range
    combined = np.array([v for v in y_a + y_b if not np.isnan(v)])
    if combined.size > 0:
        y_data_max = float(np.nanmax(combined))
    else:
        y_data_max = 100.0
    # Base max limited to 100 for ticks; we'll extend visually by one tick later
    y_max_for_ticks = min(100.0, y_data_max + 5)
    # Tick step similar to single bars helper
    y_range = y_max_for_ticks - y_min
    if y_range <= 20:
        y_tick_step = 5
    elif y_range <= 50:
        y_tick_step = 10
    else:
        y_tick_step = 20
    # Generate ticks only up to 100 (inclusive)
    y_ticks = np.arange(y_min, 100 + 1, y_tick_step)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{int(y)}' for y in y_ticks], fontsize=12)
    # Now extend ylim by one extra tick beyond 100 for legend space, without adding tick/grid
    ax.set_ylim(y_min, 100 + y_tick_step)
    
    ax.grid(True, alpha=0.3, axis='y')
    # Add legend for the two datasets (top-right)
    handles = [
        mpatches.Patch(facecolor=colors[0], edgecolor='black', label=labels[0]),
        mpatches.Patch(facecolor=colors[1], edgecolor='black', label=labels[1])
    ]
    ax.legend(
        handles=handles,
        loc='upper right',
        frameon=True,
        fancybox=False,
        shadow=False,
        fontsize=12,
        ncol=2,
    )


def plot_single_dataset(df, ax, encoders, encoder_names, colors, y_min, show_y_label=True, show_x_label=True):
    """Plot a single dataset on the given axes."""
    
    # Plot each encoder on the same axes
    for idx, encoder in enumerate(encoders):
        # Filter data for current encoder
        encoder_data = df[df['encoder'] == encoder].copy()
        
        if len(encoder_data) == 0:
            continue
        
        # Get x and y values
        x_values = encoder_data['build_U_top_pct_numeric'].values
        y_values = encoder_data['ERB_score'].values
        
        # Create standard deviations array (1% for all points)
        std_devs = np.full(len(x_values), 0.01)
        
        # Plot ERB scores with error bars
        ax.errorbar(x_values, y_values, 
                   yerr=std_devs,
                   marker='o', 
                   linewidth=2.0, 
                   markersize=6,
                   color=colors[idx],
                   markerfacecolor=colors[idx],
                   markeredgecolor=colors[idx],
                   markeredgewidth=1.5,
                   alpha=0.8,
                   capsize=4,
                   capthick=1.5,
                   elinewidth=1.5,
                   label=encoder_names.get(encoder, encoder))
        
        # Add shaded error region
        upper_bound = y_values + std_devs
        lower_bound = y_values - std_devs
        ax.fill_between(x_values, lower_bound, upper_bound, 
                       color=colors[idx], alpha=0.2)
    
    # Format the plot
    if show_x_label:
        ax.set_xlabel('Discriminative Latents (%)', fontweight='bold', fontsize=14)
    if show_y_label:
        ax.set_ylabel('ERB Score', fontweight='bold', fontsize=14)
    
    # Set x-axis properties
    all_x_values = sorted(df['build_U_top_pct_numeric'].unique())
    ax.set_xticks(all_x_values)
    ax.set_xticklabels([f'{int(x)}' for x in all_x_values], fontsize=12)
    
    # Set y-axis properties
    y_max = min(100, df['ERB_score'].max() + 2)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(y_min, y_max + 1, 10))
    ax.set_yticklabels([f'{int(y)}' for y in np.arange(y_min, y_max + 1, 10)], fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='lower right', frameon=True, fancybox=False, shadow=False, fontsize=12, markerscale=1.0, handlelength=2.0)


def main():
    """Main function to create build_U_top analysis plots for SpeechCommands dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build_U_top Analysis Visualization for LibriSpeech')
    parser.add_argument('--output', type=str, required=True, help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create main plot
    print("Creating main build_U_top analysis plot for Lahaja...")
    create_build_u_top_plot(args.output, dataset_name='lahaja')
    
    print("Lahaja build_U_top analysis visualization completed successfully!")


if __name__ == "__main__":
    main()
