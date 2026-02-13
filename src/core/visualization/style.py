import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np

# Color palettes
PALETTES = {
    'primary': ['#E91E63', '#9C27B0', '#3F51B5', '#4CAF50', '#FF5722', '#FFC107'],
    'secondary': ['#006400', '#8B0000', '#000080', '#FF8C00', '#800080', '#008080'],
    'accent': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
    'grayscale': ['#2C3E50', '#7F8C8D', '#95A5A6', '#BDC3C7', '#ECF0F1', '#F8F9FA']
}

# Typography settings
FONT_CONFIG = {
    'family': 'serif',
    'serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'size': 6,
    'weight': 'normal'
}

# Figure settings
FIGURE_CONFIG = {
    'dpi': 600,
    'facecolor': 'white',
    'edgecolor': 'none'
}

# Axes settings
AXES_CONFIG = {
    'titlesize': 7,
    'labelsize': 6,
    'linewidth': 0.8,
    'grid': True,
    'grid_alpha': 0.3,
    'grid_linewidth': 0.5
}

# Tick settings
TICK_CONFIG = {
    'labelsize': 5,
    'major_width': 0.8,
    'minor_width': 0.6
}

# Legend settings
LEGEND_CONFIG = {
    'fontsize': 5,
    'frameon': True,
    'fancybox': False,
    'shadow': False,
    'markerscale': 0.8,
    'handlelength': 1.5
}

# Save settings
SAVE_CONFIG = {
    'dpi': 600,
    'bbox': 'tight',
    'pad_inches': 0.1,
    'facecolor': 'white',
    'edgecolor': 'none'
}


def get_palette(name: str = 'primary') -> List[str]:
    """Get color palette by name."""
    return PALETTES.get(name, PALETTES['primary'])


def apply_publication_style():
    """Apply publication-grade styling to matplotlib."""
    plt.style.use('default')
    
    # Font configuration
    plt.rcParams.update({
        'font.family': FONT_CONFIG['family'],
        'font.serif': FONT_CONFIG['serif'],
        'font.size': FONT_CONFIG['size'],
        'font.weight': FONT_CONFIG['weight'],
    })
    
    # Figure configuration
    plt.rcParams.update({
        'figure.dpi': FIGURE_CONFIG['dpi'],
        'figure.facecolor': FIGURE_CONFIG['facecolor'],
        'figure.edgecolor': FIGURE_CONFIG['edgecolor'],
    })
    
    # Axes configuration
    plt.rcParams.update({
        'axes.titlesize': AXES_CONFIG['titlesize'],
        'axes.labelsize': AXES_CONFIG['labelsize'],
        'axes.linewidth': AXES_CONFIG['linewidth'],
        'axes.grid': AXES_CONFIG['grid'],
        'grid.alpha': AXES_CONFIG['grid_alpha'],
        'grid.linewidth': AXES_CONFIG['grid_linewidth'],
    })
    
    # Tick configuration
    plt.rcParams.update({
        'xtick.labelsize': TICK_CONFIG['labelsize'],
        'ytick.labelsize': TICK_CONFIG['labelsize'],
        'xtick.major.width': TICK_CONFIG['major_width'],
        'ytick.major.width': TICK_CONFIG['major_width'],
        'xtick.minor.width': TICK_CONFIG['minor_width'],
        'ytick.minor.width': TICK_CONFIG['minor_width'],
    })
    
    # Legend configuration
    plt.rcParams.update({
        'legend.fontsize': LEGEND_CONFIG['fontsize'],
        'legend.frameon': LEGEND_CONFIG['frameon'],
        'legend.fancybox': LEGEND_CONFIG['fancybox'],
        'legend.shadow': LEGEND_CONFIG['shadow'],
        'legend.markerscale': LEGEND_CONFIG['markerscale'],
        'legend.handlelength': LEGEND_CONFIG['handlelength'],
    })
    
    # Save configuration
    plt.rcParams.update({
        'savefig.dpi': SAVE_CONFIG['dpi'],
        'savefig.bbox': SAVE_CONFIG['bbox'],
        'savefig.pad_inches': SAVE_CONFIG['pad_inches'],
        'savefig.facecolor': SAVE_CONFIG['facecolor'],
        'savefig.edgecolor': SAVE_CONFIG['edgecolor'],
    })


def create_figure(figsize: Tuple[float, float], nrows: int = 1, ncols: int = 1) -> Tuple[plt.Figure, plt.Axes]:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
    return fig, ax


def create_subplots(rows: int, cols: int, figsize: Tuple[float, float]) -> Tuple[plt.Figure, np.ndarray]:
    apply_publication_style()   
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    return fig, axes


def style_plot(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "",  grid: bool = True, legend: bool = True) -> plt.Axes:
    if title:
        ax.set_title(title, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='bold')
    
    if grid:
        ax.grid(True, alpha=0.3)
    
    if legend:
        ax.legend(loc='best', frameon=True, fancybox=False, shadow=False)
    
    return ax


def save_plot(fig: plt.Figure, path: str, dpi: int = 600, bbox: str = 'tight', pad_inches: float = 0.1, facecolor: str = 'white', edgecolor: str = 'none') -> str:
    bbox_inches = bbox if bbox != 'tight' else None
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, facecolor=facecolor, edgecolor=edgecolor)
    return path


def get_color_cycle(palette: str = 'primary') -> List[str]:
    return get_palette(palette)


def set_color_cycle(ax: plt.Axes, palette: str = 'primary'):
    colors = get_palette(palette)
    ax.set_prop_cycle(color=colors)


apply_publication_style()