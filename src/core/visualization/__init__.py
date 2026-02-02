"""
Visualization module for embedding analysis and unlearning evaluation.
"""
from .embedding_plots import (
    create_embedding_visualizations,
    create_embedding_summary_stats,
)
from .latent_selection import plot_discriminability_landscape
from .visual_abstract import create_visual_abstract

__all__ = [
    'create_embedding_visualizations',
    'create_embedding_summary_stats',
    'plot_discriminability_landscape',
    'create_visual_abstract',
]
