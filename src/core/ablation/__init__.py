"""
Ablation module: INLP, SAL, IGBP, L̄EOPARD erasers for comparison with Audio-SALE.
Each method is applied on editable representations R (either Z @ U with SAE or Z without SAE).
"""

from src.core.ablation.base import AblationEraser, lift_back_to_Z, apply_ablation_eraser

__all__ = [
    "AblationEraser",
    "lift_back_to_Z",
    "apply_ablation_eraser",
]
