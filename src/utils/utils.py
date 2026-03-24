"""Utility functions for visualization.

This module provides backward compatibility by re-exporting visualization
functions from the visualization module.
"""

# Import all visualization functions from the dedicated visualization module
from src.utils.visualization import (
    draw_solution,
    visualize_concave_vertices,
    visualize_concave_vertices_cord,
    COLOR_PALETTE
)

# Re-export for backward compatibility
__all__ = [
    'draw_solution',
    'visualize_concave_vertices',
    'visualize_concave_vertices_cord',
    'COLOR_PALETTE'
]
