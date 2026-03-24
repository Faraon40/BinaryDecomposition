"""Core algorithms for binary image rectangle decomposition.

Available algorithms:
- Genetic Algorithm (GA) with multiple initialization and mutation strategies
- Quadtree hierarchical decomposition
- Delta Method (DM) decomposition
"""

from src.algorithms.genetic import run_ga
from src.algorithms.quadtree import run_quadtree
from src.algorithms.dm import run_dm

__all__ = ["run_ga", "run_quadtree", "run_dm"]