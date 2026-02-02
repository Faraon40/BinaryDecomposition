"""Core algorithms for binary image rectangle decomposition.

Available algorithms:
- Genetic Algorithm (GA) with multiple initialization and mutation strategies
- Quadtree hierarchical decomposition
"""

from src.algorithms.genetic import run_ga
from src.algorithms.quadtree import run_quadtree

__all__ = ["run_ga", "run_quadtree"]