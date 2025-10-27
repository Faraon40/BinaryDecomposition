"""
Experiment configuration for binary decomposition algorithms.

This module defines the configuration dataclass used to specify
experiment parameters for testing different algorithm variants and
mutation combinations.
"""

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run.

    Attributes
    ----------
    name : str
        Human-readable experiment name.
    seed : int
        Random seed for reproducibility.
    algorithm : str
        Algorithm variant: "ga_rle", "ga_random", "ga_quadtree", or
        "quadtree".
    pop_size : int, optional
        Population size for GA (default: 100).
    generations : int, optional
        Maximum generations for GA (default: 100).
    elite_size : int, optional
        Number of elite individuals preserved (default: 3).
    penalty : float, optional
        Penalty multiplier for invalid solutions (default: 1.5).
    patience : int, optional
        Generations without improvement before early stopping
        (default: 10).
    mutation_geometry : bool, optional
        Enable geometry mutation (G) (default: True).
    mutation_merge : bool, optional
        Enable merge mutation (M) (default: True).
    mutation_local : bool, optional
        Enable local repartition mutation (L) (default: True).
    quadtree_min_size : int, optional
        Minimum quadrant size for quadtree algorithm (default: 4).
    quadtree_trim : bool, optional
        Trim quadtree rectangles to exact coverage (default: True).

    """

    name: str
    seed: int
    algorithm: str

    # GA parameters
    pop_size: int = 100
    generations: int = 100
    elite_size: int = 3
    penalty: float = 1.5
    patience: int = 10

    # Mutation flags (GML naming scheme)
    mutation_geometry: bool = True
    mutation_merge: bool = True
    mutation_local: bool = True

    # Quadtree parameters
    quadtree_min_size: int = 4
    quadtree_trim: bool = True

    def get_mutation_combo_code(self) -> str:
        """Get mutation combination code (e.g., 'GML', 'G', 'NONE').

        Returns
        -------
        str
            Mutation combo code using G (geometry), M (merge), L (local).

        """
        if not any([
            self.mutation_geometry,
            self.mutation_merge,
            self.mutation_local
        ]):
            return "NONE"

        code = ""
        if self.mutation_geometry:
            code += "G"
        if self.mutation_merge:
            code += "M"
        if self.mutation_local:
            code += "L"

        return code