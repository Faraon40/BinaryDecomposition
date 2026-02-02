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
        (default: 5).
    p_geometry : float, optional
        Probability of geometry mutation (G) (default: 0.05).
        Set to 0.0 to disable.
    p_merge : float, optional
        Probability of merge mutation (M) (default: 0.05).
        Set to 0.0 to disable.
    p_local : float, optional
        Probability of local repartition mutation (L) (default: 0.05).
        Set to 0.0 to disable.
    quadtree_min_size : int, optional
        Minimum quadrant size for quadtree algorithm (default: 4).
    quadtree_trim : bool, optional
        Trim quadtree rectangles to exact coverage (default: True).

    """

    name: str
    seed: int
    algorithm: str

    # GA parameters
    pop_size: int = 30
    generations: int = 100
    elite_size: int = 3
    penalty: float = 2
    patience: int = 5

    # Mutation probabilities (GML naming scheme)
    # Set to 0.0 to disable a mutation
    p_geometry: float = 0.20
    p_merge: float = 0.20
    p_local: float = 0.20

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
            self.p_geometry > 0,
            self.p_merge > 0,
            self.p_local > 0
        ]):
            return "NONE"

        code = ""
        if self.p_geometry > 0:
            code += "G"
        if self.p_merge > 0:
            code += "M"
        if self.p_local > 0:
            code += "L"

        return code