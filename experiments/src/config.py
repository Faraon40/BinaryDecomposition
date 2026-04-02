"""
Experiment configuration for binary decomposition algorithms.

This module defines the configuration dataclass used to specify
experiment parameters for testing different algorithm variants and
mutation combinations.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run.

    Attributes
    ----------
    name : str
        Human-readable experiment name.
    seed : Optional[int]
        Random seed for reproducibility. If None, a seed is
        auto-generated at runtime for GA algorithms.
    algorithm : str
        Algorithm variant: "ga_dm", "ga_gdm", "ga_random", "ga_qtd",
        "ga_morph", "ga_mixed", "quadtree", "graph_based",
        "dm", "gdm", or "morphological".
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
        Probability of geometry mutation (G) (default: 0.30).
        Set to 0.0 to disable.
    p_merge : float, optional
        Probability of merge mutation (M) (default: 0.10).
        Set to 0.0 to disable.
    p_local : float, optional
        Probability of local repartition mutation (L) (default: 0.50).
        Set to 0.0 to disable.
    p_largest : float, optional
        Probability of largest-rect mutation (R) (default: 0.20).
        Set to 0.0 to disable.
    p_delete : float, optional
        Probability of delete mutation (D) (default: 0.20).
        Set to 0.0 to disable.
    p_split : float, optional
        Probability of split mutation (S) (default: 0.20).
        Set to 0.0 to disable.
    p_shift : float, optional
        Probability of shift mutation (H) (default: 0.05).
        Set to 0.0 to disable.
    p_repair : float, optional
        Probability of coverage repair after each mutation (default: 0.50).
        Overlap trimming is always applied regardless of this value.
    crossover_method : str, optional
        Crossover method to use (default: "subset_greedy").
        Options: "subset_greedy" (Subset Crossover with Greedy
        Non-overlapping Extension), "single_point", "two_point", "uniform".
    quadtree_full_decomposition : bool, optional
        If True, subdivide down to min_size=2 (precise, more rectangles).
        If False, use adaptive min_size based on image size (default: True).
    quadtree_trim : bool, optional
        Trim mixed leaf rectangles using GDM (default: True).
    morphological_coverage : float, optional
        Stop morphological decomposition once this fraction of foreground
        pixels is covered (default: 1.0 — full coverage).

    """

    name: str
    seed: Optional[int]
    algorithm: str

    # GA parameters
    pop_size: int = 30
    generations: int = 100
    elite_size: int = 3
    penalty: float = 2
    patience: int = 5

    # Mutation probabilities — set to 0.0 to disable
    p_delete: float = 0.20
    p_split: float = 0.20
    p_geometry: float = 0.30
    p_shift: float = 0.05
    p_local: float = 0.50
    p_largest: float = 0.20
    p_repair: float = 0.50
    p_merge: float = 0.10

    # Crossover method
    crossover_method: str = "subset_greedy"

    # Quadtree parameters
    quadtree_full_decomposition: bool = True
    quadtree_trim: bool = True

    # Morphological parameters
    morphological_coverage: float = 0.95

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
            self.p_local > 0,
            self.p_largest > 0,
        ]):
            return "NONE"

        code = ""
        if self.p_geometry > 0:
            code += "G"
        if self.p_merge > 0:
            code += "M"
        if self.p_local > 0:
            code += "L"
        if self.p_largest > 0:
            code += "R"

        return code