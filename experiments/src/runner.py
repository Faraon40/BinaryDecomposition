"""
Single experiment runner.

This module provides the main function to run a single experiment on
one image with specified configuration. Wraps the algorithm
implementations and provides a unified interface.
"""

import random
import time
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np

from src.algorithms.genetic import run_ga
from src.algorithms.graph_based import run_graph_based
from src.algorithms.quadtree import run_quadtree
from src.algorithms.dm import run_dm
from experiments.src.metrics import calculate_metrics


def run_single_experiment(
    image_path: str,
    config
) -> Tuple[object, Dict, List[int]]:
    """Run a single experiment on one image.

    Parameters
    ----------
    image_path : str
        Path to the binary image file (.npy).
    config : ExperimentConfig
        Experiment configuration.

    Returns
    -------
    solution : Chromosome or list of Rectangle
        The resulting solution.
    metrics : dict
        Dictionary of metrics (from calculate_metrics).
    generation_history : list of int
        Rectangle counts per generation (empty for quadtree).

    Raises
    ------
    ValueError
        If algorithm name is invalid.
    FileNotFoundError
        If image file does not exist.

    """
    # Load image
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = np.load(img_path)
    img = (img > 0).astype(int)  # Ensure 0s and 1s

    # Run algorithm based on config
    start_time = time.time()

    if config.algorithm == "quadtree":
        solution, generation_history = run_quadtree(
            img,
            min_size=config.quadtree_min_size,
            trim=config.quadtree_trim,
            verbose=False
        )
        generations_used = None

    elif config.algorithm in [
        "ga_dm", "ga_gdm", "ga_random", "ga_quadtree",
        "ga_morphological", "ga_mixed",
    ]:
        # Determine init method from algorithm name
        init_method = config.algorithm.split("_", 1)[1]

        if config.seed is None:
            config.seed = random.randint(0, 2**31 - 1)

        solution, generation_history = run_ga(
            img,
            pop_size=config.pop_size,
            generations=config.generations,
            elite_size=config.elite_size,
            penalty_extra=config.penalty,
            patience=config.patience,
            seed=config.seed,
            init_method=init_method,
            verbose=False,
            mutation_geometry=config.p_geometry,
            mutation_merge=config.p_merge,
            mutation_local=config.p_local,
            mutation_largest=config.p_largest,
            crossover_method=config.crossover_method,
        )
        generations_used = len(generation_history)

    elif config.algorithm == "graph_based":
        solution, generation_history = run_graph_based(
            img,
            verbose=False
        )
        generations_used = None

    elif config.algorithm == "dm":
        solution = run_dm(img, verbose=False)
        generation_history = []
        generations_used = None

    elif config.algorithm == "gdm":
        from src.algorithms.gdm import run_gdm
        solution = run_gdm(img, verbose=False)
        generation_history = []
        generations_used = None

    elif config.algorithm == "morphological":
        from src.algorithms.morphological import run_morphological
        solution = run_morphological(img, verbose=False)
        generation_history = []
        generations_used = None

    else:
        raise ValueError(
            f"Invalid algorithm: {config.algorithm}. "
            f"Must be 'ga_dm', 'ga_gdm', 'ga_random', 'ga_quadtree', "
            f"'ga_morphological', 'ga_mixed', 'quadtree', "
            f"'graph_based', 'dm', 'gdm', or 'morphological'."
        )

    execution_time = time.time() - start_time

    # Calculate metrics
    metrics = calculate_metrics(
        solution,
        execution_time,
        generations_used
    )

    return solution, metrics, generation_history