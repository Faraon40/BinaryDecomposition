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


def run_single_experiment_from_array(
    img: np.ndarray,
    config,
) -> Tuple[object, Dict, List[int]]:
    """Run a single experiment on a pre-loaded image array.

    Same interface as run_single_experiment but accepts an
    already-loaded numpy array instead of a file path. Useful
    when the image has been binarized in memory (e.g. from PNG).

    Parameters
    ----------
    img : np.ndarray
        Binary image array. Values are normalized to 0/1 int.
    config : ExperimentConfig
        Experiment configuration.

    Returns
    -------
    solution : Chromosome or list of Rectangle
        The resulting solution.
    metrics : dict
        Dictionary of metrics (from calculate_metrics).
    generation_history : list of int
        Rectangle counts per generation (empty for most algos).

    Raises
    ------
    ValueError
        If algorithm name is invalid.

    """
    img = (img > 0).astype(int)
    start_time = time.time()

    if config.algorithm == "quadtree":
        solution, generation_history = run_quadtree(
            img,
            full_decomposition=config.quadtree_full_decomposition,
            trim=config.quadtree_trim,
            verbose=False,
        )
        generations_used = None

    elif config.algorithm in [
        "ga_dm", "ga_gdm", "ga_random", "ga_qtd",
        "ga_lrf", "ga_mixed",
    ]:
        _init_map = {
            "ga_dm": "dm",
            "ga_gdm": "gdm",
            "ga_random": "random",
            "ga_qtd": "quadtree",
            "ga_lrf": "largest_rect",
            "ga_mixed": "mixed",
        }
        init_method = _init_map[config.algorithm]

        if config.seed is None:
            config.seed = random.randint(0, 2**31 - 1)

        solution, generation_history = run_ga(
            img,
            pop_size=config.pop_size,
            generations=config.generations,
            patience=config.patience,
            elite_size=config.elite_size,
            penalty_extra=config.penalty,
            seed=config.seed,
            init_method=init_method,
            crossover_method=config.crossover_method,
            verbose=False,
            mutation_delete=config.p_delete,
            mutation_split=config.p_split,
            mutation_geometry=config.p_geometry,
            mutation_shift=config.p_shift,
            mutation_local=config.p_local,
            mutation_largest=config.p_largest,
            mutation_merge=config.p_merge,
            repair_coverage_prob=config.p_repair
        )
        generations_used = len(generation_history)

    elif config.algorithm == "graph_based":
        solution, generation_history = run_graph_based(
            img, verbose=False,
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

    elif config.algorithm == "largest_rect":
        from src.algorithms.largest_rect import run_largest_rect
        solution = run_largest_rect(
            img,
            coverage_threshold=config.largest_rect_coverage,
            verbose=False,
        )
        generation_history = []
        generations_used = None

    else:
        raise ValueError(
            f"Invalid algorithm: {config.algorithm}. "
            f"Must be one of: ga_dm, ga_gdm, ga_random, ga_qtd, "
            f"ga_lrf, ga_mixed, quadtree, graph_based, "
            f"dm, gdm, largest_rect."
        )

    execution_time = time.time() - start_time
    metrics = calculate_metrics(
        solution, execution_time, generations_used,
    )
    return solution, metrics, generation_history


def run_single_experiment(
    image_path: str,
    config,
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
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = np.load(img_path)
    return run_single_experiment_from_array(img, config)