"""
Metrics calculation for experiment results.

This module provides functions to calculate and extract metrics from
algorithm solutions.
"""

from typing import Dict


def calculate_metrics(
    solution,
    execution_time: float,
    generations_used: int = None
) -> Dict[str, float]:
    """Calculate metrics from a solution.

    Parameters
    ----------
    solution : Chromosome or list of Rectangle
        The solution returned by the algorithm. Can be a Chromosome
        object (GA) or list of rectangles (quadtree).
    execution_time : float
        Execution time in seconds.
    generations_used : int, optional
        Number of generations used (for GA only).

    Returns
    -------
    dict
        Dictionary containing:
        - 'rectangle_count': Number of rectangles in solution
        - 'final_fitness': Fitness value (if available)
        - 'execution_time_sec': Execution time in seconds
        - 'generations_used': Generations used (if applicable)

    """
    metrics = {}

    # Handle both Chromosome objects and plain lists
    if hasattr(solution, 'rectangles'):
        # GA solution (Chromosome object)
        rect_count = len(solution.rectangles)
        metrics['rectangles'] = rect_count
        metrics['rectangle_count'] = rect_count  # Backward compatibility
        metrics['final_fitness'] = solution.fitness
    else:
        # Quadtree solution (list of rectangles)
        rect_count = len(solution)
        metrics['rectangles'] = rect_count
        metrics['rectangle_count'] = rect_count  # Backward compatibility
        metrics['final_fitness'] = None

    metrics['execution_time'] = round(execution_time, 2)
    metrics['execution_time_sec'] = round(execution_time, 2)  # Compat
    metrics['generations_used'] = generations_used

    return metrics