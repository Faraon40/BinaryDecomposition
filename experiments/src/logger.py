"""
CSV logging with checkpoint support for experiments.

This module provides a CSV logger that writes experiment results with
automatic checkpointing after each image. Supports resumable execution.
"""

import csv
from pathlib import Path
from typing import Dict, List, Set

_GA_ALGORITHMS = {"ga_dm", "ga_random", "ga_quadtree"}


class CSVLogger:
    """Logger for experiment results with CSV output.

    Creates two CSV files per algorithm and dataset:
    1. results.csv - Main results (one row per image)
    2. generations.csv - Generation history for plotting

    Files are saved in hierarchical structure:
    csv/algorithm_mutationcombo/dataset_name/results.csv

    Parameters
    ----------
    algorithm_name : str
        Name of the algorithm (e.g., "ga_dm").
    output_dir : str or Path
        Base directory to save CSV files.
    dataset_name : str
        Name of dataset directory (e.g., "objects_binary").
    mutation_combo : str
        Mutation combination code (e.g., "GML").

    """

    def __init__(
        self,
        algorithm_name: str,
        output_dir: str,
        dataset_name: str,
        mutation_combo: str
    ):
        """Initialize CSV logger.

        Parameters
        ----------
        algorithm_name : str
            Name of the algorithm.
        output_dir : str
            Base directory path for output CSVs.
        dataset_name : str
            Name of dataset directory.
        mutation_combo : str
            Mutation combination code.

        """
        self.algorithm_name = algorithm_name
        self.dataset_name = dataset_name
        self.mutation_combo = mutation_combo
        self._is_ga = algorithm_name in _GA_ALGORITHMS

        # Create hierarchical directory:
        # csv/algorithm_mutationcombo/dataset_name/
        algo_dir = f"{algorithm_name}_{mutation_combo}"
        self.output_dir = Path(output_dir) / algo_dir / dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results_csv = self.output_dir / "results.csv"
        self.generations_csv = self.output_dir / "generations.csv"

        # Create CSVs with headers if they don't exist
        self._initialize_csv_files()

    def _initialize_csv_files(self):
        """Create CSV files with headers if they don't exist."""
        if not self.results_csv.exists():
            with open(self.results_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                if self._is_ga:
                    writer.writerow([
                        'image_name',
                        'seed',
                        'mutation_combo',
                        'pop_size',
                        'generations',
                        'rectangle_count',
                        'execution_time_sec',
                        'final_fitness',
                        'generations_used'
                    ])
                else:
                    writer.writerow([
                        'image_name',
                        'rectangle_count',
                        'execution_time_sec'
                    ])

        if self._is_ga and not self.generations_csv.exists():
            with open(self.generations_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'image_name',
                    'mutation_combo',
                    'generation',
                    'best_rectangle_count'
                ])

    def log_result(
        self,
        image_name: str,
        config,
        metrics: Dict,
        generation_history: List[int] = None
    ):
        """Log experiment result to CSV.

        Parameters
        ----------
        image_name : str
            Name of the image file.
        config : ExperimentConfig
            Experiment configuration.
        metrics : dict
            Metrics dictionary from calculate_metrics().
        generation_history : list of int, optional
            Rectangle counts per generation (for GA only).

        """
        # Append to results CSV
        with open(self.results_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if self._is_ga:
                mutation_combo = config.get_mutation_combo_code()
                writer.writerow([
                    image_name,
                    config.seed,
                    mutation_combo,
                    config.pop_size,
                    config.generations,
                    metrics['rectangle_count'],
                    metrics['execution_time_sec'],
                    metrics.get('final_fitness', ''),
                    metrics.get('generations_used', '')
                ])
            else:
                writer.writerow([
                    image_name,
                    metrics['rectangle_count'],
                    metrics['execution_time_sec']
                ])

        # Append generation history if available
        if generation_history:
            with open(self.generations_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                for gen, rect_count in enumerate(generation_history):
                    writer.writerow([
                        image_name,
                        mutation_combo,
                        gen,
                        rect_count
                    ])

    def get_processed_images(
        self,
        mutation_combo: str = None
    ) -> Set[str]:
        """Get set of already-processed image names.

        Used for resuming interrupted experiments.

        Parameters
        ----------
        mutation_combo : str, optional
            Filter by mutation combo (e.g., "GML"). If None, returns
            all processed images.

        Returns
        -------
        set of str
            Set of image names already processed.

        """
        if not self.results_csv.exists():
            return set()

        processed = set()
        with open(self.results_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if mutation_combo is None:
                    processed.add(row['image_name'])
                elif row['mutation_combo'] == mutation_combo:
                    processed.add(row['image_name'])

        return processed

    def log_error(self, image_name: str, error_msg: str):
        """Log error to error log file.

        Parameters
        ----------
        image_name : str
            Name of the image that caused the error.
        error_msg : str
            Error message.

        """
        error_log = self.output_dir / "errors.log"
        with open(error_log, 'a') as f:
            f.write(f"{image_name}: {error_msg}\n")