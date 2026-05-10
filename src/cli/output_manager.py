"""Manages output: JSON rectangles, PNG visualization, CSV logging."""

import json
from pathlib import Path
from typing import List

import numpy as np

from experiments.src.logger import CSVLogger
from src.cli.registry import GA_ALGOS
from src.utils.visualization import draw_solution


class OutputManager:
    """Saves results for one algorithm + dataset combination.

    Creates three output sub-trees under base_dir:
      csv/<algo>/<dataset>/results.csv
      rectangles/<algo>/<dataset>/<stem>_rects_N.json
      visualizations/<algo>/<dataset>/<stem>_rects_N.png

    Parameters
    ----------
    base_dir : str or Path
        Root output directory (--output argument).
    algo : str
        Algorithm key used for sub-directory naming.
    dataset_name : str
        Dataset label used for sub-directory naming.

    """

    def __init__(self, base_dir: str, algo: str, dataset_name: str):
        self.base_dir = Path(base_dir)
        self.algo = algo
        self.dataset_name = dataset_name

        self._rects_dir = (
            self.base_dir / "rectangles" / algo / dataset_name
        )
        self._viz_dir = (
            self.base_dir / "visualizations" / algo / dataset_name
        )
        self._rects_dir.mkdir(parents=True, exist_ok=True)
        self._viz_dir.mkdir(parents=True, exist_ok=True)

        self._csv_logger = CSVLogger(
            algorithm_name=algo,
            output_dir=str(self.base_dir / "csv"),
            dataset_name=dataset_name,
        )

    def save(
        self,
        image_name: str,
        img: np.ndarray,
        solution,
        config,
        metrics: dict,
        generation_history: List,
        *,
        no_viz: bool = False,
        no_csv: bool = False,
    ):
        """Persist results for one processed image.

        Parameters
        ----------
        image_name : str
            Original filename (used as stem for output files).
        img : np.ndarray
            Binary image array (for visualization background).
        solution : Chromosome or list of Rectangle
            Algorithm output.
        config : ExperimentConfig
            Config used for the run (seed may be mutated by runner).
        metrics : dict
            Metrics dict from calculate_metrics().
        generation_history : list
            Per-generation rectangle counts (GA only).
        no_viz : bool
            Skip saving the visualization PNG.
        no_csv : bool
            Skip appending to the CSV log.

        """
        rects = _extract_rects(solution)
        rect_count = len(rects)
        stem = Path(image_name).stem

        self._save_json(stem, image_name, rects, rect_count, metrics)

        if not no_viz:
            viz_path = self._viz_dir / f"{stem}_rects_{rect_count}.png"
            draw_solution(
                img,
                rects,
                save_path=str(viz_path),
                show=False,
                title=f"{self.algo} — {stem} ({rect_count} rects)",
            )

        if not no_csv:
            self._csv_logger.log_result(
                image_name, config, metrics, generation_history,
            )

    def log_error(self, image_name: str, msg: str):
        """Delegate error logging to CSVLogger.

        Parameters
        ----------
        image_name : str
            Name of the image that failed.
        msg : str
            Error description.

        """
        self._csv_logger.log_error(image_name, msg)

    def _save_json(
        self,
        stem: str,
        image_name: str,
        rects: list,
        rect_count: int,
        metrics: dict,
    ):
        payload = {
            "image_name": image_name,
            "algorithm": self.algo,
            "rectangle_count": rect_count,
            "execution_time_sec": metrics.get("execution_time_sec"),
            "rectangles": [[int(v) for v in r] for r in rects],
        }
        out = self._rects_dir / f"{stem}_rects_{rect_count}.json"
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)


def _extract_rects(solution) -> list:
    if hasattr(solution, "rectangles"):
        return solution.rectangles
    return list(solution)
