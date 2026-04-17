"""Run greedy largest-rectangle decomposition experiments on binary images.

This script runs the largest-rectangle-first (histogram-based) decomposition
algorithm on image directories and saves results, rectangles, and
visualizations for analysis.
"""

import json
import time
from pathlib import Path
from typing import List, Tuple, Any, Dict

import numpy as np

from experiments.src.config import ExperimentConfig
from experiments.src.logger import CSVLogger
from experiments.src.runner import run_single_experiment
from src.utils.utils import draw_solution


def save_solution_rectangles(
    image_name: str,
    rectangles: List[Tuple[int, int, int, int]],
    rect_count: int,
    output_dir: Path,
    config: ExperimentConfig,
    metrics: dict,
    dataset_name: str,
    run_id: str = "run1",
):
    """Save solution rectangles to JSON file.

    Parameters
    ----------
    image_name : str
        Name of the image (with extension).
    rectangles : list of tuple
        List of (x, y, width, height) rectangles.
    rect_count : int
        Number of rectangles in solution.
    output_dir : Path
        Base output directory (results/rectangles/).
    config : ExperimentConfig
        Experiment configuration used.
    metrics : dict
        Metrics dictionary from experiment.
    dataset_name : str
        Name of the dataset directory (e.g., "research_leafs_binary").
    run_id : str, optional
        Run identifier for distinguishing multiple runs (default: "run1").

    Returns
    -------
    Path
        Path to saved JSON file.

    """
    image_stem = Path(image_name).stem

    save_dir = output_dir / "largest_rect" / dataset_name / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{image_stem}_rects_{rect_count}.json"
    save_path = save_dir / filename

    data = {
        "image_name": image_name,
        "rectangle_count": rect_count,
        "rectangles": [
            [int(x), int(y), int(w), int(h)]
            for x, y, w, h in rectangles
        ],
        "config": {
            "algorithm": config.algorithm,
            "largest_rect_coverage": config.largest_rect_coverage,
        },
        "metrics": metrics
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    return save_path


def load_solution_rectangles(
    json_path: str
) -> tuple[list[tuple[Any]], dict[str, Any]]:
    """Load rectangles from JSON file.

    Parameters
    ----------
    json_path : str
        Path to JSON file with saved rectangles.

    Returns
    -------
    rectangles : list of tuple
        List of (x, y, width, height) Rectangle tuples.
    metadata : dict
        Dictionary with image_name, config, and metrics.

    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rectangles = [tuple(rect) for rect in data['rectangles']]

    metadata = {
        'image_name': data['image_name'],
        'rectangle_count': data['rectangle_count'],
        'config': data['config'],
        'metrics': data['metrics']
    }

    return rectangles, metadata


def run_experiments(
    image_dir_name: str,
    max_images: int = None,
    run_id: str = "run1",
    coverage: float = 0.9,
):
    """Run largest-rectangle decomposition experiments on images.

    Parameters
    ----------
    image_dir_name : str
        Directory name under data/datasets/ (e.g., "leafs_unique_color",
        "research_leafs_binary", "objects_binary").
    max_images : int, optional
        Maximum number of images to process. If None, processes all
        images in directory (default: None).
    run_id : str, optional
        Identifier for this run (default: "run1"). Results are saved
        under a subdirectory named after run_id, so different run_ids
        never overwrite each other.
    coverage : float, optional
        Stop once this fraction of foreground pixels is covered
        (default: 1.0 — full coverage). Values below 1.0 produce
        fewer rectangles but leave some pixels uncovered (residual
        covered by GDM).

    """
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    image_dir = project_root / "data/datasets" / image_dir_name / "npy"

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}\n"
            f"Available directories in data/datasets/: "
            f"{[d.name for d in (project_root / 'data/datasets').iterdir() if d.is_dir()]}"
        )

    image_paths = sorted(image_dir.glob("*.npy"))

    if not image_paths:
        raise FileNotFoundError(
            f"No .npy files found in {image_dir}"
        )

    if max_images is not None:
        image_paths = image_paths[:max_images]

    mode_str = (
        f"TEST MODE ({len(image_paths)} images)"
        if max_images else "PRODUCTION MODE (all images)"
    )

    print("=" * 70)
    print(f"LARGEST-RECT DECOMPOSITION EXPERIMENTS - {mode_str}")
    print("=" * 70)
    print(f"Directory: {image_dir_name}")
    print(f"Run ID: {run_id}")
    print(f"Coverage threshold: {coverage}")
    print(f"Images to process: {len(image_paths)}")
    print(
        f"Algorithm: largest_rect "
        f"(greedy largest-rectangle-first, histogram-based)"
    )
    print("=" * 70)

    config = ExperimentConfig(
        name=f"largest_rect_{image_dir_name}_{run_id}",
        seed=None,
        algorithm="largest_rect",
        largest_rect_coverage=coverage,
    )

    logger = CSVLogger(
        "largest_rect",
        str(project_root / "experiments/results/csv/"),
        f"{image_dir_name}/{run_id}",
    )

    print("-" * 70)

    viz_dir = project_root / "experiments/results/visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    rect_dir = project_root / "experiments/results/rectangles"
    rect_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i:2d}/{len(image_paths)}] Processing: {img_path.name}")
        print(f"  Running... ", end="", flush=True)

        start_time = time.time()

        try:
            solution, metrics, history = run_single_experiment(
                str(img_path), config
            )

            elapsed = time.time() - start_time
            print(f"Done in {elapsed:.1f}s")

            rectangles = (
                solution.rectangles
                if hasattr(solution, 'rectangles')
                else solution
            )
            rect_count = len(rectangles)

            logger.log_result(
                img_path.name, config, metrics, history
            )

            print(f"  Rectangles: {rect_count}")
            print(f"  Time: {metrics['execution_time_sec']:.1f}s")

            rect_path = save_solution_rectangles(
                img_path.name,
                rectangles,
                rect_count,
                rect_dir,
                config,
                metrics,
                image_dir_name,
                run_id=run_id,
            )
            print(f"  Rectangles saved: "
                  f"{rect_path.relative_to(project_root)}")

            img = np.load(img_path)
            img = (img > 0).astype(int)

            viz_subdir = (
                viz_dir / "largest_rect" / image_dir_name / run_id
            )
            viz_subdir.mkdir(parents=True, exist_ok=True)

            viz_filename = f"{img_path.stem}_rects_{rect_count}.png"
            viz_path = viz_subdir / viz_filename

            draw_solution(
                img, rectangles, save_path=str(viz_path), show=False
            )
            print(f"  Visualization: "
                  f"{viz_path.relative_to(project_root)}")

        except Exception as e:
            print(f"FAILED")
            print(f"  ERROR: {e}")
            logger.log_error(img_path.name, str(e))

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Results saved to: {logger.results_csv}")
    print(f"Rectangles saved to: {rect_dir / 'largest_rect'}")
    print(f"Visualizations saved to: {viz_dir / 'largest_rect'}")


def main():
    """Main entry point - configure experiments here."""
    run_experiments(
        image_dir_name="leafs_binary_fix",
        max_images=None,
        run_id="run1",
        coverage=0.90,
    )

    # Different coverage - saves separately as run2:
    # run_experiments(
    #     image_dir_name="objects_binary",
    #     max_images=1,
    #     run_id="run2",
    #     coverage=0.95,
    # )

    # PRODUCTION MODE: Uncomment to run on all images
    # run_experiments(
    #     image_dir_name="research_leafs_binary",
    #     max_images=None,
    #     run_id="run1",
    #     coverage=1.0,
    # )


if __name__ == "__main__":
    main()
