"""
Run genetic algorithm experiments on binary images.

This is the main production script for running experiments on entire
image directories. Results, rectangles, and visualizations are saved
for analysis and validation.
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
    dataset_name: str
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
        Name of the dataset directory (e.g., "objects_binary").

    Returns
    -------
    Path
        Path to saved JSON file.

    """
    # Create directory structure:
    # rectangles/algorithm_mutationcombo/dataset/image_name/
    image_stem = Path(image_name).stem
    mutation_combo = config.get_mutation_combo_code()
    algo_name = f"{config.algorithm}_{mutation_combo}"

    save_dir = output_dir / algo_name / dataset_name / image_stem
    save_dir.mkdir(parents=True, exist_ok=True)

    # Filename: seed_XXXXX_rects_NN.json
    filename = f"seed_{config.seed}_rects_{rect_count}.json"
    save_path = save_dir / filename

    # Prepare data structure - rectangles as list of tuples
    # Format: [[x, y, w, h], [x, y, w, h], ...]
    data = {
        "image_name": image_name,
        "rectangle_count": rect_count,
        "rectangles": [
            [int(x), int(y), int(w), int(h)]
            for x, y, w, h in rectangles
        ],
        "config": {
            "algorithm": config.algorithm,
            "seed": config.seed,
            "pop_size": config.pop_size,
            "generations": config.generations,
            "patience": config.patience,
            "p_geometry": config.p_geometry,
            "p_merge": config.p_merge,
            "p_local": config.p_local,
            "mutation_combo": config.get_mutation_combo_code()
        },
        "metrics": metrics
    }

    # Save to JSON
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

    # Convert lists back to tuples
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
    p_geometry: float = 0.2,
    p_merge: float = 0.2,
    p_local: float = 0.3,
    seed: int = None,
    pop_size: int = 20,
    generations: int = 100,
    patience: int = 5,
    algorithm: str = "ga_rle",
    max_images: int = None
):
    """Run experiments on images from specified directory.

    Parameters
    ----------
    image_dir_name : str
        Directory name under data/datasets/ (e.g., "leafs_binary",
        "research_leafs_binary", "objects_binary").
    p_geometry : float, optional
        Geometry mutation probability (default: 0.2).
    p_merge : float, optional
        Merge mutation probability (default: 0.2).
    p_local : float, optional
        Local repartition mutation probability (default: 0.3).
    seed : int, optional
        Random seed for reproducibility (default: 42).
    pop_size : int, optional
        Population size for GA (default: 20).
    generations : int, optional
        Maximum generations for GA (default: 100).
    patience : int, optional
        Early stopping patience (default: 5).
    algorithm : str, optional
        Algorithm variant: "ga_rle", "ga_random", "ga_quadtree"
        (default: "ga_rle").
    max_images : int, optional
        Maximum number of images to process. If None, processes all
        images in directory (default: None).

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

    # Get image paths
    image_paths = sorted(image_dir.glob("*.npy"))

    if not image_paths:
        raise FileNotFoundError(
            f"No .npy files found in {image_dir}"
        )

    # Limit to max_images if specified
    if max_images is not None:
        image_paths = image_paths[:max_images]

    mode_str = f"TEST MODE ({len(image_paths)} images)" if max_images else "PRODUCTION MODE (all images)"

    print("=" * 70)
    print(f"GENETIC ALGORITHM EXPERIMENTS - {mode_str}")
    print("=" * 70)
    print(f"Directory: {image_dir_name}")
    print(f"Images to process: {len(image_paths)}")
    print(f"Algorithm: {algorithm}")
    print(f"Mutation probabilities: G={p_geometry}, M={p_merge}, L={p_local}")
    print(f"Population: {pop_size}, Generations: {generations}, "
          f"Patience: {patience}")
    print("=" * 70)

    # Create config first to get mutation combo
    config = ExperimentConfig(
        name=f"{algorithm}_{image_dir_name}",
        seed=seed,
        algorithm=algorithm,
        pop_size=pop_size,
        generations=generations,
        patience=patience,
        p_geometry=p_geometry,
        p_merge=p_merge,
        p_local=p_local,
    )

    mutation_combo = config.get_mutation_combo_code()

    # Setup logger with hierarchical structure
    logger = CSVLogger(
        algorithm,
        str(project_root / "experiments/results/csv/"),
        image_dir_name,
        mutation_combo
    )

    print(f"Mutation combo: {mutation_combo}")
    print(f"Seed: {seed}")
    print("-" * 70)

    # Create output directories
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

            # Extract rectangles
            rectangles = (
                solution.rectangles
                if hasattr(solution, 'rectangles')
                else solution
            )
            rect_count = len(rectangles)

            # Log result
            logger.log_result(
                img_path.name, config, metrics, history
            )

            # Print summary
            print(f"  ✓ Rectangles: {rect_count}")
            print(f"  ✓ Fitness: {metrics.get('final_fitness', 'N/A'):.4f}")
            print(f"  ✓ Time: {metrics['execution_time_sec']:.1f}s")
            print(f"  ✓ Generations: {metrics['generations_used']}")

            # Save rectangles to JSON
            rect_path = save_solution_rectangles(
                img_path.name,
                rectangles,
                rect_count,
                rect_dir,
                config,
                metrics,
                image_dir_name
            )
            print(f"  ✓ Rectangles saved: {rect_path.relative_to(project_root)}")

            # Save visualization with hierarchical structure:
            # visualizations/algorithm_mutationcombo/dataset/image_seed_XXXXX_rects_NN.png
            img = np.load(img_path)
            img = (img > 0).astype(int)

            mutation_combo = config.get_mutation_combo_code()
            algo_name = f"{config.algorithm}_{mutation_combo}"
            viz_subdir = viz_dir / algo_name / image_dir_name
            viz_subdir.mkdir(parents=True, exist_ok=True)

            viz_filename = f"{img_path.stem}_seed_{config.seed}_rects_{rect_count}.png"
            viz_path = viz_subdir / viz_filename

            draw_solution(
                img, rectangles, save_path=str(viz_path), show=False
            )
            print(f"  ✓ Visualization: {viz_path.relative_to(project_root)}")

        except Exception as e:
            print(f"FAILED")
            print(f"  ✗ ERROR: {e}")
            logger.log_error(img_path.name, str(e))

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Results saved to: {logger.results_csv}")
    print(f"Generations saved to: {logger.generations_csv}")
    print(f"Rectangles saved to: {rect_dir}")
    print(f"Visualizations saved to: {viz_dir}")


def main():
    """Main entry point - configure experiments here."""
    # TEST MODE: Quick test on 5 images
    run_experiments(
        image_dir_name="research_leafs_binary",
        max_images=1,  # Limit to 5 images for testing
        p_geometry=0.2,
        p_merge=0.2,
        p_local=0.3,
        seed=None,
        pop_size=20,
        generations=100,
        patience=5,
        algorithm="ga_rle",
    )
    # Algorithms: "ga_rle", "ga_random", "ga_quadtree"

    # PRODUCTION MODE: Uncomment to run on all images
    # run_experiments(
    #     image_dir_name="research_leafs_binary",
    #     p_geometry=0.2,
    #     p_merge=0.2,
    #     p_local=0.3,
    #     seed=45646456,
    #     pop_size=20,
    #     generations=100,
    #     patience=5,
    #     algorithm="ga_rle",
    #     max_images=None  # Process ALL images
    # )


if __name__ == "__main__":
    main()