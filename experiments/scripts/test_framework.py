"""
Test the experiment framework on 5 images.

This script tests the framework implementation before running full
experiments.
"""

import time
from pathlib import Path

import numpy as np

from experiments.src.config import ExperimentConfig
from experiments.src.logger import CSVLogger
from experiments.src.runner import run_single_experiment
from src.utils.utils import draw_solution


def main():
    """Test framework on 5 small images."""
    # Get first 5 small images
    image_dir = Path("../../res/figures/synthetic/npy/")
    image_paths = sorted(image_dir.glob("small_*.npy"))[:5]

    print(f"Testing framework on {len(image_paths)} images...")
    print("=" * 60)

    # Test one algorithm with one mutation combo
    algorithm = "ga_rle"
    logger = CSVLogger(algorithm, "../../experiments/results/csv/")

    # Test GML mutation combo
    config = ExperimentConfig(
        name="test_ga_rle_GML",
        seed=42,
        algorithm="ga_rle",
        pop_size=20,  # Smaller for testing
        generations=100,  # Fewer generations for testing
        patience=5,
        mutation_geometry=True,
        mutation_merge=True,
        mutation_local=True,
    )

    print(f"\nAlgorithm: {algorithm}")
    print(f"Mutation combo: {config.get_mutation_combo_code()}")
    print(f"Config: pop_size={config.pop_size}, "
          f"generations={config.generations}")
    print("-" * 60)

    # Create visualization directory
    viz_dir = Path("../../experiments/results/visualizations/")
    viz_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing: {img_path.name}")
        print(f"  Running... ", end="", flush=True)

        start_time = time.time()

        try:
            solution, metrics, history = run_single_experiment(
                str(img_path), config
            )

            elapsed = time.time() - start_time
            print(f"Done in {elapsed:.1f}s")

            # Log result
            logger.log_result(
                img_path.name, config, metrics, history
            )

            # Print summary
            print(f"  ✓ Rectangles: {metrics['rectangle_count']}")
            print(f"  ✓ Fitness: {metrics.get('final_fitness', 'N/A')}")
            print(f"  ✓ Time: {metrics['execution_time_sec']}s")
            print(f"  ✓ Generations: {metrics['generations_used']}")

            # Save visualization
            img = np.load(img_path)
            img = (img > 0).astype(int)
            rectangles = (
                solution.rectangles
                if hasattr(solution, 'rectangles')
                else solution
            )
            viz_path = viz_dir / f"{img_path.stem}_solution.png"
            print(f"  Saving: {viz_path.name}")
            draw_solution(
                img, rectangles, save_path=str(viz_path), show=False
            )

        except Exception as e:
            print(f"FAILED")
            print(f"  ✗ ERROR: {e}")
            logger.log_error(img_path.name, str(e))

    print("\n" + "=" * 60)
    print("Test complete!")
    print(f"Results saved to: {logger.results_csv}")
    print(f"Generations saved to: {logger.generations_csv}")


if __name__ == "__main__":
    main()