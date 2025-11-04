"""
Grid search for optimal mutation probability hyperparameters.

This script performs a comprehensive grid search across all combinations
of mutation probabilities on 30 randomly selected leaf images.

Grid parameters:
- p_geometry: [0, 0.05, 0.1, 0.2, 0.3]
- p_merge: [0, 0.05, 0.1, 0.2, 0.3]
- p_local: [0, 0.05, 0.1, 0.2, 0.3]

Total combinations: 5 × 5 × 5 = 125
Total runs: 125 combinations × 30 images = 3,750
"""

import json
import time
from pathlib import Path
from itertools import product
import csv

from experiments.src.config import ExperimentConfig
from experiments.src.runner import run_single_experiment


def main(image_dir="leafs_binary", num_images=30, seed=42):
    """Run grid search experiment.

    Args:
        image_dir: Directory name under res/figures/ (default: leafs_binary)
        num_images: Number of random images to use (default: 30)
        seed: Random seed for image selection (default: 42)
    """
    print("=" * 70)
    print("GRID SEARCH: Mutation Probability Hyperparameter Tuning")
    print("=" * 70)

    # Get absolute paths relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    # Get all .npy files from specified directory
    npy_dir = project_root / "res/figures" / image_dir / "npy"

    if not npy_dir.exists():
        raise FileNotFoundError(f"Directory not found: {npy_dir}")

    all_images = sorted([f.name for f in npy_dir.glob("*.npy")])

    if len(all_images) == 0:
        raise ValueError(f"No .npy files found in {npy_dir}")

    # Select random subset
    import random
    random.seed(seed)
    selected_images = random.sample(
        all_images,
        min(num_images, len(all_images))
    )
    selected_images = sorted(selected_images)

    print(f"\nImage directory: {image_dir}")
    print(f"Total available: {len(all_images)} images")
    print(f"Selected {len(selected_images)} images for experiment")

    # Define grid
    prob_values = [0.0, 0.05, 0.1, 0.2]

    # Generate all combinations
    combinations = list(product(prob_values, repeat=3))
    print(f"\nGrid combinations: {len(combinations)}")
    print(f"Total runs: {len(combinations)} × {len(selected_images)} "
          f"= {len(combinations) * len(selected_images)}")

    # Prepare output CSV
    output_file = (
        project_root / "experiments/results/csv/grid_search_mutations.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # CSV header
    fieldnames = [
        'image_name',
        'p_geometry',
        'p_merge',
        'p_local',
        'rectangles',
        'execution_time',
        'generations_used',
        'fitness'
    ]

    # Fixed parameters for all experiments
    base_seed = 42
    algorithm = "ga_rle"
    pop_size = 20
    generations = 100
    patience = 5

    print(f"\nFixed parameters:")
    print(f"  Algorithm: {algorithm}")
    print(f"  Population size: {pop_size}")
    print(f"  Max generations: {generations}")
    print(f"  Patience: {patience}")
    print(f"  Base seed: {base_seed}")

    # Start experiment
    start_time = time.time()
    completed = 0
    total_runs = len(combinations) * len(selected_images)

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over all combinations
        for combo_idx, (p_geo, p_merge, p_local) in enumerate(
            combinations, 1
        ):
            print(f"\n{'─' * 70}")
            print(
                f"Combination {combo_idx}/{len(combinations)}: "
                f"G={p_geo:.2f}, M={p_merge:.2f}, L={p_local:.2f}"
            )
            print(f"{'─' * 70}")

            # Run on all images
            for img_idx, image_name in enumerate(selected_images, 1):
                image_path = npy_dir / image_name

                # Create config
                config = ExperimentConfig(
                    name=f"grid_{combo_idx}_{img_idx}",
                    seed=base_seed,
                    algorithm=algorithm,
                    pop_size=pop_size,
                    generations=generations,
                    patience=patience,
                    p_geometry=p_geo,
                    p_merge=p_merge,
                    p_local=p_local
                )

                try:
                    print(
                        f"  [{completed + 1:4d}/{total_runs}] "
                        f"Running {image_name}...",
                        end=" ",
                        flush=True
                    )
                    run_start = time.time()

                    solution, metrics, _ = run_single_experiment(
                        str(image_path),
                        config
                    )

                    run_time = time.time() - run_start

                    writer.writerow({
                        'image_name': image_name,
                        'p_geometry': p_geo,
                        'p_merge': p_merge,
                        'p_local': p_local,
                        'rectangles': metrics['rectangles'],
                        'execution_time': metrics['execution_time'],
                        'generations_used': metrics.get(
                            'generations_used', None
                        ),
                        'fitness': solution.fitness if hasattr(
                            solution, 'fitness'
                        ) else None
                    })

                    completed += 1
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    remaining = (total_runs - completed) * avg_time

                    print(
                        f"✓ {metrics['rectangles']} rects in {run_time:.1f}s "
                        f"(ETA: {remaining/60:.1f}m)"
                    )

                except Exception as e:
                    print(f"✗ ERROR: {e}")
                    continue

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETED")
    print("=" * 70)
    print(f"Total runs: {completed}/{total_runs}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per run: {total_time/completed:.2f} seconds")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    # Change parameters here:
    # - image_dir: "leafs_binary" or "objects_binary"
    # - num_images: number of images to use
    # - seed: random seed for reproducibility
    main(image_dir="leafs_binary", num_images=10, seed=75674575764)