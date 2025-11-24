"""
Test grid search on a small subset to verify everything works.

Tests 4 combinations on 2 images = 8 runs (should take ~1-2 minutes).
"""

import json
import time
from pathlib import Path
import csv

from experiments.src.config import ExperimentConfig
from experiments.src.runner import run_single_experiment


def main(image_dir="leafs_binary", num_images=2, seed=42):
    """Run small test of grid search.

    Args:
        image_dir: Directory name under res/figures/ (default: leafs_binary)
        num_images: Number of random images to test (default: 2)
        seed: Random seed for image selection (default: 42)
    """
    print("=" * 70)
    print("TEST: Grid Search on Small Subset")
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
    print(f"Selected {len(selected_images)} images for testing:")
    for img in selected_images:
        print(f"  - {img}")

    # Define small grid (4 combinations)
    test_combinations = [
        (0.0, 0.0, 0.0),    # No mutations
        (0.05, 0.0, 0.0),   # Only geometry
        (0.05, 0.05, 0.0),  # Geometry + merge
        (0.05, 0.05, 0.05)  # All mutations
    ]

    print(f"\nTesting {len(test_combinations)} combinations:")
    for i, (g, m, l) in enumerate(test_combinations, 1):
        print(f"  {i}. G={g:.2f}, M={m:.2f}, L={l:.2f}")

    total_runs = len(test_combinations) * len(selected_images)
    print(f"\nTotal test runs: {total_runs}")

    # Prepare output CSV
    output_file = project_root / "experiments/results/csv/test_grid_search.csv"
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

    # Fixed parameters
    base_seed = 42
    algorithm = "ga_rle"
    pop_size = 20
    generations = 100
    patience = 5

    print(f"\nFixed parameters:")
    print(f"  Algorithm: {algorithm}")
    print(f"  Pop size: {pop_size}, Gens: {generations}, "
          f"Patience: {patience}")
    print(f"  Seed: {base_seed}\n")

    # Start experiment
    start_time = time.time()
    completed = 0

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for combo_idx, (p_geo, p_merge, p_local) in enumerate(
            test_combinations, 1
        ):
            print(f"\nCombo {combo_idx}/{len(test_combinations)}: "
                  f"G={p_geo:.2f}, M={p_merge:.2f}, L={p_local:.2f}")
            print("-" * 70)

            for img_idx, image_name in enumerate(selected_images, 1):
                image_path = npy_dir / image_name

                config = ExperimentConfig(
                    name=f"test_{combo_idx}_{img_idx}",
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
                    print(f"  Running {image_name}...", end=" ")
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
                    print(
                        f"✓ {metrics['rectangles']} rects in "
                        f"{run_time:.1f}s"
                    )

                except Exception as e:
                    print(f"✗ ERROR: {e}")
                    continue

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)
    print(f"Runs completed: {completed}/{total_runs}")
    print(f"Total time: {total_time:.1f} seconds "
          f"({total_time/60:.2f} minutes)")
    print(f"Average per run: {total_time/completed:.1f} seconds")
    print(f"\nResults saved to: {output_file}")
    print(f"\nIf test successful, run full grid search with:")
    print(f"  python experiments/scripts/grid_search_mutations.py")


if __name__ == "__main__":
    # Change parameters here:
    # - image_dir: "leafs_binary" or "objects_binary"
    # - num_images: number of images to test
    # - seed: random seed for reproducibility
    main(image_dir="objects_binary", num_images=10, seed=4534546454453)