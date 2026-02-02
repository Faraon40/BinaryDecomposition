"""
Three-phase mutation probability optimization.

Phase 1: Find best mutation probabilities (1 image, 1 seed, all mutations ON)
Phase 2: Validate best combination on multiple seeds (1 image, 10 seeds)
Phase 3: Compare best vs no mutations (baseline proof, 10 seeds each)

Total estimated time: ~4-5 hours for leafs_binary images
"""

import time
from pathlib import Path
from itertools import product
import csv
import random

from experiments.src.config import ExperimentConfig
from experiments.src.runner import run_single_experiment


def phase1_find_best_probabilities(
    image_path,
    base_seed=42,
    prob_values=None
):
    """Phase 1: Find best mutation probability combination.

    Args:
        image_path: Path to the test image
        base_seed: Seed for GA runs
        prob_values: List of probability values to test

    Returns:
        tuple: (best_combination, results_list)
    """
    if prob_values is None:
        prob_values = [0.05, 0.1, 0.2]

    print("\n" + "=" * 70)
    print("PHASE 1: Finding Best Mutation Probabilities")
    print("=" * 70)
    print(f"Image: {Path(image_path).name}")
    print(f"Seed: {base_seed}")
    print(f"Testing probabilities: {prob_values}")

    # Generate all combinations (all mutations ON)
    combinations = list(product(prob_values, repeat=3))
    print(f"Total combinations: {len(combinations)}")

    results = []
    best_combo = None
    best_rects = float('inf')

    start_time = time.time()

    for idx, (p_geo, p_merge, p_local) in enumerate(combinations, 1):
        print(
            f"\n[{idx:2d}/{len(combinations)}] "
            f"Testing G={p_geo:.2f}, M={p_merge:.2f}, L={p_local:.2f}...",
            end=" ",
            flush=True
        )

        config = ExperimentConfig(
            name=f"phase1_{idx}",
            seed=base_seed,
            algorithm="ga_rle",
            pop_size=20,
            generations=100,
            patience=5,
            p_geometry=p_geo,
            p_merge=p_merge,
            p_local=p_local
        )

        try:
            run_start = time.time()
            solution, metrics, _ = run_single_experiment(image_path, config)
            run_time = time.time() - run_start

            rect_count = metrics['rectangles']
            results.append({
                'p_geometry': p_geo,
                'p_merge': p_merge,
                'p_local': p_local,
                'rectangles': rect_count,
                'execution_time': run_time,
                'fitness': solution.fitness if hasattr(
                    solution, 'fitness'
                ) else None
            })

            print(f"✓ {rect_count} rects in {run_time:.1f}s")

            if rect_count < best_rects:
                best_rects = rect_count
                best_combo = (p_geo, p_merge, p_local)
                print(f"    NEW BEST!")

        except Exception as e:
            print(f"✗ ERROR: {e}")
            continue

    elapsed = time.time() - start_time
    print(f"\nPhase 1 completed in {elapsed/60:.1f} minutes")
    print(f"Best combination: G={best_combo[0]:.2f}, "
          f"M={best_combo[1]:.2f}, L={best_combo[2]:.2f}")
    print(f"Best result: {best_rects} rectangles")

    return best_combo, results


def phase2_validate_on_seeds(image_path, best_combo, num_seeds=10):
    """Phase 2: Validate best combination on multiple seeds.

    Args:
        image_path: Path to the test image
        best_combo: Best (p_geo, p_merge, p_local) from Phase 1
        num_seeds: Number of different seeds to test

    Returns:
        list: Results for each seed
    """
    print("\n" + "=" * 70)
    print("PHASE 2: Validating Best Combination on Multiple Seeds")
    print("=" * 70)
    print(f"Image: {Path(image_path).name}")
    print(f"Best combination: G={best_combo[0]:.2f}, "
          f"M={best_combo[1]:.2f}, L={best_combo[2]:.2f}")
    print(f"Testing on {num_seeds} different seeds")

    # Generate random seeds
    random.seed(42)
    seeds = [random.randint(1, 1000000) for _ in range(num_seeds)]

    results = []
    start_time = time.time()

    for idx, seed in enumerate(seeds, 1):
        print(
            f"[{idx:2d}/{num_seeds}] "
            f"Testing with seed {seed}...",
            end=" ",
            flush=True
        )

        config = ExperimentConfig(
            name=f"phase2_{idx}",
            seed=seed,
            algorithm="ga_rle",
            pop_size=20,
            generations=100,
            patience=5,
            p_geometry=best_combo[0],
            p_merge=best_combo[1],
            p_local=best_combo[2]
        )

        try:
            run_start = time.time()
            solution, metrics, _ = run_single_experiment(image_path, config)
            run_time = time.time() - run_start

            results.append({
                'seed': seed,
                'rectangles': metrics['rectangles'],
                'execution_time': run_time,
                'fitness': solution.fitness if hasattr(
                    solution, 'fitness'
                ) else None
            })

            print(f"✓ {metrics['rectangles']} rects in {run_time:.1f}s")

        except Exception as e:
            print(f"✗ ERROR: {e}")
            continue

    elapsed = time.time() - start_time

    # Calculate statistics
    rect_counts = [r['rectangles'] for r in results]
    avg_rects = sum(rect_counts) / len(rect_counts)
    min_rects = min(rect_counts)
    max_rects = max(rect_counts)

    print(f"\nPhase 2 completed in {elapsed/60:.1f} minutes")
    print(f"Results across {len(results)} seeds:")
    print(f"  Average: {avg_rects:.1f} rectangles")
    print(f"  Best: {min_rects} rectangles")
    print(f"  Worst: {max_rects} rectangles")

    return results


def phase3_compare_baseline(
    image_path,
    best_combo,
    num_seeds=10
):
    """Phase 3: Compare best combo vs no mutations baseline.

    Args:
        image_path: Path to the test image
        best_combo: Best (p_geo, p_merge, p_local) from Phase 1
        num_seeds: Number of seeds to test each configuration

    Returns:
        dict: Results for both configurations
    """
    print("\n" + "=" * 70)
    print("PHASE 3: Comparing Best vs No Mutations (Baseline)")
    print("=" * 70)
    print(f"Image: {Path(image_path).name}")
    print(f"Testing {num_seeds} seeds for each configuration")

    # Generate random seeds
    random.seed(42)
    seeds = [random.randint(1, 1000000) for _ in range(num_seeds)]

    results = {
        'baseline': [],
        'best': []
    }

    start_time = time.time()

    # Test baseline (no mutations)
    print("\n--- Testing BASELINE (no mutations) ---")
    for idx, seed in enumerate(seeds, 1):
        print(
            f"[{idx:2d}/{num_seeds}] "
            f"Baseline seed {seed}...",
            end=" ",
            flush=True
        )

        config = ExperimentConfig(
            name=f"phase3_baseline_{idx}",
            seed=seed,
            algorithm="ga_rle",
            pop_size=20,
            generations=100,
            patience=5,
            p_geometry=0.0,
            p_merge=0.0,
            p_local=0.0
        )

        try:
            run_start = time.time()
            solution, metrics, _ = run_single_experiment(image_path, config)
            run_time = time.time() - run_start

            results['baseline'].append({
                'seed': seed,
                'rectangles': metrics['rectangles'],
                'execution_time': run_time,
                'fitness': solution.fitness if hasattr(
                    solution, 'fitness'
                ) else None
            })

            print(f"✓ {metrics['rectangles']} rects in {run_time:.1f}s")

        except Exception as e:
            print(f"✗ ERROR: {e}")
            continue

    # Test best combination
    print(f"\n--- Testing BEST "
          f"(G={best_combo[0]:.2f}, M={best_combo[1]:.2f}, "
          f"L={best_combo[2]:.2f}) ---")
    for idx, seed in enumerate(seeds, 1):
        print(
            f"[{idx:2d}/{num_seeds}] "
            f"Best seed {seed}...",
            end=" ",
            flush=True
        )

        config = ExperimentConfig(
            name=f"phase3_best_{idx}",
            seed=seed,
            algorithm="ga_rle",
            pop_size=20,
            generations=100,
            patience=5,
            p_geometry=best_combo[0],
            p_merge=best_combo[1],
            p_local=best_combo[2]
        )

        try:
            run_start = time.time()
            solution, metrics, _ = run_single_experiment(image_path, config)
            run_time = time.time() - run_start

            results['best'].append({
                'seed': seed,
                'rectangles': metrics['rectangles'],
                'execution_time': run_time,
                'fitness': solution.fitness if hasattr(
                    solution, 'fitness'
                ) else None
            })

            print(f"✓ {metrics['rectangles']} rects in {run_time:.1f}s")

        except Exception as e:
            print(f"✗ ERROR: {e}")
            continue

    elapsed = time.time() - start_time

    # Calculate statistics
    baseline_rects = [r['rectangles'] for r in results['baseline']]
    best_rects = [r['rectangles'] for r in results['best']]

    baseline_avg = sum(baseline_rects) / len(baseline_rects)
    best_avg = sum(best_rects) / len(best_rects)
    improvement = ((baseline_avg - best_avg) / baseline_avg) * 100

    print(f"\nPhase 3 completed in {elapsed/60:.1f} minutes")
    print(f"\n{'─' * 70}")
    print("COMPARISON RESULTS:")
    print(f"{'─' * 70}")
    print(f"Baseline (no mutations):")
    print(f"  Average: {baseline_avg:.1f} rectangles")
    print(f"  Best: {min(baseline_rects)} rectangles")
    print(f"  Worst: {max(baseline_rects)} rectangles")
    print(f"\nBest combination "
          f"(G={best_combo[0]:.2f}, M={best_combo[1]:.2f}, "
          f"L={best_combo[2]:.2f}):")
    print(f"  Average: {best_avg:.1f} rectangles")
    print(f"  Best: {min(best_rects)} rectangles")
    print(f"  Worst: {max(best_rects)} rectangles")
    print(f"\n⭐ IMPROVEMENT: {improvement:+.2f}%")

    return results


def save_results(
    phase1_results,
    phase2_results,
    phase3_results,
    best_combo,
    output_dir
):
    """Save all results to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1 results
    with open(
        output_dir / "phase1_probability_search.csv",
        'w',
        newline='',
        encoding='utf-8'
    ) as f:
        if phase1_results:
            writer = csv.DictWriter(
                f,
                fieldnames=phase1_results[0].keys()
            )
            writer.writeheader()
            writer.writerows(phase1_results)

    # Phase 2 results
    with open(
        output_dir / "phase2_seed_validation.csv",
        'w',
        newline='',
        encoding='utf-8'
    ) as f:
        if phase2_results:
            writer = csv.DictWriter(
                f,
                fieldnames=phase2_results[0].keys()
            )
            writer.writeheader()
            writer.writerows(phase2_results)

    # Phase 3 results
    with open(
        output_dir / "phase3_baseline_comparison.csv",
        'w',
        newline='',
        encoding='utf-8'
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['configuration', 'seed', 'rectangles',
                        'execution_time', 'fitness']
        )
        writer.writeheader()

        for result in phase3_results['baseline']:
            writer.writerow({
                'configuration': 'baseline',
                **result
            })

        for result in phase3_results['best']:
            writer.writerow({
                'configuration': 'best',
                **result
            })

    # Summary
    with open(
        output_dir / "summary.txt",
        'w',
        encoding='utf-8'
    ) as f:
        f.write("MUTATION PROBABILITY OPTIMIZATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Best combination found:\n")
        f.write(f"  p_geometry: {best_combo[0]:.2f}\n")
        f.write(f"  p_merge: {best_combo[1]:.2f}\n")
        f.write(f"  p_local: {best_combo[2]:.2f}\n\n")

        if phase3_results['baseline'] and phase3_results['best']:
            baseline_rects = [
                r['rectangles'] for r in phase3_results['baseline']
            ]
            best_rects = [
                r['rectangles'] for r in phase3_results['best']
            ]
            baseline_avg = sum(baseline_rects) / len(baseline_rects)
            best_avg = sum(best_rects) / len(best_rects)
            improvement = ((baseline_avg - best_avg) / baseline_avg) * 100

            f.write(f"Baseline average: {baseline_avg:.1f} rectangles\n")
            f.write(f"Best average: {best_avg:.1f} rectangles\n")
            f.write(f"Improvement: {improvement:+.2f}%\n")

    print(f"\nResults saved to: {output_dir}")


def main(
    image_name="Acer_pseudoplatanus_binary.npy",
    image_dir="leafs_binary",
    prob_values=None
):
    """Run all three phases of mutation optimization.

    Args:
        image_name: Name of the test image
        image_dir: Directory under data/datasets/
        prob_values: Probability values to test in Phase 1
    """
    if prob_values is None:
        prob_values = [0.05, 0.1, 0.2]

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    image_path = (
        project_root / "data/datasets" / image_dir / "npy" / image_name
    )
    output_dir = project_root / "experiments/results/csv/optimization"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("=" * 70)
    print("MUTATION PROBABILITY OPTIMIZATION")
    print("=" * 70)
    print(f"Image: {image_name}")
    print(f"Probability values: {prob_values}")
    print(f"Total combinations in Phase 1: {len(prob_values) ** 3}")

    total_start = time.time()

    # Phase 1: Find best probabilities
    best_combo, phase1_results = phase1_find_best_probabilities(
        str(image_path),
        base_seed=42,
        prob_values=prob_values
    )

    # Phase 2: Validate on multiple seeds
    phase2_results = phase2_validate_on_seeds(
        str(image_path),
        best_combo,
        num_seeds=10
    )

    # Phase 3: Compare with baseline
    phase3_results = phase3_compare_baseline(
        str(image_path),
        best_combo,
        num_seeds=10
    )

    total_elapsed = time.time() - total_start

    # Save results
    save_results(
        phase1_results,
        phase2_results,
        phase3_results,
        best_combo,
        output_dir
    )

    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETED")
    print("=" * 70)
    print(f"Total time: {total_elapsed/60:.1f} minutes "
          f"({total_elapsed/3600:.2f} hours)")
    print(f"\nBest mutation probabilities:")
    print(f"  p_geometry = {best_combo[0]:.2f}")
    print(f"  p_merge = {best_combo[1]:.2f}")
    print(f"  p_local = {best_combo[2]:.2f}")


if __name__ == "__main__":
    # Configure experiment here:
    # - image_name: Name of .npy file to use
    # - image_dir: "leafs_binary" or "objects_binary"
    # - prob_values: List of probabilities to test
    main(
        image_name="Colutea_arborescens_binary.npy",
        image_dir="leafs_binary",
        prob_values=[0.05, 0.1, 0.15, 0.2]
    )