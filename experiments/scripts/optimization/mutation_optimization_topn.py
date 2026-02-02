"""
Three-phase mutation probability optimization with TOP-N approach.

Phase 1: Find TOP N best mutation probabilities (1 image, 1 seed)
Phase 2: Validate TOP N combinations on multiple seeds (10 seeds each)
Phase 3: Compare best validated vs no mutations (baseline proof)

Total estimated time: ~5-6 hours for leafs_binary images
"""

import time
from pathlib import Path
from itertools import product
import csv
import random

from experiments.src.config import ExperimentConfig
from experiments.src.runner import run_single_experiment


def phase1_find_top_probabilities(
    image_path,
    base_seed=42,
    prob_values=None,
    top_n=5
):
    """Phase 1: Find TOP N best mutation probability combinations.

    Args:
        image_path: Path to the test image
        base_seed: Seed for GA runs
        prob_values: List of probability values to test
        top_n: Number of top combinations to keep (default: 5)

    Returns:
        tuple: (top_combinations_list, all_results_list)
    """
    if prob_values is None:
        prob_values = [0.05, 0.1, 0.2]

    print("\n" + "=" * 70)
    print(f"PHASE 1: Finding TOP {top_n} Mutation Probability Combinations")
    print("=" * 70)
    print(f"Image: {Path(image_path).name}")
    print(f"Seed: {base_seed}")
    print(f"Testing probabilities: {prob_values}")

    # Generate all combinations (all mutations ON)
    combinations = list(product(prob_values, repeat=3))
    print(f"Total combinations: {len(combinations)}")

    results = []
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
            solution, metrics, _ = run_single_experiment(
                str(image_path),
                config
            )
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

        except Exception as e:
            print(f"✗ ERROR: {e}")
            continue

    elapsed = time.time() - start_time

    # Sort by rectangles (ascending) and get top N
    results_sorted = sorted(results, key=lambda x: x['rectangles'])
    top_combinations = results_sorted[:top_n]

    print(f"\nPhase 1 completed in {elapsed/60:.1f} minutes")
    print(f"\nTOP {top_n} combinations:")
    for i, combo in enumerate(top_combinations, 1):
        print(
            f"  {i}. G={combo['p_geometry']:.2f}, "
            f"M={combo['p_merge']:.2f}, "
            f"L={combo['p_local']:.2f} "
            f"→ {combo['rectangles']} rects"
        )

    return top_combinations, results


def phase2_validate_top_combinations(
    image_path,
    top_combinations,
    num_seeds=10
):
    """Phase 2: Validate TOP N combinations on multiple seeds.

    Args:
        image_path: Path to the test image
        top_combinations: List of top combinations from Phase 1
        num_seeds: Number of different seeds to test

    Returns:
        dict: Results for each combination
    """
    print("\n" + "=" * 70)
    print(f"PHASE 2: Validating TOP {len(top_combinations)} "
          f"Combinations on {num_seeds} Seeds")
    print("=" * 70)
    print(f"Image: {Path(image_path).name}")

    # Generate random seeds
    random.seed(42)
    seeds = [random.randint(1, 1000000) for _ in range(num_seeds)]

    all_results = {}
    start_time = time.time()

    for combo_idx, combo in enumerate(top_combinations, 1):
        p_geo = combo['p_geometry']
        p_merge = combo['p_merge']
        p_local = combo['p_local']
        combo_key = f"G={p_geo:.2f}_M={p_merge:.2f}_L={p_local:.2f}"

        print(
            f"\n--- Combination {combo_idx}/{len(top_combinations)}: "
            f"G={p_geo:.2f}, M={p_merge:.2f}, L={p_local:.2f} ---"
        )

        results = []

        for seed_idx, seed in enumerate(seeds, 1):
            print(
                f"  [{seed_idx:2d}/{num_seeds}] Seed {seed}...",
                end=" ",
                flush=True
            )

            config = ExperimentConfig(
                name=f"phase2_combo{combo_idx}_seed{seed_idx}",
                seed=seed,
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
                solution, metrics, _ = run_single_experiment(
                    str(image_path),
                    config
                )
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

        # Calculate statistics
        rect_counts = [r['rectangles'] for r in results]
        avg_rects = sum(rect_counts) / len(rect_counts)
        min_rects = min(rect_counts)
        max_rects = max(rect_counts)

        print(
            f"  Stats: Avg={avg_rects:.1f}, "
            f"Best={min_rects}, Worst={max_rects}"
        )

        all_results[combo_key] = {
            'combination': {
                'p_geometry': p_geo,
                'p_merge': p_merge,
                'p_local': p_local
            },
            'results': results,
            'avg': avg_rects,
            'min': min_rects,
            'max': max_rects
        }

    elapsed = time.time() - start_time

    # Find best combination based on average
    best_combo_key = min(
        all_results.keys(),
        key=lambda k: all_results[k]['avg']
    )
    best_combo = all_results[best_combo_key]

    print(f"\nPhase 2 completed in {elapsed/60:.1f} minutes")
    print("\n" + "─" * 70)
    print("VALIDATION SUMMARY (sorted by average):")
    print("─" * 70)

    sorted_combos = sorted(
        all_results.items(),
        key=lambda x: x[1]['avg']
    )

    for rank, (key, data) in enumerate(sorted_combos, 1):
        c = data['combination']
        print(
            f"{rank}. G={c['p_geometry']:.2f}, "
            f"M={c['p_merge']:.2f}, "
            f"L={c['p_local']:.2f} "
            f"→ Avg: {data['avg']:.1f}, "
            f"Best: {data['min']}, "
            f"Worst: {data['max']}"
        )

    print(f"\n BEST validated combination: {best_combo_key}")
    print(f"   Average: {best_combo['avg']:.1f} rectangles")

    return all_results, best_combo


def phase3_compare_baseline(
    image_path,
    best_combo,
    num_seeds=10
):
    """Phase 3: Compare best validated combo vs no mutations baseline.

    Args:
        image_path: Path to the test image
        best_combo: Best validated combination dict from Phase 2
        num_seeds: Number of seeds to test for best combo
                   (baseline only runs once - deterministic)

    Returns:
        dict: Results for both configurations
    """
    combo = best_combo['combination']

    print("\n" + "=" * 70)
    print("PHASE 3: Comparing Best Validated vs No Mutations (Baseline)")
    print("=" * 70)
    print(f"Image: {Path(image_path).name}")
    print(
        f"Best combo: G={combo['p_geometry']:.2f}, "
        f"M={combo['p_merge']:.2f}, "
        f"L={combo['p_local']:.2f}"
    )
    print(f"Baseline: 1 run (deterministic - RLE init, no mutations)")
    print(f"Best combo: {num_seeds} seeds")

    # Generate random seeds for best combo
    random.seed(42)
    seeds = [random.randint(1, 1000000) for _ in range(num_seeds)]

    results = {
        'baseline': [],
        'best': []
    }

    start_time = time.time()

    # Test baseline (no mutations) - ONLY ONCE
    print("\n--- Testing BASELINE (no mutations) ---")
    print("Running baseline (deterministic)...", end=" ", flush=True)

    config = ExperimentConfig(
        name="phase3_baseline",
        seed=42,  # Seed doesn't matter for baseline
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
        solution, metrics, _ = run_single_experiment(
            str(image_path),
            config
        )
        run_time = time.time() - run_start

        results['baseline'].append({
            'seed': 42,
            'rectangles': metrics['rectangles'],
            'execution_time': run_time,
            'fitness': solution.fitness if hasattr(
                solution, 'fitness'
            ) else None
        })

        print(f"✓ {metrics['rectangles']} rects in {run_time:.1f}s")

    except Exception as e:
        print(f"✗ ERROR: {e}")

    # Test best combination
    print(
        f"\n--- Testing BEST "
        f"(G={combo['p_geometry']:.2f}, "
        f"M={combo['p_merge']:.2f}, "
        f"L={combo['p_local']:.2f}) ---"
    )
    for idx, seed in enumerate(seeds, 1):
        print(
            f"[{idx:2d}/{num_seeds}] Best seed {seed}...",
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
            p_geometry=combo['p_geometry'],
            p_merge=combo['p_merge'],
            p_local=combo['p_local']
        )

        try:
            run_start = time.time()
            solution, metrics, _ = run_single_experiment(
                str(image_path),
                config
            )
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
    print(
        f"\nBest validated combination "
        f"(G={combo['p_geometry']:.2f}, "
        f"M={combo['p_merge']:.2f}, "
        f"L={combo['p_local']:.2f}):"
    )
    print(f"  Average: {best_avg:.1f} rectangles")
    print(f"  Best: {min(best_rects)} rectangles")
    print(f"  Worst: {max(best_rects)} rectangles")
    print(f"\n IMPROVEMENT: {improvement:+.2f}%")

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

    # Phase 1: All tested combinations
    with open(
        output_dir / "phase1_all_combinations.csv",
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

    # Phase 2: Validation results for all top combinations
    with open(
        output_dir / "phase2_validation_all.csv",
        'w',
        newline='',
        encoding='utf-8'
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'combination',
                'p_geometry',
                'p_merge',
                'p_local',
                'seed',
                'rectangles',
                'execution_time',
                'fitness'
            ]
        )
        writer.writeheader()

        for combo_key, data in phase2_results.items():
            c = data['combination']
            for result in data['results']:
                writer.writerow({
                    'combination': combo_key,
                    'p_geometry': c['p_geometry'],
                    'p_merge': c['p_merge'],
                    'p_local': c['p_local'],
                    **result
                })

    # Phase 2: Summary statistics
    with open(
        output_dir / "phase2_validation_summary.csv",
        'w',
        newline='',
        encoding='utf-8'
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'combination',
                'p_geometry',
                'p_merge',
                'p_local',
                'avg_rectangles',
                'min_rectangles',
                'max_rectangles'
            ]
        )
        writer.writeheader()

        for combo_key, data in phase2_results.items():
            c = data['combination']
            writer.writerow({
                'combination': combo_key,
                'p_geometry': c['p_geometry'],
                'p_merge': c['p_merge'],
                'p_local': c['p_local'],
                'avg_rectangles': data['avg'],
                'min_rectangles': data['min'],
                'max_rectangles': data['max']
            })

    # Phase 3: Baseline comparison
    with open(
        output_dir / "phase3_baseline_comparison.csv",
        'w',
        newline='',
        encoding='utf-8'
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'configuration',
                'seed',
                'rectangles',
                'execution_time',
                'fitness'
            ]
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
    combo = best_combo['combination']
    with open(
        output_dir / "summary.txt",
        'w',
        encoding='utf-8'
    ) as f:
        f.write("MUTATION PROBABILITY OPTIMIZATION SUMMARY (TOP-N)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Best validated combination:\n")
        f.write(f"  p_geometry: {combo['p_geometry']:.2f}\n")
        f.write(f"  p_merge: {combo['p_merge']:.2f}\n")
        f.write(f"  p_local: {combo['p_local']:.2f}\n\n")

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
    image_name="image.npy",
    image_dir="leafs_binary",
    prob_values=None,
    top_n=5,
    base_seed=42,
):
    """Run all three phases of mutation optimization with TOP-N.

    Args:
        image_name: Name of the test image
        image_dir: Directory under data/datasets/
        prob_values: Probability values to test in Phase 1
        top_n: Number of top combinations to validate in Phase 2
        base_seed: Seed for phase1 runs
    """
    if prob_values is None:
        prob_values = [0.05, 0.1, 0.2]

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    image_path = (
        project_root / "data/datasets" / image_dir / "npy" / image_name
    )

    # Create output directory: results/optimization/image_name/
    image_basename = Path(image_name).stem  # Removes .npy extension
    output_dir = (
        project_root / "experiments/results/optimization" / image_basename
    )

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("=" * 70)
    print("MUTATION PROBABILITY OPTIMIZATION (TOP-N APPROACH)")
    print("=" * 70)
    print(f"Image: {image_name}")
    print(f"Probability values: {prob_values}")
    print(f"Total combinations in Phase 1: {len(prob_values) ** 3}")
    print(f"TOP N to validate: {top_n}")

    total_start = time.time()

    # Phase 1: Find TOP N best probabilities
    top_combinations, phase1_all_results = phase1_find_top_probabilities(
        str(image_path),
        base_seed=base_seed,
        prob_values=prob_values,
        top_n=top_n
    )

    # Phase 2: Validate TOP N on multiple seeds
    phase2_results, best_combo = phase2_validate_top_combinations(
        str(image_path),
        top_combinations,
        num_seeds=10
    )

    # Phase 3: Compare best validated with baseline
    phase3_results = phase3_compare_baseline(
        str(image_path),
        best_combo,
        num_seeds=10
    )

    total_elapsed = time.time() - total_start

    # Save results
    save_results(
        phase1_all_results,
        phase2_results,
        phase3_results,
        best_combo,
        output_dir
    )

    combo = best_combo['combination']
    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETED")
    print("=" * 70)
    print(
        f"Total time: {total_elapsed/60:.1f} minutes "
        f"({total_elapsed/3600:.2f} hours)"
    )
    print(f"\nBest mutation probabilities (validated on {10} seeds):")
    print(f"  p_geometry = {combo['p_geometry']:.2f}")
    print(f"  p_merge = {combo['p_merge']:.2f}")
    print(f"  p_local = {combo['p_local']:.2f}")
    print(f"  Average result: {best_combo['avg']:.1f} rectangles")


if __name__ == "__main__":
    # Configure experiment here:
    # - image_name: Name of .npy file to use
    # - image_dir: "leafs_binary" or "objects_binary"
    # - prob_values: List of probabilities to test
    # - top_n: Number of top combinations to validate (3-5 recommended)
    main(
        image_name="Chaenomeles_japonica_binary.npy",
        image_dir="leafs_binary",
        prob_values=[0.05, 0.1, 0.2, 0.3],
        top_n=5,
        base_seed=67578345
    )