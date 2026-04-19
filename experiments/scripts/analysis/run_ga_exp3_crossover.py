"""EXP-3: Compare all GA crossover methods.

Runs GA with each of the 5 crossover variants across multiple seeds.
Results saved under ``exp3_crossover/<method>/`` run_ids.

Prerequisites:
  Run EXP-2 (run_ga_exp2_init) first to determine the best init method.
  Update BEST_INIT_ALGO below after analyzing EXP-2 results.

Usage:
  python -m experiments.scripts.analysis.run_ga_exp3_crossover
"""

import random

from experiments.scripts.run_ga import run_experiments

# Best init method from EXP-2 (update after EXP-2 results are analyzed)
BEST_INIT_ALGO = "ga_dm"

CROSSOVER_METHODS: list[str] = [
    "subset_greedy",
    "subset_greedy_relaxed",
    "single_point",
    "two_point",
    "uniform",
]

POP_SIZE = 20
GENERATIONS = 100
PATIENCE = 8

SEEDS = random.sample(range(10**8), 2)

DATASETS: list[tuple[str, int | None]] = [
    ("analysis/objects_unique", 10),
    ("analysis/leafs_subset", 10),
]

MUTATION_DEFAULTS = dict(
    p_delete=0.2,
    p_split=0.2,
    p_geometry=0.3,
    p_shift=0.05,
    p_local=0.5,
    p_largest=0.2,
    p_repair=0.5,
    p_merge=0.1,
)


def main() -> None:
    """Run all crossover method comparisons."""
    total_runs = len(CROSSOVER_METHODS) * len(SEEDS) * len(DATASETS)
    run_idx = 0
    print(f"Seeds: {SEEDS}")

    for dataset_name, max_images in DATASETS:
        for crossover in CROSSOVER_METHODS:
            for seed in SEEDS:
                run_idx += 1
                run_id = f"exp3_crossover/{crossover}"
                print(
                    f"\n{'='*60}\n"
                    f"RUN {run_idx}/{total_runs}: "
                    f"crossover={crossover}, seed={seed}, "
                    f"dataset={dataset_name}\n"
                    f"{'='*60}"
                )
                run_experiments(
                    image_dir_name=dataset_name,
                    seed=seed,
                    pop_size=POP_SIZE,
                    generations=GENERATIONS,
                    patience=PATIENCE,
                    algorithm=BEST_INIT_ALGO,
                    crossover_method=crossover,
                    max_images=max_images,
                    run_id=run_id,
                    **MUTATION_DEFAULTS,
                )


if __name__ == "__main__":
    main()
