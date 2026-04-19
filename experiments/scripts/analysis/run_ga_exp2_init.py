"""EXP-2: Compare all GA initialization methods + convergence curves.

Runs GA with each initialization method across multiple seeds.
Results saved under ``exp2_init/<method>/`` run_ids for dashboard analysis.
Per-generation history (generations.csv) is logged automatically,
providing convergence curves with 3 seeds (replaces convergence_analysis).

Prerequisites:
  Run EXP-1 (run_ga_exp1_popsize) first to determine optimal pop_size.

Usage:
  python -m experiments.scripts.analysis.run_ga_exp2_init
"""

import random

from experiments.scripts.run_ga import run_experiments

# Initialization method -> algorithm name mapping
INIT_METHODS: dict[str, str] = {
    "dm": "ga_dm",
    "gdm": "ga_gdm",
    "random": "ga_random",
    "quadtree": "ga_qtd",
    "largest_rect": "ga_lrf",
}

# Fixed GA hyperparameters for fair comparison
POP_SIZE = 20
GENERATIONS = 150
PATIENCE = 6
CROSSOVER = "subset_greedy_relaxed"

# 3 seeds for statistical reliability
SEEDS = random.sample(range(10**8), 3)

# Datasets and image limits
DATASETS: list[tuple[str, int | None]] = [
    ("analysis/objects_unique", 10),
    ("analysis/leafs_subset", 10),
]

# Shared default mutation probabilities
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
    """Run all init method comparisons."""
    total_runs = len(INIT_METHODS) * len(SEEDS) * len(DATASETS)
    run_idx = 0
    print(f"Seeds: {SEEDS}")

    for dataset_name, max_images in DATASETS:
        for method_name, algo in INIT_METHODS.items():
            for seed in SEEDS:
                run_idx += 1
                run_id = f"exp2_init/{method_name}"
                print(
                    f"\n{'='*60}\n"
                    f"RUN {run_idx}/{total_runs}: "
                    f"init={method_name}, seed={seed}, "
                    f"dataset={dataset_name}\n"
                    f"{'='*60}"
                )
                run_experiments(
                    image_dir_name=dataset_name,
                    seed=seed,
                    pop_size=POP_SIZE,
                    generations=GENERATIONS,
                    patience=PATIENCE,
                    algorithm=algo,
                    crossover_method=CROSSOVER,
                    max_images=max_images,
                    run_id=run_id,
                    **MUTATION_DEFAULTS,
                )


if __name__ == "__main__":
    main()
