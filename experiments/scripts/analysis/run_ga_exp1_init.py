"""EXP-1: Compare all GA initialization methods + convergence curves.

Runs GA with each initialization method across multiple seeds.
Results saved under ``exp1_init/<method>/`` run_ids for dashboard analysis.
Per-generation history (generations.csv) is logged automatically,
providing convergence curves with 3 seeds (replaces convergence_analysis).

Usage:
  python -m experiments.scripts.analysis.run_ga_exp1_init
"""

import random
from concurrent.futures import ProcessPoolExecutor, as_completed

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
POP_SIZE = 15
GENERATIONS = 150
PATIENCE = 12
CROSSOVER = "subset_greedy_relaxed"

N_SEEDS = 5
SEEDS = random.sample(range(10**8), N_SEEDS)

# Datasets — quartile-stratified subsets (20 imgs each, 5 per size quartile)
DATASETS: list[tuple[str, int | None]] = [
    # ("analysis/objects_quartile", None),
    ("analysis/leafs_quartile", None),
]

# Shared default mutation probabilities
MUTATION_DEFAULTS = dict(
    p_delete=0.2,
    p_split=0.2,
    p_geometry=0.2,
    p_shift=0.1,
    p_local=0.5,
    p_largest=0.2,
    p_merge=0.2,
    p_repair=0.5,
)


def _run_seed(seed: int) -> None:
    """Run all methods and datasets for one seed (worker entrypoint)."""
    for dataset_name, max_images in DATASETS:
        for method_name, algo in INIT_METHODS.items():
            run_id = f"exp1_init_run2/{method_name}"
            print(f"[seed={seed}] {method_name} | {dataset_name}")
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


def main() -> None:
    """Run all init method comparisons — one process per seed."""
    print(f"Seeds: {SEEDS}")
    print(f"Workers: {N_SEEDS} (one per seed)")

    with ProcessPoolExecutor(max_workers=N_SEEDS) as executor:
        futures = {executor.submit(_run_seed, seed): seed for seed in SEEDS}
        for future in as_completed(futures):
            seed = futures[future]
            exc = future.exception()
            if exc:
                print(f"[seed={seed}] FAILED: {exc}")
            else:
                print(f"[seed={seed}] DONE")


if __name__ == "__main__":
    main()
