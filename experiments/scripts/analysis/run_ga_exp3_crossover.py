"""EXP-3: Compare all GA crossover methods.
...
"""

import random
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiments.scripts.run_ga import run_experiments

BEST_INIT_ALGO = "ga_dm"

CROSSOVER_METHODS: list[str] = [
    "subset_greedy",
    "subset_greedy_relaxed",
    "single_point",
    "two_point",
    "uniform",
]

POP_SIZE = 15
GENERATIONS = 150
PATIENCE = 12

N_SEEDS = 5
SEEDS = random.sample(range(10**8), N_SEEDS)
# Zafixujeme prvý seed pre uniformné kríženie, aby sme ho vedeli identifikovať
FIRST_SEED = SEEDS[0]

DATASETS: list[tuple[str, int | None]] = [
    ("analysis/objects_quartile", None),
    ("analysis/leafs_quartile", None),
]

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
    """Run all crossover methods and datasets for one seed (worker entrypoint)."""
    for dataset_name, max_images in DATASETS:
        for crossover in CROSSOVER_METHODS:

            # TEMPORÁLNA ÚPRAVA: Uniformné kríženie beží iba pre prvý seed
            if crossover == "uniform" and seed != FIRST_SEED:
                # print(f"[seed={seed}] SKIPPING crossover={crossover} (too slow, only 1 seed allowed)")
                continue

            print(f"[seed={seed}] crossover={crossover} | {dataset_name}")
            run_experiments(
                image_dir_name=dataset_name,
                seed=seed,
                pop_size=POP_SIZE,
                generations=GENERATIONS,
                patience=PATIENCE,
                algorithm=BEST_INIT_ALGO,
                crossover_method=crossover,
                max_images=max_images,
                run_id=f"exp3_crossover/{crossover}",
                **MUTATION_DEFAULTS,
            )


def main() -> None:
    """Run all crossover method comparisons — one process per seed."""
    print(f"Seeds: {SEEDS}")
    print(f"Workers: {N_SEEDS}")
    print(f"Note: 'uniform' crossover will only run for seed {FIRST_SEED} (temporal restriction)")

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