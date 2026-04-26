"""EXP-4: Vplyv p_local a p_merge na kvalitu GA.

Dve cielené otázky:
  A) Aký vplyv má p_local? (p_merge fixné = 0.20)
  B) Aký vplyv má p_merge? (p_local fixné = 0.50)

Seeds bežia paralelne (jeden proces na seed).

Prerequisites:
  Run EXP-1 first and update BEST_POP_SIZE below.

Usage:
  python -m experiments.scripts.analysis.run_ga_exp4_mutation
"""

import random
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiments.scripts.run_ga import run_experiments

ALGO = "ga_dm"
CROSSOVER = "subset_greedy_relaxed"
POP_SIZE = 15
GENERATIONS = 150
PATIENCE = 12


N_SEEDS = 5
SEEDS = random.sample(range(10**8), N_SEEDS)

DATASETS: list[tuple[str, int | None]] = [
    ("analysis/objects_quartile", None),
    ("analysis/leafs_quartile", None),
]

# A) p_local sweep — p_merge fixné
P_LOCAL_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FIXED_P_MERGE_FOR_LOCAL = 0.20

# B) p_merge sweep — p_local fixné
P_MERGE_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FIXED_P_LOCAL_FOR_MERGE = 0.50

BASE_MUTATIONS = dict(
    p_delete=0.2,
    p_split=0.2,
    p_geometry=0.2,
    p_shift=0.1,
    p_largest=0.2,
    p_repair=0.5,
)


def _run_seed(seed: int) -> None:
    """Run all mutation sweep combinations for one seed (worker entrypoint)."""
    for dataset_name, max_images in DATASETS:
        for p_local in P_LOCAL_VALUES:
            print(f"[seed={seed}] p_local={p_local} | {dataset_name}")
            run_experiments(
                image_dir_name=dataset_name,
                seed=seed,
                pop_size=POP_SIZE,
                generations=GENERATIONS,
                patience=PATIENCE,
                algorithm=ALGO,
                crossover_method=CROSSOVER,
                max_images=max_images,
                run_id=f"exp4_local/{p_local:.2f}",
                p_local=p_local,
                p_merge=FIXED_P_MERGE_FOR_LOCAL,
                **BASE_MUTATIONS,
            )

    for dataset_name, max_images in DATASETS:
        for p_merge in P_MERGE_VALUES:
            print(f"[seed={seed}] p_merge={p_merge} | {dataset_name}")
            run_experiments(
                image_dir_name=dataset_name,
                seed=seed,
                pop_size=POP_SIZE,
                generations=GENERATIONS,
                patience=PATIENCE,
                algorithm=ALGO,
                crossover_method=CROSSOVER,
                max_images=max_images,
                run_id=f"exp4_merge/{p_merge:.2f}",
                p_local=FIXED_P_LOCAL_FOR_MERGE,
                p_merge=p_merge,
                **BASE_MUTATIONS,
            )


def main() -> None:
    """Run mutation analysis experiments — one process per seed."""
    print(f"Seeds: {SEEDS}")
    print(f"Workers: {N_SEEDS}")

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
