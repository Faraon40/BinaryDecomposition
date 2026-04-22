"""EXP-4: Vplyv Local decomposition a p_merge na kvalitu GA.

Dve cielené otázky:
  A) Aký vplyv má p_local na objektoch? (p_merge fixné = 0.10)
  B) Potvrdiť že p_merge >= 0.10 je dôležité? (p_local fixné = 0.50)

Init: ga_gdm. Ostatné mutácie fixné.
Aktualizuj BEST_POP_SIZE po analýze EXP-2.

Prerequisites:
  Run EXP-2 (run_ga_exp2_popsize) first and update BEST_POP_SIZE below.

Usage:
  python -m experiments.scripts.analysis.run_ga_exp4_mutation
"""

import random

from experiments.scripts.run_ga import run_experiments

ALGO = "ga_dm"
CROSSOVER = "subset_greedy_relaxed"
GENERATIONS = 100
PATIENCE = 6


BEST_POP_SIZE = 20

SEEDS = random.sample(range(10**8), 3)

DATASETS: list[tuple[str, int | None]] = [
    ("analysis/objects_unique", 5),
    ("analysis/leafs_subset", 5),
]

# A) Vplyv p_local — p_merge fixné
P_LOCAL_VALUES = [0.0, 0.25, 0.5, 0.75]
FIXED_P_MERGE_FOR_LOCAL = 0.20

# B) Vplyv p_merge — p_local fixné
P_MERGE_VALUES = [0.05, 0.10, 0.20]
FIXED_P_LOCAL_FOR_MERGE = 0.50

BASE_MUTATIONS = dict(
    p_delete=0.2,
    p_split=0.2,
    p_geometry=0.3,
    p_shift=0.05,
    p_largest=0.2,
    p_repair=0.5,
)


def main() -> None:
    """Run mutation analysis experiments."""
    runs_a = len(P_LOCAL_VALUES) * len(SEEDS) * len(DATASETS)
    runs_b = len(P_MERGE_VALUES) * len(SEEDS) * len(DATASETS)
    total_runs = runs_a + runs_b
    run_idx = 0

    print(f"Seeds: {SEEDS}")
    print(f"EXP-4A: p_local sweep ({runs_a} runs)")
    print(f"EXP-4B: p_merge sweep ({runs_b} runs)")
    print(f"Total: {total_runs} runs\n")

    # ── A) p_local sweep ─────────────────────────────────────────────
    for dataset_name, max_images in DATASETS:
        for p_local in P_LOCAL_VALUES:
            for seed in SEEDS:
                run_idx += 1
                run_id = f"exp5_local/{p_local:.2f}"
                print(
                    f"\n{'='*60}\n"
                    f"RUN {run_idx}/{total_runs} [4A]: "
                    f"p_local={p_local}, seed={seed}, "
                    f"dataset={dataset_name}\n"
                    f"{'='*60}"
                )
                run_experiments(
                    image_dir_name=dataset_name,
                    seed=seed,
                    pop_size=BEST_POP_SIZE,
                    generations=GENERATIONS,
                    patience=PATIENCE,
                    algorithm=ALGO,
                    crossover_method=CROSSOVER,
                    max_images=max_images,
                    run_id=run_id,
                    p_local=p_local,
                    p_merge=FIXED_P_MERGE_FOR_LOCAL,
                    **BASE_MUTATIONS,
                )

    # ── B) p_merge sweep ─────────────────────────────────────────────
    for dataset_name, max_images in DATASETS:
        for p_merge in P_MERGE_VALUES:
            for seed in SEEDS:
                run_idx += 1
                run_id = f"exp5_merge_local/{p_merge:.2f}"
                print(
                    f"\n{'='*60}\n"
                    f"RUN {run_idx}/{total_runs} [4B]: "
                    f"p_merge={p_merge}, seed={seed}, "
                    f"dataset={dataset_name}\n"
                    f"{'='*60}"
                )
                run_experiments(
                    image_dir_name=dataset_name,
                    seed=seed,
                    pop_size=BEST_POP_SIZE,
                    generations=GENERATIONS,
                    patience=PATIENCE,
                    algorithm=ALGO,
                    crossover_method=CROSSOVER,
                    max_images=max_images,
                    run_id=run_id,
                    p_local=FIXED_P_LOCAL_FOR_MERGE,
                    p_merge=p_merge,
                    **BASE_MUTATIONS,
                )


if __name__ == "__main__":
    main()
