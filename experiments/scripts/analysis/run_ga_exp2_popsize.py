"""EXP-2 (DM): Vplyv pop_size a patience na kvalitu a rýchlosť GA s DM init.

Dva samostatné sweepy:
  A) pop_size ∈ [3, 5, 8, 10, 15, 20, 30, 50, 100] — patience fixné
  B) patience ∈ [3, 5, 8, 12, 15, 20, 25]          — pop_size fixné

Seeds bežia paralelne (jeden proces na seed).

Usage:
  python -m experiments.scripts.analysis.run_ga_exp2_popsize
  python -m experiments.scripts.analysis.run_ga_exp2_popsize --sweep popsize
  python -m experiments.scripts.analysis.run_ga_exp2_popsize --sweep patience
  python -m experiments.scripts.analysis.run_ga_exp2_popsize --dataset objects
"""

import argparse
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiments.scripts.run_ga import run_experiments

ALGO = "ga_dm"
CROSSOVER = "subset_greedy_relaxed"
GENERATIONS = 100

N_SEEDS = 5
SEEDS = random.sample(range(10**8), N_SEEDS)

DATASETS: dict[str, tuple[str, int | None]] = {
    "objects": ("analysis/objects_quartile", None),
    "leafs":   ("analysis/leafs_quartile", None),
}

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

# ── A) pop_size sweep ─────────────────────────────────────────────────
POP_SIZES = [3, 5, 8, 10, 15, 20, 30, 50, 100]
FIXED_PATIENCE_FOR_POP = 10

# ── B) patience sweep ─────────────────────────────────────────────────
PATIENCE_VALUES = [3, 5, 8, 12, 15, 20, 25]
FIXED_POP_SIZE_FOR_PATIENCE = 20


def _run_seed(
    seed: int,
    datasets: list[tuple[str, int | None]],
    sweeps: list[str],
) -> None:
    """Run all sweep combinations for one seed (worker entrypoint)."""
    if "popsize" in sweeps:
        for dataset_name, max_images in datasets:
            for pop_size in POP_SIZES:
                print(f"[seed={seed}] popsize={pop_size} | {dataset_name}")
                run_experiments(
                    image_dir_name=dataset_name,
                    seed=seed,
                    pop_size=pop_size,
                    generations=GENERATIONS,
                    patience=FIXED_PATIENCE_FOR_POP,
                    algorithm=ALGO,
                    crossover_method=CROSSOVER,
                    max_images=max_images,
                    run_id=f"exp2_popsize/{pop_size}",
                    **MUTATION_DEFAULTS,
                )

    if "patience" in sweeps:
        for dataset_name, max_images in datasets:
            for patience in PATIENCE_VALUES:
                print(f"[seed={seed}] patience={patience} | {dataset_name}")
                run_experiments(
                    image_dir_name=dataset_name,
                    seed=seed,
                    pop_size=FIXED_POP_SIZE_FOR_PATIENCE,
                    generations=GENERATIONS,
                    patience=patience,
                    algorithm=ALGO,
                    crossover_method=CROSSOVER,
                    max_images=max_images,
                    run_id=f"exp2_patience/{patience}",
                    **MUTATION_DEFAULTS,
                )


def main() -> None:
    """Run pop_size and patience sweeps — one process per seed."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep", choices=["popsize", "patience"],
        help="Which sweep to run (default: both)",
    )
    parser.add_argument(
        "--dataset", choices=list(DATASETS.keys()),
        help="Dataset to process (default: all)",
    )
    args = parser.parse_args()

    datasets = [DATASETS[args.dataset]] if args.dataset else list(DATASETS.values())
    sweeps = [args.sweep] if args.sweep else ["popsize", "patience"]

    print(f"Seeds: {SEEDS}")
    print(f"Workers: {N_SEEDS} | Sweeps: {sweeps}")

    with ProcessPoolExecutor(max_workers=N_SEEDS) as executor:
        futures = {
            executor.submit(_run_seed, seed, datasets, sweeps): seed
            for seed in SEEDS
        }
        for future in as_completed(futures):
            seed = futures[future]
            exc = future.exception()
            if exc:
                print(f"[seed={seed}] FAILED: {exc}")
            else:
                print(f"[seed={seed}] DONE")


if __name__ == "__main__":
    main()
