"""EXP-2 (DM): Vplyv pop_size a patience na kvalitu a rýchlosť GA s DM init.

Cieľ: porovnať či DM init (väčšia diverzita populácie) reaguje inak na
pop_size a patience ako GDM init.

Dva samostatné sweepy:
  A) pop_size ∈ [3, 5, 8, 10, 15, 20, 30, 50, 100] — patience fixné
  B) patience ∈ [3, 5, 8, 12, 15]                  — pop_size fixné

Init metóda: dm (ga_dm).

Usage (celý experiment):
  python -m experiments.scripts.analysis.run_ga_exp2_dm_popsize

Usage (4 paralelné procesy):
  python -m experiments.scripts.analysis.run_ga_exp2_popsize --sweep popsize --dataset objects
  python -m experiments.scripts.analysis.run_ga_exp2_popsize --sweep popsize --dataset leafs
  python -m experiments.scripts.analysis.run_ga_exp2_popsize --sweep patience --dataset objects
  python -m experiments.scripts.analysis.run_ga_exp2_popsize --sweep patience --dataset leafs
"""


import argparse
import random

from experiments.scripts.run_ga import run_experiments

ALGO = "ga_dm"
CROSSOVER = "subset_greedy_relaxed"
GENERATIONS = 100

SEEDS = random.sample(range(10**8), 4)

DATASETS: dict[str, tuple[str, int | None]] = {
    "objects": ("analysis/objects_unique", 10),
    "leafs": ("analysis/leafs_subset", 10),
}

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

# ── A) pop_size sweep ─────────────────────────────────────────────────
POP_SIZES = [3, 5, 8, 10, 15, 20, 30, 50, 100]
FIXED_PATIENCE_FOR_POP = 10

# ── B) patience sweep ─────────────────────────────────────────────────
PATIENCE_VALUES = [3, 5, 8, 12, 15, 20, 25]
FIXED_POP_SIZE_FOR_PATIENCE = 20


def run_popsize_sweep(datasets: list[tuple[str, int | None]]) -> None:
    """Run pop_size sweep for the given datasets."""
    total = len(POP_SIZES) * len(SEEDS) * len(datasets)
    run_idx = 0
    print(f"EXP-2A (DM): pop_size sweep — {total} runs\n")
    for dataset_name, max_images in datasets:
        for pop_size in POP_SIZES:
            for seed in SEEDS:
                run_idx += 1
                print(
                    f"\n{'='*60}\n"
                    f"RUN {run_idx}/{total} [2A-DM]: "
                    f"pop_size={pop_size}, patience={FIXED_PATIENCE_FOR_POP}, "
                    f"seed={seed}, dataset={dataset_name}\n"
                    f"{'='*60}"
                )
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


def run_patience_sweep(datasets: list[tuple[str, int | None]]) -> None:
    """Run patience sweep for the given datasets."""
    total = len(PATIENCE_VALUES) * len(SEEDS) * len(datasets)
    run_idx = 0
    print(f"EXP-2B (DM): patience sweep — {total} runs\n")
    for dataset_name, max_images in datasets:
        for patience in PATIENCE_VALUES:
            for seed in SEEDS:
                run_idx += 1
                print(
                    f"\n{'='*60}\n"
                    f"RUN {run_idx}/{total} [2B-DM]: "
                    f"pop_size={FIXED_POP_SIZE_FOR_PATIENCE}, "
                    f"patience={patience}, "
                    f"seed={seed}, dataset={dataset_name}\n"
                    f"{'='*60}"
                )
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
    """Run pop_size and patience sweeps with DM init."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep",
        choices=["popsize", "patience"],
        help="Which sweep to run (default: both)",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        help="Dataset to process: objects | leafs (default: all)",
    )
    args = parser.parse_args()

    datasets = [DATASETS[args.dataset]] if args.dataset else list(DATASETS.values())
    sweeps = [args.sweep] if args.sweep else ["popsize", "patience"]

    print(f"Seeds: {SEEDS}")
    print(f"Datasets: {[d for d, _ in datasets]}")
    print(f"Sweeps: {sweeps}\n")

    if "popsize" in sweeps:
        run_popsize_sweep(datasets)
    if "patience" in sweeps:
        run_patience_sweep(datasets)


if __name__ == "__main__":
    main()
