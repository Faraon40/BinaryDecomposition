"""EXP-1: Vplyv pop_size a patience na kvalitu a rýchlosť GA.

Spúšťa sa ako prvý experiment — výsledky určujú pop_size pre EXP-2–4.

Cieľ: nájsť najmenší pop_size a patience kde GA stále prekonáva GDM
v počte obdĺžnikov a zostáva rýchlejší ako Graph-Based (GBD).

Dva samostatné sweepy:
  A) pop_size ∈ [5, 10, 15, 20, 30] — patience fixné
  B) patience ∈ [5, 10, 15, 25]     — pop_size fixné

Init metóda: gdm (ga_gdm).

Usage:
  python -m experiments.scripts.analysis.run_ga_exp1_popsize
"""

import random

from experiments.scripts.run_ga import run_experiments

ALGO = "ga_gdm"
CROSSOVER = "subset_greedy"
GENERATIONS = 100

SEEDS = random.sample(range(10**8), 3)

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

# ── A) pop_size sweep ─────────────────────────────────────────────────
POP_SIZES = [3, 5, 8, 10, 15]
FIXED_PATIENCE_FOR_POP = 8

# ── B) patience sweep ─────────────────────────────────────────────────
PATIENCE_VALUES = [3, 5, 8, 12]
FIXED_POP_SIZE_FOR_PATIENCE = 10


def main() -> None:
    """Run pop_size and patience sweeps."""
    runs_a = len(POP_SIZES) * len(SEEDS) * len(DATASETS)
    runs_b = len(PATIENCE_VALUES) * len(SEEDS) * len(DATASETS)
    total_runs = runs_a + runs_b
    run_idx = 0

    print(f"Seeds: {SEEDS}")
    print(f"EXP-1A: pop_size sweep  ({runs_a} runs)")
    print(f"EXP-1B: patience sweep  ({runs_b} runs)")
    print(f"Total:  {total_runs} runs\n")

    # ── A) pop_size sweep ─────────────────────────────────────────────
    for dataset_name, max_images in DATASETS:
        for pop_size in POP_SIZES:
            for seed in SEEDS:
                run_idx += 1
                print(
                    f"\n{'='*60}\n"
                    f"RUN {run_idx}/{total_runs} [1A]: "
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
                    run_id=f"exp4_popsize/{pop_size}",
                    **MUTATION_DEFAULTS,
                )

    # ── B) patience sweep ─────────────────────────────────────────────
    for dataset_name, max_images in DATASETS:
        for patience in PATIENCE_VALUES:
            for seed in SEEDS:
                run_idx += 1
                print(
                    f"\n{'='*60}\n"
                    f"RUN {run_idx}/{total_runs} [1B]: "
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
                    run_id=f"exp4_patience/{patience}",
                    **MUTATION_DEFAULTS,
                )


if __name__ == "__main__":
    main()
