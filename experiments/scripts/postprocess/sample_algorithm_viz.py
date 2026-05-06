"""Find a shared image across all algorithms and copy one PNG per algorithm to sample_selection/<image>.

Run from repo root:
    python -m experiments.scripts.sample_algorithm_viz
"""

import random
import re
import shutil
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
VIZ_ROOT = _REPO.joinpath("experiments", "results", "visualizations")
OUT_BASE = VIZ_ROOT.joinpath("sample_selection")


def image_stem(png: Path) -> str:
    return re.sub(r"_rects_\d+$", "", png.stem)


def main() -> None:
    # Set to a specific image stem to fix the choice, or None for random
    leafs_image: str | None = "Betula_pendula"
    objects_image: str | None = "tree-15_binary"

    rng = random.Random()

    algorithms = sorted(
        p.name for p in VIZ_ROOT.iterdir()
        if p.is_dir() and p.name != "sample_selection"
    )

    for dataset, forced in (("leafs_selected", leafs_image), ("objects_selected", objects_image)):
        index: dict[str, dict[str, list[Path]]] = {}
        for alg in algorithms:
            for png in (VIZ_ROOT / alg).rglob("*.png"):
                if dataset not in str(png):
                    continue
                stem = image_stem(png)
                index.setdefault(stem, {}).setdefault(alg, []).append(png)

        algs_with_data = [a for a in algorithms if a in {alg for m in index.values() for alg in m}]
        common = [
            stem for stem, alg_map in index.items()
            if all(a in alg_map for a in algs_with_data)
        ]

        if not common:
            print(f"{dataset}: no image found in all algorithms, skipping")
            continue

        chosen_stem = forced if forced is not None else rng.choice(sorted(common))
        out_dir = OUT_BASE.joinpath(dataset, chosen_stem)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{dataset}] Selected image: {chosen_stem}")

        for alg in algs_with_data:
            png = rng.choice(index[chosen_stem][alg])
            dest = out_dir / f"{alg}__{png.name}"
            shutil.copy2(png, dest)
            print(f"  {alg}  →  {dest.name}")

    print(f"\nDone — {OUT_BASE}")


if __name__ == "__main__":
    main()