"""Render clean white-background PNGs from sample_selection JSONs.

Identical to draw_solution (same palette, alpha, figsize, dpi, bbox_inches)
except: white background (inverted image) and no title.

Run from repo root:
    python -m experiments.scripts.render_clean_viz
"""

import json
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.visualization import COLOR_PALETTE

_REPO = Path(__file__).resolve().parents[2]
SAMPLE_DIR = _REPO.joinpath("experiments", "results", "visualizations", "sample_selection")
RECTS_ROOT = _REPO.joinpath("experiments", "results", "rectangles")
DATASETS_ROOT = _REPO.joinpath("data", "datasets")


def find_json(alg: str, dataset: str, json_name: str) -> Path | None:
    for p in RECTS_ROOT.joinpath(alg).rglob(json_name):
        if dataset in str(p):
            return p
    return None


def load_image(image_name: str, dataset: str) -> np.ndarray | None:
    p = DATASETS_ROOT.joinpath(dataset, "npy", image_name)
    if not p.exists():
        return None
    img = np.load(p).astype(float)
    if img.max() > 1:
        img = img / 255.0
    return img


def render(png: Path, dataset: str) -> None:
    m = re.match(r"^(.+?)__(.+_rects_\d+)$", png.stem)
    if not m:
        print(f"  [skip] cannot parse: {png.name}")
        return

    alg, base = m.group(1), m.group(2)
    json_path = find_json(alg, dataset, base + ".json")
    if json_path is None:
        print(f"  [skip] JSON not found: {alg}/{dataset}/{base}.json")
        return

    data = json.loads(json_path.read_text())
    img = load_image(data["image_name"], dataset)
    if img is None:
        print(f"  [skip] image not found: {data['image_name']}")
        return

    rectangles = data["rectangles"]
    height, width = img.shape

    palette = COLOR_PALETTE[:]
    random.shuffle(palette)

    # --- identical to draw_solution except: 1-img (white bg), no title ---
    plt.figure(figsize=(10, 10))
    plt.imshow(1.0 - img, cmap="gray", origin="upper", extent=[0, width, height, 0])

    for idx, rect in enumerate(rectangles):
        x, y, w, h = map(int, rect)
        color = palette[idx % len(palette)]
        plt.gca().add_patch(plt.Rectangle(
            (x, y), w, h,
            facecolor=color,
            alpha=0.75,
            edgecolor=color,
            linewidth=0.5,
        ))

    plt.axis("off")

    out = png.parent / f"{png.stem}_clean.png"
    plt.savefig(str(out), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  {alg}  →  {out.name}")


def main() -> None:
    for dataset_dir in sorted(SAMPLE_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name

        for image_dir in sorted(dataset_dir.iterdir()):
            if not image_dir.is_dir():
                continue
            print(f"\n[{dataset}] {image_dir.name}")

            for png in sorted(image_dir.glob("*.png")):
                if png.stem.endswith("_clean"):
                    continue
                render(png, dataset)

    print("\nDone.")


if __name__ == "__main__":
    main()