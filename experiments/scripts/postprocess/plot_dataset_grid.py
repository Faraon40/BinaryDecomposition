"""Display a 4x4 sample grid for each dataset as separate figures."""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATASETS = {
    "leafs": Path("../../../data/datasets/leafs_selected/npy"),
    "objects": Path("../../../data/datasets/objects_selected/npy"),
}
COLS = 4
ROWS = 4
SEED = 42
OUTPUT_DIR = Path("../../notebooks/dashboard")


def load_image(path: Path) -> np.ndarray:
    img = np.load(path).astype(float)
    if img.max() > 1:
        img = img / 255.0
    return 1.0 - img  # invert: object black on white background


def plot_grid(label: str, npy_dir: Path) -> None:
    rng = random.Random(SEED)
    n = COLS * ROWS
    files = sorted(npy_dir.glob("*.npy"))
    sample = rng.sample(files, min(n, len(files)))

    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 1.8, ROWS * 2.0))

    for i, fpath in enumerate(sample):
        ax = axes[i // COLS, i % COLS]
        ax.imshow(load_image(fpath), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")

    plt.tight_layout(h_pad=0.4, w_pad=0.4)
    out = OUTPUT_DIR / f"fig_dataset_grid_{label}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


def main() -> None:
    for label, npy_dir in DATASETS.items():
        plot_grid(label, npy_dir)


if __name__ == "__main__":
    main()