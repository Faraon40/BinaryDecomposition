"""Demo: All decomposition algorithms on the same binary image."""

import numpy as np

from src.algorithms.dm import run_dm
from src.algorithms.gdm import run_gdm
from src.algorithms.graph_based import run_graph_based
from src.algorithms.largest_rect import run_largest_rect
from src.algorithms.quadtree import run_quadtree
from experiments.src.demo_utils import DOCS_FIGURES, save_decomposition

DATA_PATH = (
    "../../data/datasets/objects_binary/npy/butterfly-14_binary.npy"
)

img = (np.load(DATA_PATH) > 0).astype(np.uint8)
img_name = DATA_PATH.split("/")[-1].replace(".npy", "")

algorithms = [
    ("DM",          lambda: run_dm(img)),
    ("GDM",         lambda: run_gdm(img)),
    ("Quadtree",    lambda: run_quadtree(img)[0]),
    ("Largest-Rect", lambda: run_largest_rect(img)),
    ("GBD",         lambda: run_graph_based(img)[0]),
]

for name, run_fn in algorithms:
    rects = run_fn()
    save_decomposition(
        img, rects,
        path=f"{DOCS_FIGURES}/demo/{img_name}_{name.lower().replace('-', '_')}.png",
        title=f"{name}: {len(rects)} rectangles",
    )
    print(f"{name:<14} {len(rects)} rectangles")

print(f"\nSaved to {DOCS_FIGURES}")