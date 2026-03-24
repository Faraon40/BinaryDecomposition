"""Generalized Delta Method (GDM) decomposition for binary images.

This module implements the Generalized Delta Method as a standalone
algorithm and as a GA initialization strategy.

GDM extends the Delta Method by merging adjacent rows (or columns) that
share identical intervals into taller rectangles, producing significantly
fewer rectangles than DM.

References:
    Spiliotis, I. M., & Mertzios, B. G. (1998). Real-time computation of
    two-dimensional moments on binary images using image block
    representation. IEEE Transactions on Image Processing, 7(11),
    1609-1615. https://doi.org/10.1109/83.725368

    Suk, T., Höschl IV, C., & Flusser, J. (2012). Rectangular
    decomposition of binary images. In Advanced Concepts for Intelligent
    Vision Systems (ACIVS), LNCS 7517, pp. 215-224. Springer.
    https://doi.org/10.1007/978-3-642-33140-4_19
"""

import random
import time
from typing import Dict, List, Tuple

import numpy as np

from src.utils.types import Chromosome, Rectangle


def gdm_decomposition(
    img: np.ndarray,
direction: str = "random",
) -> List[Rectangle]:
    """Decompose a binary image using the Generalized Delta Method.

    Scans the image row by row (or column by column), tracking active
    rectangular blocks.  When a row's intervals exactly match those of
    the previous row, the blocks are extended downward.  When intervals
    change, completed blocks are closed and new ones are opened.

    Args:
        img: Binary image (2-D NumPy array of 0/1 values).
        direction: Scan direction - ``"row"`` for row-wise,
            ``"col"`` for column-wise, ``"auto"`` to pick based on
            aspect ratio (row-wise when width >= height),
            ``"random"`` to choose uniformly at random.

    Returns:
        List of non-overlapping rectangles covering all 1-pixels.

    """
    height, width = img.shape

    if direction == "random":
        direction = random.choice(["row", "col"])
    elif direction == "auto":
        direction = "row" if width >= height else "col"

    rectangles: List[Rectangle] = []

    if direction == "row":
        # active: maps (x, w) -> y_start of the current block
        active: Dict[Tuple[int, int], int] = {}

        for y in range(height):
            current: Dict[Tuple[int, int], None] = {}
            x = 0
            while x < width:
                if img[y, x] == 1:
                    x_start = x
                    while x < width and img[y, x] == 1:
                        x += 1
                    current[(x_start, x - x_start)] = None
                else:
                    x += 1

            new_active: Dict[Tuple[int, int], int] = {}
            for key, y_start in active.items():
                if key in current:
                    new_active[key] = y_start
                else:
                    xk, wk = key
                    rectangles.append((xk, y_start, wk, y - y_start))

            for key in current:
                if key not in new_active:
                    new_active[key] = y

            active = new_active

        for (xk, wk), y_start in active.items():
            rectangles.append((xk, y_start, wk, height - y_start))

    else:
        # active: maps (y, h) -> x_start of the current block
        active_col: Dict[Tuple[int, int], int] = {}

        for x in range(width):
            current_col: Dict[Tuple[int, int], None] = {}
            y = 0
            while y < height:
                if img[y, x] == 1:
                    y_start = y
                    while y < height and img[y, x] == 1:
                        y += 1
                    current_col[(y_start, y - y_start)] = None
                else:
                    y += 1

            new_active_col: Dict[Tuple[int, int], int] = {}
            for key, x_start in active_col.items():
                if key in current_col:
                    new_active_col[key] = x_start
                else:
                    yk, hk = key
                    rectangles.append((x_start, yk, x - x_start, hk))

            for key in current_col:
                if key not in new_active_col:
                    new_active_col[key] = x

            active_col = new_active_col

        for (yk, hk), x_start in active_col.items():
            rectangles.append((x_start, yk, width - x_start, hk))

    return rectangles


def run_gdm(img: np.ndarray, verbose: bool = False) -> List[Rectangle]:
    """Run Generalized Delta Method decomposition on a binary image.

    Args:
        img: Binary image (2-D NumPy array of 0/1 values).
        verbose: If True, print rectangle count and elapsed time.

    Returns:
        List of non-overlapping rectangles covering all 1-pixels.

    """
    t0 = time.time()
    rects = gdm_decomposition(img)
    if verbose:
        elapsed = time.time() - t0
        print(
            f"Generalized Delta Method decomposition: {len(rects)} rectangles "
            f"in {elapsed:.4f}s"
        )
    return rects


def init_population_gdm(
    img: np.ndarray, integral: np.ndarray, pop_size: int
) -> List[Chromosome]:
    """Initialize GA population using GDM decomposition (shuffled variants).

    Each chromosome is built from a fresh GDM decomposition with a
    randomly chosen scan direction, then rectangles are shuffled.

    Args:
        img: Binary image.
        integral: Precomputed integral image (kept for API consistency).
        pop_size: Number of chromosomes to generate.

    Returns:
        List of initialized Chromosome objects.

    """
    population: List[Chromosome] = []
    for _ in range(pop_size):
        rects = gdm_decomposition(img, direction="random")
        random.shuffle(rects)
        population.append(Chromosome(rects))
    return population
