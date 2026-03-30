"""Delta Method (DM) decomposition for binary images.

This module implements the Delta Method rectangle decomposition as a
standalone algorithm, producing 1-pixel-tall (or 1-pixel-wide) rectangles
that cover all foreground pixels without overlap.

The decomposition direction is chosen automatically:
- Row-wise when image width >= height
- Column-wise otherwise
"""

import random
import time
from typing import List

import numpy as np

from src.utils.types import Chromosome, Rectangle


def build_integral(img: np.ndarray) -> np.ndarray:
    """Build integral image for O(1) rectangle sum queries."""
    return img.cumsum(axis=0).cumsum(axis=1)


def rect_sum(
    integral: np.ndarray, x: int, y: int, w: int, h: int
) -> int:
    """Return sum of rectangle (x, y, w, h) using integral image."""
    x2, y2 = x + w - 1, y + h - 1
    total = integral[y2, x2]
    if x > 0:
        total -= integral[y2, x - 1]
    if y > 0:
        total -= integral[y - 1, x2]
    if x > 0 and y > 0:
        total += integral[y - 1, x - 1]
    return total


def is_valid_rectangle_integral(
    integral: np.ndarray, rect: Rectangle
) -> bool:
    """Check if rectangle is within bounds and fully covered by 1-pixels.

    Args:
        integral: Integral image array.
        rect: Rectangle tuple (x, y, width, height).

    Returns:
        True if rectangle is within bounds and fully covered.

    """
    x, y, w, h = rect
    if (
        x < 0
        or y < 0
        or x + w > integral.shape[1]
        or y + h > integral.shape[0]
    ):
        return False
    return rect_sum(integral, x, y, w, h) == w * h


def dm_decomposition(
    img: np.ndarray,
    integral: np.ndarray,
    direction: str = "random",
) -> List[Rectangle]:
    """Perform a single Delta Method pass over the image.

    Each run of consecutive 1-pixels becomes one rectangle.

    Args:
        img: Binary image (2-D NumPy array).
        integral: Precomputed integral image.
        direction: Scan direction - ``"row"`` for row-wise (1-pixel-tall
            rectangles), ``"col"`` for column-wise (1-pixel-wide
            rectangles), ``"auto"`` to pick based on aspect ratio
            (row-wise when width >= height, column-wise otherwise),
            ``"random"`` to choose uniformly at random.

    Returns:
        List of non-overlapping rectangles covering all 1-pixels.

    """
    height, width = img.shape
    rects: List[Rectangle] = []

    if direction == "random":
        direction = random.choice(["row", "col"])
    elif direction == "auto":
        direction = "row" if width >= height else "col"

    if direction == "row":
        for y in range(height):
            x = 0
            while x < width:
                if img[y, x] == 1:
                    x_start = x
                    while x < width and img[y, x] == 1:
                        x += 1
                    rect = (x_start, y, x - x_start, 1)
                    if is_valid_rectangle_integral(integral, rect):
                        rects.append(rect)
                else:
                    x += 1
    else:
        for x in range(width):
            y = 0
            while y < height:
                if img[y, x] == 1:
                    y_start = y
                    while y < height and img[y, x] == 1:
                        y += 1
                    rect = (x, y_start, 1, y - y_start)
                    if is_valid_rectangle_integral(integral, rect):
                        rects.append(rect)
                else:
                    y += 1

    return rects


def run_dm(img: np.ndarray, verbose: bool = False) -> List[Rectangle]:
    """Run Delta Method decomposition on a binary image.

    Args:
        img: Binary image (2-D NumPy array of 0/1 values).
        verbose: If True, print rectangle count and elapsed time.

    Returns:
        List of non-overlapping rectangles covering all 1-pixels.

    """
    t0 = time.time()
    integral = build_integral(img)
    rects = dm_decomposition(img, integral)
    if verbose:
        elapsed = time.time() - t0
        print(
            f"Delta Method decomposition: {len(rects)} rectangles "
            f"in {elapsed:.4f}s"
        )
    return rects


def init_population_dm(
    img: np.ndarray,
    integral: np.ndarray,
    pop_size: int,
    direction: str = "mixed",
) -> List[Chromosome]:
    """Initialize GA population using Delta Method decomposition (shuffled variants).

    Args:
        img: Binary image.
        integral: Precomputed integral image.
        pop_size: Number of chromosomes to generate.
        direction: ``"row"`` for row-wise only, ``"col"`` for column-wise
            only, ``"mixed"`` (default) for half-and-half.

    Returns:
        List of initialized Chromosome objects.

    """
    if direction == "mixed":
        rects_row = dm_decomposition(img, integral, direction="row")
        rects_col = dm_decomposition(img, integral, direction="col")
        half = pop_size // 2
        sources = [(rects_row, half), (rects_col, pop_size - half)]
    else:
        rects = dm_decomposition(img, integral, direction=direction)
        sources = [(rects, pop_size)]

    population: List[Chromosome] = []
    for rects, count in sources:
        for _ in range(count):
            chromosome_rects = list(rects)
            random.shuffle(chromosome_rects)
            population.append(Chromosome(chromosome_rects))

    random.shuffle(population)
    return population