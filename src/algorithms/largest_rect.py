"""Greedy largest-rectangle-first decomposition for binary images.

This module implements a greedy decomposition that at each step finds
and places the largest axis-aligned rectangle fitting entirely within
uncovered foreground pixels, using the classic histogram-based
largest-rectangle-in-a-matrix algorithm.  The process repeats until
all foreground pixels are covered.

The algorithm is also used as a GA initialization strategy
(``init_population_largest_rect``).
"""

import random
import time
from typing import List, Optional, Tuple

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


def largest_rect_in_histogram(
    heights: List[int],
) -> Tuple[int, int, int]:
    """Find the largest rectangle in a histogram using a stack algorithm.

    Classic O(n) stack-based algorithm.  The sentinel value ``h=0``
    appended at the end ensures the stack is fully drained.

    Args:
        heights: List of non-negative bar heights.

    Returns:
        Tuple ``(x, width, height)`` of the largest rectangle found.
        Returns ``(0, 0, 0)`` when all bars are zero.

    """
    stack: List[int] = []
    best_area = 0
    best_x = 0
    best_w = 0
    best_h = 0

    n = len(heights)
    for i in range(n + 1):
        h = heights[i] if i < n else 0
        while stack and heights[stack[-1]] > h:
            top = stack.pop()
            bar_h = heights[top]
            left = stack[-1] + 1 if stack else 0
            width = i - left
            area = bar_h * width
            if area > best_area:
                best_area = area
                best_x = left
                best_w = width
                best_h = bar_h
        stack.append(i)

    return best_x, best_w, best_h


def largest_rect_in_image(
    img: np.ndarray,
    covered: np.ndarray,
) -> Optional[Rectangle]:
    """Find the largest rectangle of uncovered foreground pixels.

    Scans each row, building a histogram of consecutive uncovered
    foreground pixels above that row, then applies
    ``largest_rect_in_histogram`` to find the best rectangle ending at
    that row.

    Args:
        img: Binary image (2-D array of 0/1 values).
        covered: Boolean/uint8 mask of already-covered pixels.

    Returns:
        Rectangle ``(x, y, width, height)`` of the largest uncovered
        region, or ``None`` if all foreground pixels are already
        covered.

    """
    height, width = img.shape
    heights = [0] * width
    best_area = 0
    best_rect: Optional[Rectangle] = None

    for row in range(height):
        for col in range(width):
            if img[row, col] == 1 and covered[row, col] == 0:
                heights[col] += 1
            else:
                heights[col] = 0

        bx, bw, bh = largest_rect_in_histogram(heights)

        if bw == 0:
            continue

        area = bw * bh
        if area > best_area:
            best_area = area
            best_rect = (bx, row - bh + 1, bw, bh)

    return best_rect


def largest_rect_decomposition(
    img: np.ndarray,
    coverage_threshold: float = 0.95,
) -> List[Rectangle]:
    """Decompose a binary image using greedy largest-rectangle-first strategy.

    Repeatedly places the largest valid rectangle of uncovered
    foreground pixels until ``coverage_threshold`` fraction of all
    foreground pixels is covered (or all pixels are covered when
    threshold is 1.0).  Convergence is guaranteed because each
    iteration covers at least one pixel.

    Args:
        img: Binary image (2-D array of 0/1 values).
        coverage_threshold: Stop once this fraction of foreground pixels
            is covered (default: 0.95).

    Returns:
        List of non-overlapping rectangles covering foreground pixels
        up to ``coverage_threshold``.

    """
    covered = np.zeros_like(img, dtype=np.uint8)
    rects: List[Rectangle] = []
    total = int(img.sum())

    if total == 0:
        return rects

    while True:
        covered_count = int(covered[img == 1].sum())
        if covered_count >= total * coverage_threshold:
            break
        rect = largest_rect_in_image(img, covered)
        if rect is None:
            break
        x, y, w, h = rect
        rects.append(rect)
        covered[y: y + h, x: x + w] = 1

    return rects


def run_largest_rect(
    img: np.ndarray,
    coverage_threshold: float = 0.95,
    verbose: bool = False,
) -> List[Rectangle]:
    """Run largest-rectangle-first decomposition, completing residual with GDM.

    The greedy largest-rectangle pass stops once ``coverage_threshold``
    fraction of foreground pixels is covered, then GDM covers the
    remaining pixels.  This avoids the prohibitive runtime of a full
    pass on large images while still producing large rectangles for the
    dominant foreground regions.

    Args:
        img: Binary image (2-D NumPy array of 0/1 values).
        coverage_threshold: Stop the greedy pass once this fraction of
            foreground pixels is covered, then complete with GDM
            (default: 0.95).  Use 1.0 for a pure greedy pass (slow on
            large images).
        verbose: If True, print rectangle counts and elapsed time.

    Returns:
        List of non-overlapping rectangles covering all 1-pixels.

    """
    from src.algorithms.gdm import gdm_decomposition

    t0 = time.time()
    rects = largest_rect_decomposition(
        img, coverage_threshold=coverage_threshold
    )

    if coverage_threshold < 1.0:
        covered = np.zeros_like(img, dtype=np.uint8)
        for x, y, w, h in rects:
            covered[y:y + h, x:x + w] = 1

        residual = ((img == 1) & (covered == 0)).astype(np.uint8)

        if residual.any():
            gdm_rects = gdm_decomposition(residual, direction="auto")
            rects.extend(gdm_rects)

    if verbose:
        elapsed = time.time() - t0
        mode = (
            f"largest_rect+GDM (coverage={coverage_threshold})"
            if coverage_threshold < 1.0 else "largest_rect"
        )
        print(f"{mode}: {len(rects)} rectangles in {elapsed:.4f}s")
    return rects


def init_population_largest_rect(
    img: np.ndarray,
    integral: np.ndarray,
    pop_size: int,
) -> List[Chromosome]:
    """Initialize GA population using largest-rectangle decomposition.

    Decomposition is run once deterministically.  Each individual
    receives a shuffled copy of the resulting rectangles, so the
    population is diverse with respect to rectangle order, which
    affects crossover and overlap-repair outcomes.

    Args:
        img: Binary image.
        integral: Precomputed integral image (unused, kept for API
            consistency with other init functions).
        pop_size: Number of chromosomes to generate.

    Returns:
        List of initialized Chromosome objects.

    """
    rects = largest_rect_decomposition(img, coverage_threshold=0.95)

    population: List[Chromosome] = []
    for _ in range(pop_size):
        shuffled = rects[:]
        random.shuffle(shuffled)
        population.append(Chromosome(shuffled))
    return population