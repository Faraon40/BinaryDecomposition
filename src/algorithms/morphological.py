"""Morphological decomposition for binary images.

This module implements largest-rectangle-first decomposition as a
standalone algorithm and as a GA initialization strategy.  At each
step the largest axis-aligned rectangle that fits entirely within
uncovered foreground pixels is found and placed, until all foreground
pixels are covered.
"""

import os
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
    rng: Optional[random.Random] = None,
) -> Optional[Rectangle]:
    """Find the largest rectangle of uncovered foreground pixels.

    Scans each row, building a histogram of consecutive uncovered
    foreground pixels above that row, then applies
    ``largest_rect_in_histogram`` to find the best rectangle ending at
    that row.

    Args:
        img: Binary image (2-D array of 0/1 values).
        covered: Boolean/uint8 mask of already-covered pixels.
        rng: Optional :class:`random.Random` instance.  When provided,
            a small random perturbation ``[0, 2)`` is added to each
            positive histogram bar before the histogram scan to break
            ties diversely without collecting all ties explicitly.

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

        if rng is not None:
            scan_heights = [
                h + rng.random() * 2 if h > 0 else 0
                for h in heights
            ]
        else:
            scan_heights = heights

        bx, bw, bh_scan = largest_rect_in_histogram(scan_heights)

        if bw == 0:
            continue

        # Recover integer height from the perturbed histogram
        bh = int(bh_scan) if rng is None else min(
            int(bh_scan), heights[bx]
        )
        # Re-derive height from unperturbed histogram for correctness
        if rng is not None:
            # Actual height is the minimum unperturbed bar in [bx, bx+bw)
            bh = min(heights[bx: bx + bw])

        area = bw * bh
        if area > best_area:
            best_area = area
            y = row - bh + 1
            best_rect = (bx, y, bw, bh)

    return best_rect


def morphological_decomposition(
    img: np.ndarray,
    rng: Optional[random.Random] = None,
    coverage_threshold: float = 0.95,
) -> List[Rectangle]:
    """Decompose a binary image using largest-rectangle-first strategy.

    Repeatedly places the largest valid rectangle of uncovered
    foreground pixels until ``coverage_threshold`` fraction of all
    foreground pixels is covered (or all pixels are covered when
    threshold is 1.0).  Convergence is guaranteed because each
    iteration covers at least one pixel.

    Args:
        img: Binary image (2-D array of 0/1 values).
        rng: Optional :class:`random.Random` instance for stochastic
            tie-breaking.  Pass ``None`` for the deterministic version.
        coverage_threshold: Stop once this fraction of foreground pixels
            is covered (default: 1.0 — full coverage).

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
        rect = largest_rect_in_image(img, covered, rng=rng)
        if rect is None:
            break
        x, y, w, h = rect
        rects.append(rect)
        covered[y: y + h, x: x + w] = 1

    return rects


def run_morphological(
    img: np.ndarray,
    coverage_threshold: float = 0.95,
    verbose: bool = False,
) -> List[Rectangle]:
    """Run morphological decomposition, completing residual with GDM.

    Morphological stops once ``coverage_threshold`` fraction of
    foreground pixels is covered, then GDM covers the remaining
    pixels.  This avoids the prohibitive runtime of full morphological
    decomposition on large images while still producing large
    rectangles for the dominant foreground regions.

    Args:
        img: Binary image (2-D NumPy array of 0/1 values).
        coverage_threshold: Stop morphological once this fraction of
            foreground pixels is covered, then complete with GDM
            (default: 0.95).  Use 1.0 for pure morphological
            decomposition (slow on large images).
        verbose: If True, print rectangle counts and elapsed time.

    Returns:
        List of non-overlapping rectangles covering all 1-pixels.

    """
    from src.algorithms.gdm import gdm_decomposition

    t0 = time.time()
    rects = morphological_decomposition(
        img, rng=None, coverage_threshold=coverage_threshold
    )

    if coverage_threshold < 1.0:
        # Build covered mask from morphological rectangles
        covered = np.zeros_like(img, dtype=np.uint8)
        for x, y, w, h in rects:
            covered[y:y + h, x:x + w] = 1

        # Residual: foreground pixels not yet covered
        residual = ((img == 1) & (covered == 0)).astype(np.uint8)

        if residual.any():
            gdm_rects = gdm_decomposition(residual, direction="auto")
            rects.extend(gdm_rects)

    if verbose:
        elapsed = time.time() - t0
        mode = (
            f"morphological+GDM (coverage={coverage_threshold})"
            if coverage_threshold < 1.0 else "morphological"
        )
        print(f"{mode}: {len(rects)} rectangles in {elapsed:.4f}s")
    return rects


def init_population_morphological(
    img: np.ndarray,
    integral: np.ndarray,
    pop_size: int,
) -> List[Chromosome]:
    """Initialize GA population using stochastic morphological decomposition.

    Each member of the population is produced by a fresh
    :class:`random.Random` instance seeded from ``os.urandom``, so
    every individual is independently randomised.

    Args:
        img: Binary image.
        integral: Precomputed integral image (unused here, kept for
            API consistency with other init functions).
        pop_size: Number of chromosomes to generate.

    Returns:
        List of initialized Chromosome objects.

    """
    from src.algorithms.genetic import repair

    population: List[Chromosome] = []
    integral = build_integral(img)
    rng = random.Random()
    seed_bytes = os.urandom(8)
    rng.seed(int.from_bytes(seed_bytes, "big"))
    rects = morphological_decomposition(img, rng=rng, coverage_threshold=0.95)
    for _ in range(pop_size):
        population.append(Chromosome(rects))
    return population
