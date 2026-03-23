"""Genetic algorithm for binary image rectangle decomposition."""

import random
import time
from typing import List
from scipy import ndimage

import matplotlib.pyplot as plt
import numpy as np

from src.utils.types import Chromosome, Rectangle
from src.utils.utils import draw_solution
from src.algorithms.rle import init_population_rle
from src.algorithms.quadtree import init_population_quadtree
from src.algorithms.morphological import (
    init_population_morphological,
    largest_rect_in_image,
)


def build_integral(img: np.ndarray) -> np.ndarray:
    """Build integral image for O(1) rectangle sum queries."""
    return img.cumsum(axis=0).cumsum(axis=1)


def rect_sum(integral: np.ndarray, x: int, y: int, w: int, h: int) -> int:
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


def is_valid_rectangle_integral(integral: np.ndarray, rect: Rectangle) -> bool:
    """Check if rectangle is valid within bounds and fully covered.

    Args:
        integral: Integral image array.
        rect: Rectangle tuple (x, y, width, height).

    Returns:
        True if rectangle is within bounds and fully covered.

    """
    x, y, w, h = rect
    # Boundary check
    if (
        x < 0
        or y < 0
        or x + w > integral.shape[1]
        or y + h > integral.shape[0]
    ):
        return False
    return rect_sum(integral, x, y, w, h) == w * h


def fitness(
    chrom: Chromosome,
    img: np.ndarray,
    penalty_extra: float = 2.0,
    penalty_overlap: float = 5.0,
    penalty_count: float = 10.0,
) -> float:
    """Calculate fitness of a chromosome using a two-phase hierarchical scoring.

    Phase 1 (incomplete coverage): drives evolution toward full coverage
    using a continuous gradient based on coverage ratio.
    Phase 2 (full coverage): optimizes for fewer rectangles, no background
    coverage, and no overlaps.

    Args:
        chrom: Chromosome with rectangles.
        img: Binary image to decompose (0s and 1s).
        penalty_extra: Penalty coefficient for pixels outside the object
            (background pixels covered by rectangles).
        penalty_overlap: Penalty coefficient for overlapping pixels
            (pixels covered by more than one rectangle).
        penalty_count: Penalty coefficient for each rectangle in the solution.

    Returns:
        Fitness score (higher is better).

    """
    covered = np.zeros_like(img, dtype=np.uint8)
    for x, y, w, h in chrom.rectangles:
        covered[y: y + h, x: x + w] += 1

    missing_pixels = np.sum((img == 1) & (covered == 0))
    extra_pixels = np.sum((img == 0) & (covered > 0))
    overlap_pixels = np.sum(covered > 1)
    total_obj_pixels = np.sum(img == 1)

    if missing_pixels > 0:
        # Phase 1: drive toward full coverage
        coverage_ratio = (total_obj_pixels - missing_pixels) / total_obj_pixels
        score = -1e6 * (1.0 - coverage_ratio) - (missing_pixels ** 2)
    else:
        # Phase 2: optimize within feasible region
        score = 10000.0
        score -= extra_pixels * penalty_extra
        score -= overlap_pixels * penalty_overlap
        score -= len(chrom.rectangles) * penalty_count

    return float(score)


def _build_covered(
    rects: List[Rectangle], img: np.ndarray
) -> np.ndarray:
    """Build a binary mask of pixels covered by the given rectangles."""
    covered = np.zeros_like(img, dtype=np.uint8)
    for x, y, w, h in rects:
        covered[y : y + h, x : x + w] = 1
    return covered


def _trim_to_uncovered(
    rect: Rectangle, covered: np.ndarray, integral: np.ndarray
) -> Rectangle | None:
    """Trim rect to fit within uncovered region using 4-directional clipping.

    Tries clipping from right, left, bottom, and top; returns the largest
    valid non-overlapping result, or None if no valid trim exists.

    Args:
        rect: Rectangle to trim.
        covered: Binary mask of already-covered pixels.
        integral: Integral image for validity checks.

    Returns:
        Largest valid trimmed rectangle, or None.

    """
    x, y, w, h = rect
    overlap = covered[y : y + h, x : x + w]

    # Per-column / per-row overlap flags
    cols = np.any(overlap, axis=0)   # shape (w,)
    rows = np.any(overlap, axis=1)   # shape (h,)
    overlap_cols = np.where(cols)[0]
    overlap_rows = np.where(rows)[0]

    candidates: List[Rectangle] = []

    # Trim from right: keep columns before first overlapping column
    new_w = int(overlap_cols[0])
    if new_w > 0:
        r = (x, y, new_w, h)
        if is_valid_rectangle_integral(integral, r):
            candidates.append(r)

    # Trim from left: keep columns after last overlapping column
    new_x = x + int(overlap_cols[-1]) + 1
    new_w = w - (int(overlap_cols[-1]) + 1)
    if new_w > 0:
        r = (new_x, y, new_w, h)
        if is_valid_rectangle_integral(integral, r):
            candidates.append(r)

    # Trim from bottom: keep rows before first overlapping row
    new_h = int(overlap_rows[0])
    if new_h > 0:
        r = (x, y, w, new_h)
        if is_valid_rectangle_integral(integral, r):
            candidates.append(r)

    # Trim from top: keep rows after last overlapping row
    new_y = y + int(overlap_rows[-1]) + 1
    new_h = h - (int(overlap_rows[-1]) + 1)
    if new_h > 0:
        r = (x, new_y, w, new_h)
        if is_valid_rectangle_integral(integral, r):
            candidates.append(r)

    if not candidates:
        return None
    return max(candidates, key=lambda r: r[2] * r[3])


def repair_overlaps(
    rects: List[Rectangle], img: np.ndarray, integral: np.ndarray
) -> List[Rectangle]:
    """Repair overlapping rectangles by trimming instead of discarding.

    Processes rectangles in order. For each rectangle that overlaps with
    already-covered pixels, tries to clip it from one of 4 directions
    (right, left, bottom, top) and keeps the largest valid result.
    Rectangles that cannot be trimmed to a valid non-overlapping form
    are discarded.

    Args:
        rects: List of rectangles to repair.
        img: Binary image.
        integral: Integral image.

    Returns:
        List of non-overlapping valid rectangles.

    """
    covered = np.zeros_like(img, dtype=np.uint8)
    repaired: List[Rectangle] = []

    for rect in rects:
        x, y, w, h = rect
        # Skip out-of-bounds rectangles
        if (
            x < 0
            or y < 0
            or x + w > img.shape[1]
            or y + h > img.shape[0]
        ):
            continue

        if np.all(covered[y : y + h, x : x + w] == 0):
            if is_valid_rectangle_integral(integral, rect):
                repaired.append(rect)
                covered[y : y + h, x : x + w] = 1
        else:
            trimmed = _trim_to_uncovered(rect, covered, integral)
            if trimmed is not None:
                tx, ty, tw, th = trimmed
                repaired.append(trimmed)
                covered[ty : ty + th, tx : tx + tw] = 1

    return repaired


def repair_coverage(
    rects: List[Rectangle],
    img: np.ndarray,
    integral: np.ndarray,
    p_coverage: float = 0.5,
) -> List[Rectangle]:
    """Cover uncovered foreground pixels with greedy rectangle expansion.

    Args:
        rects: List of already non-overlapping rectangles.
        img: Binary image.
        integral: Integral image.
        p_coverage: Fraction of uncovered pixels to attempt to cover
            (0.0–1.0). Values below 1.0 randomly subsample uncovered
            pixels, leaving the rest penalised by fitness instead.

    Returns:
        Extended list with additional rectangles covering uncovered pixels.

    """
    covered = _build_covered(rects, img)
    repaired = list(rects)

    ys, xs = np.where((img == 1) & (covered == 0))
    if len(ys) == 0:
        return repaired

    if p_coverage < 1.0:
        n = max(1, int(len(ys) * p_coverage))
        idx = np.random.permutation(len(ys))[:n]
        ys, xs = ys[idx], xs[idx]

    for x, y in zip(xs, ys):
        if covered[y, x] == 1:
            continue
        # Expand width
        w = 1
        while x + w < img.shape[1] and np.all(
            (img[y : y + 1, x : x + w + 1] == 1)
            & (covered[y : y + 1, x : x + w + 1] == 0)
        ):
            if is_valid_rectangle_integral(integral, (x, y, w + 1, 1)):
                w += 1
            else:
                break
        max_w = w
        # Expand height with chosen width
        h = 1
        while y + h < img.shape[0] and np.all(
            (img[y : y + h + 1, x : x + max_w] == 1)
            & (covered[y : y + h + 1, x : x + max_w] == 0)
        ):
            if is_valid_rectangle_integral(integral, (x, y, max_w, h + 1)):
                h += 1
            else:
                break
        max_h = h
        repaired.append((x, y, max_w, max_h))
        covered[y : y + max_h, x : x + max_w] = 1

    return repaired


def repair(
    rects: List[Rectangle], img: np.ndarray, integral: np.ndarray
) -> List[Rectangle]:
    """Repair solution: trim overlaps then cover remaining pixels.

    Thin wrapper combining repair_overlaps and repair_coverage.

    Args:
        rects: List of rectangles to repair.
        img: Binary image.
        integral: Integral image.

    Returns:
        Repaired list of valid non-overlapping rectangles.

    """
    rects = repair_overlaps(rects, img, integral)
    return repair_coverage(rects, img, integral)


def init_population_random(
    img: np.ndarray,
    integral: np.ndarray,
    pop_size: int,
    max_attempts: int = 1000,
) -> List[Chromosome]:
    """Initialize rectangles with random rectangles.

       Args:
           img: Binary image.
           integral: Integral image.
           pop_size: Population size.
           max_attempts: Max attempts per individual.

       Returns:
           List of initialized chromosomes.

   """
    height, width = img.shape
    population = []

    # Precompute all pixels with value 1
    ones = [tuple(p) for p in np.argwhere(img == 1)]

    for _ in range(pop_size):

        rects = []

        # Local copy of uncovered pixels
        uncovered = set(ones)

        attempts = 0

        while uncovered and attempts < max_attempts:

            # Random uncovered pixel
            y0, x0 = random.choice(tuple(uncovered))

            # Random rectangle size
            max_w = width - x0
            max_h = height - y0

            w = random.randint(1, max_w)
            h = random.randint(1, max_h)

            rect = (x0, y0, w, h)

            if is_valid_rectangle_integral(integral, rect):

                rects.append(rect)

                # Remove covered pixels from uncovered set
                for y in range(y0, y0 + h):
                    for x in range(x0, x0 + w):
                        if (y, x) in uncovered:
                            uncovered.remove((y, x))

            attempts += 1

        # Repair remaining uncovered pixels (full coverage during init)
        rects = repair_overlaps(rects, img, integral)
        rects = repair_coverage(rects, img, integral)

        population.append(Chromosome(rects))

    return population


def is_overlap(r1: Rectangle, r2: Rectangle) -> bool:
    """Check if two rectangles overlap.

    Args:
        r1: First rectangle.
        r2: Second rectangle.

    Returns:
        True if rectangles overlap.

    """
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return not (
        x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1
    )


def subset_greedy_crossover(
    p1: Chromosome,
    p2: Chromosome,
    img: np.ndarray,
    integral: np.ndarray,
) -> Chromosome:
    """Subset Crossover with Greedy Non-overlapping Extension.

    Merge compatible genes from both parents into child by selecting
    a random subset from Parent 1, then greedily adding non-overlapping
    rectangles from Parent 2.

    Args:
        p1: First parent chromosome.
        p2: Second parent chromosome.
        img: Binary image.
        integral: Integral image.

    Returns:
        New child chromosome.

    """
    rects = []
    if not p1.rectangles or not p2.rectangles:
        return Chromosome(p1.rectangles or p2.rectangles)

    # Select random subset from Parent 1
    subset_size = random.randint(1, len(p1.rectangles))
    rects.extend(random.sample(p1.rectangles, subset_size))

    # Try to add non-overlapping rectangles from Parent 2
    for r in p2.rectangles:
        if not any(is_overlap(r, r2) for r2 in rects):
            if is_valid_rectangle_integral(integral, r):
                rects.append(r)
    # rects = repair(rects, img, integral)

    return Chromosome(rects)


def single_point_crossover(
    p1: Chromosome,
    p2: Chromosome,
    img: np.ndarray,
    integral: np.ndarray,
) -> Chromosome:
    """Single-point crossover - split parents at random point and merge.

    Args:
        p1: First parent chromosome.
        p2: Second parent chromosome.
        img: Binary image.
        integral: Integral image.

    Returns:
        New child chromosome.

    """
    if not p1.rectangles or not p2.rectangles:
        return Chromosome(p1.rectangles or p2.rectangles)

    # Single-point crossover: choose random cut point for each parent
    cut1 = random.randint(0, len(p1.rectangles))
    cut2 = random.randint(0, len(p2.rectangles))

    # Take first part from p1, second part from p2
    rects = list(p1.rectangles[:cut1])

    # Try to add rectangles from second part of p2
    for r in p2.rectangles[cut2:]:
        if not any(is_overlap(r, r2) for r2 in rects):
            if is_valid_rectangle_integral(integral, r):
                rects.append(r)

    # Repair to ensure complete coverage
    # rects = repair(rects, img, integral)
    return Chromosome(rects)


def two_point_crossover(
    p1: Chromosome,
    p2: Chromosome,
    img: np.ndarray,
    integral: np.ndarray,
) -> Chromosome:
    """Two-point crossover - take middle section from one parent, ends from other.

    Args:
        p1: First parent chromosome.
        p2: Second parent chromosome.
        img: Binary image.
        integral: Integral image.

    Returns:
        New child chromosome.

    """
    if not p1.rectangles or not p2.rectangles:
        return Chromosome(p1.rectangles or p2.rectangles)

    # Two-point crossover: choose two cut points for each parent
    cut1_a, cut1_b = sorted(random.sample(range(len(p1.rectangles) + 1), 2))
    cut2_a, cut2_b = sorted(random.sample(range(len(p2.rectangles) + 1), 2))

    # Take middle section from p1, ends from p2
    rects = []

    # Add beginning from p2
    for r in p2.rectangles[:cut2_a]:
        if is_valid_rectangle_integral(integral, r):
            rects.append(r)

    # Add middle section from p1
    for r in p1.rectangles[cut1_a:cut1_b]:
        if not any(is_overlap(r, r2) for r2 in rects):
            if is_valid_rectangle_integral(integral, r):
                rects.append(r)

    # Add end from p2
    for r in p2.rectangles[cut2_b:]:
        if not any(is_overlap(r, r2) for r2 in rects):
            if is_valid_rectangle_integral(integral, r):
                rects.append(r)

    # Repair to ensure complete coverage
    # rects = repair(rects, img, integral)
    return Chromosome(rects)


def uniform_crossover(
    p1: Chromosome,
    p2: Chromosome,
    img: np.ndarray,
    integral: np.ndarray,
    p: float = 0.5,
) -> Chromosome:
    """Uniform crossover - each gene selected independently with probability p.

    Args:
        p1: First parent chromosome.
        p2: Second parent chromosome.
        img: Binary image.
        integral: Integral image.
        p: Probability of taking gene from p1 (default 0.5).

    Returns:
        New child chromosome.

    """
    if not p1.rectangles or not p2.rectangles:
        return Chromosome(p1.rectangles or p2.rectangles)

    rects = []

    # Process each rectangle from p1 with probability p
    for r in p1.rectangles:
        if random.random() < p:
            if not any(is_overlap(r, r2) for r2 in rects):
                if is_valid_rectangle_integral(integral, r):
                    rects.append(r)

    # Process each rectangle from p2 with probability (1-p)
    for r in p2.rectangles:
        if random.random() < (1 - p):
            if not any(is_overlap(r, r2) for r2 in rects):
                if is_valid_rectangle_integral(integral, r):
                    rects.append(r)

    # Repair to ensure complete coverage
    # rects = repair(rects, img, integral)

    return Chromosome(rects)


def mutate_geometry(
    rects: List[Rectangle],
    img: np.ndarray,
    integral: np.ndarray,
    max_step: int = 50,
    p: float = 0.2,
) -> List[Rectangle]:
    """Expand or shrink one rectangle randomly.

    Args:
        rects: List of rectangles.
        img: Binary image.
        integral: Integral image.
        max_step: Maximum step size.
        p: Probability of mutation.

    Returns:
        Mutated rectangle list.

    """
    if not rects or random.random() >= p:
        return rects

    rects = rects.copy()
    idx = random.randrange(len(rects))
    x, y, w, h = rects[idx]
    height, width = img.shape
    step = random.randint(1, max_step)
    choice = random.choice(["expand_w", "expand_h", "shrink_w", "shrink_h"])

    new_rect = (x, y, w, h)
    if choice == "expand_w":
        new_w = min(w + step, width - x)
        new_rect = (x, y, new_w, h)
    elif choice == "expand_h":
        new_h = min(h + step, height - y)
        new_rect = (x, y, w, new_h)
    elif choice == "shrink_w" and w > step:
        new_rect = (x, y, w - step, h)
    elif choice == "shrink_h" and h > step:
        new_rect = (x, y, w, h - step)

    # accept only valid rectangles
    if is_valid_rectangle_integral(integral, new_rect):
        rects[idx] = new_rect
    # rects = repair(rects, img, integral)
    return rects


def mutate_merge(
    rects: List[Rectangle],
    integral: np.ndarray,
    p_merge: float = 0.05,
) -> List[Rectangle]:
    """Merge adjacent rectangles probabilistically.

    Args:
        rects: List of rectangles.
        integral: Integral image.
        p_merge: Probability of merging.

    Returns:
        Rectangle list with merged adjacent.

    """
    rects = rects.copy()
    i = 0
    while i < len(rects):
        r1 = rects[i]
        j = i + 1
        while j < len(rects):
            r2 = rects[j]
            horiz_adj = (
                r1[1] == r2[1]
                and r1[3] == r2[3]
                and (r1[0] + r1[2] == r2[0] or r2[0] + r2[2] == r1[0])
            )
            vert_adj = (
                r1[0] == r2[0]
                and r1[2] == r2[2]
                and (r1[1] + r1[3] == r2[1] or r2[1] + r2[3] == r1[1])
            )

            if (horiz_adj or vert_adj) and random.random() < p_merge:
                # merge attempt
                x = min(r1[0], r2[0])
                y = min(r1[1], r2[1])
                w = max(r1[0] + r1[2], r2[0] + r2[2]) - x
                h = max(r1[1] + r1[3], r2[1] + r2[3]) - y
                new_rect = (x, y, w, h)

                if is_valid_rectangle_integral(integral, new_rect):
                    rects[i] = new_rect
                    rects.pop(j)
                    j = i + 1
                    continue
            j += 1
        i += 1
    return rects


def mutate_merge_v2(
    rects: List[Rectangle],
    integral: np.ndarray,
    p_merge: float = 0.05,
) -> List[Rectangle]:
    """Merge randomly selected pairs of adjacent rectangles.

    Repeats up to n_attempts times: picks one rectangle at random,
    finds its adjacent neighbours, picks one at random and attempts to
    merge them.  Each successful merge reduces the rectangle count by
    one and updates the list for the next attempt.

    Args:
        rects: List of rectangles.
        integral: Integral image.
        p_merge: Probability that the mutation fires at all.
        n_attempts: Maximum number of merge attempts per call.

    Returns:
        Rectangle list with up to n_attempts merges applied.

    """
    if not rects or random.random() >= p_merge:
        return rects

    rects = rects.copy()
    n_attempts = random.randint(1, max(1, len(rects) // 2))
    for _ in range(n_attempts):
        if len(rects) < 2:
            break

        idx = random.randrange(len(rects))
        r1 = rects[idx]

        adjacent = []
        for j, r2 in enumerate(rects):
            if j == idx:
                continue
            horiz_adj = (
                r1[1] == r2[1]
                and r1[3] == r2[3]
                and (r1[0] + r1[2] == r2[0] or r2[0] + r2[2] == r1[0])
            )
            vert_adj = (
                r1[0] == r2[0]
                and r1[2] == r2[2]
                and (r1[1] + r1[3] == r2[1] or r2[1] + r2[3] == r1[1])
            )
            if horiz_adj or vert_adj:
                adjacent.append(j)

        if not adjacent:
            continue

        j = random.choice(adjacent)
        r2 = rects[j]

        x = min(r1[0], r2[0])
        y = min(r1[1], r2[1])
        w = max(r1[0] + r1[2], r2[0] + r2[2]) - x
        h = max(r1[1] + r1[3], r2[1] + r2[3]) - y
        new_rect = (x, y, w, h)

        if is_valid_rectangle_integral(integral, new_rect):
            high, low = (idx, j) if idx > j else (j, idx)
            rects.pop(high)
            rects.pop(low)
            rects.append(new_rect)

    return rects


def local_decomposition(
    sub_img: np.ndarray,
    sub_integral: np.ndarray,
    offset_x: int,
    offset_y: int,
):
    """Decompose local binary patch heuristically.

    Simple bounding box approach for local region.

    Args:
        sub_img: Sub-image patch.
        sub_integral: Sub-integral image.
        offset_x: X offset in full image.
        offset_y: Y offset in full image.

    Returns:
        List of rectangles for this patch.

    """
    rects = []
    ys, xs = np.where(sub_img == 1)
    if len(xs) == 0:
        return rects
    # Naive rectangular cover: bounding box
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    rects.append(
        (
            offset_x + x_min,
            offset_y + y_min,
            x_max - x_min + 1,
            y_max - y_min + 1,
        )
    )
    return rects


def mutate_local_repartition(
    rects: List[Rectangle],
    img: np.ndarray,
    integral: np.ndarray,
    p_local: float = 0.1,
    max_area: int = 200,
) -> List[Rectangle]:
    """Re-decompose a local area.

    Removes rectangles in region and re-decomposes.

    Args:
        rects: List of rectangles.
        img: Binary image.
        integral: Integral image.
        p_local: Probability of local repartition.
        max_area: Maximum area (unused).

    Returns:
        Modified rectangle list.

    """
    if not rects or random.random() >= p_local:
        return rects

    rects = rects.copy()
    height, width = img.shape

    # Choose random center region
    cx = random.randint(0, width - 1)
    cy = random.randint(0, height - 1)
    region_size = random.randint(10, 30)
    x1, y1 = (
        max(0, cx - region_size),
        max(0, cy - region_size),
    )
    x2, y2 = (
        min(width, cx + region_size),
        min(height, cy + region_size),
    )

    # Remove rectangles fully inside region
    remaining = []
    for r in rects:
        rx, ry, rw, rh = r
        if rx >= x1 and ry >= y1 and rx + rw <= x2 and ry + rh <= y2:
            continue
        remaining.append(r)

    # Re-decompose that local patch
    sub_img = img[y1:y2, x1:x2]
    new_rects = local_decomposition(sub_img, integral[y1:y2, x1:x2], x1, y1)
    return remaining + new_rects


def mutate_local_repartition_v2(
    rects: List[Rectangle],
    img: np.ndarray,
    integral: np.ndarray,
    p_local: float = 0.1,
) -> List[Rectangle]:
    """Re-decompose a local area using random greedy coverage.

    Selects a random region, removes rectangles fully contained within
    it, then re-covers the freed pixels with randomly generated valid
    rectangles.  Unlike mutate_local_repartition the re-decomposition
    is stochastic: starting pixels are shuffled and rectangle sizes are
    chosen randomly, so each call produces a different result for the
    same region.

    Args:
        rects: List of rectangles.
        img: Binary image.
        integral: Integral image.
        p_local: Probability that the mutation fires at all.

    Returns:
        Modified rectangle list.

    """
    if not rects or random.random() >= p_local:
        return rects

    rects = rects.copy()
    height, width = img.shape

    cx = random.randint(0, width - 1)
    cy = random.randint(0, height - 1)
    region_size = random.randint(10, 30)
    x1 = max(0, cx - region_size)
    y1 = max(0, cy - region_size)
    x2 = min(width, cx + region_size)
    y2 = min(height, cy + region_size)

    remaining = [
        r for r in rects
        if not (
            r[0] >= x1 and r[1] >= y1
            and r[0] + r[2] <= x2 and r[1] + r[3] <= y2
        )
    ]

    # Build coverage mask from rectangles that survived removal
    combined_covered = np.zeros_like(img, dtype=np.uint8)
    for rx, ry, rw, rh in remaining:
        combined_covered[ry: ry + rh, rx: rx + rw] = 1

    # Uncovered 1-pixels inside the region
    region_mask = (
        (img[y1:y2, x1:x2] == 1)
        & (combined_covered[y1:y2, x1:x2] == 0)
    )
    ys, xs = np.where(region_mask)
    if len(xs) == 0:
        return remaining

    # Shuffle starting pixels for random greedy coverage
    candidates = list(zip((xs + x1).tolist(), (ys + y1).tolist()))
    random.shuffle(candidates)

    new_rects = []
    for x0, y0 in candidates:
        if combined_covered[y0, x0] == 1:
            continue

        max_w = x2 - x0
        max_h = y2 - y0
        w = random.randint(1, max_w)
        h = random.randint(1, max_h)
        rect = (x0, y0, w, h)

        if (
            is_valid_rectangle_integral(integral, rect)
            and np.all(combined_covered[y0: y0 + h, x0: x0 + w] == 0)
        ):
            new_rects.append(rect)
            combined_covered[y0: y0 + h, x0: x0 + w] = 1
    rects = remaining + new_rects
    # rects = repair(rects, img, integral)
    return rects


def mutate_largest_rect(
    rects: List[Rectangle],
    img: np.ndarray,
    integral: np.ndarray,
    p_largest: float = 0.05,
) -> List[Rectangle]:
    """Replace part of a region with the largest uncovered rectangle.

    Picks a random region, removes rectangles fully inside it, then
    finds the largest valid rectangle among the remaining uncovered
    foreground pixels and appends it.  The ``repair()`` call in the
    outer mutation pipeline fills any remaining gaps.

    Args:
        rects: List of rectangles.
        img: Binary image.
        integral: Integral image (kept for API consistency).
        p_largest: Probability that the mutation fires at all.

    Returns:
        Modified rectangle list.

    """
    if not rects or random.random() >= p_largest:
        return rects

    rects = rects.copy()
    height, width = img.shape
    cx = random.randint(0, width - 1)
    cy = random.randint(0, height - 1)
    region_size = random.randint(10, 50)
    x1 = max(0, cx - region_size)
    y1 = max(0, cy - region_size)
    x2 = min(width, cx + region_size)
    y2 = min(height, cy + region_size)

    remaining = [
        r for r in rects
        if not (
            r[0] >= x1 and r[1] >= y1
            and r[0] + r[2] <= x2 and r[1] + r[3] <= y2
        )
    ]

    # Build covered mask from ALL surviving rects to avoid overlap
    covered = np.zeros_like(img, dtype=np.uint8)
    for rx, ry, rw, rh in remaining:
        covered[ry: ry + rh, rx: rx + rw] = 1

    best = largest_rect_in_image(img, covered)
    if best is not None:
        remaining = remaining + [best]

    return remaining


def mutate_split(
    rects: List[Rectangle],
    integral: np.ndarray,
    p_split: float = 0.05,
) -> List[Rectangle]:
    """Split one randomly selected rectangle into two along a random axis.

    Picks one rectangle at random and divides it either horizontally or
    vertically at a random cut point.  Both halves are guaranteed valid
    because any sub-rectangle of a fully-covered rectangle is also
    fully covered.  This is the inverse operation of mutate_merge and
    enables the search to escape local optima where a single large
    rectangle blocks a better configuration.

    Args:
        rects: List of rectangles.
        integral: Integral image.
        p_split: Probability that the mutation fires at all.

    Returns:
        Rectangle list with one rectangle replaced by two.

    """
    if not rects or random.random() >= p_split:
        return rects

    rects = rects.copy()
    idx = random.randrange(len(rects))
    x, y, w, h = rects[idx]

    can_split_h = w > 1
    can_split_v = h > 1

    if not can_split_h and not can_split_v:
        return rects

    axes = []
    if can_split_h:
        axes.append("h")
    if can_split_v:
        axes.append("v")
    axis = random.choice(axes)

    if axis == "h":
        k = random.randint(1, w - 1)
        r1 = (x, y, k, h)
        r2 = (x + k, y, w - k, h)
    else:
        k = random.randint(1, h - 1)
        r1 = (x, y, w, k)
        r2 = (x, y + k, w, h - k)

    rects.pop(idx)
    rects.extend([r1, r2])
    return rects


def mutate_shift(
    rects: List[Rectangle],
    img: np.ndarray,
    integral: np.ndarray,
    p_shift: float = 0.05,
    max_step: int = 5,
) -> List[Rectangle]:
    """Translate one randomly selected rectangle by a random offset.

    Unlike mutate_geometry which changes rectangle size, this mutation
    moves the rectangle to a different position while keeping its
    dimensions unchanged.  Only valid translations (within image bounds
    and covering only 1-pixels) are accepted.

    Args:
        rects: List of rectangles.
        img: Binary image.
        integral: Integral image.
        p_shift: Probability that the mutation fires at all.
        max_step: Maximum translation step in pixels.

    Returns:
        Rectangle list with one rectangle potentially moved.

    """
    if not rects or random.random() >= p_shift:
        return rects

    rects = rects.copy()
    idx = random.randrange(len(rects))
    x, y, w, h = rects[idx]

    dx = random.randint(-max_step, max_step)
    dy = random.randint(-max_step, max_step)
    new_rect = (x + dx, y + dy, w, h)

    if is_valid_rectangle_integral(integral, new_rect):
        rects[idx] = new_rect

    return rects


def mutate_delete(
    rects: List[Rectangle],
    p_delete: float = 0.05,
) -> List[Rectangle]:
    """Remove one randomly selected rectangle from the solution.

    Deleting a rectangle creates uncovered pixels that repair() will
    fill with a fresh local decomposition.  This forces the search to
    explore alternative decompositions for the freed region without
    being constrained by the deleted rectangle's boundaries.

    Args:
        rects: List of rectangles.
        p_delete: Probability that the mutation fires at all.

    Returns:
        Rectangle list with one rectangle removed.

    """
    if len(rects) <= 1 or random.random() >= p_delete:
        return rects

    rects = rects.copy()
    idx = random.randrange(len(rects))
    rects.pop(idx)
    return rects


def mutation(
    chrom: Chromosome,
    img: np.ndarray,
    integral: np.ndarray,
    p_geo: float = 0.05,
    p_merge: float = 0.05,
    p_local: float = 0.05,
    p_split: float = 0.05,
    p_shift: float = 0.05,
    p_delete: float = 0.05,
    p_largest: float = 0.05,
    p_repair: float = 0.5,
    max_step: int = 5,
) -> "Chromosome":
    """Apply mutation operators to chromosome.

    Pipeline order:
    1. delete        — remove a rectangle, creating gaps for repair
    2. split         — break a rectangle into two
    3. geometry      — resize one rectangle
    4. shift         — translate one rectangle
    5. local         — re-decompose a random region
    6. largest_rect  — replace region with largest uncovered rectangle
    7. repair_overlaps — trim overlapping rectangles (always applied)
    8. repair_coverage — fill uncovered pixels (applied with prob p_repair)
    9. merge         — consolidate adjacent rectangles

    Args:
        chrom: Chromosome to mutate.
        img: Binary image.
        integral: Integral image.
        p_geo: Probability of geometry mutation.
        p_merge: Probability of merge mutation.
        p_local: Probability of local repartition mutation.
        p_split: Probability of split mutation.
        p_shift: Probability of shift mutation.
        p_delete: Probability of delete mutation.
        p_largest: Probability of largest-rect mutation.
        p_repair: Probability of applying coverage repair after mutations.
            Overlap trimming is always applied regardless of this value.
            Set to 1.0 (default) to always repair full coverage, or lower
            to let fitness penalise uncovered pixels instead.
        max_step: Maximum step size for geometry and shift mutations.

    Returns:
        Mutated chromosome.

    """
    rects = chrom.rectangles
    if not rects:
        return chrom

    rects = mutate_delete(rects, p_delete)
    rects = mutate_split(rects, integral, p_split)
    rects = mutate_geometry(rects, img, integral, max_step, p_geo)
    rects = mutate_shift(rects, img, integral, p_shift, max_step)
    rects = mutate_local_repartition_v2(rects, img, integral, p_local)
    rects = mutate_largest_rect(rects, img, integral, p_largest)
    rects = mutate_merge_v2(rects, integral, p_merge)
    rects = repair_overlaps(rects, img, integral)
    if random.random() < p_repair:
        rects = repair_coverage(rects, img, integral)

    return Chromosome(rects)


def run_ga(
    img: np.ndarray,
    pop_size=20,
    generations=100,
    elite_size=3,
    penalty_extra=2.0,
    penalty_overlap=5.0,
    penalty_count=10.0,
    patience=10,
    seed=None,
    init_method="rle",
    verbose=False,
    mutation_geometry=0.05,
    mutation_merge=0.05,
    mutation_local=0.05,
    mutation_split=0.05,
    mutation_shift=0.05,
    mutation_delete=0.05,
    mutation_largest=0.05,
    repair_coverage_prob=0.5,
    crossover_method="subset_greedy",
):
    """Run the Genetic Algorithm for binary image decomposition.

    Includes early stopping when best fitness stagnates.

    Args:
        img: Binary image to decompose (0s and 1s).
        pop_size: Population size.
        generations: Maximum generations.
        elite_size: Number of elites to keep.
        penalty_extra: Penalty coefficient for background pixels covered by
            rectangles (extra coverage).
        penalty_overlap: Penalty coefficient for pixels covered by more than
            one rectangle (default: 5.0).
        penalty_count: Penalty coefficient per rectangle in the solution
            (default: 10.0).
        patience: Generations without improvement before stopping.
        seed: Random seed for reproducibility (default: None).
        init_method: Population initialization method:
            "rle" (default), "random", "quadtree", "morphological",
            or "mixed" (pop_size//3 each of rle, morphological, random).
        verbose: Print progress information (default: True).
        mutation_geometry: Probability of geometry mutation (G)
            (default: 0.05). Set to 0.0 to disable.
        mutation_merge: Probability of merge mutation (M)
            (default: 0.05). Set to 0.0 to disable.
        mutation_local: Probability of local repartition mutation (L)
            (default: 0.05). Set to 0.0 to disable.
        mutation_split: Probability of split mutation (default: 0.05).
            Splits one rectangle into two along a random axis.
        mutation_shift: Probability of shift mutation (default: 0.05).
            Translates one rectangle to a new position.
        mutation_delete: Probability of delete mutation (default: 0.05).
            Removes one rectangle; repair() refills the freed region.
        mutation_largest: Probability of largest-rect mutation (R)
            (default: 0.05). Replaces a region with the largest
            uncovered rectangle found by the morphological scan.
        repair_coverage_prob: Probability of applying coverage repair after
            each mutation (default: 1.0). Overlap trimming is always applied.
            Values below 1.0 allow uncovered pixels to persist, penalised
            by fitness, which promotes more exploratory mutations.
        crossover_method: Crossover method to use (default: "subset_greedy").
            Options: "subset_greedy" (Subset Crossover with Greedy
            Non-overlapping Extension), "single_point", "two_point",
            "uniform".

    Returns:
        Tuple of (best_chromosome, generation_history).
        generation_history: List of best rectangle counts per generation.

    """
    start_time = time.time()

    # Set random seeds for reproducibility
    if seed is None:
        seed = random.randint(0, 2 ** 31 - 1)
    random.seed(seed)
    np.random.seed(seed)

    if verbose:
        print(f"Seed: {seed}")

    integral = build_integral(img)

    # Map crossover method name to function
    crossover_methods = {
        "subset_greedy": subset_greedy_crossover,
        "single_point": single_point_crossover,
        "two_point": two_point_crossover,
        "uniform": uniform_crossover,
    }

    if crossover_method not in crossover_methods:
        raise ValueError(
            f"Unknown crossover_method: {crossover_method}. "
            f"Use 'subset_greedy', 'single_point', 'two_point', or 'uniform'."
        )

    crossover_fn = crossover_methods[crossover_method]

    # Initialize population based on method
    if verbose:
        print(f"Initializing population (method: {init_method})...")

    if init_method == "rle":
        population = init_population_rle(img, integral, pop_size)
    elif init_method == "random":
        population = init_population_random(img, integral, pop_size)
    elif init_method == "quadtree":
        population = init_population_quadtree(img, integral, pop_size)
    elif init_method == "morphological":
        population = init_population_morphological(img, integral, pop_size)
    elif init_method == "mixed":
        n_rle = pop_size // 3
        n_morph = pop_size // 3
        n_rand = pop_size - n_rle - n_morph
        population = (
            init_population_rle(img, integral, n_rle)
            + init_population_morphological(img, integral, n_morph)
            + init_population_random(img, integral, n_rand)
        )
    else:
        raise ValueError(
            f"Unknown init_method: {init_method}. "
            f"Use 'rle', 'random', 'quadtree', 'morphological', or 'mixed'."
        )

    if verbose:
        print(f"Initial population: {len(population[0].rectangles)} rectangles")
        # draw_solution(img, population[0], show=True)

    best_fitness = float("-inf")
    stagnant_generations = 0
    best_chrom = None
    generation_history = []  # Track rectangle counts per generation

    for g in range(generations):
        # Evaluate fitness
        for chrom in population:
            chrom.fitness = fitness(chrom, img, penalty_extra, penalty_overlap, penalty_count)

        # Sort by fitness descending
        population.sort(key=lambda c: c.fitness, reverse=True)
        best = population[0]

        # Track history
        generation_history.append(len(best.rectangles))

        if verbose:
            print(
                f"Gen {g:3d}: Fitness={best.fitness:7.2f}, "
                f"Rects={len(best.rectangles):3d}"
            )

        # Track improvement
        if best.fitness > best_fitness:
            best_fitness = best.fitness
            stagnant_generations = 0
            best_chrom = best
        else:
            stagnant_generations += 1

        # Early stopping check
        if stagnant_generations >= patience:
            if verbose:
                print(
                    f"Early stopping at generation {g}: "
                    f"No improvement for {patience} generations."
                )
            break

        # Elitism: keep the best few chromosomes
        new_pop = population[:elite_size]

        # Generate offspring
        while len(new_pop) < pop_size:
            top_candidates = population[: len(population) // 10]
            p1, p2 = random.sample(top_candidates, 2)
            child = crossover_fn(p1, p2, img, integral)
            # Apply mutations with configured probabilities
            child = mutation(
                child,
                img,
                integral,
                p_geo=mutation_geometry,
                p_merge=mutation_merge,
                p_local=mutation_local,
                p_split=mutation_split,
                p_shift=mutation_shift,
                p_delete=mutation_delete,
                p_largest=mutation_largest,
                p_repair=repair_coverage_prob,
            )
            new_pop.append(child)

        population = new_pop

    execution_time = time.time() - start_time

    if verbose:
        print(f"\nCompleted in {execution_time:.2f} seconds")
        print(f"Best solution: {len(best_chrom.rectangles)} rectangles, "
            f"fitness={best_chrom.fitness:.2f}"
        )

    # Return best found (not just last generation)
    return best_chrom if best_chrom else population[0], generation_history


def main():
    """Run GA demo on sample binary image."""

    img = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])

    img_bugged_1 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    img_bugged_2 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    img_large = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    img_bugged_5 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    img_holes_1 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 0, 1, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    # Load the .npy file
    img_loaded = np.load("../../data/datasets/objects_binary/npy/crown-3_binary.npy")
    # img_loaded = np.load("../../data/datasets/research_leafs_binary/npy/Acer_campestre_1_binary.npy")
    # img_loaded = np.load("../../data/datasets/objects_binary/npy/hat-5_binary.npy")
    # img_loaded = np.load("../../data/datasets/objects_binary/npy/butterfly-4_binary.npy")

    # img_loaded = np.load("../../data/datasets/objects_binary/npy/hat-20_binary.npy")

    img = img_loaded
    img = (img > 0).astype(int)

    # Display the image
    plt.imshow(img, cmap="gray")
    plt.title("Loaded Binary Image")
    plt.axis("off")
    plt.show()

    print(img.shape)

    best, history = run_ga(
        img,
        pop_size=30,
        generations=500,
        patience=10,
        seed=None,
        init_method="morphological",
        verbose=True,
        mutation_geometry=0.3,
        mutation_merge=0.2,
        mutation_local=0.4,
        mutation_split=0.2,
        mutation_shift=0.5,
        mutation_delete=0.2,
        mutation_largest=0.3,
        repair_coverage_prob=0.2,
        crossover_method="subset_greedy"
    )

    draw_solution(img, best.rectangles, show=True)


if __name__ == "__main__":
    main()
