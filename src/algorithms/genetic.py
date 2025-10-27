"""Genetic algorithm for binary image rectangle decomposition."""

import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from src.utils.types import Chromosome, Rectangle
from src.utils.utils import draw_solution


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


def fitness(chrom: Chromosome, img: np.ndarray, penalty: float) -> float:
    """Calculate fitness of chromosome solution.

    Balances rectangle count, coverage errors, and total area.

    Args:
        chrom: Chromosome with rectangles.
        img: Binary image to decompose.
        penalty: Penalty factor for extra coverage.

    Returns:
        Fitness score (higher is better).

    """
    covered = np.zeros_like(img)
    total_area = 0

    for x, y, w, h in chrom.rectangles:
        covered[y : y + h, x : x + w] = 1
        total_area += w * h

    missing = np.sum((img == 1) & (covered == 0))
    extra = np.sum((img == 0) & (covered == 1))

    if missing > 0:
        return -1e6  # invalid: not all 1s covered

    # Balance: fewer rectangles, fewer extras, larger rectangles
    return -len(chrom.rectangles) - penalty * extra + 0.01 * total_area


def repair(
    rects: List[Rectangle], img: np.ndarray, integral: np.ndarray
) -> List[Rectangle]:
    """Repair solution by removing overlaps and covering missing pixels.

    Args:
        rects: List of rectangles to repair.
        img: Binary image.
        integral: Integral image.

    Returns:
        Repaired list of valid non-overlapping rectangles.

    """
    covered = np.zeros_like(img, dtype=np.uint8)
    repaired = []

    # Keep valid non-overlapping rects
    for x, y, w, h in rects:
        if np.all(covered[y : y + h, x : x + w] == 0) and (
            is_valid_rectangle_integral(integral, (x, y, w, h))
        ):
            repaired.append((x, y, w, h))
            covered[y : y + h, x : x + w] = 1

    # Cover remaining uncovered pixels (where img == 1)
    ys, xs = np.where((img == 1) & (covered == 0))
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
        # Place the biggest rectangle found
        new_rect = (x, y, max_w, max_h)
        repaired.append(new_rect)
        covered[y : y + max_h, x : x + max_w] = 1
    return repaired


def init_population_rle(
    img: np.ndarray, integral: np.ndarray, pop_size: int
) -> List[Chromosome]:
    """Initialize GA rectangles using run-length decomposition.

    Decomposes by rows or columns depending on image shape.

    Args:
        img: Binary image.
        integral: Integral image.
        pop_size: Population size.

    Returns:
        List of initialized chromosomes.

    """
    height, width = img.shape
    population = []

    for _ in range(pop_size):
        rects = []

        mode = "row" if width >= height else "col"

        # ROW-wise decomposition
        if mode == "row":
            for y in range(height):
                x = 0
                while x < width:
                    if img[y, x] == 1:
                        x_start = x
                        while x < width and img[y, x] == 1:
                            x += 1
                        w = x - x_start
                        rect = (x_start, y, w, 1)
                        if is_valid_rectangle_integral(integral, rect):
                            rects.append(rect)
                    else:
                        x += 1

        # COLUMN-wise decomposition
        elif mode == "col":
            for x in range(width):
                y = 0
                while y < height:
                    if img[y, x] == 1:
                        y_start = y
                        while y < height and img[y, x] == 1:
                            y += 1
                        h = y - y_start
                        rect = (x, y_start, 1, h)
                        if is_valid_rectangle_integral(integral, rect):
                            rects.append(rect)
                    else:
                        y += 1

        # Shuffle for variation
        random.shuffle(rects)
        population.append(Chromosome(rects))

    return population


def init_population_random(
    img: np.ndarray,
    integral: np.ndarray,
    pop_size: int,
    max_attempts: int = 100,
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
    ones_count = np.sum(img)

    for _ in range(pop_size):
        rects = []
        covered = np.zeros_like(img)
        attempts = 0

        while np.sum(covered) < ones_count and attempts < max_attempts:
            # Randomly select top-left on uncovered 1
            ys, xs = np.where((img == 1) & (covered == 0))
            if len(xs) == 0:
                break
            idx = random.randrange(len(xs))
            x0, y0 = xs[idx], ys[idx]

            # Randomly select width and height
            max_w = width - x0
            max_h = height - y0
            w = random.randint(1, max_w)
            h = random.randint(1, max_h)

            rect = (x0, y0, w, h)
            if is_valid_rectangle_integral(integral, rect):
                rects.append(rect)
                covered[y0 : y0 + h, x0 : x0 + w] = 1

            attempts += 1

        # Repair remaining uncovered 1s
        rects = repair(rects, img, integral)
        population.append(Chromosome(rects))
    return population


def overlap(r1: Rectangle, r2: Rectangle) -> bool:
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


def crossover(
    p1: Chromosome,
    p2: Chromosome,
    img: np.ndarray,
    integral: np.ndarray,
) -> Chromosome:
    """Merge compatible genes from both parents into child.

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
        if not any(overlap(r, r2) for r2 in rects):
            if is_valid_rectangle_integral(integral, r):
                rects.append(r)

    rects = repair(rects, img, integral)
    return Chromosome(rects)


def mutate_geometry(
    rects: List[Rectangle],
    img: np.ndarray,
    integral: np.ndarray,
    max_step: int = 5,
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
        Rectangle list with merged adjacents.

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


def mutation(
    chrom: Chromosome,
    img: np.ndarray,
    integral: np.ndarray,
    p_geo: float = 0.05,
    p_merge: float = 0.05,
    p_local: float = 0.05,
    max_step: int = 5,
) -> "Chromosome":
    """Apply mutation operators to chromosome.

    Applies geometry, merge, and local repartition mutations.

    Args:
        chrom: Chromosome to mutate.
        img: Binary image.
        integral: Integral image.
        p_geo: Probability of geometry mutation.
        p_merge: Probability of merge mutation.
        p_local: Probability of local repartition.
        max_step: Maximum step for geometry mutation.

    Returns:
        Mutated chromosome.

    """
    rects = chrom.rectangles
    if not rects:
        return chrom

    # Local geometric tweak
    rects = mutate_geometry(rects, img, integral, max_step, p_geo)

    # Low-probability merge of compatible rectangles
    rects = mutate_merge(rects, integral, p_merge)

    # Local re-decomposition of a random patch
    rects = mutate_local_repartition(rects, img, integral, p_local)

    # Final repair to ensure feasibility
    rects = repair(rects, img, integral)

    return Chromosome(rects)


def run_ga(
    img: np.ndarray,
    pop_size=100,
    generations=10,
    elite_size=3,
    penalty=1.5,
    patience=10,
    seed=None,
    init_method="rle",
    verbose=True,
):
    """Run the Genetic Algorithm for binary image decomposition.

    Includes early stopping when best fitness stagnates.

    Args:
        img: Binary image to decompose (0s and 1s).
        pop_size: Population size.
        generations: Maximum generations.
        elite_size: Number of elites to keep.
        penalty: Penalty for extra coverage.
        patience: Generations without improvement before stopping.
        seed: Random seed for reproducibility (default: None).
        init_method: Population initialization method:
            "rle" (default), "random", or "quadtree".
        verbose: Print progress information (default: True).

    Returns:
        Tuple of (best_chromosome, generation_history).
        generation_history: List of best rectangle counts per generation.

    """
    import time

    start_time = time.time()

    # Set random seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    integral = build_integral(img)

    # Initialize population based on method
    if verbose:
        print(f"Initializing population (method: {init_method})...")

    if init_method == "rle":
        population = init_population_rle(img, integral, pop_size)
    elif init_method == "random":
        population = init_population_random(img, integral, pop_size)
    elif init_method == "quadtree":
        from src.algorithms.quadtree import init_population_quadtree

        population = init_population_quadtree(img, integral, pop_size)
    else:
        raise ValueError(
            f"Unknown init_method: {init_method}. "
            f"Use 'rle', 'random', or 'quadtree'."
        )

    if verbose:
        print(
            f"Initial population: {len(population[0].rectangles)} rectangles"
        )

    best_fitness = float("-inf")
    stagnant_generations = 0
    best_chrom = None
    generation_history = []  # Track rectangle counts per generation

    for g in range(generations):
        # Evaluate fitness
        for chrom in population:
            chrom.fitness = fitness(chrom, img, penalty)

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
            child = crossover(p1, p2, img, integral)
            child = mutation(child, img, integral)
            new_pop.append(child)

        population = new_pop

    execution_time = time.time() - start_time

    if verbose:
        print(f"\nCompleted in {execution_time:.2f} seconds")
        print(
            f"Best solution: {len(best_chrom.rectangles)} rectangles, "
            f"fitness={best_chrom.fitness:.2f}"
        )

    # Return best found (not just last generation)
    return best_chrom if best_chrom else population[0], generation_history


def main():
    """Run GA demo on sample binary image."""
    img = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    # Load the .npy file
    # Make sure it's 0s and 1s
    img = np.load(
        "../../res/figures/objects_binary/npy/crown-6_binary.npy"
    )
    # img = np.load(
    #     "../../res/figures/leafs_binary/npy/Vitis_riparia_5_binary.npy"
    # )
    # img = np.load(
    #     "../../docs/figures/leafs_binary/npy/Vitis_riparia_binary.npy"
    # )
    img = (img > 0).astype(int)
    # img = 1 - img  # if image is loaded we have to invert 0s and 1s

    # Display the image
    plt.imshow(img, cmap="gray")
    plt.title("Loaded Binary Image")
    plt.axis("off")
    plt.show()

    best, history = run_ga(img, pop_size=100, generations=100, seed=42)

    draw_solution(img, best.rectangles)


if __name__ == "__main__":
    main()
