import numpy as np
import random
from typing import List, Tuple
import matplotlib.pyplot as plt

Rectangle = Tuple[int, int, int, int]  # (x, y, width, height)


def is_valid_rectangle(img: np.ndarray, rect: Rectangle) -> bool:
    """Check if the given rectangle area contains all ones."""
    x, y, w, h = rect
    if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
        return False
    sub = img[y:y + h, x:x + w]
    return np.all(sub == 1)


def overlap(r1: Rectangle, r2: Rectangle) -> bool:
    """"""
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


class Chromosome:
    """"""
    def __init__(self, rectangles: List[Rectangle]):
        """"""
        self.rectangles = rectangles
        self.fitness = None


def fitness(chrom: Chromosome, img: np.ndarray) -> float:
    """"""
    covered = np.zeros_like(img)
    total_area = 0
    penalty = 0.5

    for (x, y, w, h) in chrom.rectangles:
        covered[y:y + h, x:x + w] = 1
        total_area += w * h

    missing = np.sum((img == 1) & (covered == 0))
    extra = np.sum((img == 0) & (covered == 1))

    if missing > 0:
        return -1e6  # invalid: not all 1s covered

    # balance between fewer rectangles, fewer extras, and larger rectangles
    return -len(chrom.rectangles) - penalty - extra + 0.01 * total_area


def repair(rects: List[Rectangle], img: np.ndarray, integral: np.ndarray) -> List[Rectangle]:
    covered = np.zeros_like(img, dtype=np.uint8)
    repaired = []

    # Keep valid non-overlapping rects
    for x, y, w, h in rects:
        if np.all(covered[y:y+h, x:x+w] == 0) and is_valid_rectangle_integral(integral, (x, y, w, h)):
            repaired.append((x, y, w, h))
            covered[y:y+h, x:x+w] = 1

    # Cover remaining uncovered pixels (where img == 1)
    ys, xs = np.where((img == 1) & (covered == 0))

    for x, y in zip(xs, ys):
        if covered[y, x] == 1:
            continue

        # Try to expand to a bigger rectangle
        max_w, max_h = 1, 1

        # Expand width
        w = 1
        while x + w < img.shape[1] and np.all((img[y:y+1, x:x+w+1] == 1) & (covered[y:y+1, x:x+w+1] == 0)):
            if is_valid_rectangle_integral(integral, (x, y, w+1, 1)):
                w += 1
            else:
                break
        max_w = w

        # Expand height with chosen width
        h = 1
        while y + h < img.shape[0] and np.all((img[y:y+h+1, x:x+max_w] == 1) & (covered[y:y+h+1, x:x+max_w] == 0)):
            if is_valid_rectangle_integral(integral, (x, y, max_w, h+1)):
                h += 1
            else:
                break
        max_h = h

        # Place the biggest rectangle found
        new_rect = (x, y, max_w, max_h)
        repaired.append(new_rect)
        covered[y:y+max_h, x:x+max_w] = 1

    return repaired


def build_integral(img: np.ndarray) -> np.ndarray:
    """Build integral image for O(1) rectangle sum queries."""
    return img.cumsum(axis=0).cumsum(axis=1)


def rect_sum(integral: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    """Return sum of rectangle (x, y, w, h) using integral image."""
    x2, y2 = x + w - 1, y + h - 1
    total = integral[y2, x2]
    if x > 0:
        total -= integral[y2, x-1]
    if y > 0:
        total -= integral[y-1, x2]
    if x > 0 and y > 0:
        total += integral[y-1, x-1]
    return total


def is_valid_rectangle_integral(integral: np.ndarray, rect: Rectangle) -> bool:
    x, y, w, h = rect
    return rect_sum(integral, x, y, w, h) == w * h


def init_population_by_rows(img: np.ndarray, pop_size: int) -> List[Chromosome]:
    height, width = img.shape
    integral = build_integral(img)
    population = []

    for _ in range(pop_size):
        rects = []
        # Scan each row
        for y in range(height):
            x = 0
            while x < width:
                # Find start of consecutive 1s
                if img[y, x] == 1:
                    x_start = x
                    while x < width and img[y, x] == 1:
                        x += 1
                    w = x - x_start
                    rect = (x_start, y, w, 1)
                    # Only add valid rectangles
                    if is_valid_rectangle_integral(integral, rect):
                        rects.append(rect)
                else:
                    x += 1

        # Optional: shuffle rectangles slightly for diversity
        random.shuffle(rects)
        population.append(Chromosome(rects))
    return population


def init_population_random(img: np.ndarray, pop_size: int, max_attempts: int = 100) -> List[Chromosome]:
    height, width = img.shape
    population = []
    integral = build_integral(img)
    ones_count = np.sum(img)

    for _ in range(pop_size):
        rects = []
        covered = np.zeros_like(img)
        attempts = 0

        while np.sum(covered) < ones_count and attempts < max_attempts:
            # Randomly select a top-left corner on a 1 that is not covered
            ys, xs = np.where((img == 1) & (covered == 0))
            if len(xs) == 0:
                break
            idx = random.randrange(len(xs))
            x0, y0 = xs[idx], ys[idx]

            # Randomly select width and height biased toward bigger rectangles
            max_w = width - x0
            max_h = height - y0
            w = random.randint(1, max_w)
            h = random.randint(1, max_h)

            rect = (x0, y0, w, h)
            if is_valid_rectangle_integral(integral, rect):
                rects.append(rect)
                covered[y0:y0 + h, x0:x0 + w] = 1

            attempts += 1

        # Repair remaining uncovered 1s with 1x1 rectangles
        rects = repair(rects, img, integral)
        population.append(Chromosome(rects))

    return population


def mutate(chrom: Chromosome, img: np.ndarray, integral: np.ndarray,
    p: float = 0.2, p_merge: float = 0.6, max_step: int = 3) -> Chromosome:
    rects = chrom.rectangles.copy()
    if not rects:
        return chrom

    # Expand/shrink mutation
    if random.random() < p:
        idx = random.randrange(len(rects))
        x, y, w, h = rects[idx]

        choice = random.choice(["expand_w", "expand_h", "shrink_w", "shrink_h"])
        step = random.randint(1, max_step)  # bigger random step

        new_rect = (x, y, w, h)  # fallback

        H, W = img.shape  # height, width

        if choice == "expand_w":
            new_w = min(w + step, W - x)
            new_rect = (x, y, new_w, h)
        elif choice == "expand_h":
            new_h = min(h + step, H - y)
            new_rect = (x, y, w, new_h)
        elif choice == "shrink_w" and w > step:
            new_rect = (x, y, w - step, h)
        elif choice == "shrink_h" and h > step:
            new_rect = (x, y, w, h - step)

        # Only accept valid rectangles
        if is_valid_rectangle_integral(integral, new_rect):
            rects[idx] = new_rect

    # Probabilistic merging of neighbors
    rects = merge_rectangles_probabilistic(rects, p_merge)

    # Repair remaining uncovered 1s
    rects = repair(rects, img, integral)
    return Chromosome(rects)


def crossover(p1: Chromosome, p2: Chromosome, img: np.ndarray, integral: np.ndarray) -> Chromosome:
    rects = []
    if not p1.rectangles or not p2.rectangles:
        return Chromosome(p1.rectangles or p2.rectangles)

    subset_size = random.randint(1, len(p1.rectangles))
    rects.extend(random.sample(p1.rectangles, subset_size))
    for r in p2.rectangles:
        if not any(overlap(r, r2) for r2 in rects):
            if is_valid_rectangle_integral(integral, r):
                rects.append(r)

    rects = repair(rects, img, integral)
    return Chromosome(rects)


def merge_adjacent_rectangles(rects: List[Rectangle], integral: np.ndarray, p_merge: float = 0.5) -> List[Rectangle]:
    """
    Merge horizontally or vertically adjacent rectangles with probability p_merge.
    """
    if not rects:
        return rects

    # Sort by y, then x for deterministic merging
    rects = sorted(rects, key=lambda r: (r[1], r[0]))
    merged = []
    skip = set()

    for i, r1 in enumerate(rects):
        if i in skip:
            continue
        x1, y1, w1, h1 = r1
        merged_rect = r1

        for j in range(i + 1, len(rects)):
            if j in skip:
                continue
            x2, y2, w2, h2 = rects[j]

            # Horizontal merge: same row, consecutive x
            if y1 == y2 and h1 == h2 and x1 + w1 == x2:
                if random.random() < p_merge:
                    new_rect = (x1, y1, w1 + w2, h1)
                    if is_valid_rectangle_integral(integral, new_rect):
                        merged_rect = new_rect
                        skip.add(j)
                        w1 += w2

            # Vertical merge: same x, consecutive y, same width
            elif x1 == x2 and w1 == w2 and y1 + h1 == y2:
                if random.random() < p_merge:
                    new_rect = (x1, y1, w1, h1 + h2)
                    if is_valid_rectangle_integral(integral, new_rect):
                        merged_rect = new_rect
                        skip.add(j)
                        h1 += h2

        merged.append(merged_rect)

    return merged


def merge_rectangles_probabilistic(rects: List[Rectangle], p_merge: float = 0.5) -> List[Rectangle]:
    """
    Probabilistically merge rectangles if they are neighbors.
    rects: list of (x, y, w, h)
    p_merge: probability to attempt a merge
    """
    rects = rects.copy()
    i = 0
    while i < len(rects):
        r1 = rects[i]
        j = i + 1
        while j < len(rects):
            r2 = rects[j]
            # Check horizontal adjacency
            horiz_adj = (r1[1] == r2[1] and r1[3] == r2[3] and (r1[0] + r1[2] == r2[0] or r2[0] + r2[2] == r1[0]))
            # Check vertical adjacency
            vert_adj = (r1[0] == r2[0] and r1[2] == r2[2] and (r1[1] + r1[3] == r2[1] or r2[1] + r2[3] == r1[1]))

            if (horiz_adj or vert_adj) and random.random() < p_merge:
                # Merge them
                x = min(r1[0], r2[0])
                y = min(r1[1], r2[1])
                w = max(r1[0] + r1[2], r2[0] + r2[2]) - x
                h = max(r1[1] + r1[3], r2[1] + r2[3]) - y
                rects[i] = (x, y, w, h)
                rects.pop(j)
                j = i + 1  # restart merge check for this rectangle
            else:
                j += 1
        i += 1
    return rects


def run_ga(img: np.ndarray, pop_size=20, generations=50, elite_size=2):
    integral = build_integral(img)
    population = init_population_by_rows(img, pop_size)
    print(f"Init population...: {len(population[0].rectangles)}")
    draw_solution(img, population[0].rectangles)
    print("Generations...")

    for g in range(generations):
        for chrom in population:
            chrom.fitness = fitness(chrom, img)

        population.sort(key=lambda c: c.fitness, reverse=True)
        best = population[0]
        print(f"Gen {g}: Best fitness={best.fitness}, Rects={len(best.rectangles)}")

        new_pop = population[:elite_size]
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population[:10], 2)
            child = crossover(p1, p2, img, integral)
            child = mutate(child, img, integral)
            new_pop.append(child)

        population = new_pop

    return population[0]


def draw_solution(img: np.ndarray, chromosomes: List[Chromosome]):
    """"""
    height, width = img.shape
    plt.imshow(img, cmap="gray", origin="upper", extent=[0, width, height, 0])

    # get rectangles from Chromosome object or list
    rects = chromosomes.rects if hasattr(chromosomes, "rects") else chromosomes

    for rect in rects:
        x, y, w, h = map(int, rect)
        color = (random.random(), random.random(), random.random())
        plt.gca().add_patch(
            plt.Rectangle((x, y), w, h, facecolor=color, alpha=1, edgecolor=color)
        )
    plt.axis("off")
    plt.show()


def main():
    """"""
    img = np.array([
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
    ])

    # Load the .npy file
    # Make sure it's 0s and 1s
    img = np.load("../../docs/figures/objects_binary/npy/device3-6_binary.npy")
    img = (img > 0).astype(int)
    # img = 1 - img  # if image is loaded we have to invert 0s and 1s

    # Display the image
    plt.imshow(img, cmap="gray")
    plt.title("Loaded Binary Image")
    plt.axis("off")  # optional: hide axes
    plt.show()

    best = run_ga(img, pop_size=100, generations=20, elite_size=4)
    # print("Best solution:", best.rectangles)

    draw_solution(img, best.rectangles)


if __name__ == "__main__":
    main()
