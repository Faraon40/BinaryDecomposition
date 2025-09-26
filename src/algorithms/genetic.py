import numpy as np
import random
from typing import List, Tuple

Rectangle = Tuple[int, int, int, int]  # (x, y, w, h)

# ------------------------------
# Rectangle utilities
# ------------------------------
def is_valid_rectangle(img: np.ndarray, rect: Rectangle) -> bool:
    x, y, w, h = rect
    if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
        return False
    sub = img[y:y+h, x:x+w]
    return np.all(sub == 1)

def overlap(r1: Rectangle, r2: Rectangle) -> bool:
    x1,y1,w1,h1 = r1
    x2,y2,w2,h2 = r2
    return not (x1+w1 <= x2 or x2+w2 <= x1 or y1+h1 <= y2 or y2+h2 <= y1)

# ------------------------------
# Chromosome
# ------------------------------
class Chromosome:
    def __init__(self, rectangles: List[Rectangle]):
        self.rectangles = rectangles
        self.fitness = None

# ------------------------------
# Fitness function
# ------------------------------
def fitness(chrom: Chromosome, img: np.ndarray) -> float:
    covered = np.zeros_like(img)
    for (x, y, w, h) in chrom.rectangles:
        covered[y:y+h, x:x+w] = 1

    missing = np.sum((img == 1) & (covered == 0))
    extra   = np.sum((img == 0) & (covered == 1))

    if missing > 0:
        return -1e6  # invalid: not all 1s covered

    return -len(chrom.rectangles) - extra  # fewer rects + fewer extras is better

# ------------------------------
# Repair operator
# ------------------------------
def repair(rects: List[Rectangle], img: np.ndarray) -> List[Rectangle]:
    covered = np.zeros_like(img)
    new_rects = []
    for r in rects:
        if not any(overlap(r, r2) for r2 in new_rects):
            if is_valid_rectangle(img, r):
                new_rects.append(r)
                x,y,w,h = r
                covered[y:y+h, x:x+w] = 1

    # cover missing 1s with 1x1 rectangles (greedy repair)
    ys, xs = np.where((img == 1) & (covered == 0))
    for y, x in zip(ys, xs):
        new_rects.append((x,y,1,1))

    return new_rects

# ------------------------------
# Initialization
# ------------------------------
def init_population(img: np.ndarray, pop_size: int) -> List[Chromosome]:
    H, W = img.shape
    population = []
    ones = np.argwhere(img == 1)

    for _ in range(pop_size):
        rects = []
        for _ in range(random.randint(1, len(ones)//2 + 1)):
            y, x = random.choice(ones)
            w, h = random.randint(1, W-x), random.randint(1, H-y)
            rect = (x, y, w, h)
            if is_valid_rectangle(img, rect):
                rects.append(rect)
        rects = repair(rects, img)
        population.append(Chromosome(rects))
    return population

# ------------------------------
# Mutation
# ------------------------------
def mutate(chrom: Chromosome, img: np.ndarray, p: float=0.3) -> Chromosome:
    rects = chrom.rectangles.copy()
    if not rects:
        return chrom

    if random.random() < p:
        idx = random.randrange(len(rects))
        x, y, w, h = rects[idx]
        choice = random.choice(["expand_w", "expand_h", "shrink_w", "shrink_h"])
        new_rect = (x, y, w, h)
        if choice == "expand_w": new_rect = (x, y, w+1, h)
        elif choice == "expand_h": new_rect = (x, y, w, h+1)
        elif choice == "shrink_w" and w > 1: new_rect = (x, y, w-1, h)
        elif choice == "shrink_h" and h > 1: new_rect = (x, y, w, h-1)
        if is_valid_rectangle(img, new_rect):
            rects[idx] = new_rect

    rects = repair(rects, img)
    return Chromosome(rects)

# ------------------------------
# Crossover
# ------------------------------
def crossover(p1: Chromosome, p2: Chromosome, img: np.ndarray) -> Chromosome:
    rects = []
    if not p1.rectangles or not p2.rectangles:
        return Chromosome(p1.rectangles or p2.rectangles)

    # subset swap
    subset_size = random.randint(1, len(p1.rectangles))
    rects.extend(random.sample(p1.rectangles, subset_size))
    for r in p2.rectangles:
        if not any(overlap(r, r2) for r2 in rects):
            if is_valid_rectangle(img, r):
                rects.append(r)

    rects = repair(rects, img)
    return Chromosome(rects)

# ------------------------------
# GA Loop
# ------------------------------
def run_ga(img: np.ndarray, pop_size=20, generations=50, elite_size=2):
    population = init_population(img, pop_size)

    for g in range(generations):
        # evaluate fitness
        for chrom in population:
            chrom.fitness = fitness(chrom, img)

        # sort by fitness (descending, since fewer rectangles = better)
        population.sort(key=lambda c: c.fitness, reverse=True)
        best = population[0]
        print(f"Gen {g}: Best fitness={best.fitness}, Rects={len(best.rectangles)}")

        # elitism
        new_pop = population[:elite_size]

        # generate rest
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population[:10], 2)  # tournament from top 10
            child = crossover(p1, p2, img)
            child = mutate(child, img)
            new_pop.append(child)

        population = new_pop

    return population[0]  # best found

# Simple 6x6 binary image
img = np.array([
    [0,0,0,0,255,1,0,0,0,0],
    [0,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,1,1,0,0,0,0],
    [0,0,0,0,1,1,0,0,0,0],
])

# img = np.load("../../docs/figures/leafs_binary/npy/Syringa_chinensis_binary.npy")
import numpy as np
import matplotlib.pyplot as plt

# Load the .npy file
# Make sure it's 0s and 1s
img = (img > 0).astype(int)
img = 1 - img
# Display the image
plt.imshow(img, cmap="gray")
plt.title("Loaded Binary Image")
plt.axis("off")  # optional: hide axes
plt.show()

# best = run_ga(img, pop_size=100, generations=300)
# print("Best solution:", best.rectangles)
