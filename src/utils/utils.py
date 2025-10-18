import numpy as np
import random

from matplotlib import pyplot as plt
from typing import List
from src.utils.types import Chromosome


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
