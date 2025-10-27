"""Utility functions for visualization."""

import random
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from src.utils.types import Rectangle


def draw_solution(
    img: np.ndarray,
    chromosomes: List[Rectangle],
    save_path: str = None,
    show: bool = False
):
    """Draw binary image with colored rectangle overlay.

    Args:
        img: Binary image.
        chromosomes: List of Chromosome objects or rectangles.
        save_path: Path to save figure (optional).
        show: Whether to display figure (default: True).

    """
    height, width = img.shape
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="gray", origin="upper", extent=[0, width, height, 0])

    # Get rectangles from Chromosome object or list
    rects = chromosomes.rects if hasattr(chromosomes, "rects") else chromosomes

    for rect in rects:
        x, y, w, h = map(int, rect)
        color = (random.random(), random.random(), random.random())
        plt.gca().add_patch(
            plt.Rectangle(
                (x, y), w, h, facecolor=color, alpha=1, edgecolor=color
            )
        )
    plt.axis("off")
    plt.title(f"Solution: {len(rects)} rectangles")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    if show:
        plt.show()
    else:
        plt.close()
