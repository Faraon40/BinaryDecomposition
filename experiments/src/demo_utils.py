"""Shared utilities for demo visualization scripts."""

import random

import matplotlib.pyplot as plt
import numpy as np

from src.utils.visualization import COLOR_PALETTE

DOCS_FIGURES = (
    "/home/antonio/PycharmProjects/BinaryDecompositionDocs/src/figures"
)


def save_decomposition(
    img: np.ndarray,
    rects: list,
    path: str,
    title: str,
    figsize: tuple = (8, 8),
) -> None:
    """Save decomposition visualization with white background.

    Args:
        img: Binary image (2D numpy array).
        rects: List of (x, y, w, h) rectangles.
        path: Output file path.
        title: Plot title.
        figsize: Figure size in inches (default 8x8).
    """
    palette = COLOR_PALETTE[:]
    random.shuffle(palette)
    height, width = img.shape
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")
    ax.imshow(img, cmap="gray_r", origin="upper", extent=[0, width, height, 0])
    for idx, (x, y, w, h) in enumerate(rects):
        color = palette[idx % len(palette)]
        ax.add_patch(plt.Rectangle(
            (x, y), w, h,
            facecolor=color, alpha=0.75,
            edgecolor=color, linewidth=0.5,
        ))
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)