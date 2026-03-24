"""Visualization utilities for binary image decomposition algorithms.

This module provides visualization functions for displaying decomposition
results, including rectangle overlays and concave vertex visualization.
"""

from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

from src.utils.types import Rectangle


# Shared color palette for all visualizations
COLOR_PALETTE = [
    # Pastel colors
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
    '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#A8E6CF',

    # Vivid colors
    '#FF5733', '#33FF57', '#3357FF', '#FF33F5', '#F5FF33',
    '#33FFF5', '#F533FF', '#57FF33', '#5733FF', '#FF3357',

    # Warm tones
    '#FF9999', '#FFCC99', '#FFFF99', '#CCFF99', '#99FF99',
    '#99FFCC', '#99FFFF', '#99CCFF', '#9999FF', '#CC99FF',

    # Cool tones
    '#99CCCC', '#6699CC', '#336699', '#003366', '#006666',
    '#009999', '#00CCCC', '#66CCCC', '#99CCFF', '#CCFFFF',

    # Earth tones
    '#D2691E', '#CD853F', '#DEB887', '#F4A460', '#DAA520',
    '#B8860B', '#BC8F8F', '#CD5C5C', '#F08080', '#FA8072',

    # Neon colors
    '#39FF14', '#DFFF00', '#FFFF00', '#FF6600', '#FF00FF',
    '#00FFFF', '#FF007F', '#7FFF00', '#00FF7F', '#FF1493'
]


def draw_solution(
    img: np.ndarray,
    rectangles: List[Rectangle],
    save_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
):
    """Draw binary image with colored rectangle overlay.

    General-purpose visualization function for displaying decomposition
    results as colored rectangles overlaid on the binary image.
    Each rectangle is drawn with a thin white border for visual separation.

    Args:
        img: Binary image (numpy array).
        rectangles: List of Rectangle tuples (x, y, width, height) or
            Chromosome object with rectangles attribute.
        save_path: Path to save figure (optional).
        show: Whether to display figure (default: False).
        title: Custom title for the plot (optional).

    """
    height, width = img.shape
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="gray", origin="upper", extent=[0, width, height, 0])

    rects = (
        rectangles.rectangles
        if hasattr(rectangles, "rectangles")
        else rectangles
    )

    for idx, rect in enumerate(rects):
        x, y, w, h = map(int, rect)
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        plt.gca().add_patch(
            plt.Rectangle(
                (x, y), w, h,
                facecolor=color,
                alpha=0.75,
                edgecolor=color,
                linewidth=0.5,
            )
        )

    plt.axis("off")
    plt.title(title if title else f"Solution: {len(rects)} rectangles")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def visualize_concave_vertices(grid, concave_vertices):
    """Visualize concave vertices on a binary grid using Matplotlib.

    Active pixels (1) are shown in white, and background pixels (0) in black.
    Concave vertices are highlighted as red pixels with yellow ID labels.

    Args:
        grid (np.ndarray): 2D binary numpy array (0s and 1s).
        concave_vertices (list): List of tuples (id, x, y, corners, cx, cy).

    """
    rows, cols = grid.shape

    image = np.zeros((rows, cols, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1:
                image[i, j] = [255, 255, 255]
            else:
                image[i, j] = [0, 0, 0]

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")

    for idx, x, y, corners, cx, cy in concave_vertices:
        if 0 <= y < rows and 0 <= x < cols:
            image[y, x] = [255, 0, 0]

        plt.text(
            x, y, str(idx),
            color="yellow", fontsize=10, ha="center", va="center",
            fontweight="bold"
        )

    plt.imshow(image)
    plt.title("Detected Concave Vertices")
    plt.show()


def visualize_concave_vertices_cord(grid, concave_vertices):
    """Visualize the binary grid with pixel coordinates and highlighted concave vertices.

    Renders the input grid, labels every active pixel with its (x, y)
    coordinates, and highlights detected concave vertices with their IDs
    and precise geometric centers.

    Args:
        grid (np.ndarray): 2D binary numpy array (0s and 1s).
        concave_vertices (list): List of tuples (id, x, y, corners, cx, cy).

    """
    rows, cols = grid.shape

    image = np.zeros((rows, cols, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1:
                image[i, j] = [255, 255, 255]
            else:
                image[i, j] = [0, 0, 0]

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")

    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1:
                plt.text(
                    j, i, f"({j},{i})",
                    color="gray", fontsize=7, ha="center", va="center",
                    bbox=dict(
                        boxstyle="square,pad=0.1",
                        facecolor="none",
                        edgecolor="gray",
                        linewidth=0.4
                    )
                )

    for idx, x, y, corners, cx, cy in concave_vertices:
        plt.plot(cx, cy, "mo", markersize=5)

        if 0 <= y < rows and 0 <= x < cols:
            image[y, x] = [255, 0, 0]

        plt.text(
            x, y - 0.2, str(idx),
            color="yellow", fontsize=10, ha="center", va="center",
            fontweight="bold"
        )
        plt.text(
            x, y + 0.2, f"({x},{y})",
            color="yellow", fontsize=7, ha="center", va="center",
        )

    plt.imshow(image)
    plt.title("Concave Vertices Visualization")
    plt.show()
