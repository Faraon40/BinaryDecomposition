"""Visualization utilities for binary image decomposition algorithms.

This module provides visualization functions for displaying decomposition
results, including rectangle overlays, concave vertex visualization,
and algorithm-specific visualizations.
"""

import random
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
    use_palette: bool = False
):
    """Draw binary image with colored rectangle overlay.

    General-purpose visualization function for displaying decomposition
    results as colored rectangles overlaid on the binary image.

    Args:
        img: Binary image (numpy array).
        rectangles: List of Rectangle tuples (x, y, width, height) or
            Chromosome object with rectangles attribute.
        save_path: Path to save figure (optional).
        show: Whether to display figure (default: False).
        title: Custom title for the plot (optional).
        use_palette: Whether to use the shared color palette instead of
            random colors (default: False).

    """
    height, width = img.shape
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="gray", origin="upper", extent=[0, width, height, 0])

    # Get rectangles from Chromosome object or list
    rects = rectangles.rectangles if hasattr(rectangles, "rectangles") else rectangles

    for idx, rect in enumerate(rects):
        x, y, w, h = map(int, rect)

        if use_palette:
            color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        else:
            color = (random.random(), random.random(), random.random())

        plt.gca().add_patch(
            plt.Rectangle(
                (x, y), w, h, facecolor=color, alpha=1, edgecolor=color
            )
        )

    plt.axis("off")

    if title:
        plt.title(title)
    else:
        plt.title(f"Solution: {len(rects)} rectangles")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def visualize_concave_vertices(
    grid: np.ndarray,
    concave_vertices: List[Tuple[int, int]],
    output_path: str = 'concave_vertices.png'
):
    """Visualize concave vertices on binary image.

    Creates RGB image with white pixels for 1s, black for 0s, and red dots
    for concave vertices. Useful for debugging and understanding the
    FER algorithm's input.

    Args:
        grid: Binary image (numpy array).
        concave_vertices: List of (x, y) concave vertex coordinates.
        output_path: Path to save output image.

    """
    rows, cols = grid.shape

    # Create RGB image (3 channels)
    image = np.zeros((rows, cols, 3), dtype=np.uint8)

    # Set pixels according to binary grid
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1:
                image[i, j] = [255, 255, 255]  # White
            else:
                image[i, j] = [0, 0, 0]  # Black

    # Mark concave vertices with red color
    for x, y in concave_vertices:
        if 0 <= y < rows and 0 <= x < cols:
            image[y, x] = [255, 0, 0]  # Red on main pixel

    # Save image
    img = Image.fromarray(image)
    img.save(output_path)

    print(f"Image saved: {output_path}")
    print(f"Size: {cols}x{rows} pixels")
    print(f"Number of concave vertices: {len(concave_vertices)}")
    print(f"Vertex coordinates (first 10): {concave_vertices[:10]}")


def visualize_fer_decomposition(
    grid: np.ndarray,
    result: Dict,
    save_path: Optional[str] = None,
    show: bool = True
):
    """Visualize FER decomposition as colored rectangles on binary image.

    Specialized visualization for FER (graph-based) algorithm results.
    Shows the decomposition with a diverse color palette.

    Args:
        grid: Binary image (numpy array).
        result: Dict with FER algorithm results containing 'rectangles' key.
            Each rectangle should be a dict with 'min_x', 'min_y', 'width',
            and 'height' keys.
        save_path: Path to save image (optional).
        show: Whether to display image (default: True).

    """
    rows, cols = grid.shape

    plt.figure(figsize=(7, 7))
    plt.imshow(grid, cmap="gray", origin="upper", extent=[0, cols, rows, 0])

    # Draw rectangles with color palette
    for idx, rect_data in enumerate(result['rectangles']):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]

        plt.gca().add_patch(
            plt.Rectangle(
                (rect_data['min_x'], rect_data['min_y']),
                rect_data['width'],
                rect_data['height'],
                facecolor=color,
                alpha=1,
                edgecolor=color
            )
        )

    plt.axis("off")
    plt.title(f"FER Solution: {len(result['rectangles'])} rectangles")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    if show:
        plt.show()
    else:
        plt.close()


def visualize_rectangles_dict(
    grid: np.ndarray,
    rectangles: List[Dict],
    save_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None
):
    """Visualize rectangles provided as dictionaries.

    Convenience function for visualizing rectangles that are stored as
    dictionaries with 'min_x', 'min_y', 'width', 'height' keys.

    Args:
        grid: Binary image (numpy array).
        rectangles: List of rectangle dictionaries with keys:
            'min_x', 'min_y', 'width', 'height'.
        save_path: Path to save image (optional).
        show: Whether to display image (default: False).
        title: Custom title for the plot (optional).

    """
    # Convert dict format to tuple format (x, y, w, h)
    rect_tuples = [
        (r['min_x'], r['min_y'], r['width'], r['height'])
        for r in rectangles
    ]

    draw_solution(
        grid,
        rect_tuples,
        save_path=save_path,
        show=show,
        title=title,
        use_palette=True
    )
