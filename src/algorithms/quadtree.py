"""Quadtree-based binary image decomposition.

This module implements quadtree decomposition for binary images,
converting the hierarchical structure into rectangles compatible
with the genetic algorithm.

References:
    - Samet, H. (1984). "The Quadtree and Related Hierarchical
      Data Structures"
    - Finkel, R.A., Bentley, J.L. (1974). "Quad trees: A data
      structure for retrieval on composite keys"
"""

from typing import List, Tuple

import numpy as np

from src.utils.types import Chromosome, Rectangle


class QuadNode:
    """Node in quadtree structure.

    Represents a rectangular region that is either:
    - Leaf: homogeneous region (all 0s or all 1s)
    - Internal: heterogeneous region with 4 children
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        is_leaf: bool = False,
    ):
        """Initialize quadtree node.

        Args:
            x: Top-left x coordinate.
            y: Top-left y coordinate.
            width: Region width.
            height: Region height.
            is_leaf: Whether this is a leaf node.

        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.is_leaf = is_leaf
        self.value = None  # 0 or 1 for leaf nodes
        self.children = []  # 4 children for internal nodes

    def get_rectangle(self) -> Rectangle:
        """Get rectangle representation of this node.

        Returns:
            Rectangle tuple (x, y, width, height).

        """
        return (self.x, self.y, self.width, self.height)


def is_homogeneous(
    img: np.ndarray, x: int, y: int, w: int, h: int
) -> Tuple[bool, int]:
    """Check if region is homogeneous (all same value).

    Args:
        img: Binary image.
        x: Region x coordinate.
        y: Region y coordinate.
        w: Region width.
        h: Region height.

    Returns:
        Tuple of (is_homogeneous, value).
        Value is 0 or 1 if homogeneous, None otherwise.

    """
    region = img[y : y + h, x : x + w]
    first_val = region[0, 0]

    if np.all(region == first_val):
        return True, int(first_val)
    return False, None


def build_quadtree(
    img: np.ndarray,
    x: int = 0,
    y: int = 0,
    width: int = None,
    height: int = None,
    min_size: int = 2,
) -> QuadNode:
    """Build quadtree from binary image using recursive subdivision.

    Algorithm:
    1. Check if region is homogeneous
    2. If yes or too small, create leaf node
    3. Otherwise, subdivide into 4 quadrants and recurse

    Args:
        img: Binary image (0s and 1s).
        x: Region x coordinate.
        y: Region y coordinate.
        width: Region width (default: image width).
        height: Region height (default: image height).
        min_size: Minimum region size (stop subdivision).

    Returns:
        Root node of quadtree.

    """
    if width is None:
        width = img.shape[1]
    if height is None:
        height = img.shape[0]

    # Check homogeneity
    homogeneous, value = is_homogeneous(img, x, y, width, height)

    # Create leaf ONLY if homogeneous
    if homogeneous:
        node = QuadNode(x, y, width, height, is_leaf=True)
        node.value = value
        return node

    # If too small but not homogeneous, don't create leaf - subdivide
    if width <= min_size or height <= min_size:
        # For very small heterogeneous regions, check if any 1s exist
        region = img[y : y + height, x : x + width]
        has_ones = np.any(region == 1)
        node = QuadNode(x, y, width, height, is_leaf=True)
        # Only mark as 1 if it contains ANY 1s (will be fixed by repair)
        node.value = 1 if has_ones else 0
        return node

    # Subdivide into 4 quadrants
    node = QuadNode(x, y, width, height, is_leaf=False)

    # Calculate quadrant dimensions
    mid_w = width // 2
    mid_h = height // 2

    # NW quadrant (top-left)
    node.children.append(build_quadtree(img, x, y, mid_w, mid_h, min_size))

    # NE quadrant (top-right)
    if width - mid_w > 0:
        node.children.append(
            build_quadtree(
                img, x + mid_w, y, width - mid_w, mid_h, min_size
            )
        )

    # SW quadrant (bottom-left)
    if height - mid_h > 0:
        node.children.append(
            build_quadtree(
                img, x, y + mid_h, mid_w, height - mid_h, min_size
            )
        )

    # SE quadrant (bottom-right)
    if width - mid_w > 0 and height - mid_h > 0:
        node.children.append(
            build_quadtree(
                img,
                x + mid_w,
                y + mid_h,
                width - mid_w,
                height - mid_h,
                min_size,
            )
        )

    return node


def collect_leaf_rectangles(
    node: QuadNode, only_ones: bool = True
) -> List[Rectangle]:
    """Collect rectangles from quadtree leaf nodes.

    Args:
        node: Quadtree node.
        only_ones: If True, only collect leaves with value 1.

    Returns:
        List of rectangles from leaf nodes.

    """
    rectangles = []

    if node.is_leaf:
        # Only add if value is 1 (or collect all if only_ones=False)
        if not only_ones or node.value == 1:
            rectangles.append(node.get_rectangle())
    else:
        # Recursively collect from children
        for child in node.children:
            rectangles.extend(
                collect_leaf_rectangles(child, only_ones)
            )

    return rectangles


def trim_rectangles_to_ones(
    img: np.ndarray, rects: List[Rectangle]
) -> List[Rectangle]:
    """Trim rectangles to contain only 1s by finding bounding box.

    - If region is all 1s -> kept as is.
    - If region is all 0s -> skipped.
    - If region is mixed -> decomposed into minimal rectangular
     covers of 1s.

    Args:
        img: Binary image.
        rects: List of rectangles to trim.

    Returns:
        List of trimmed rectangles containing only 1s.

    """
    trimmed = []

    for rect in rects:
        x, y, w, h = rect
        region = img[y:y + h, x:x + w]

        # Skip fully empty regions
        if not np.any(region == 1):
            continue

        # Keep if already homogeneous
        if np.all(region == 1):
            trimmed.append(rect)
            continue

        # Copy to allow destructive marking
        region_copy = region.copy()

        # Decompose mixed region into pure-1 rectangles
        while np.any(region_copy == 1):
            ys, xs = np.where(region_copy == 1)
            y0, x0 = ys[0], xs[0]

            # Find maximal rectangle of 1s starting at (x0, y0)
            max_w = 1
            while x0 + max_w < region_copy.shape[1] and np.all(region_copy[y0, x0:x0 + max_w + 1] == 1):
                max_w += 1

            max_h = 1
            while y0 + max_h < region_copy.shape[0] and np.all(region_copy[y0 + max_h, x0:x0 + max_w] == 1):
                max_h += 1

            # Record rectangle in absolute coordinates
            trimmed.append((x + x0, y + y0, max_w, max_h))

            # Zero out the used area
            region_copy[y0:y0 + max_h, x0:x0 + max_w] = 0

    return trimmed


def quadtree_decomposition(
    img: np.ndarray,
    min_size: int = 2,
    trim: bool = True,
) -> List[Rectangle]:
    """Decompose binary image using quadtree algorithm.

    Args:
        img: Binary image (0s and 1s).
        min_size: Minimum quadrant size.
        trim: Whether to trim rectangles to exact 1s coverage (default True).

    Returns:
        List of rectangles covering all 1s in image.

    """
    # Build quadtree
    root = build_quadtree(img, min_size=min_size)

    # Collect rectangles from leaf nodes (only 1s)
    rectangles = collect_leaf_rectangles(root, only_ones=True)

    # Trim rectangles to contain only 1s
    if trim:
        rectangles = trim_rectangles_to_ones(img, rectangles)

    return rectangles


def init_population_quadtree(
    img: np.ndarray,
    integral: np.ndarray,
    pop_size: int,
    min_size_range: Tuple[int, int] = (2, 8),
) -> List[Chromosome]:
    """Initialize GA population using quadtree decomposition.

    Creates diverse population by varying min_size parameter across
    the specified range. Each individual uses pure quadtree algorithm
    without additional merging.

    Args:
        img: Binary image.
        integral: Integral image (for compatibility, not used).
        pop_size: Population size.
        min_size_range: Range of min_size values for diversity.

    Returns:
        List of initialized chromosomes.

    """
    population = []
    min_sizes = list(range(min_size_range[0], min_size_range[1] + 1))

    for i in range(pop_size):
        # Vary min_size for diversity
        min_size = min_sizes[i % len(min_sizes)]

        # Generate quadtree decomposition
        rectangles = quadtree_decomposition(img, min_size)

        population.append(Chromosome(rectangles))

    return population


def run_quadtree(
    img: np.ndarray,
    min_size: int = 2,
    trim: bool = True,
    verbose: bool = True,
) -> Tuple[List[Rectangle], List[int]]:
    """Run quadtree decomposition on binary image.

    Args:
        img: Binary image to decompose (0s and 1s).
        min_size: Minimum quadrant size before stopping subdivision.
        trim: Trim rectangles to exact 1s coverage (default: True).
        verbose: Print progress information (default: True).

    Returns:
        Tuple of (rectangles, generation_history).
        rectangles: List of rectangles covering all 1s.
        generation_history: Empty list (for compatibility with GA).

    """
    import time

    start_time = time.time()

    if verbose:
        print(f"Running quadtree decomposition (min_size={min_size})...")

    # Run quadtree decomposition
    rectangles = quadtree_decomposition(img, min_size=min_size, trim=trim)

    execution_time = time.time() - start_time

    if verbose:
        print(f"Completed in {execution_time:.2f} seconds")
        print(f"Solution: {len(rectangles)} rectangles")

    # Return empty generation history for API compatibility with GA
    return rectangles, []


if __name__ == "__main__":
    """Test quadtree decomposition."""
    from src.utils.utils import draw_solution

    print("=" * 60)
    print("Quadtree Decomposition Test")
    print("=" * 60)

    # Load test image
    img = np.load(
        "../../res/figures/objects_binary/npy/crown-3_binary.npy"
    )
    img = (img > 0).astype(int)

    # Show original image first
    print("Original image (without decomposition):")
    draw_solution(img, [])

    # Run quadtree decomposition
    rects, _ = run_quadtree(img, min_size=2, trim=True, verbose=True)

    print("\nDecomposed image:")
    draw_solution(img, rects)
