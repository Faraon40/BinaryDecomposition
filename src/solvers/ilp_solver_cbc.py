"""Integer Linear Programming solvers for exact rectangle decomposition.

This module provides an exact solvers for the minimum rectangle covering
problem using Integer Linear Programming (ILP). It finds the provably
optimal solution (minimum number of rectangles) for binary images.

Note: ILP is computationally expensive and only feasible for small to
medium-sized images (typically <= 100x100 pixels).
"""

import time
from typing import Dict, List, Tuple

import numpy as np
import pulp

from src.utils.types import Rectangle


def prefix_sum(img: np.ndarray) -> np.ndarray:
    """Build prefix sum array for O(1) rectangle sum queries.

    Args:
        img: Binary image array (0s and 1s).

    Returns:
        Prefix sum array of shape (h+1, w+1).

    """
    h, w = img.shape
    ps = np.zeros((h + 1, w + 1), dtype=np.int64)
    ps[1:, 1:] = img.cumsum(axis=0).cumsum(axis=1)
    return ps


def rect_sum(ps: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    """Compute sum of rectangle using prefix sums.

    Args:
        ps: Prefix sum array.
        x: Rectangle x-coordinate (left).
        y: Rectangle y-coordinate (top).
        w: Rectangle width.
        h: Rectangle height.

    Returns:
        Sum of all pixels in the rectangle.

    """
    x2, y2 = x + w, y + h
    return ps[y2, x2] - ps[y, x2] - ps[y2, x] + ps[y, x]


def enumerate_maximal_rects(
    img: np.ndarray, max_candidates: int = 100000, verbose: bool = False
) -> List[Rectangle]:
    """Enumerate maximal rectangles (cannot be extended in any direction).

    This generates far fewer candidates than enumerate_full_rects by only
    including rectangles that are maximal - they cannot be extended left,
    right, up, or down while remaining valid.

    Args:
        img: Binary image (0s and 1s).
        max_candidates: Maximum number of candidates to generate.
        verbose: Print progress during enumeration.

    Returns:
        List of maximal rectangles as (x, y, width, height) tuples.

    """
    h, w = img.shape
    ps = prefix_sum(img)  # Build prefix sum for validation
    candidates = set()  # Use set to avoid duplicates

    if verbose:
        print(f"   Enumerating maximal rectangles...")

    # For each starting row
    for start_row in range(h):
        # Build histogram for this starting row
        heights = [0] * w

        for row in range(start_row, h):
            # Update histogram: increment height if pixel is 1
            for col in range(w):
                if img[row, col] == 1:
                    heights[col] += 1
                else:
                    heights[col] = 0

            # Find all maximal rectangles in this histogram
            for col in range(w):
                if heights[col] == 0:
                    continue

                # Find extent of rectangle with height = heights[col]
                left = col
                right = col

                # Expand left
                while left > 0 and heights[left - 1] >= heights[col]:
                    left -= 1

                # Expand right
                while right < w - 1 and heights[right + 1] >= heights[col]:
                    right += 1

                # Create candidate rectangle
                rect_width = right - left + 1
                rect_height = heights[col]

                # CRITICAL: Validate that ALL pixels in rectangle equal 1
                # The histogram approach can create rectangles that extend
                # into 0-pixel areas, so we must validate using prefix sums
                if rect_sum(ps, left, start_row, rect_width, rect_height) == rect_width * rect_height:
                    rect = (left, start_row, rect_width, rect_height)
                    candidates.add(rect)

                # Safety limit
                if len(candidates) >= max_candidates:
                    if verbose:
                        print(
                            f"Hit candidate limit: {max_candidates:,} rectangles"
                        )
                    return list(candidates)

    if verbose:
        print(f"   Found {len(candidates):,} maximal rectangles")

    return list(candidates)


def enumerate_full_rects(
    img: np.ndarray, max_candidates: int = 100000, verbose: bool = False
) -> List[Rectangle]:
    """Enumerate all valid rectangles (rectangles where all pixels equal 1).

    WARNING: This generates exponentially many rectangles and is not
    recommended for images larger than ~50x50. Use enumerate_maximal_rects
    for larger images.

    Args:
        img: Binary image (0s and 1s).
        max_candidates: Maximum number of candidates to generate (safety
            limit to prevent memory exhaustion on large images).
        verbose: Print progress during enumeration.

    Returns:
        List of valid rectangles as (x, y, width, height) tuples.

    """
    h, w = img.shape
    ps = prefix_sum(img)
    candidates = []

    # Progress tracking
    total_pixels = h * w
    pixels_processed = 0
    last_progress = 0

    for y in range(h):
        for x in range(w):
            pixels_processed += 1

            # Progress reporting (every 10%)
            if verbose:
                progress = int(pixels_processed / total_pixels * 100)
                if progress >= last_progress + 10:
                    print(f"Enumeration progress: {progress}% "
                          f"({len(candidates):,} candidates)")
                    last_progress = progress

            if img[y, x] == 0:
                continue

            # Find maximum horizontal span at this position
            max_width = w - x
            for dx in range(w - x):
                if img[y, x + dx] == 0:
                    max_width = dx
                    break

            # Grow rectangle vertically and horizontally
            for y2 in range(y, h):
                height = y2 - y + 1

                # Try extending horizontally (with early termination)
                for x2 in range(x, x + max_width):
                    width = x2 - x + 1
                    area = height * width

                    if rect_sum(ps, x, y, width, height) == area:
                        candidates.append((x, y, width, height))
                    else:
                        # Can't extend further horizontally at this height
                        break

                # Safety limit to prevent memory issues
                if len(candidates) >= max_candidates:
                    print(
                        f"Hit candidate limit: {max_candidates:,} rectangles"
                    )
                    print(f"Image size: {h}×{w}, processed "
                          f"{pixels_processed}/{total_pixels} pixels"
                          f" ({pixels_processed/total_pixels * 100:.1f}%)")
                    return candidates

    return candidates


def validate_solution(
    img: np.ndarray, rectangles: List[Rectangle]
) -> Tuple[bool, str]:
    """Validate that rectangles correctly decompose the image.

    Checks that:
    1. All rectangles are within bounds
    2. No rectangles overlap
    3. All 1-pixels are covered exactly once
    4. No 0-pixels are covered

    Args:
        img: Binary image.
        rectangles: List of rectangles to validate.

    Returns:
        Tuple of (is_valid, message) where is_valid is True if solution
        is valid, and message contains details.

    """
    h, w = img.shape
    covered = np.zeros((h, w), dtype=np.uint8)

    for x, y, rw, rh in rectangles:
        # Check bounds
        if x < 0 or y < 0 or x + rw > w or y + rh > h:
            return False, f"Rectangle out of bounds: ({x}, {y}, {rw}, {rh})"

        # Check for overlaps
        if np.any(covered[y : y + rh, x : x + rw] > 0):
            return (
                False,
                f"Overlap detected at rectangle ({x}, {y}, {rw}, {rh})",
            )

        covered[y : y + rh, x : x + rw] = 1

    # Check exact coverage
    if not np.array_equal(covered, img):
        missing = np.sum((img == 1) & (covered == 0))
        extra = np.sum((img == 0) & (covered == 1))
        return (
            False,
            f"Coverage mismatch: {missing} missing pixels, {extra} extra pixels",
        )

    return True, "Valid solution"


def run_ilp(
    img: np.ndarray,
    timeout: int = 300,
    max_candidates: int = 100000,
    verbose: bool = True,
) -> Tuple[List[Rectangle], Dict]:
    """Solve minimum rectangle covering using Integer Linear Programming.

    This function finds the provably optimal solution (minimum number of
    non-overlapping rectangles) that exactly covers all 1-pixels in the
    binary image.

    The algorithm works in two phases:
    1. Enumerate all valid candidate rectangles
    2. Solve set covering ILP to select minimum subset

    Args:
        img: Binary image to decompose. Can be 0/1 or 0/255 format.
            Will be automatically converted to 0/1 if needed.
        timeout: Maximum solving time in seconds (default: 300).
        max_candidates: Maximum candidate rectangles to generate
            (default: 500000).
        verbose: Print progress information (default: True).

    Returns:
        Tuple of (rectangles, stats) where:
        - rectangles: List of (x, y, width, height) tuples
        - stats: Dictionary containing solving statistics:
            - 'status': 'optimal', 'timeout', 'infeasible', etc.
            - 'rectangle_count': Number of rectangles in solution
            - 'candidates': Number of candidate rectangles generated
            - 'solve_time': Total time in seconds
            - 'enum_time': Time for candidate enumeration
            - 'ilp_time': Time for ILP solving

    Raises:
        ValueError: If some pixels cannot be covered by any rectangles.

    """
    start_time = time.time()

    # Normalize image to 0/1 format (handle 0/255 images)
    img = (img > 0).astype(np.uint8)

    # Get filled pixels that need to be covered
    filled_pixels = [
        (x, y)
        for y in range(img.shape[0])
        for x in range(img.shape[1])
        if img[y, x] == 1
    ]

    if verbose:
        print(
            f"Image: {img.shape[1]}×{img.shape[0]}, "
            f"Pixels to cover: {len(filled_pixels):,}"
        )

    if len(filled_pixels) == 0:
        # Empty image - no rectangles needed
        stats = {
            "status": "optimal",
            "rectangle_count": 0,
            "candidates": 0,
            "solve_time": 0.0,
            "enum_time": 0.0,
            "ilp_time": 0.0,
        }
        return [], stats

    # Generate candidate rectangles
    # Use maximal rectangles for efficiency (small trade-off in optimality)
    if verbose:
        print(f"Using maximal rectangle enumeration...")
    # candidates = enumerate_maximal_rects(
    #     img,
    #     max_candidates=max_candidates,
    #     verbose=verbose
    # )

    # Full enumeration (for validation images only - requires
    # millions of candidates for crown-6)
    candidates = enumerate_full_rects(
        img,
        max_candidates=max_candidates,
        verbose=verbose
    )

    enum_time = time.time() - start_time

    if verbose:
        print(
            f"Generated {len(candidates):,} candidate rectangles "
            f"in {enum_time:.2f}s"
        )

    # Build pixel-to-rectangles coverage map
    cover_map = {p: [] for p in filled_pixels}
    for j, (x, y, w, h) in enumerate(candidates):
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                if (xx, yy) in cover_map:
                    cover_map[(xx, yy)].append(j)

    # Check feasibility - ensure all pixels can be covered
    uncoverable = [p for p, js in cover_map.items() if len(js) == 0]
    if uncoverable:
        error_msg = (
            f"️  {len(uncoverable)} pixels have no covering rectangles!\n"
            f"First uncoverable pixel: {uncoverable[0]}\n"
            f"This may indicate insufficient candidates "
            f"(current limit: {max_candidates:,})"
        )
        raise ValueError(error_msg)

    # Build ILP model
    prob = pulp.LpProblem("min_rect_cover", pulp.LpMinimize)

    # Decision variables: one binary variable per candidate rectangle
    x_vars = [
        pulp.LpVariable(f"x_{j}", cat="Binary")
        for j in range(len(candidates))
    ]

    # Objective: minimize total number of selected rectangles
    prob += pulp.lpSum(x_vars)

    # Coverage constraints: each pixel must be covered exactly once
    for p, js in cover_map.items():
        prob += pulp.lpSum([x_vars[j] for j in js]) == 1

    # Solve ILP
    if verbose:
        print(
            f"Solving ILP ({len(x_vars):,} variables, "
            f"{len(filled_pixels):,} constraints)..."
        )

    ilp_start = time.time()
    solver = pulp.PULP_CBC_CMD(msg=verbose, timeLimit=timeout)
    status = prob.solve(solver)
    ilp_time = time.time() - ilp_start

    solve_time = time.time() - start_time

    # Extract solution based on status
    if status == pulp.LpStatusOptimal:
        chosen = [
            candidates[j]
            for j, var in enumerate(x_vars)
            if var.value() >= 0.5
        ]

        if verbose:
            print(
                f"Optimal solution: {len(chosen)} rectangles "
                f"(solved in {solve_time:.2f}s)"
            )

        stats = {
            "status": "optimal",
            "rectangle_count": len(chosen),
            "candidates": len(candidates),
            "solve_time": solve_time,
            "enum_time": enum_time,
            "ilp_time": ilp_time,
        }

        return chosen, stats

    elif status == pulp.LpStatusNotSolved:
        if verbose:
            print(f"⚠️  Solver timeout after {timeout}s")

        stats = {
            "status": "timeout",
            "rectangle_count": 0,
            "candidates": len(candidates),
            "solve_time": solve_time,
            "enum_time": enum_time,
            "ilp_time": ilp_time,
        }

        return [], stats

    else:
        status_name = pulp.LpStatus[status]
        if verbose:
            print(f"✗ Solver failed with status: {status_name}")

        stats = {
            "status": status_name,
            "rectangle_count": 0,
            "candidates": len(candidates),
            "solve_time": solve_time,
            "enum_time": enum_time,
            "ilp_time": ilp_time,
        }

        return [], stats


if __name__ == "__main__":
    """Demo: run ILP solvers on images from a directory."""
    import sys
    from pathlib import Path

    # Default configuration
    DEFAULT_DIR = "synthetic"  # Can be: objects_binary, synthetic, etc.
    DEFAULT_MAX_IMAGES = 5
    DEFAULT_MAX_CANDIDATES = 200000

    # Parse command-line arguments
    if len(sys.argv) > 1:
        image_dir_name = sys.argv[1]
    else:
        image_dir_name = DEFAULT_DIR

    if len(sys.argv) > 2:
        max_images = int(sys.argv[2])
    else:
        max_images = DEFAULT_MAX_IMAGES

    if len(sys.argv) > 3:
        max_candidates = int(sys.argv[3])
    else:
        max_candidates = DEFAULT_MAX_CANDIDATES

    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    image_dir = project_root / "res/figures" / image_dir_name / "npy"

    if not image_dir.exists():
        available = [
            d.name
            for d in (project_root / "res/figures").iterdir()
            if d.is_dir()
        ]
        print(f" Image directory not found: {image_dir}")
        print(f"Available directories: {available}")
        print(
            f"\nUsage: python -m src.algorithms.ilp_solver "
            f"[directory_name] [max_images] [max_candidates]"
        )
        print(f"Example: python -m src.algorithms.ilp_solver synthetic 3 500000")
        sys.exit(1)

    # Get image paths
    image_paths = sorted(image_dir.glob("*.npy"))

    if not image_paths:
        print(f" No .npy files found in {image_dir}")
        sys.exit(1)

    # Limit to max_images if specified
    if max_images is not None:
        image_paths = image_paths[:max_images]

    print("=" * 70)
    print("ILP SOLVER - EXACT RECTANGLE DECOMPOSITION")
    print("=" * 70)
    print(f"Directory: {image_dir_name}")
    print(f"Images to process: {len(image_paths)}")
    print(f"Timeout per image: 300s")
    print(f"Max candidates: {max_candidates:,}")
    print("=" * 70)

    total_start = time.time()
    results = []

    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i:2d}/{len(image_paths)}] Processing: {img_path.name}")
        print("-" * 70)

        try:
            # Load image
            img = np.load(img_path)
            h, w = img.shape
            pixel_count = np.sum(img > 0)

            print(f"Image size: {w}×{h}, Pixels to cover: {pixel_count:,}")

            # Run ILP solvers
            rectangles, stats = run_ilp(
                img, timeout=300, max_candidates=max_candidates, verbose=True
            )

            # Validate solution
            if rectangles:
                is_valid, msg = validate_solution(img, rectangles)
                if is_valid:
                    print(f"✓ Validation: {msg}")
                else:
                    print(f"✗ Validation failed: {msg}")
            else:
                print(
                    f"️  No solution found (status: {stats['status']})"
                )

            results.append(
                {
                    "image": img_path.name,
                    "size": f"{w}×{h}",
                    "rectangles": stats["rectangle_count"],
                    "time": stats["solve_time"],
                    "status": stats["status"],
                }
            )

        except ValueError as e:
            print(f"Error: {e}")
            results.append(
                {
                    "image": img_path.name,
                    "size": "N/A",
                    "rectangles": 0,
                    "time": 0,
                    "status": "error",
                }
            )

        except Exception as e:
            print(f" Unexpected error: {e}")
            results.append(
                {
                    "image": img_path.name,
                    "size": "N/A",
                    "rectangles": 0,
                    "time": 0,
                    "status": "error",
                }
            )

    total_time = time.time() - total_start

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"{'Image':<30} {'Size':<10} {'Rects':<8} {'Time':<8} {'Status':<10}"
    )
    print("-" * 70)

    for r in results:
        print(
            f"{r['image']:<30} {r['size']:<10} {r['rectangles']:<8} "
            f"{r['time']:<8.2f} {r['status']:<10}"
        )

    print("=" * 70)
    print(f"Total time: {total_time:.2f}s")
    print(
        f"Average time per image: "
        f"{total_time / len(image_paths):.2f}s"
    )

    successful = sum(1 for r in results if r["status"] == "optimal")
    print(
        f"Successful: {successful}/{len(image_paths)} "
        f"({successful/len(image_paths)*100:.1f}%)"
    )