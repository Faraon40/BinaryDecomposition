"""
Validate and visualize saved rectangle solutions.

This script loads saved solutions from JSON files and allows you to:
- Validate the solution (check coverage and overlaps)
- Visualize the solution
- Compare multiple solutions for the same image
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np

from experiments.scripts.run import load_solution_rectangles
from src.utils.utils import draw_solution


def validate_rectangles(
    image: np.ndarray,
    rectangles: List[Tuple[int, int, int, int]]
) -> dict:
    """Validate rectangle solution against image.

    Parameters
    ----------
    image : np.ndarray
        Binary image (0s and 1s).
    rectangles : list of tuple
        List of (x, y, width, height) rectangles.

    Returns
    -------
    dict
        Validation results with coverage stats and overlap info.

    """
    height, width = image.shape
    coverage = np.zeros_like(image)

    # Mark covered pixels
    for x, y, w, h in rectangles:
        coverage[y:y+h, x:x+w] += 1

    # Calculate statistics
    pixels_to_cover = np.sum(image == 1)
    covered_pixels = np.sum((coverage > 0) & (image == 1))
    uncovered_pixels = pixels_to_cover - covered_pixels
    overlapping_pixels = np.sum(coverage > 1)
    background_covered = np.sum((coverage > 0) & (image == 0))

    is_valid = (
        uncovered_pixels == 0
        and overlapping_pixels == 0
        and background_covered == 0
    )

    return {
        "is_valid": is_valid,
        "rectangle_count": len(rectangles),
        "pixels_to_cover": int(pixels_to_cover),
        "covered_pixels": int(covered_pixels),
        "uncovered_pixels": int(uncovered_pixels),
        "overlapping_pixels": int(overlapping_pixels),
        "background_covered": int(background_covered),
        "coverage_percentage": (
            100.0 * covered_pixels / pixels_to_cover
            if pixels_to_cover > 0 else 0
        )
    }


def main():
    """Example: Load and validate a saved solution."""
    # Example usage - adjust paths as needed
    project_root = Path(__file__).parent.parent.parent

    # Load solution
    solution_path = (
        project_root
        / "experiments/results/rectangles"
        / "Quercus_robur_2_binary"  # Image name
        / "42.json"  # Rectangle count
    )

    if not solution_path.exists():
        print(f"Solution not found: {solution_path}")
        print("\nAvailable solutions:")
        rect_dir = project_root / "experiments/results/rectangles"
        if rect_dir.exists():
            for img_dir in sorted(rect_dir.iterdir()):
                if img_dir.is_dir():
                    solutions = list(img_dir.glob("*.json"))
                    print(f"  {img_dir.name}:")
                    for sol in sorted(solutions):
                        print(f"    - {sol.name}")
        return

    print(f"Loading solution from: {solution_path.name}")
    rectangles, metadata = load_solution_rectangles(str(solution_path))

    print(f"\n{'='*60}")
    print("SOLUTION METADATA")
    print(f"{'='*60}")
    print(f"Image: {metadata['image_name']}")
    print(f"Rectangles: {metadata['rectangle_count']}")
    print(f"Algorithm: {metadata['config']['algorithm']}")
    print(f"Mutation combo: {metadata['config']['mutation_combo']}")
    print(f"Seed: {metadata['config']['seed']}")
    print(f"Final fitness: {metadata['metrics'].get('final_fitness', 'N/A')}")
    print(f"Execution time: {metadata['metrics']['execution_time_sec']:.1f}s")

    # Load image
    image_name = metadata['image_name']
    # Try to find the image in common locations
    possible_dirs = [
        "research_leafs_binary",
        "leafs_binary",
        "objects_binary"
    ]

    image_path = None
    for dir_name in possible_dirs:
        test_path = (
            project_root / "res/figures" / dir_name / "npy" / image_name
        )
        if test_path.exists():
            image_path = test_path
            break

    if not image_path:
        print(f"\nWarning: Could not find image {image_name}")
        print("Skipping validation.")
        return

    # Load and validate
    img = np.load(image_path)
    img = (img > 0).astype(int)

    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}")
    validation = validate_rectangles(img, rectangles)

    for key, value in validation.items():
        if key == "is_valid":
            status = "✓ VALID" if value else "✗ INVALID"
            print(f"{key}: {status}")
        else:
            print(f"{key}: {value}")

    # Visualize
    print(f"\n{'='*60}")
    print("VISUALIZATION")
    print(f"{'='*60}")
    viz_path = (
        project_root
        / "experiments/results/visualizations"
        / f"{Path(image_name).stem}_validated.png"
    )
    viz_path.parent.mkdir(parents=True, exist_ok=True)

    draw_solution(img, rectangles, save_path=str(viz_path), show=False)
    print(f"Saved visualization to: {viz_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()