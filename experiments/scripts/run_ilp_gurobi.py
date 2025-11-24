"""
Run Gurobi ILP solvers to establish baseline/ground truth for small test images.

Saves results in separate directory structure to distinguish from CBC:
- experiments/solver_results/gurobi/rectangles/
- experiments/solver_results/gurobi/visualizations/
- experiments/solver_results/gurobi/csv/
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
from src.solvers.ilp_solver_gurobi import run_ilp_gurobi, validate_solution
from src.utils.utils import draw_solution

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))


def run_ilp_gurobi_baseline(
    image_dir_name: str,
    max_images: int = None,
    timeout: int = 300,
    max_candidates: int = 100000,
    threads: int = 0,
    mip_gap: float = 0.0,
):
    """Run Gurobi ILP solvers on small test images to establish ground truth.

    Parameters
    ----------
    image_dir_name : str
        Directory name under res/figures/ (e.g., "validation", "synthetic").
    max_images : int, optional
        Maximum number of images to process. If None, processes all.
    timeout : int, optional
        Maximum solving time per image in seconds (default: 300).
    max_candidates : int, optional
        Maximum candidate rectangles to generate (default: 100000).
    threads : int, optional
        Number of threads for Gurobi (0 = auto, default: 0).
    mip_gap : float, optional
        MIP optimality gap tolerance (default: 0.0 = exact optimal).

    """
    # Setup paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    image_dir = project_root / "res/figures" / image_dir_name / "npy"

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}\n"
            f"Available directories in res/figures/: "
            f"{[d.name for d in (project_root / 'res/figures').iterdir() if d.is_dir()]}"
        )

    # Get image paths
    image_paths = sorted(image_dir.glob("*.npy"))

    if not image_paths:
        raise FileNotFoundError(f"No .npy files found in {image_dir}")

    # Limit to max_images if specified
    if max_images is not None:
        image_paths = image_paths[:max_images]

    mode_str = f"TEST MODE ({len(image_paths)} images)" if max_images else f"ALL IMAGES ({len(image_paths)})"

    print("=" * 70)
    print(f"GUROBI ILP BASELINE SOLVER - {mode_str}")
    print("=" * 70)
    print(f"Directory: {image_dir_name}")
    print(f"Images to process: {len(image_paths)}")
    print(f"Timeout per image: {timeout}s")
    print(f"Max candidates: {max_candidates:,}")
    print(f"Threads: {threads} (0 = auto)")
    print(f"MIP gap: {mip_gap} (0.0 = exact optimal)")
    print("=" * 70)

    # Create output directories in solver_results/gurobi
    base_dir = project_root / "experiments/solver_results/gurobi"

    rect_dir = base_dir / "rectangles" / image_dir_name
    rect_dir.mkdir(parents=True, exist_ok=True)

    viz_dir = base_dir / "visualizations" / image_dir_name
    viz_dir.mkdir(parents=True, exist_ok=True)

    csv_dir = base_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    # CSV file for results (replace slashes for subdirectories)
    csv_filename = image_dir_name.replace('/', '_').replace('\\', '_') + "_results.csv"
    csv_path = csv_dir / csv_filename

    # CSV fieldnames for incremental writing
    import csv
    csv_fieldnames = ['image', 'size', 'pixels', 'rectangles', 'candidates',
                     'solve_time', 'enum_time', 'ilp_time', 'mip_gap', 'status']

    # Write CSV header if file doesn't exist
    if not csv_path.exists():
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()

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

            # Run Gurobi ILP solvers
            rectangles, stats = run_ilp_gurobi(
                img,
                timeout=timeout,
                max_candidates=max_candidates,
                verbose=True,
                threads=threads,
                mip_gap=mip_gap,
            )

            # Validate and save solution
            if rectangles:
                # Normalize image for validation and visualization
                img_normalized = (img > 0).astype(np.uint8)
                is_valid, msg = validate_solution(img_normalized, rectangles)

                print(f"Validation: {msg}")

                # Save rectangles to JSON
                image_stem = Path(img_path.name).stem
                rect_filename = f"{image_stem}_gurobi_{len(rectangles)}_rects.json"
                rect_path = rect_dir / rect_filename

                data = {
                    "image_name": img_path.name,
                    "algorithm": "ilp_gurobi",
                    "optimal": stats["status"] == "optimal",
                    "rectangle_count": len(rectangles),
                    "rectangles": [
                        [int(x), int(y), int(w), int(h)]
                        for x, y, w, h in rectangles
                    ],
                    "stats": {
                        "status": stats["status"],
                        "solve_time": stats["solve_time"],
                        "enum_time": stats["enum_time"],
                        "ilp_time": stats["ilp_time"],
                        "candidates": stats["candidates"],
                        "mip_gap": stats["mip_gap"],
                        "objective": stats["objective"],
                    },
                    "validation": {"valid": is_valid, "message": msg}
                }

                with open(rect_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)

                print(f"Saved rectangles: {rect_path}")

                # Save visualization
                viz_filename = f"{image_stem}_gurobi_{len(rectangles)}_rects.png"
                viz_path = viz_dir / viz_filename
                draw_solution(img_normalized, rectangles, save_path=str(viz_path))
                print(f"Saved visualization: {viz_path}")

            else:
                print(f"No solution found (status: {stats['status']})")

            # Store result
            result = {
                'image': img_path.name,
                'size': f"{w}×{h}",
                'pixels': pixel_count,
                'rectangles': stats['rectangle_count'],
                'candidates': stats['candidates'],
                'solve_time': stats['solve_time'],
                'enum_time': stats['enum_time'],
                'ilp_time': stats['ilp_time'],
                'mip_gap': stats['mip_gap'],
                'status': stats['status']
            }
            results.append(result)

            # Append result to CSV immediately
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                writer.writerow(result)

        except ValueError as e:
            print(f"✗ Error: {e}")
            result = {
                'image': img_path.name,
                'size': 'N/A',
                'pixels': 0,
                'rectangles': 0,
                'candidates': 0,
                'solve_time': 0,
                'enum_time': 0,
                'ilp_time': 0,
                'mip_gap': float('inf'),
                'status': 'error'
            }
            results.append(result)

            # Append error result to CSV immediately
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                writer.writerow(result)

        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            result = {
                'image': img_path.name,
                'size': 'N/A',
                'pixels': 0,
                'rectangles': 0,
                'candidates': 0,
                'solve_time': 0,
                'enum_time': 0,
                'ilp_time': 0,
                'mip_gap': float('inf'),
                'status': 'error'
            }
            results.append(result)

            # Append error result to CSV immediately
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                writer.writerow(result)

    total_time = time.time() - total_start

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Image':<30} {'Size':<10} {'Rects':<8} {'Time':<10} {'Gap':<8} {'Status':<12}")
    print("-" * 70)

    for r in results:
        gap_str = f"{r['mip_gap']:.4f}" if r['mip_gap'] != float('inf') else "N/A"
        print(
            f"{r['image']:<30} {r['size']:<10} {r['rectangles']:<8} "
            f"{r['solve_time']:<10.2f} {gap_str:<8} {r['status']:<12}"
        )

    print("=" * 70)
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per image: {total_time / len(image_paths):.2f}s")

    successful = sum(1 for r in results if r['status'] == 'optimal')
    print(f"Successful: {successful}/{len(image_paths)} ({successful/len(image_paths)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("OUTPUT STRUCTURE:")
    print("=" * 70)
    print(f"Rectangles: {rect_dir}")
    print(f"Visualizations: {viz_dir}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    import sys

    # Parse arguments
    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
    else:
        image_dir = "validation"

    if len(sys.argv) > 2:
        max_imgs = int(sys.argv[2])
    else:
        max_imgs = 6

    if len(sys.argv) > 3:
        max_cands = int(sys.argv[3])
    else:
        max_cands = 1_200_000

    if len(sys.argv) > 4:
        num_threads = int(sys.argv[4])
    else:
        num_threads = 0  # Auto

    print(f"\nUsage: python run_ilp_gurobi.py "
          f"[directory] [max_images] [max_candidates] [threads]")
    print(f"Running with: directory={image_dir}, max_images={max_imgs}, "
          f"max_candidates={max_cands}, threads={num_threads}\n")

    # Run experiments
    run_ilp_gurobi_baseline(
        image_dir_name=image_dir,
        max_images=max_imgs,
        timeout=3600,
        max_candidates=max_cands,
        threads=num_threads,
        mip_gap=0.0,  # Exact optimal
    )