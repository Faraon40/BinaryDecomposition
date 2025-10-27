"""Generate random binary images for algorithm testing and benchmarking.

This module provides functions to create synthetic binary images with
pure random noise at various densities for testing rectangle decomposition
algorithms.
"""

import json
from pathlib import Path

import cv2
import numpy as np


def generate_random_noise(
    width: int, height: int, density: float = 0.5, seed: int = None
) -> np.ndarray:
    """Generate random binary image with specified density.

    Each pixel is independently set to 1 with probability equal to
    density parameter. Creates pure random noise without structure.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        density: Fraction of 1s (foreground), range [0, 1].
        seed: Random seed for reproducibility.

    Returns:
        Binary image (0s and 1s) as uint8 array.

    """
    if seed is not None:
        np.random.seed(seed)

    img = np.random.rand(height, width)
    return (img < density).astype(np.uint8)


def save_generated_image(
    img: np.ndarray,
    output_dir: str,
    name: str,
    metadata: dict = None,
):
    """Save generated image with metadata.

    Saves image in three formats:
    - PNG (visualization, 0/255 values)
    - NPY (algorithm input, 0/1 values)
    - JSON (metadata with image properties)

    Args:
        img: Binary image to save (0s and 1s).
        output_dir: Output directory path.
        name: Base filename (without extension).
        metadata: Optional metadata dictionary.

    """
    output_path = Path(output_dir)

    # Create subdirectories for each format
    png_dir = output_path / "png"
    npy_dir = output_path / "npy"
    json_dir = output_path / "json"

    png_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # Save PNG (visualization)
    png_path = png_dir / f"{name}.png"
    cv2.imwrite(str(png_path), img * 255)

    # Save NPY (algorithm input)
    npy_path = npy_dir / f"{name}.npy"
    np.save(npy_path, img)

    # Save metadata JSON
    meta = metadata or {}
    meta.update(
        {
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "ones_count": int(np.sum(img)),
            "density": float(np.mean(img)),
        }
    )

    json_path = json_dir / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {name} ({img.shape[1]}x{img.shape[0]})")


def generate_dataset(
    output_dir: str = "../../res/figures/synthetic",
):
    """Generate synthetic dataset of pure random noise images.

    Creates images at various sizes and noise densities for systematic
    testing of decomposition algorithms.

    Images generated:
    - Small (50-100 pixels): 15 images (3 sizes × 5 densities)
    - Medium (200-500 pixels): 15 images (3 sizes × 5 densities)
    - Large (1000-1500 pixels): 10 images (2 sizes × 5 densities)
    Total: 40 images

    Args:
        output_dir: Directory to save generated images.

    """
    print("Generating synthetic random noise dataset...")
    print(f"Output directory: {output_dir}\n")

    densities = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Small images (50-100 pixels)
    print("=== Small Images ===")
    seed_offset = 100
    for size in [50, 75, 100]:
        for i, density in enumerate(densities):
            seed = seed_offset + i
            img = generate_random_noise(
                size, size, density=density, seed=seed
            )
            save_generated_image(
                img,
                output_dir,
                f"small_{size}x{size}_density{int(density*100)}",
                {"size": "small", "density": density},
            )
        seed_offset += 100

    # Medium images (200-500 pixels)
    print("\n=== Medium Images ===")
    seed_offset = 400
    for size in [200, 350, 500]:
        for i, density in enumerate(densities):
            seed = seed_offset + i
            img = generate_random_noise(
                size, size, density=density, seed=seed
            )
            save_generated_image(
                img,
                output_dir,
                f"medium_{size}x{size}_density{int(density*100)}",
                {"size": "medium", "density": density},
            )
        seed_offset += 100

    # Large images (1000-1500 pixels)
    print("\n=== Large Images ===")
    seed_offset = 800
    for size in [1000, 1500]:
        for i, density in enumerate(densities):
            seed = seed_offset + i
            img = generate_random_noise(
                size, size, density=density, seed=seed
            )
            save_generated_image(
                img,
                output_dir,
                f"large_{size}x{size}_density{int(density*100)}",
                {"size": "large", "density": density},
            )
        seed_offset += 100

    print("\n✅ Dataset generation complete!")
    print("Total images generated: 40")
    print(f"Sizes: 3 small + 3 medium + 2 large")
    print(f"Densities: {densities}")


if __name__ == "__main__":
    generate_dataset()