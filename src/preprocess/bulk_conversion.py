"""Bulk conversion of images to binary format."""

import cv2
import numpy as np
from pathlib import Path

# Import earlier preprocessing function
from src.preprocess.preprocessing import image_to_binary


def bulk_convert_to_binary(
    input_dir: str,
    output_dir: str,
    save_numpy: bool = True,
    invert: bool = True,
):
    """Convert all images in input_dir to binary masks.

    Saves PNGs and optionally NumPy arrays in separate folders.

    Args:
        input_dir: Directory with input images.
        output_dir: Base output directory.
        save_numpy: If True, save .npy arrays in subfolder.
        invert: If True, use THRESH_BINARY_INV (dark object on light
            background). If False, use THRESH_BINARY (already-binary
            image with white object on black background).

    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create separate subfolders for PNGs and NPys
    png_dir = output_path / "png"
    npy_dir = output_path / "npy"

    png_dir.mkdir(parents=True, exist_ok=True)
    if save_numpy:
        npy_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    for img_file in input_path.glob("*.*"):
        if img_file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        # Convert to binary (leaf=255, background=0)
        binary = image_to_binary(str(img_file), invert=invert)

        # Save PNG (human-friendly)
        out_png = png_dir / f"{img_file.stem}_binary.png"
        cv2.imwrite(str(out_png), binary)

        # Save NumPy array (machine-friendly, exact)
        if save_numpy:
            out_npy = npy_dir / f"{img_file.stem}.npy"
            np.save(out_npy, binary)


if __name__ == "__main__":
    bulk_convert_to_binary("../../data/datasets/leafs_binary/png/",
                           "../../data/datasets/leafs_binary_fix/", invert=True)
    print("Bulk conversion completed.")
