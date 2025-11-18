"""Bulk conversion of images to binary format."""

import cv2
import numpy as np
from pathlib import Path

# Import earlier preprocessing function
from src.preprocess.preprocessing import image_to_binary


def bulk_convert_to_binary(
    input_dir: str, output_dir: str, save_numpy: bool = True
):
    """Convert all images in input_dir to binary masks.

    Saves PNGs and optionally NumPy arrays in separate folders.

    Args:
        input_dir: Directory with input images.
        output_dir: Base output directory.
        save_numpy: If True, save .npy arrays in subfolder.

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

        # Convert to binary (leaf=0, background=255)
        binary = image_to_binary(str(img_file))

        # Save PNG (human-friendly)
        out_png = png_dir / f"{img_file.stem}_binary.png"
        cv2.imwrite(str(out_png), binary)

        # Save NumPy array (machine-friendly, exact)
        if save_numpy:
            out_npy = npy_dir / f"{img_file.stem}_binary.npy"
            np.save(out_npy, binary)


if __name__ == "__main__":
    bulk_convert_to_binary("../../res/figures/research_leafs",
                           "../../res/figures/research_leafs_binary")
    print("Bulk conversion completed.")
