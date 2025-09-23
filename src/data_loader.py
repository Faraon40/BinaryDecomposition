"""Import one of each leaf image into project directory."""

import random
import shutil
from pathlib import Path
from typing import List


def load_and_copy_one_image_per_leaf(
    root_dir: str,
    dest_dir: str,
    random_choice: bool = False,
    extensions=(".png", ".jpg", ".jpeg")
) -> List[Path]:
    """
    Collect one image from each leaf subfolder and copy it into dest_dir.

    Args:
        root_dir (str): Path to main dataset directory (e.g. 'trees_directories')
        dest_dir (str): Destination folder inside project (e.g. '../docs/figures/leafs')
        random_choice (bool): If True, pick a random image per folder.
                              If False, pick the first sorted image.
        extensions (tuple): Allowed image file extensions.

    Returns:
        List[Path]: List of copied image paths (inside dest_dir)
    """
    root = Path(root_dir)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    copied_paths = []

    # Each subdirectory = one leaf
    for leaf_dir in sorted(root.iterdir()):
        if leaf_dir.is_dir():
            # Collect all image files in this folder
            imgs = [p for p in leaf_dir.iterdir() if p.suffix.lower() in extensions]
            if not imgs:
                continue  # skip empty folders

            # Pick one image (first sorted or random)
            if random_choice:
                chosen = random.choice(imgs)
            else:
                chosen = sorted(imgs)[0]

            # Copy to destination, keeping folder name in filename
            dest_file = dest / f"{leaf_dir.name}{chosen.suffix.lower()}"
            shutil.copy(chosen, dest_file)
            copied_paths.append(dest_file)

    return copied_paths


if __name__ == "__main__":
    dataset_dir = "trees_directories"
    output_dir = "../docs/figures/leafs"

    images = load_and_copy_one_image_per_leaf(dataset_dir, output_dir, random_choice=False)

    print(f"Copied {len(images)} images into {output_dir}")
