"""
Copy leaf images from the dataset to the research_leafs directory.

This script reads rectangles_fixed.csv and copies all corresponding images
from the trees_directories dataset to res/figures/research_leafs.
"""

import csv
import os
import shutil
from pathlib import Path


def parse_image_name(image_title):
    """
    Parse image title to extract species name and number.

    Args:
        image_title: String like 'acer_campestre_1'

    Returns:
        tuple: (directory_name, filename) like ('Acer_campestre',
        'Acer_campestre_1.png')
    """
    parts = image_title.split('_')

    # Convert to proper case for directory name
    # e.g., 'acer_campestre' -> 'Acer_campestre'
    genus = parts[0].capitalize()
    species = '_'.join(parts[1:-1])
    number = parts[-1]

    directory_name = f"{genus}_{species}"
    filename = f"{directory_name}_{number}.png"

    return directory_name, filename


def main():
    """Copy images from dataset to research_leafs directory."""
    # Define paths
    csv_path = Path(
        "/experiments/results/csv/rectangles_fixed.csv"
    )
    dataset_root = Path(
        "D:/Vysoká Škola/Ing .Semester 3/DP - Diplomova Praca/"
        "Datasets/trees_directories/trees_directories"
    )
    output_dir = Path(
        "/res/figures/research_leafs"
    )

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track statistics
    copied = 0
    not_found = []

    # Read CSV and copy images
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_title = row['image_title'].strip()

            # Skip empty rows
            if not image_title:
                continue

            try:
                # Parse image name
                directory_name, filename = parse_image_name(image_title)

                # Construct source path
                source_path = dataset_root / directory_name / filename

                # Construct destination path
                dest_path = output_dir / filename

                # Copy file if it exists
                if source_path.exists():
                    shutil.copy2(source_path, dest_path)
                    copied += 1
                    print(f"Copied: {filename}")
                else:
                    not_found.append(str(source_path))
                    print(f"NOT FOUND: {source_path}")

            except Exception as e:
                print(f"Error processing {image_title}: {e}")

    # Print summary
    print("\n" + "="*60)
    print(f"Successfully copied: {copied} images")
    print(f"Not found: {len(not_found)} images")

    if not_found:
        print("\nMissing images:")
        for path in not_found:
            print(f"  - {path}")


if __name__ == "__main__":
    main()