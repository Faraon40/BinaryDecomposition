"""Convert GIF images to PNG format."""

import os

from PIL import Image


def convert_gifs_to_png(input_dir: str, output_dir: str):
    """Convert all GIF images in input_dir into PNG format."""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".gif"):
            gif_path = os.path.join(input_dir, filename)

            # Take only first frame from GIFs
            with Image.open(gif_path) as img:
                # Ensure correct transparency handling
                img = img.convert("RGBA")
                png_filename = os.path.splitext(filename)[0] + ".png"
                png_path = os.path.join(output_dir, png_filename)
                img.save(png_path, "PNG")

            print(f"Converted: {filename} → {png_filename}")


if __name__ == "__main__":
    input_dir = "../../data/datasets/objects_gifs"  # folder with GIFs
    output_dir = "../../data/datasets/objects"  # folder for PNGs
    convert_gifs_to_png(input_dir, output_dir)
