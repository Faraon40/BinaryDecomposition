"""CLI command: preprocess — binarize images to .npy / PNG."""

import sys
from pathlib import Path

import numpy as np


def handle_preprocess(args):
    """Binarize one image or a directory of images.

    Saves output as .npy arrays and/or binary PNGs depending on
    --format. Skips files that are already .npy arrays.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Expected attributes:
          input  — source file or directory
          output — destination directory
          invert — bool, True for dark-object-on-light-bg
          format — 'npy', 'png', or 'both'

    """
    from src.preprocess.preprocessing import image_to_binary

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    _IMAGE_EXT = frozenset(
        {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )

    if input_path.is_file():
        candidates = [input_path]
    elif input_path.is_dir():
        candidates = sorted(
            f for f in input_path.iterdir()
            if f.suffix.lower() in _IMAGE_EXT
        )
    else:
        print(f"Error: input path not found: {input_path}")
        sys.exit(1)

    if not candidates:
        print(f"No raster images found in: {input_path}")
        sys.exit(1)

    fmt = getattr(args, "format", "both")
    npy_dir = output_dir / "npy"
    png_dir = output_dir / "png"

    if fmt in ("npy", "both"):
        npy_dir.mkdir(parents=True, exist_ok=True)
    if fmt in ("png", "both"):
        png_dir.mkdir(parents=True, exist_ok=True)

    print(f"Binarizing {len(candidates)} image(s) → {output_dir}")

    for i, src in enumerate(candidates, 1):
        stem = src.stem
        print(f"  [{i}/{len(candidates)}] {src.name}", end="  ")

        try:
            binary = image_to_binary(str(src), invert=args.invert)

            if fmt in ("npy", "both"):
                arr = (binary > 0).astype(np.uint8)
                np.save(npy_dir / f"{stem}.npy", arr)

            if fmt in ("png", "both"):
                import cv2
                cv2.imwrite(str(png_dir / f"{stem}_binary.png"), binary)

            print("OK")

        except Exception as exc:
            print(f"ERROR: {exc}")

    print(f"Done. Output: {output_dir}")
