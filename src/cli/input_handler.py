"""Input path resolution and image classification for the CLI."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
)


@dataclass
class InputSpec:
    """Describes a single image to be processed.

    Attributes
    ----------
    path : Path
        Absolute path to the image file.
    is_npy : bool
        True if file is a .npy array; False if raster image.
    needs_binarization : bool
        True if image_to_binary() must be called before running an algo.
    invert : bool
        Passed to image_to_binary() — True for dark-object-on-light-bg.

    """

    path: Path
    is_npy: bool
    needs_binarization: bool
    invert: bool = True


def resolve_input(
    path_str: str,
    invert: bool = True,
    max_images: int = None,
) -> List[InputSpec]:
    """Resolve an input path to a list of InputSpec objects.

    Handles single files and directories. For directories, all .npy,
    .png, .jpg (and related) files are collected and sorted.

    Parameters
    ----------
    path_str : str
        Path to a single image file or a directory.
    invert : bool
        Passed to image_to_binary() for raster inputs.
    max_images : int, optional
        Limit to first N images (useful for quick tests).

    Returns
    -------
    list of InputSpec
        One spec per image, ready for the execution loop.

    Raises
    ------
    FileNotFoundError
        If path does not exist or directory contains no images.

    """
    path = Path(path_str).resolve()

    if path.is_file():
        return [_classify(path, invert)]

    if path.is_dir():
        candidates = sorted(
            f for f in path.iterdir()
            if f.suffix == ".npy"
            or f.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not candidates:
            raise FileNotFoundError(
                f"No supported images found in: {path}\n"
                f"Supported: .npy, .png, .jpg, .jpeg, .bmp, .tif"
            )
        specs = [_classify(f, invert) for f in candidates]
        if max_images is not None:
            specs = specs[:max_images]
        return specs

    raise FileNotFoundError(f"Input path not found: {path}")


def load_image(spec: InputSpec) -> np.ndarray:
    """Load an image described by an InputSpec into a binary array.

    Parameters
    ----------
    spec : InputSpec
        Describes the file to load.

    Returns
    -------
    np.ndarray
        2D binary array with values in {0, 1}.

    """
    if spec.is_npy:
        img = np.load(spec.path)
        return (img > 0).astype(int)

    from src.preprocess.preprocessing import image_to_binary
    binary = image_to_binary(str(spec.path), invert=spec.invert)
    return (binary > 0).astype(int)


def _classify(path: Path, invert: bool) -> InputSpec:
    if path.suffix == ".npy":
        return InputSpec(
            path=path,
            is_npy=True,
            needs_binarization=False,
            invert=invert,
        )
    return InputSpec(
        path=path,
        is_npy=False,
        needs_binarization=True,
        invert=invert,
    )


def derive_dataset_name(path_str: str) -> str:
    """Derive a dataset label from the input path.

    Used to build output directory hierarchy.

    Parameters
    ----------
    path_str : str
        Original --input argument value.

    Returns
    -------
    str
        A short name: directory name for dirs, 'single' for files.

    """
    path = Path(path_str).resolve()
    if path.is_dir():
        return path.name
    return "single"
