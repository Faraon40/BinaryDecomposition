"""Convert image into binary image.

References:
    https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def image_to_binary(
    path: str, save_path: str = None, invert: bool = True
) -> np.ndarray:
    """
    Convert a leaf image with white background into a binary image.

    Object (leaf) -> white (255), background -> black (0).

    Args:
        path (str): Path to input image
        save_path (str, optional): If given, save binary image to path
        invert (bool): If True, use THRESH_BINARY_INV (dark object on light
            background, e.g. colored leaf on white background).
            If False, use THRESH_BINARY (light object on dark background,
            e.g. already-binary image where object is white).

    Returns:
        np.ndarray: Binary image (uint8, 0/255)
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu threshold - automatically finds best threshold
    # Use invert=True  (THRESH_BINARY_INV): dark object on light background
    #                   -> object=255, background=0
    # Use invert=False (THRESH_BINARY):     light object on dark background
    #                   -> object=255, background=0
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(gray, 0, 255, thresh_type + cv2.THRESH_OTSU)

    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Fill internal holes: flood-fill background from top-left corner,
    # then OR with original to recover any interior pixels lost by Otsu.
    # Works regardless of hole size (more robust than large MORPH_CLOSE).
    h, w = binary.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    filled = binary.copy()
    cv2.floodFill(filled, flood_mask, (0, 0), 255)
    # Pixels still black after flood-fill are interior holes -> set to white
    interior = cv2.bitwise_not(filled)
    binary = cv2.bitwise_or(binary, interior)

    if save_path:
        cv2.imwrite(save_path, binary)

    return binary
