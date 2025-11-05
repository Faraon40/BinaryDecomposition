"""Convert image into binary image.

References:
    https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

"""

import cv2
import numpy as np


def image_to_binary(path: str, save_path: str = None) -> np.ndarray:
    """
    Convert a leaf image with white background into a binary image.

    Object (leaf) -> white (255), background -> black (0).

    Args:
        path (str): Path to input image
        save_path (str, optional): If given, save binary image to path

    Returns:
        np.ndarray: Binary image (uint8, 0/255)
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define white background range
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 40, 255], dtype=np.uint8)

    # Mask white regions
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Invert: object (leaf) = 255 (1), background = 0
    binary = cv2.bitwise_not(mask_white)

    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    # Removing noise on our object, closing small holes
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # Removing background noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Save if requested
    if save_path:
        cv2.imwrite(save_path, binary)

    return binary


image_to_binary(
    "../../res/figures/original/Vitis_riparia_5.png",
    save_path="../../res/figures/Vitis_riparia_5_binary.png",
)
