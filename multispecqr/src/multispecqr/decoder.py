"""
Multispectral QR – baseline RGB decoder.

Function:
    decode_rgb(img) -> list[str]
"""
from __future__ import annotations

from typing import List

import numpy as np
import cv2
from PIL import Image


def _decode_single_layer(layer_img: np.ndarray) -> str | None:
    """
    Try to decode a monochrome QR layer (0/255 uint8).
    Returns the decoded text, or None if decoding fails.
    """
    detector = cv2.QRCodeDetector()
    data, _, _ = detector.detectAndDecode(layer_img)
    return data or None


def decode_rgb(img: Image.Image) -> List[str]:
    """
    Split an RGB QR image into R, G, B layers, threshold each,
    and return a list of decoded strings (order: R, G, B).

    Layers that fail to decode are returned as an empty string.
    """
    if img.mode != "RGB":
        raise ValueError("Expected an RGB image")

    arr = np.array(img)  # H x W x 3
    results: List[str] = []

    for c in range(3):  # R, G, B
        channel = arr[:, :, c]
        # Simple global threshold: treat <128 as black
        _, binary = cv2.threshold(channel, 128, 255, cv2.THRESH_BINARY_INV)
        decoded = _decode_single_layer(binary)
        results.append(decoded or "")

    return results
