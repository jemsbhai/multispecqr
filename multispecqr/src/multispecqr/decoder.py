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


def decode_layers(img: Image.Image, num_layers: int | None = None) -> List[str]:
    """
    Decode a multi-layer QR image encoded with the 6-color palette.

    Args:
        img: RGB PIL Image encoded with encode_layers()
        num_layers: Number of layers to decode (1-6). If None, auto-detect.

    Returns:
        List of decoded strings, one per layer.
    """
    from .palette import inverse_palette_6

    if img.mode != "RGB":
        raise ValueError("Expected an RGB image")

    # Default to 6 layers if not specified
    if num_layers is None:
        num_layers = 6

    arr = np.array(img)  # H x W x 3
    h, w = arr.shape[:2]

    # Build inverse lookup and color array for nearest-neighbor matching
    inv_palette = inverse_palette_6()
    palette_colors = np.array(list(inv_palette.keys()), dtype=np.uint8)  # (7, 3)
    palette_bitvecs = list(inv_palette.values())

    # Reshape image for efficient color matching
    pixels = arr.reshape(-1, 3)  # (H*W, 3)

    # Find nearest palette color for each pixel using Euclidean distance
    # Compute distances: (H*W, 7)
    distances = np.linalg.norm(
        pixels[:, np.newaxis, :].astype(np.float32) - palette_colors[np.newaxis, :, :].astype(np.float32),
        axis=2
    )
    nearest_idx = np.argmin(distances, axis=1)  # (H*W,)

    # Map each pixel to its bit-vector
    bitvec_array = np.array([palette_bitvecs[i] for i in nearest_idx])  # (H*W, 6)
    bitvec_array = bitvec_array.reshape(h, w, 6)  # (H, W, 6)

    # Extract and decode each layer
    results: List[str] = []
    for layer_idx in range(num_layers):
        # Extract layer: 1 = black module, 0 = white module
        layer = bitvec_array[:, :, layer_idx]

        # Convert to QR-decodable format: 0 = black, 255 = white
        # Then invert for OpenCV detector which expects black modules to be 0
        binary = ((1 - layer) * 255).astype(np.uint8)

        decoded = _decode_single_layer(binary)
        results.append(decoded or "")

    return results
