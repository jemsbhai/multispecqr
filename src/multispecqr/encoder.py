"""
Multispectral QR encoder.

Provides:
    encode_rgb(data_r, data_g, data_b, version=4, ec="M") -> PIL.Image
    encode_layers(data_list, version=4, ec="M") -> PIL.Image
"""
from __future__ import annotations

import numpy as np
from PIL import Image
import qrcode

from .palette import _select_palette


def _make_layer(data: str, version: int, ec: str) -> np.ndarray:
    """Return a binary (0/1) numpy array for one QR layer."""
    qr = qrcode.QRCode(
        version=version,
        error_correction=getattr(qrcode.constants, f"ERROR_CORRECT_{ec}"),
    )
    qr.add_data(data)
    qr.make(fit=False)
    img = qr.make_image(fill_color="black", back_color="white").convert("1")
    return (np.array(img) == 0).astype(np.uint8)  # 1 = black module


def encode_rgb(
    data_r: str,
    data_g: str,
    data_b: str,
    *,
    version: int = 4,
    ec: str = "M",
) -> Image.Image:
    """
    Combine three payloads into a single RGB QR image.

    Each payload is encoded as an independent monochrome QR layer,
    then assigned to one color channel: R, G, B.

    Args:
        data_r: Payload string for Red channel
        data_g: Payload string for Green channel
        data_b: Payload string for Blue channel
        version: QR code version 1-40. Higher versions hold more data. Default: 4
        ec: Error correction level - "L", "M", "Q", or "H". Default: "M"

    Returns:
        PIL.Image in RGB mode
    """
    r = _make_layer(data_r, version, ec)
    g = _make_layer(data_g, version, ec)
    b = _make_layer(data_b, version, ec)

    if not (r.shape == g.shape == b.shape):
        raise ValueError("Layers ended up different sizes; pick same version.")

    rgb_stack = np.stack([r * 255, g * 255, b * 255], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb_stack)


def encode_layers(data_list: list[str], *, version: int = 4, ec: str = "M") -> Image.Image:
    """
    Encode N binary QR layers into a color QR using an appropriate color palette.

    Automatically selects palette based on number of layers:
    - 1-6 layers: 64-color palette (6-bit)
    - 7-8 layers: 256-color palette (8-bit)
    - 9 layers: 512-color palette (9-bit)

    Args:
        data_list: List of payload strings (1-9 layers)
        version: QR code version 1-40. Default: 4
        ec: Error correction level ("L", "M", "Q", "H"). Default: "M"

    Returns:
        PIL.Image in RGB mode

    Raises:
        ValueError: If more than 9 layers provided
    """
    num_layers = len(data_list)
    if num_layers > 9:
        raise ValueError("Maximum of 9 layers supported.")

    # Select appropriate palette based on layer count
    codebook, _, num_bits = _select_palette(num_layers)

    layers = [_make_layer(data, version, ec) for data in data_list]
    shape = layers[0].shape
    if not all(l.shape == shape for l in layers):
        raise ValueError("QR layers must all have the same shape.")

    h, w = shape
    img_arr = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            # Build bit-vector, padding with zeros for unused layers
            bits = [layer[y, x] for layer in layers] + [0] * (num_bits - num_layers)
            key = tuple(bits)
            img_arr[y, x] = codebook.get(key, (255, 255, 255))  # default to white

    return Image.fromarray(img_arr)
