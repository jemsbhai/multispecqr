"""
Multispectral QR – RGB encoder prototype.

Provides:
    encode_rgb(data_r, data_g, data_b, version=4, ec="M") -> PIL.Image
"""
from __future__ import annotations

import numpy as np
from PIL import Image
import qrcode

# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
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
    """
    r = _make_layer(data_r, version, ec)
    g = _make_layer(data_g, version, ec)
    b = _make_layer(data_b, version, ec)

    if not (r.shape == g.shape == b.shape):
        raise ValueError("Layers ended up different sizes; pick same version.")

    rgb_stack = np.stack([r * 255, g * 255, b * 255], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb_stack, mode="RGB")



from .palette import palette_6

def encode_layers(data_list: list[str], *, version: int = 4, ec: str = "M") -> Image.Image:
    """
    Encode N binary QR layers into a color QR using the RGB_CMY 6-color palette.
    Up to 6 layers. Returns a PIL.Image in RGB mode.
    """
    if len(data_list) > 6:
        raise ValueError("Maximum of 6 layers supported in palette_6 mode.")

    layers = [_make_layer(data, version, ec) for data in data_list]
    shape = layers[0].shape
    if not all(l.shape == shape for l in layers):
        raise ValueError("QR layers must all have the same shape.")

    h, w = shape
    img_arr = np.zeros((h, w, 3), dtype=np.uint8)
    codebook = palette_6()

    num_layers = len(layers)
    for y in range(h):
        for x in range(w):
            # Build 6-element bit-vector, padding with zeros for unused layers
            bits = [layer[y, x] for layer in layers] + [0] * (6 - num_layers)
            key = tuple(bits)
            img_arr[y, x] = codebook.get(key, (255, 255, 255))  # default to white

    return Image.fromarray(img_arr, mode="RGB")





"""QR layer encoder — to be implemented."""