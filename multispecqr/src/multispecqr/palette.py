"""
Color palettes & codebooks for extended-layer QR.

`RGB_CMY_6` gives 6 orthogonal colors suitable for first prototype,
mapping a layer-activation bit-vector -> RGB triplet.
"""
from __future__ import annotations

from typing import Dict, Tuple

# Ordered list of colors (R, G, B, C, M, Y)
_PALETTE_6 = {
    (1, 0, 0, 0, 0, 0): (255,   0,   0),  # Red
    (0, 1, 0, 0, 0, 0): (  0, 255,   0),  # Green
    (0, 0, 1, 0, 0, 0): (  0,   0, 255),  # Blue
    (0, 0, 0, 1, 0, 0): (  0, 255, 255),  # Cyan
    (0, 0, 0, 0, 1, 0): (255,   0, 255),  # Magenta
    (0, 0, 0, 0, 0, 1): (255, 255,   0),  # Yellow
    (0, 0, 0, 0, 0, 0): (255, 255, 255),  # White (no layer)
}

def palette_6() -> Dict[Tuple[int, ...], Tuple[int, int, int]]:
    """Return the 6-layer RGB ↔ bit-vector codebook."""
    return _PALETTE_6.copy()
