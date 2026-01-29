"""
Color palettes & codebooks for extended-layer QR.

Provides a full 64-color palette that supports all 2^6 combinations
of 6 binary layers, mapping each bit-vector to a unique RGB color.

Color encoding scheme:
- Bits 0,1 determine R level: {0, 85, 170, 255}
- Bits 2,3 determine G level: {0, 85, 170, 255}
- Bits 4,5 determine B level: {0, 85, 170, 255}

This gives 4^3 = 64 unique, evenly-spaced colors in the RGB cube.
"""
from __future__ import annotations

from typing import Dict, Tuple
import itertools


def _bitvec_to_color(bits: Tuple[int, ...]) -> Tuple[int, int, int]:
    """Convert a 6-bit vector to an RGB color."""
    # Ensure we have exactly 6 bits
    bits = tuple(bits) + (0,) * (6 - len(bits))

    # Combine pairs of bits into 2-bit values (0-3), then scale to 0-255
    r_level = bits[0] + bits[1] * 2  # 0-3
    g_level = bits[2] + bits[3] * 2  # 0-3
    b_level = bits[4] + bits[5] * 2  # 0-3

    return (r_level * 85, g_level * 85, b_level * 85)


def _color_to_bitvec(r: int, g: int, b: int) -> Tuple[int, ...]:
    """Convert an RGB color to a 6-bit vector."""
    # Quantize to nearest level (0, 85, 170, 255)
    r_level = min(3, max(0, round(r / 85)))
    g_level = min(3, max(0, round(g / 85)))
    b_level = min(3, max(0, round(b / 85)))

    # Extract individual bits
    return (
        r_level % 2, r_level // 2,
        g_level % 2, g_level // 2,
        b_level % 2, b_level // 2,
    )


# Build the full 64-color palette
def _build_palette_64() -> Dict[Tuple[int, ...], Tuple[int, int, int]]:
    """Generate all 64 bit-vector to color mappings."""
    palette = {}
    for bits in itertools.product([0, 1], repeat=6):
        palette[bits] = _bitvec_to_color(bits)
    return palette


_PALETTE_64 = _build_palette_64()


def palette_6() -> Dict[Tuple[int, ...], Tuple[int, int, int]]:
    """Return the 6-layer bit-vector -> RGB codebook (64 colors)."""
    return _PALETTE_64.copy()


def inverse_palette_6() -> Dict[Tuple[int, int, int], Tuple[int, ...]]:
    """Return the inverse codebook: RGB triplet -> bit-vector."""
    return {color: bitvec for bitvec, color in _PALETTE_64.items()}
