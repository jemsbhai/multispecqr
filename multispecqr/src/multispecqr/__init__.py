
from .__about__ import __version__

# Public API
from .encoder import encode_rgb, encode_layers
from .decoder import decode_rgb, decode_layers
from .palette import palette_6, inverse_palette_6
from .calibration import generate_calibration_card, compute_calibration, apply_calibration

__all__ = [
    "__version__",
    "encode_rgb",
    "encode_layers",
    "decode_rgb",
    "decode_layers",
    "palette_6",
    "inverse_palette_6",
    "generate_calibration_card",
    "compute_calibration",
    "apply_calibration",
]
