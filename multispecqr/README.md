# multispecqr

[![PyPI - Version](https://img.shields.io/pypi/v/multispecqr.svg)](https://pypi.org/project/multispecqr)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/multispecqr.svg)](https://pypi.org/project/multispecqr)

**Multi-spectral QR codes** — encode multiple independent data payloads in a single QR code image using color channels.

## Features

- **3-Layer RGB Mode**: Encode 3 independent payloads using Red, Green, and Blue channels
- **6-Layer Palette Mode**: Encode up to 6 independent payloads using a 64-color palette
- **Robustness Features**: Adaptive thresholding, preprocessing, and color calibration for real-world images
- **Full round-trip support**: Encode and decode with high fidelity
- **Simple API**: Easy-to-use Python functions for encoding and decoding
- **CLI included**: Full-featured command-line interface for quick operations

## Installation

```console
pip install multispecqr
```

## Quick Start

### Python API

#### RGB Mode (3 layers)

Encode three separate pieces of data into a single QR code:

```python
from multispecqr import encode_rgb, decode_rgb

# Encode three payloads
img = encode_rgb("Hello Red", "Hello Green", "Hello Blue", version=2)
img.save("rgb_qr.png")

# Decode back
decoded = decode_rgb(img)
print(decoded)  # ['Hello Red', 'Hello Green', 'Hello Blue']
```

#### 6-Layer Palette Mode

Encode up to six separate pieces of data:

```python
from multispecqr import encode_layers, decode_layers

# Encode up to 6 payloads
data = ["Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5", "Layer 6"]
img = encode_layers(data, version=2)
img.save("palette_qr.png")

# Decode back
decoded = decode_layers(img, num_layers=6)
print(decoded)  # ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6']
```

#### Robustness Features

For decoding real-world images (photos of printed QR codes):

```python
from multispecqr import decode_rgb, decode_layers

# Use adaptive thresholding for uneven lighting
decoded = decode_rgb(img, threshold_method="otsu")

# Use preprocessing to reduce noise
decoded = decode_rgb(img, preprocess="denoise")

# Combine multiple options
decoded = decode_rgb(img, threshold_method="otsu", preprocess="blur")
```

#### Color Calibration

For accurate color matching when decoding photographed QR codes:

```python
from multispecqr import (
    generate_calibration_card,
    compute_calibration,
    decode_layers
)

# 1. Generate and print a calibration card
card = generate_calibration_card()
card.save("calibration_card.png")

# 2. Photograph the printed card alongside your QR code
# 3. Load both the reference and photographed card
photographed_card = Image.open("photographed_card.jpg")

# 4. Compute calibration
calibration = compute_calibration(card, photographed_card)

# 5. Use calibration when decoding
decoded = decode_layers(qr_image, calibration=calibration)
```

### Command Line Interface

The CLI supports both RGB and palette modes with full control over QR code parameters.

#### Basic Usage

```bash
# Show help
python -m multispecqr --help
python -m multispecqr encode --help
python -m multispecqr decode --help
```

#### RGB Mode (default)

```bash
# Encode three payloads into an RGB QR code
python -m multispecqr encode "Red data" "Green data" "Blue data" output.png

# Decode an RGB QR code
python -m multispecqr decode output.png
```

#### Palette Mode (up to 6 layers)

```bash
# Encode up to 6 payloads using palette mode
python -m multispecqr encode "L1" "L2" "L3" "L4" "L5" "L6" output.png --mode palette

# Decode a palette QR code (specify number of layers)
python -m multispecqr decode output.png --mode palette --layers 6
```

#### Advanced Options

```bash
# Encode with higher QR version (more capacity) and error correction
python -m multispecqr encode "R" "G" "B" output.png --version 4 --ec H

# Short form options
python -m multispecqr encode "R" "G" "B" output.png -v 4 -e H -m rgb
```

#### CLI Options Reference

**Encode command:**
| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--mode` | `-m` | Encoding mode: `rgb` (3 layers) or `palette` (1-6 layers) | `rgb` |
| `--version` | `-v` | QR code version (1-40). Higher = more capacity | `2` |
| `--ec` | `-e` | Error correction: `L` (7%), `M` (15%), `Q` (25%), `H` (30%) | `M` |

**Decode command:**
| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--mode` | `-m` | Decoding mode: `rgb` or `palette` | `rgb` |
| `--layers` | `-l` | Number of layers to decode (palette mode only) | `6` |

## API Reference

### Encoding Functions

#### `encode_rgb(data_r, data_g, data_b, *, version=4, ec="M")`

Encode three payloads into an RGB QR code using channel separation.

- **data_r, data_g, data_b** (`str`): Payload strings for Red, Green, Blue channels
- **version** (`int`): QR code version 1-40. Higher versions hold more data. Default: 4
- **ec** (`str`): Error correction level - "L", "M", "Q", or "H". Default: "M"
- **Returns**: `PIL.Image.Image` in RGB mode

#### `encode_layers(data_list, *, version=4, ec="M")`

Encode 1-6 payloads using the 64-color palette system.

- **data_list** (`list[str]`): List of 1-6 payload strings
- **version** (`int`): QR code version 1-40. Default: 4
- **ec** (`str`): Error correction level. Default: "M"
- **Returns**: `PIL.Image.Image` in RGB mode
- **Raises**: `ValueError` if more than 6 payloads provided

### Decoding Functions

#### `decode_rgb(img, *, threshold_method="global", preprocess=None, calibration=None)`

Decode an RGB QR code back into three payloads.

- **img** (`PIL.Image.Image`): RGB image to decode
- **threshold_method** (`str`): Thresholding algorithm:
  - `"global"`: Simple threshold at 128 (default, fastest)
  - `"otsu"`: Otsu's automatic threshold selection
  - `"adaptive_gaussian"`: Adaptive threshold with Gaussian weights
  - `"adaptive_mean"`: Adaptive threshold with mean of neighborhood
- **preprocess** (`str | None`): Optional preprocessing:
  - `None` or `"none"`: No preprocessing
  - `"blur"`: Gaussian blur to reduce noise
  - `"denoise"`: Non-local means denoising
- **calibration** (`dict | None`): Calibration data from `compute_calibration()`
- **Returns**: `list[str]` of 3 strings (R, G, B channels). Empty string for failed layers.
- **Raises**: `ValueError` if image is not RGB mode

#### `decode_layers(img, num_layers=None, *, preprocess=None, calibration=None)`

Decode a palette-encoded QR code.

- **img** (`PIL.Image.Image`): RGB image to decode
- **num_layers** (`int | None`): Number of layers to decode (1-6). Default: 6
- **preprocess** (`str | None`): Optional preprocessing (same options as `decode_rgb`)
- **calibration** (`dict | None`): Calibration data from `compute_calibration()`
- **Returns**: `list[str]` of decoded strings. Empty string for failed layers.
- **Raises**: `ValueError` if image is not RGB mode

### Calibration Functions

#### `generate_calibration_card(patch_size=50, padding=5)`

Generate a calibration card containing all 64 palette colors.

- **patch_size** (`int`): Size of each color patch in pixels. Default: 50
- **padding** (`int`): Padding between patches. Default: 5
- **Returns**: `PIL.Image.Image` containing the calibration card

#### `compute_calibration(reference, sample, *, patch_size=50, padding=5)`

Compute color calibration from a reference and sample calibration card.

- **reference** (`PIL.Image.Image`): Original calibration card (from `generate_calibration_card()`)
- **sample** (`PIL.Image.Image`): Photographed calibration card
- **Returns**: `dict` containing calibration data (matrix, offset, method)

#### `apply_calibration(img, calibration)`

Apply color calibration to an image.

- **img** (`PIL.Image.Image`): Input image to calibrate
- **calibration** (`dict`): Calibration data from `compute_calibration()`
- **Returns**: `PIL.Image.Image` with corrected colors

### Palette Functions

#### `palette_6()`

Get the 64-color palette mapping bit-vectors to RGB colors.

- **Returns**: `dict[tuple[int, ...], tuple[int, int, int]]`

#### `inverse_palette_6()`

Get the inverse palette mapping RGB colors to bit-vectors.

- **Returns**: `dict[tuple[int, int, int], tuple[int, ...]]`

## How It Works

### RGB Mode

Each payload is encoded as an independent monochrome QR code, then assigned to one color channel (R, G, or B). The decoder separates the channels using thresholding and decodes each independently.

```
Payload 1 → QR Layer → Red Channel   ─┐
Payload 2 → QR Layer → Green Channel ─┼→ Combined RGB Image
Payload 3 → QR Layer → Blue Channel  ─┘
```

### 6-Layer Palette Mode

Uses a systematic 64-color palette to encode all 2^6 possible combinations of 6 binary layers. Each pixel's color encodes which of the 6 layers have a "black module" at that position.

**Color encoding scheme:**
- Bits 0-1 → Red level: {0, 85, 170, 255}
- Bits 2-3 → Green level: {0, 85, 170, 255}
- Bits 4-5 → Blue level: {0, 85, 170, 255}

This creates 4³ = 64 unique colors, one for each possible combination of 6 binary bits. The decoder uses nearest-neighbor color matching to recover the bit-vectors, then reconstructs each layer.

```
6 Payloads → 6 QR Layers → Pixel-wise bit-vectors → 64-color palette → RGB Image
```

### Robustness Features

For real-world usage (photographed QR codes), the library provides:

1. **Adaptive Thresholding**: Handles uneven lighting conditions
   - Otsu's method: Automatic threshold selection based on image histogram
   - Adaptive Gaussian/Mean: Local thresholding for varying illumination

2. **Preprocessing**: Reduces image noise
   - Gaussian blur: Smooths out small noise artifacts
   - Non-local means denoising: Advanced noise reduction

3. **Color Calibration**: Corrects for camera/display color differences
   - Generate a calibration card with all palette colors
   - Photograph the card under the same conditions as your QR code
   - Compute and apply color correction

## Requirements

- Python 3.9+
- opencv-python
- qrcode[pil]
- numpy
- Pillow

## License

`multispecqr` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
