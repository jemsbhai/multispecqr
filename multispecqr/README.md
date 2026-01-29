# multispecqr

[![PyPI - Version](https://img.shields.io/pypi/v/multispecqr.svg)](https://pypi.org/project/multispecqr)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/multispecqr.svg)](https://pypi.org/project/multispecqr)

**Multi-spectral QR codes** — encode multiple independent data payloads in a single QR code image using color channels.

## Features

- **3-Layer RGB Mode**: Encode 3 independent payloads using Red, Green, and Blue channels
- **6-Layer Palette Mode**: Encode up to 6 independent payloads using a 64-color palette
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

#### `decode_rgb(img)`

Decode an RGB QR code back into three payloads.

- **img** (`PIL.Image.Image`): RGB image to decode
- **Returns**: `list[str]` of 3 strings (R, G, B channels). Empty string for failed layers.
- **Raises**: `ValueError` if image is not RGB mode

#### `decode_layers(img, num_layers=None)`

Decode a palette-encoded QR code.

- **img** (`PIL.Image.Image`): RGB image to decode
- **num_layers** (`int | None`): Number of layers to decode (1-6). Default: 6
- **Returns**: `list[str]` of decoded strings. Empty string for failed layers.
- **Raises**: `ValueError` if image is not RGB mode

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

## Requirements

- Python 3.9+
- opencv-python
- qrcode[pil]
- numpy
- Pillow

## License

`multispecqr` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
