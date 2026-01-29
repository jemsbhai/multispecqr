# multispecqr

[![PyPI - Version](https://img.shields.io/pypi/v/multispecqr.svg)](https://pypi.org/project/multispecqr)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/multispecqr.svg)](https://pypi.org/project/multispecqr)

**Multi-spectral QR codes** — encode multiple independent data payloads in a single QR code image using color channels.

## Features

- **3-Layer RGB Mode**: Encode 3 independent payloads using Red, Green, and Blue channels
- **6-Layer Palette Mode**: Encode up to 6 independent payloads using a 64-color palette
- **Full round-trip support**: Encode and decode with high fidelity
- **Simple API**: Easy-to-use functions for encoding and decoding
- **CLI included**: Command-line interface for quick operations

## Installation

```console
pip install multispecqr
```

## Quick Start

### RGB Mode (3 layers)

Encode three separate pieces of data into a single QR code:

```python
from multispecqr import encode_rgb, decode_rgb

# Encode three payloads
img = encode_rgb("Hello Red", "Hello Green", "Hello Blue", version=3)
img.save("rgb_qr.png")

# Decode back
decoded = decode_rgb(img)
print(decoded)  # ['Hello Red', 'Hello Green', 'Hello Blue']
```

### 6-Layer Palette Mode

Encode up to six separate pieces of data:

```python
from multispecqr import encode_layers, decode_layers

# Encode up to 6 payloads
data = ["Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5", "Layer 6"]
img = encode_layers(data, version=4)
img.save("palette_qr.png")

# Decode back
decoded = decode_layers(img, num_layers=6)
print(decoded)  # ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6']
```

## Command Line Interface

```bash
# Encode (RGB mode)
python -m multispecqr encode "Red data" "Green data" "Blue data" output.png

# Decode
python -m multispecqr decode output.png
```

## API Reference

### Encoding Functions

#### `encode_rgb(data_r, data_g, data_b, *, version=4, ec="M")`

Encode three payloads into an RGB QR code.

- **data_r, data_g, data_b**: String payloads for each color channel
- **version**: QR code version (1-40), controls capacity
- **ec**: Error correction level ("L", "M", "Q", "H")
- **Returns**: PIL Image in RGB mode

#### `encode_layers(data_list, *, version=4, ec="M")`

Encode up to 6 payloads using the 64-color palette.

- **data_list**: List of 1-6 string payloads
- **version**: QR code version (1-40)
- **ec**: Error correction level
- **Returns**: PIL Image in RGB mode

### Decoding Functions

#### `decode_rgb(img)`

Decode an RGB QR code into three payloads.

- **img**: PIL Image in RGB mode
- **Returns**: List of 3 strings (R, G, B channels)

#### `decode_layers(img, num_layers=None)`

Decode a palette-encoded QR code.

- **img**: PIL Image in RGB mode
- **num_layers**: Number of layers to decode (1-6). Defaults to 6.
- **Returns**: List of decoded strings

## How It Works

### RGB Mode

Each payload is encoded as an independent monochrome QR code, then assigned to one color channel (R, G, or B). The decoder separates the channels and decodes each independently.

### 6-Layer Palette Mode

Uses a systematic 64-color palette to encode all 2^6 possible combinations of 6 binary layers:

- Bits 0-1 determine Red level: {0, 85, 170, 255}
- Bits 2-3 determine Green level: {0, 85, 170, 255}
- Bits 4-5 determine Blue level: {0, 85, 170, 255}

This allows each pixel to represent any combination of the 6 layers being "on" or "off", enabling full overlap support.

## Requirements

- Python 3.9+
- opencv-python
- qrcode[pil]
- numpy
- Pillow

## License

`multispecqr` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
