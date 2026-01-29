# MultiSpecQR Examples

Example scripts demonstrating the multispecqr library.

## Running Examples

```bash
cd examples
python 01_basic_rgb.py
python 02_palette_mode.py
python 03_robustness.py
python 04_calibration.py
python 05_batch_processing.py
```

## Examples Overview

| File | Description |
|------|-------------|
| `01_basic_rgb.py` | Encode/decode 3 payloads using RGB channels |
| `02_palette_mode.py` | Encode up to 9 layers using color palettes |
| `03_robustness.py` | Threshold methods and preprocessing for noisy images |
| `04_calibration.py` | Color calibration workflow for printed QR codes |
| `05_batch_processing.py` | Process multiple QR codes in batch |

## CLI Examples

```bash
# Encode RGB mode
multispecqr encode "Red" "Green" "Blue" output.png

# Encode palette mode (6 layers)
multispecqr encode "A" "B" "C" "D" "E" "F" output.png --mode palette

# Decode with robustness options
multispecqr decode image.png --threshold otsu --preprocess denoise

# Generate calibration card
multispecqr calibrate card.png

# Batch decode
multispecqr batch-decode *.png --json
```
