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
python 06_advanced_options.py
python 07_ml_decoder_training.py  # Requires: pip install multispecqr[ml]
```

## Examples Overview

| File | Description |
|------|-------------|
| `01_basic_rgb.py` | Encode/decode 3 payloads using RGB channels |
| `02_palette_mode.py` | Encode up to 9 layers using color palettes |
| `03_robustness.py` | Threshold methods and preprocessing for noisy images |
| `04_calibration.py` | Color calibration workflow for printed QR codes |
| `05_batch_processing.py` | Process multiple QR codes in batch |
| `06_advanced_options.py` | QR parameters, scaling, file I/O, ML decoder |
| `07_ml_decoder_training.py` | ML decoder training and comprehensive evaluation |

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
