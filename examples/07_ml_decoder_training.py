"""ML decoder: separate RGB and Palette decoders test.

Tests the dual-decoder architecture:
- RGBMLDecoder: trained on RGB data, outputs 3 layers
- PaletteMLDecoder: trained on palette data, outputs 6 layers
"""

import io
import numpy as np
from PIL import Image
import torch

from multispecqr import encode_rgb, decode_rgb, encode_layers, decode_layers
from multispecqr.ml_decoder import RGBMLDecoder, PaletteMLDecoder

# Config
TRAIN_EPOCHS = 50
SAMPLES_PER_EPOCH = 200
BATCH_SIZE = 16
QR_VERSION = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Device: {DEVICE}")
print(f"Config: {TRAIN_EPOCHS} epochs, {SAMPLES_PER_EPOCH} samples/epoch")

# =============================================================================
# Train RGB Decoder
# =============================================================================
print("\n" + "="*60)
print("Training RGB Decoder (3 outputs)")
print("="*60)

rgb_decoder = RGBMLDecoder(device=DEVICE)

for epoch in range(TRAIN_EPOCHS):
    loss = rgb_decoder.train_epoch(num_samples=SAMPLES_PER_EPOCH, batch_size=BATCH_SIZE, version=QR_VERSION)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:3d}/{TRAIN_EPOCHS}: loss = {loss:.4f}")

# =============================================================================
# Train Palette Decoder
# =============================================================================
print("\n" + "="*60)
print("Training Palette Decoder (6 outputs)")
print("="*60)

palette_decoder = PaletteMLDecoder(device=DEVICE)

for epoch in range(TRAIN_EPOCHS):
    loss = palette_decoder.train_epoch(num_samples=SAMPLES_PER_EPOCH, batch_size=BATCH_SIZE, version=QR_VERSION)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:3d}/{TRAIN_EPOCHS}: loss = {loss:.4f}")

# =============================================================================
# Test RGB Decoder
# =============================================================================
print("\n" + "="*60)
print("Testing RGB Decoder")
print("="*60)

test_cases_rgb = [
    ("A", "B", "C"),
    ("Hello", "World", "Test"),
    ("Red", "Green", "Blue"),
    ("QR1", "QR2", "QR3"),
]

rgb_correct = 0
for r, g, b in test_cases_rgb:
    img = encode_rgb(r, g, b, version=QR_VERSION)
    
    result_threshold = decode_rgb(img, method="threshold")
    result_ml = rgb_decoder.decode(img)
    
    expected = [r, g, b]
    threshold_ok = result_threshold == expected
    ml_ok = result_ml == expected
    
    if ml_ok:
        rgb_correct += 1
    
    print(f"Expected: {expected}")
    print(f"  Threshold: {result_threshold} {'✓' if threshold_ok else '✗'}")
    print(f"  ML:        {result_ml} {'✓' if ml_ok else '✗'}")

print(f"\nRGB ML accuracy: {rgb_correct}/{len(test_cases_rgb)}")

# =============================================================================
# Test Palette Decoder
# =============================================================================
print("\n" + "="*60)
print("Testing Palette Decoder")
print("="*60)

test_cases_palette = [
    ["A", "B", "C", "D", "E", "F"],
    ["L1", "L2", "L3", "L4", "L5", "L6"],
    ["QR", "ML", "AI", "CV", "NN", "DL"],
]

palette_correct = 0
for data in test_cases_palette:
    img = encode_layers(data, version=QR_VERSION)
    
    result_threshold = decode_layers(img, num_layers=6, method="threshold")
    result_ml = palette_decoder.decode(img, num_layers=6)
    
    threshold_ok = result_threshold == data
    ml_ok = result_ml == data
    
    if ml_ok:
        palette_correct += 1
    
    print(f"Expected: {data}")
    print(f"  Threshold: {result_threshold} {'✓' if threshold_ok else '✗'}")
    print(f"  ML:        {result_ml} {'✓' if ml_ok else '✗'}")

print(f"\nPalette ML accuracy: {palette_correct}/{len(test_cases_palette)}")

# =============================================================================
# Debug: Check model output distributions
# =============================================================================
print("\n" + "="*60)
print("Debug: Model Output Distributions")
print("="*60)

# RGB decoder output
test_img = encode_rgb("A", "B", "C", version=QR_VERSION)
rgb_decoder.model.eval()
with torch.no_grad():
    x = rgb_decoder.preprocess(test_img)
    output = rgb_decoder.model(x).squeeze(0).cpu().numpy()
    
    print("RGB Decoder on RGB image:")
    for i in range(3):
        layer = output[i]
        print(f"  Layer {i}: min={layer.min():.3f}, max={layer.max():.3f}, mean={layer.mean():.3f}")

# Palette decoder output
test_img = encode_layers(["A", "B", "C", "D", "E", "F"], version=QR_VERSION)
palette_decoder.model.eval()
with torch.no_grad():
    x = palette_decoder.preprocess(test_img)
    output = palette_decoder.model(x).squeeze(0).cpu().numpy()
    
    print("\nPalette Decoder on palette image:")
    for i in range(6):
        layer = output[i]
        print(f"  Layer {i}: min={layer.min():.3f}, max={layer.max():.3f}, mean={layer.mean():.3f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
RGB Decoder:     {rgb_correct}/{len(test_cases_rgb)} correct
Palette Decoder: {palette_correct}/{len(test_cases_palette)} correct

The dual-decoder approach uses separate models for each encoding mode:
- RGBMLDecoder: 3 output channels for R, G, B layers
- PaletteMLDecoder: 6 output channels for palette bit-vector

Usage:
    from multispecqr.ml_decoder import RGBMLDecoder, PaletteMLDecoder
    
    # For RGB mode
    rgb_decoder = RGBMLDecoder()
    for _ in range(50):
        rgb_decoder.train_epoch(num_samples=200, version=2)
    result = rgb_decoder.decode(img)
    
    # For palette mode
    palette_decoder = PaletteMLDecoder()
    for _ in range(50):
        palette_decoder.train_epoch(num_samples=200, version=2)
    result = palette_decoder.decode(img, num_layers=6)
""")
