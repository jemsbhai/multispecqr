"""ML decoder: RGB and Palette decoders with multi-layer support.

Demonstrates the ML decoder architecture:
- RGBMLDecoder: trained on RGB data, outputs 3 layers
- PaletteMLDecoder: trained on palette data, supports 1-9 layers
  - 1-6 layers: uses 64-color palette (6-bit model)
  - 7-8 layers: uses 256-color palette (8-bit model)  
  - 9 layers: uses 512-color palette (9-bit model)
"""

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
# Train RGB Decoder (3 layers)
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
# Train Palette Decoder - 6 layers (64-color palette)
# =============================================================================
print("\n" + "="*60)
print("Training Palette Decoder - 6 layers (64-color palette)")
print("="*60)

palette_decoder_6 = PaletteMLDecoder(num_layers=6, device=DEVICE)
print(f"Model bits: {palette_decoder_6.model_bits} (64 colors)")

for epoch in range(TRAIN_EPOCHS):
    loss = palette_decoder_6.train_epoch(num_samples=SAMPLES_PER_EPOCH, batch_size=BATCH_SIZE, version=QR_VERSION)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:3d}/{TRAIN_EPOCHS}: loss = {loss:.4f}")

# =============================================================================
# Train Palette Decoder - 8 layers (256-color palette)
# =============================================================================
print("\n" + "="*60)
print("Training Palette Decoder - 8 layers (256-color palette)")
print("="*60)

palette_decoder_8 = PaletteMLDecoder(num_layers=8, device=DEVICE)
print(f"Model bits: {palette_decoder_8.model_bits} (256 colors)")

for epoch in range(TRAIN_EPOCHS):
    loss = palette_decoder_8.train_epoch(num_samples=SAMPLES_PER_EPOCH, batch_size=BATCH_SIZE, version=QR_VERSION)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:3d}/{TRAIN_EPOCHS}: loss = {loss:.4f}")

# =============================================================================
# Train Palette Decoder - 9 layers (512-color palette)
# =============================================================================
print("\n" + "="*60)
print("Training Palette Decoder - 9 layers (512-color palette)")
print("="*60)

palette_decoder_9 = PaletteMLDecoder(num_layers=9, device=DEVICE)
print(f"Model bits: {palette_decoder_9.model_bits} (512 colors)")

for epoch in range(TRAIN_EPOCHS):
    loss = palette_decoder_9.train_epoch(num_samples=SAMPLES_PER_EPOCH, batch_size=BATCH_SIZE, version=QR_VERSION)
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
# Test Palette Decoder - 6 layers
# =============================================================================
print("\n" + "="*60)
print("Testing Palette Decoder - 6 layers")
print("="*60)

test_cases_6 = [
    ["A", "B", "C", "D", "E", "F"],
    ["L1", "L2", "L3", "L4", "L5", "L6"],
    ["QR", "ML", "AI", "CV", "NN", "DL"],
]

palette_6_correct = 0
for data in test_cases_6:
    img = encode_layers(data, version=QR_VERSION)
    
    result_threshold = decode_layers(img, num_layers=6, method="threshold")
    result_ml = palette_decoder_6.decode(img)
    
    threshold_ok = result_threshold == data
    ml_ok = result_ml == data
    
    if ml_ok:
        palette_6_correct += 1
    
    print(f"Expected: {data}")
    print(f"  Threshold: {result_threshold} {'✓' if threshold_ok else '✗'}")
    print(f"  ML:        {result_ml} {'✓' if ml_ok else '✗'}")

print(f"\n6-layer ML accuracy: {palette_6_correct}/{len(test_cases_6)}")

# =============================================================================
# Test Palette Decoder - 8 layers
# =============================================================================
print("\n" + "="*60)
print("Testing Palette Decoder - 8 layers")
print("="*60)

test_cases_8 = [
    ["A", "B", "C", "D", "E", "F", "G", "H"],
    ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"],
]

palette_8_correct = 0
for data in test_cases_8:
    img = encode_layers(data, version=QR_VERSION)
    
    result_threshold = decode_layers(img, num_layers=8, method="threshold")
    result_ml = palette_decoder_8.decode(img)
    
    threshold_ok = result_threshold == data
    ml_ok = result_ml == data
    
    if ml_ok:
        palette_8_correct += 1
    
    print(f"Expected: {data}")
    print(f"  Threshold: {result_threshold} {'✓' if threshold_ok else '✗'}")
    print(f"  ML:        {result_ml} {'✓' if ml_ok else '✗'}")

print(f"\n8-layer ML accuracy: {palette_8_correct}/{len(test_cases_8)}")

# =============================================================================
# Test Palette Decoder - 9 layers
# =============================================================================
print("\n" + "="*60)
print("Testing Palette Decoder - 9 layers")
print("="*60)

test_cases_9 = [
    ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
    ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"],
]

palette_9_correct = 0
for data in test_cases_9:
    img = encode_layers(data, version=QR_VERSION)
    
    result_threshold = decode_layers(img, num_layers=9, method="threshold")
    result_ml = palette_decoder_9.decode(img)
    
    threshold_ok = result_threshold == data
    ml_ok = result_ml == data
    
    if ml_ok:
        palette_9_correct += 1
    
    print(f"Expected: {data}")
    print(f"  Threshold: {result_threshold} {'✓' if threshold_ok else '✗'}")
    print(f"  ML:        {result_ml} {'✓' if ml_ok else '✗'}")

print(f"\n9-layer ML accuracy: {palette_9_correct}/{len(test_cases_9)}")

# =============================================================================
# Test 7 layers (uses 8-bit model internally)
# =============================================================================
print("\n" + "="*60)
print("Testing 7 layers (uses 8-bit model)")
print("="*60)

# 7 layers uses the same 8-bit model, just returns 7 results
test_data_7 = ["A", "B", "C", "D", "E", "F", "G"]
img = encode_layers(test_data_7, version=QR_VERSION)

# Can use the 8-layer decoder and request only 7 layers
result_ml = palette_decoder_8.decode(img, num_layers=7)

print(f"Expected: {test_data_7}")
print(f"  ML (8-bit model, 7 layers): {result_ml} {'✓' if result_ml == test_data_7 else '✗'}")

# Or create a dedicated 7-layer decoder
palette_decoder_7 = PaletteMLDecoder(num_layers=7, device=DEVICE)
print(f"\nCreated dedicated 7-layer decoder (model_bits={palette_decoder_7.model_bits})")

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
    
    print("RGB Decoder (3 outputs):")
    for i in range(3):
        layer = output[i]
        print(f"  Layer {i}: min={layer.min():.3f}, max={layer.max():.3f}, mean={layer.mean():.3f}")

# 6-layer palette decoder output
test_img = encode_layers(["A", "B", "C", "D", "E", "F"], version=QR_VERSION)
palette_decoder_6.model.eval()
with torch.no_grad():
    x = palette_decoder_6.preprocess(test_img)
    output = palette_decoder_6.model(x).squeeze(0).cpu().numpy()
    
    print("\nPalette Decoder 6-layer (6 outputs):")
    for i in range(6):
        layer = output[i]
        print(f"  Layer {i}: min={layer.min():.3f}, max={layer.max():.3f}, mean={layer.mean():.3f}")

# 8-layer palette decoder output
test_img = encode_layers(["A", "B", "C", "D", "E", "F", "G", "H"], version=QR_VERSION)
palette_decoder_8.model.eval()
with torch.no_grad():
    x = palette_decoder_8.preprocess(test_img)
    output = palette_decoder_8.model(x).squeeze(0).cpu().numpy()
    
    print("\nPalette Decoder 8-layer (8 outputs):")
    for i in range(8):
        layer = output[i]
        print(f"  Layer {i}: min={layer.min():.3f}, max={layer.max():.3f}, mean={layer.mean():.3f}")

# 9-layer palette decoder output
test_img = encode_layers(["A", "B", "C", "D", "E", "F", "G", "H", "I"], version=QR_VERSION)
palette_decoder_9.model.eval()
with torch.no_grad():
    x = palette_decoder_9.preprocess(test_img)
    output = palette_decoder_9.model(x).squeeze(0).cpu().numpy()
    
    print("\nPalette Decoder 9-layer (9 outputs):")
    for i in range(9):
        layer = output[i]
        print(f"  Layer {i}: min={layer.min():.3f}, max={layer.max():.3f}, mean={layer.mean():.3f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"""
RGB Decoder:        {rgb_correct}/{len(test_cases_rgb)} correct
6-layer Palette:    {palette_6_correct}/{len(test_cases_6)} correct
8-layer Palette:    {palette_8_correct}/{len(test_cases_8)} correct
9-layer Palette:    {palette_9_correct}/{len(test_cases_9)} correct

PaletteMLDecoder supports 1-9 layers with automatic palette selection:
- 1-6 layers: 64-color palette (6-bit model)
- 7-8 layers: 256-color palette (8-bit model)
- 9 layers: 512-color palette (9-bit model)

Usage:
    from multispecqr.ml_decoder import RGBMLDecoder, PaletteMLDecoder
    
    # For RGB mode (3 layers)
    rgb_decoder = RGBMLDecoder()
    for _ in range(50):
        rgb_decoder.train_epoch(num_samples=200)
    result = rgb_decoder.decode(img)  # Returns 3 strings
    
    # For palette mode - 6 layers (default)
    decoder_6 = PaletteMLDecoder(num_layers=6)
    for _ in range(50):
        decoder_6.train_epoch(num_samples=200)
    result = decoder_6.decode(img)  # Returns 6 strings
    
    # For palette mode - 8 layers
    decoder_8 = PaletteMLDecoder(num_layers=8)
    for _ in range(50):
        decoder_8.train_epoch(num_samples=200)
    result = decoder_8.decode(img)  # Returns 8 strings
    
    # For palette mode - 9 layers (max capacity)
    decoder_9 = PaletteMLDecoder(num_layers=9)
    for _ in range(50):
        decoder_9.train_epoch(num_samples=200)
    result = decoder_9.decode(img)  # Returns 9 strings
    
    # 7 layers uses 8-bit model internally
    decoder_7 = PaletteMLDecoder(num_layers=7)
    # Or use an 8-layer decoder and request 7:
    result = decoder_8.decode(img, num_layers=7)  # Returns 7 strings
""")
