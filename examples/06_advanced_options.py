"""Advanced options: QR parameters, scaling, and file I/O."""

from pathlib import Path
from PIL import Image
from multispecqr import encode_rgb, decode_rgb, encode_layers, decode_layers

# -----------------------------------------------------------------------------
# QR Version and Error Correction
# -----------------------------------------------------------------------------
# Version 1-40: Higher version = more data capacity but larger QR code
# Error correction: L(7%), M(15%), Q(25%), H(30%) recovery

# Low capacity, small QR
img_small = encode_rgb("A", "B", "C", version=1, ec="L")
print(f"Version 1, EC=L: {img_small.size[0]}x{img_small.size[1]} pixels")

# High capacity, larger QR with maximum error correction
img_large = encode_rgb(
    "This is a longer message for the red channel",
    "This is a longer message for the green channel",
    "This is a longer message for the blue channel",
    version=10,
    ec="H"
)
print(f"Version 10, EC=H: {img_large.size[0]}x{img_large.size[1]} pixels")

# -----------------------------------------------------------------------------
# Scaling for Print
# -----------------------------------------------------------------------------
# Scale up for printing - each QR module becomes scale x scale pixels

img = encode_rgb("Print", "Ready", "QR", version=2, ec="M")
print(f"Original size: {img.size[0]}x{img.size[1]}")

# Scale 10x for printing (using nearest neighbor to keep sharp edges)
scaled = img.resize((img.size[0] * 10, img.size[1] * 10), Image.NEAREST)
scaled.save("print_ready.png")
print(f"Scaled size: {scaled.size[0]}x{scaled.size[1]}")

# -----------------------------------------------------------------------------
# File I/O Workflow
# -----------------------------------------------------------------------------
# Typical workflow: encode -> save -> (transfer/print/scan) -> load -> decode

# Encode and save
data = ["Layer1", "Layer2", "Layer3", "Layer4"]
img = encode_layers(data, version=3)
img.save("saved_qr.png")

# Later: load and decode
loaded = Image.open("saved_qr.png")
decoded = decode_layers(loaded, num_layers=4)
print(f"File I/O round-trip: {decoded}")

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
Path("print_ready.png").unlink(missing_ok=True)
Path("saved_qr.png").unlink(missing_ok=True)

# -----------------------------------------------------------------------------
# ML Decoder Note
# -----------------------------------------------------------------------------
print("\nFor ML decoder examples, see: 07_ml_decoder_training.py")
