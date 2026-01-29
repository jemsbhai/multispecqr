"""Basic RGB mode: encode and decode 3 payloads."""

from multispecqr import encode_rgb, decode_rgb

# Encode three separate messages into one QR code
img = encode_rgb(
    "Red channel data",
    "Green channel data",
    "Blue channel data",
    version=2,
    ec="M"
)
img.save("output_rgb.png")
print("Saved: output_rgb.png")

# Decode it back
decoded = decode_rgb(img)
print(f"Decoded: {decoded}")
