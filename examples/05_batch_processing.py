"""Batch encode/decode multiple QR codes."""

from pathlib import Path
from PIL import Image
from multispecqr import encode_rgb, decode_rgb

# Batch encode
messages = [
    ("Hello", "World", "One"),
    ("Foo", "Bar", "Baz"),
    ("Alpha", "Beta", "Gamma"),
]

output_dir = Path("batch_output")
output_dir.mkdir(exist_ok=True)

for i, (r, g, b) in enumerate(messages):
    img = encode_rgb(r, g, b, version=2)
    img.save(output_dir / f"qr_{i:03d}.png")

print(f"Encoded {len(messages)} QR codes to {output_dir}/")

# Batch decode
for path in sorted(output_dir.glob("*.png")):
    img = Image.open(path)
    decoded = decode_rgb(img)
    print(f"{path.name}: {decoded}")
