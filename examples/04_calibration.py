"""Color calibration workflow for photographed QR codes."""

from multispecqr import (
    generate_calibration_card,
    compute_calibration,
    encode_layers,
    decode_layers,
)

# Step 1: Generate a calibration card
# Print this and photograph it alongside your QR codes
card = generate_calibration_card(patch_size=50)
card.save("calibration_card.png")
print("Saved: calibration_card.png")
print("Print this card and photograph it with your QR codes.")

# Step 2: Create a sample QR code
data = ["Secret1", "Secret2", "Secret3", "Secret4"]
qr = encode_layers(data, version=3)
qr.save("sample_qr.png")
print("Saved: sample_qr.png")

# Step 3: After photographing, compute calibration
# (Here we use the original card as both reference and sample for demo)
calibration = compute_calibration(reference=card, sample=card)
print(f"Calibration computed: {calibration['method']}")

# Step 4: Use calibration when decoding photographed QR codes
# decoded = decode_layers(photographed_qr, num_layers=4, calibration=calibration)
