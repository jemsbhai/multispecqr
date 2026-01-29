"""Robustness features for decoding real-world images."""

from PIL import Image
from multispecqr import encode_rgb, decode_rgb

# Create a test image
img = encode_rgb("Test1", "Test2", "Test3", version=2)

# Simulate a noisy/photographed image by saving as JPEG
img.save("temp.jpg", quality=70)
noisy = Image.open("temp.jpg")

# Try different threshold methods
print("Global threshold:", decode_rgb(noisy, threshold_method="global"))
print("Otsu threshold:", decode_rgb(noisy, threshold_method="otsu"))
print("Adaptive Gaussian:", decode_rgb(noisy, threshold_method="adaptive_gaussian"))

# Try preprocessing options
print("With blur:", decode_rgb(noisy, preprocess="blur"))
print("With denoise:", decode_rgb(noisy, preprocess="denoise"))

# Combine options for best results
print("Otsu + denoise:", decode_rgb(noisy, threshold_method="otsu", preprocess="denoise"))

# Cleanup
import os
os.remove("temp.jpg")
