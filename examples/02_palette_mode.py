"""Palette mode: encode up to 9 independent layers."""

from multispecqr import encode_layers, decode_layers

# 6-layer example (uses 64-color palette)
data_6 = ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6"]
img_6 = encode_layers(data_6, version=3)
img_6.save("output_6layer.png")
print("Saved: output_6layer.png")

decoded_6 = decode_layers(img_6, num_layers=6)
print(f"6-layer decoded: {decoded_6}")

# 9-layer example (uses 512-color palette)
data_9 = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
img_9 = encode_layers(data_9, version=4)
img_9.save("output_9layer.png")
print("Saved: output_9layer.png")

decoded_9 = decode_layers(img_9, num_layers=9)
print(f"9-layer decoded: {decoded_9}")
