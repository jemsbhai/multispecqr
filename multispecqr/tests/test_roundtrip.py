from multispecqr.encoder import encode_rgb
from multispecqr.decoder import decode_rgb


def test_roundtrip_rgb():
    original = ["RED", "GREEN", "BLUE"]
    img = encode_rgb(*original, version=2)
    decoded = decode_rgb(img)
    assert decoded == original