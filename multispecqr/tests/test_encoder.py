from multispecqr.encoder import encode_rgb


def test_encode_rgb_returns_image():
    img = encode_rgb("RED", "GREEN", "BLUE", version=2)
    assert img.mode == "RGB"
    w, h = img.size
    assert w == h and w > 0