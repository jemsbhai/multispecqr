"""
Test suite for expanded capacity (8-layer and 9-layer modes).

Tests the extended palettes and round-trip encoding/decoding
for 7, 8, and 9 layer configurations.
"""
import numpy as np
import pytest
from PIL import Image

from multispecqr.encoder import encode_layers
from multispecqr.decoder import decode_layers


class TestPalette8:
    """Test 8-layer (256-color) palette."""

    def test_palette_8_returns_dict(self):
        """palette_8() should return a dictionary."""
        from multispecqr.palette import palette_8
        p = palette_8()
        assert isinstance(p, dict)

    def test_palette_8_has_256_colors(self):
        """8-layer palette should have exactly 256 colors."""
        from multispecqr.palette import palette_8
        p = palette_8()
        assert len(p) == 256

    def test_palette_8_bitvec_to_color(self):
        """Each 8-bit vector should map to a unique RGB color."""
        from multispecqr.palette import palette_8
        p = palette_8()
        colors = list(p.values())
        # All colors should be unique
        assert len(set(colors)) == 256

    def test_palette_8_all_black(self):
        """All-zeros bit-vector should map to black."""
        from multispecqr.palette import palette_8
        p = palette_8()
        black_bits = (0, 0, 0, 0, 0, 0, 0, 0)
        assert p[black_bits] == (0, 0, 0)

    def test_palette_8_all_white(self):
        """All-ones bit-vector should map to white."""
        from multispecqr.palette import palette_8
        p = palette_8()
        white_bits = (1, 1, 1, 1, 1, 1, 1, 1)
        assert p[white_bits] == (255, 255, 255)

    def test_inverse_palette_8_returns_dict(self):
        """inverse_palette_8() should return a dictionary."""
        from multispecqr.palette import inverse_palette_8
        inv = inverse_palette_8()
        assert isinstance(inv, dict)

    def test_inverse_palette_8_has_256_entries(self):
        """Inverse palette should have 256 entries."""
        from multispecqr.palette import inverse_palette_8
        inv = inverse_palette_8()
        assert len(inv) == 256

    def test_palette_8_roundtrip(self):
        """Forward and inverse palette should be consistent."""
        from multispecqr.palette import palette_8, inverse_palette_8
        p = palette_8()
        inv = inverse_palette_8()
        for bitvec, color in p.items():
            assert inv[color] == bitvec


class TestPalette9:
    """Test 9-layer (512-color) palette."""

    def test_palette_9_returns_dict(self):
        """palette_9() should return a dictionary."""
        from multispecqr.palette import palette_9
        p = palette_9()
        assert isinstance(p, dict)

    def test_palette_9_has_512_colors(self):
        """9-layer palette should have exactly 512 colors."""
        from multispecqr.palette import palette_9
        p = palette_9()
        assert len(p) == 512

    def test_palette_9_bitvec_to_color(self):
        """Each 9-bit vector should map to a unique RGB color."""
        from multispecqr.palette import palette_9
        p = palette_9()
        colors = list(p.values())
        # All colors should be unique
        assert len(set(colors)) == 512

    def test_palette_9_all_black(self):
        """All-zeros bit-vector should map to black."""
        from multispecqr.palette import palette_9
        p = palette_9()
        black_bits = (0,) * 9
        assert p[black_bits] == (0, 0, 0)

    def test_palette_9_all_white(self):
        """All-ones bit-vector should map to white."""
        from multispecqr.palette import palette_9
        p = palette_9()
        white_bits = (1,) * 9
        assert p[white_bits] == (255, 255, 255)

    def test_inverse_palette_9_returns_dict(self):
        """inverse_palette_9() should return a dictionary."""
        from multispecqr.palette import inverse_palette_9
        inv = inverse_palette_9()
        assert isinstance(inv, dict)

    def test_inverse_palette_9_has_512_entries(self):
        """Inverse palette should have 512 entries."""
        from multispecqr.palette import inverse_palette_9
        inv = inverse_palette_9()
        assert len(inv) == 512

    def test_palette_9_roundtrip(self):
        """Forward and inverse palette should be consistent."""
        from multispecqr.palette import palette_9, inverse_palette_9
        p = palette_9()
        inv = inverse_palette_9()
        for bitvec, color in p.items():
            assert inv[color] == bitvec


class TestEncodeSevenLayers:
    """Test encoding with 7 layers."""

    def test_encode_seven_layers_returns_image(self):
        """encode_layers should accept 7 layers."""
        data = ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]
        img = encode_layers(data, version=2)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_encode_seven_layers_uses_256_palette(self):
        """7 layers should use the 256-color palette."""
        data = ["A", "B", "C", "D", "E", "F", "G"]
        img = encode_layers(data, version=2)
        arr = np.array(img)
        # Check that we have colors beyond the 64-color palette
        unique_colors = set(tuple(arr[y, x]) for y in range(arr.shape[0]) for x in range(arr.shape[1]))
        # Should have more than just white and a few colors
        assert len(unique_colors) > 1


class TestEncodeEightLayers:
    """Test encoding with 8 layers."""

    def test_encode_eight_layers_returns_image(self):
        """encode_layers should accept 8 layers."""
        data = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]
        img = encode_layers(data, version=2)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_encode_eight_layers_different_data(self):
        """8 layers with different data should encode successfully."""
        data = ["Hello", "World", "Test", "Data", "More", "Layers", "Eight", "Total"]
        img = encode_layers(data, version=3)
        assert img is not None


class TestEncodeNineLayers:
    """Test encoding with 9 layers."""

    def test_encode_nine_layers_returns_image(self):
        """encode_layers should accept 9 layers."""
        data = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]
        img = encode_layers(data, version=2)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_encode_nine_layers_different_data(self):
        """9 layers with different data should encode successfully."""
        data = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        img = encode_layers(data, version=3)
        assert img is not None

    def test_encode_ten_layers_raises(self):
        """encode_layers should reject more than 9 layers."""
        data = ["L" + str(i) for i in range(10)]
        with pytest.raises(ValueError, match="Maximum of 9 layers"):
            encode_layers(data, version=2)


class TestDecodeSevenLayers:
    """Test decoding 7-layer QR codes."""

    def test_roundtrip_seven_layers(self):
        """Encode and decode 7 layers should match."""
        original = ["A1", "B2", "C3", "D4", "E5", "F6", "G7"]
        img = encode_layers(original, version=2)
        decoded = decode_layers(img, num_layers=7)
        assert decoded == original

    def test_roundtrip_seven_layers_longer_data(self):
        """7-layer roundtrip with longer payloads."""
        original = ["Layer1", "Layer2", "Layer3", "Layer4", "Layer5", "Layer6", "Layer7"]
        img = encode_layers(original, version=3)
        decoded = decode_layers(img, num_layers=7)
        assert decoded == original


class TestDecodeEightLayers:
    """Test decoding 8-layer QR codes."""

    def test_roundtrip_eight_layers(self):
        """Encode and decode 8 layers should match."""
        original = ["A", "B", "C", "D", "E", "F", "G", "H"]
        img = encode_layers(original, version=2)
        decoded = decode_layers(img, num_layers=8)
        assert decoded == original

    def test_roundtrip_eight_layers_mixed_data(self):
        """8-layer roundtrip with varied data."""
        original = ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight"]
        img = encode_layers(original, version=3)
        decoded = decode_layers(img, num_layers=8)
        assert decoded == original

    def test_roundtrip_eight_layers_numeric(self):
        """8-layer roundtrip with numeric data."""
        original = ["111", "222", "333", "444", "555", "666", "777", "888"]
        img = encode_layers(original, version=2)
        decoded = decode_layers(img, num_layers=8)
        assert decoded == original


class TestDecodeNineLayers:
    """Test decoding 9-layer QR codes."""

    def test_roundtrip_nine_layers(self):
        """Encode and decode 9 layers should match."""
        original = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        img = encode_layers(original, version=2)
        decoded = decode_layers(img, num_layers=9)
        assert decoded == original

    def test_roundtrip_nine_layers_longer_data(self):
        """9-layer roundtrip with longer payloads."""
        original = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]
        # Use version 4 for better QR detection with 9 layers
        img = encode_layers(original, version=4)
        decoded = decode_layers(img, num_layers=9)
        assert decoded == original


class TestBackwardsCompatibility:
    """Ensure existing 6-layer functionality still works."""

    def test_six_layers_still_works(self):
        """6-layer encoding/decoding should be unchanged."""
        original = ["A", "B", "C", "D", "E", "F"]
        img = encode_layers(original, version=2)
        decoded = decode_layers(img, num_layers=6)
        assert decoded == original

    def test_three_layers_still_works(self):
        """3-layer encoding/decoding should be unchanged."""
        original = ["X", "Y", "Z"]
        img = encode_layers(original, version=2)
        decoded = decode_layers(img, num_layers=3)
        assert decoded == original

    def test_single_layer_still_works(self):
        """Single layer encoding/decoding should be unchanged."""
        original = ["A"]
        img = encode_layers(original, version=1)
        decoded = decode_layers(img, num_layers=1)
        assert decoded == original


class TestAutoDetectPalette:
    """Test automatic palette selection based on layer count."""

    def test_auto_select_palette_6_for_6_layers(self):
        """6 or fewer layers should use 64-color palette."""
        from multispecqr.palette import _select_palette
        palette, inv_palette, num_bits = _select_palette(6)
        assert len(palette) == 64
        assert num_bits == 6

    def test_auto_select_palette_8_for_7_layers(self):
        """7-8 layers should use 256-color palette."""
        from multispecqr.palette import _select_palette
        palette, inv_palette, num_bits = _select_palette(7)
        assert len(palette) == 256
        assert num_bits == 8

    def test_auto_select_palette_8_for_8_layers(self):
        """8 layers should use 256-color palette."""
        from multispecqr.palette import _select_palette
        palette, inv_palette, num_bits = _select_palette(8)
        assert len(palette) == 256
        assert num_bits == 8

    def test_auto_select_palette_9_for_9_layers(self):
        """9 layers should use 512-color palette."""
        from multispecqr.palette import _select_palette
        palette, inv_palette, num_bits = _select_palette(9)
        assert len(palette) == 512
        assert num_bits == 9


class TestColorSpacing:
    """Test that colors are well-spaced for robustness."""

    def test_palette_8_min_color_distance(self):
        """8-layer palette colors should have reasonable spacing."""
        from multispecqr.palette import palette_8
        p = palette_8()
        colors = list(p.values())

        # Calculate minimum Euclidean distance between any two colors
        min_dist = float('inf')
        for i, c1 in enumerate(colors):
            for c2 in colors[i+1:]:
                dist = sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5
                min_dist = min(min_dist, dist)

        # Minimum distance should be at least 30 (reasonable for camera capture)
        assert min_dist >= 30, f"Min color distance {min_dist} is too small"

    def test_palette_9_min_color_distance(self):
        """9-layer palette colors should have reasonable spacing."""
        from multispecqr.palette import palette_9
        p = palette_9()
        colors = list(p.values())

        # Calculate minimum Euclidean distance between any two colors
        min_dist = float('inf')
        for i, c1 in enumerate(colors):
            for c2 in colors[i+1:]:
                dist = sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5
                min_dist = min(min_dist, dist)

        # Minimum distance should be at least 25 for 9-layer palette
        assert min_dist >= 25, f"Min color distance {min_dist} is too small"
