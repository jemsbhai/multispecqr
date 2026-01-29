"""
Test suite for 6-layer palette encoder/decoder round-trip.

Tests follow TDD approach - written before implementation.
"""
import pytest
import numpy as np
from PIL import Image

from multispecqr.encoder import encode_layers
from multispecqr.decoder import decode_layers
from multispecqr.palette import palette_6, inverse_palette_6


class TestInversePalette:
    """Test the inverse palette lookup function."""

    def test_inverse_palette_returns_dict(self):
        """inverse_palette_6() should return a dictionary."""
        inv = inverse_palette_6()
        assert isinstance(inv, dict)

    def test_inverse_palette_maps_colors_to_bitvectors(self):
        """Each color should map back to its bit-vector."""
        palette = palette_6()
        inv = inverse_palette_6()

        # Verify round-trip: bit-vector -> color -> bit-vector
        for bitvec, color in palette.items():
            assert inv[color] == bitvec, f"Color {color} should map to {bitvec}"

    def test_inverse_palette_has_all_64_colors(self):
        """Should have all 64 colors (2^6 combinations)."""
        inv = inverse_palette_6()
        assert len(inv) == 64

    def test_inverse_black_maps_to_zeros(self):
        """Black (0,0,0) should map to all-zeros bit-vector."""
        inv = inverse_palette_6()
        assert inv[(0, 0, 0)] == (0, 0, 0, 0, 0, 0)

    def test_inverse_white_maps_to_ones(self):
        """White (255,255,255) should map to all-ones bit-vector."""
        inv = inverse_palette_6()
        assert inv[(255, 255, 255)] == (1, 1, 1, 1, 1, 1)


class TestDecodeLayersSingleLayer:
    """Test decode_layers with single layer inputs."""

    def test_roundtrip_single_layer_0(self):
        """Single layer in position 0 (Red channel)."""
        original = ["LAYER0"]
        img = encode_layers(original, version=2)
        decoded = decode_layers(img, num_layers=1)
        assert decoded == original

    def test_roundtrip_single_layer_1(self):
        """Single layer in position 1 (Green channel)."""
        original = ["LAYER1"]
        img = encode_layers(original, version=2)
        decoded = decode_layers(img, num_layers=1)
        assert decoded == original

    def test_roundtrip_single_layer_5(self):
        """Single layer in position 5 (Yellow channel)."""
        original = ["LAYER5"]
        img = encode_layers(original, version=2)
        decoded = decode_layers(img, num_layers=1)
        assert decoded == original


class TestDecodeLayersMultipleLayers:
    """Test decode_layers with multiple layer inputs."""

    def test_roundtrip_two_layers(self):
        """Two layers should round-trip correctly."""
        original = ["FIRST", "SECOND"]
        img = encode_layers(original, version=2)
        decoded = decode_layers(img, num_layers=2)
        assert decoded == original

    def test_roundtrip_three_layers(self):
        """Three layers should round-trip correctly."""
        original = ["AAA", "BBB", "CCC"]
        img = encode_layers(original, version=2)
        decoded = decode_layers(img, num_layers=3)
        assert decoded == original

    def test_roundtrip_four_layers(self):
        """Four layers should round-trip correctly."""
        original = ["L1", "L2", "L3", "L4"]
        img = encode_layers(original, version=2)
        decoded = decode_layers(img, num_layers=4)
        assert decoded == original

    def test_roundtrip_five_layers(self):
        """Five layers should round-trip correctly."""
        original = ["A", "B", "C", "D", "E"]
        img = encode_layers(original, version=2)
        decoded = decode_layers(img, num_layers=5)
        assert decoded == original

    def test_roundtrip_all_six_layers(self):
        """All six layers should round-trip correctly."""
        original = ["RED", "GREEN", "BLUE", "CYAN", "MAGENTA", "YELLOW"]
        img = encode_layers(original, version=2)
        decoded = decode_layers(img, num_layers=6)
        assert decoded == original


class TestDecodeLayersAutoDetect:
    """Test decode_layers with automatic layer count detection."""

    def test_auto_detect_single_layer(self):
        """Should auto-detect 1 layer when num_layers not specified."""
        original = ["SINGLE"]
        img = encode_layers(original, version=2)
        decoded = decode_layers(img)
        # Should return at least the first layer correctly
        assert decoded[0] == original[0]

    def test_auto_detect_six_layers(self):
        """Should auto-detect 6 layers when num_layers not specified."""
        original = ["R", "G", "B", "C", "M", "Y"]
        img = encode_layers(original, version=2)
        decoded = decode_layers(img)
        assert decoded == original


class TestDecodeLayersEdgeCases:
    """Test edge cases and error handling."""

    def test_decode_requires_rgb_image(self):
        """Should raise error for non-RGB images."""
        gray_img = Image.new("L", (100, 100), color=128)
        with pytest.raises(ValueError, match="RGB"):
            decode_layers(gray_img)

    def test_decode_empty_result_for_failed_layer(self):
        """Failed layer decoding should return empty string, not crash."""
        # Create a non-QR image
        noise_img = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        # Should not raise, but return empty strings
        result = decode_layers(noise_img, num_layers=1)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_roundtrip_with_longer_data(self):
        """Test with longer payload data."""
        original = ["Hello World from Layer 1!", "Greetings from Layer 2!"]
        img = encode_layers(original, version=4)  # Higher version for more data
        decoded = decode_layers(img, num_layers=2)
        assert decoded == original

    def test_roundtrip_with_numeric_data(self):
        """Test with numeric string data."""
        original = ["12345", "67890", "11111"]
        img = encode_layers(original, version=2)
        decoded = decode_layers(img, num_layers=3)
        assert decoded == original

    def test_roundtrip_with_special_characters(self):
        """Test with URL-like data containing special chars."""
        original = ["https://example.com", "test@email.com"]
        img = encode_layers(original, version=4)
        decoded = decode_layers(img, num_layers=2)
        assert decoded == original


class TestDecodeLayersConsistency:
    """Test consistency between encode_layers and decode_layers."""

    def test_decode_returns_list_of_strings(self):
        """decode_layers should always return List[str]."""
        original = ["TEST"]
        img = encode_layers(original, version=2)
        decoded = decode_layers(img, num_layers=1)
        assert isinstance(decoded, list)
        assert all(isinstance(s, str) for s in decoded)

    def test_decode_returns_correct_number_of_layers(self):
        """decode_layers should return exactly num_layers strings."""
        original = ["A", "B", "C"]
        img = encode_layers(original, version=2)
        decoded = decode_layers(img, num_layers=3)
        assert len(decoded) == 3

    def test_different_versions_roundtrip(self):
        """Test round-trip works with different QR versions."""
        for version in [1, 2, 3, 4]:
            original = ["V" + str(version)]
            img = encode_layers(original, version=version)
            decoded = decode_layers(img, num_layers=1)
            assert decoded == original, f"Failed for version {version}"

    def test_different_error_correction_roundtrip(self):
        """Test round-trip works with different error correction levels."""
        for ec in ["L", "M", "Q", "H"]:
            original = ["EC_" + ec]
            img = encode_layers(original, version=2, ec=ec)
            decoded = decode_layers(img, num_layers=1)
            assert decoded == original, f"Failed for EC level {ec}"
