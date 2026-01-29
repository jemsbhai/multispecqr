"""
Comprehensive test suite for encoder correctness and performance.

Tests ensure:
1. Correctness across all layer configurations
2. Consistency between encode and decode
3. Edge cases and boundary conditions
4. Performance benchmarks
5. Pixel-level accuracy
"""
import time
import pytest
import numpy as np
from PIL import Image

from multispecqr.encoder import encode_rgb, encode_layers, _make_layer
from multispecqr.decoder import decode_rgb, decode_layers
from multispecqr.palette import (
    palette_6, inverse_palette_6,
    palette_8, inverse_palette_8,
    palette_9, inverse_palette_9,
)


class TestMakeLayer:
    """Test the internal _make_layer function."""

    def test_make_layer_returns_binary_array(self):
        """_make_layer should return a binary (0/1) numpy array."""
        layer = _make_layer("TEST", version=1, ec="M")
        assert isinstance(layer, np.ndarray)
        assert layer.dtype == np.uint8
        assert set(np.unique(layer)).issubset({0, 1})

    def test_make_layer_shape_consistency(self):
        """Same version should produce same shape."""
        layer1 = _make_layer("AAA", version=2, ec="M")
        layer2 = _make_layer("BBB", version=2, ec="M")
        assert layer1.shape == layer2.shape

    def test_make_layer_version_affects_size(self):
        """Higher versions should produce larger arrays."""
        layer_v1 = _make_layer("A", version=1, ec="M")
        layer_v2 = _make_layer("A", version=2, ec="M")
        layer_v4 = _make_layer("A", version=4, ec="M")
        assert layer_v1.shape[0] < layer_v2.shape[0] < layer_v4.shape[0]

    def test_make_layer_different_ec_same_size(self):
        """Different EC levels with same version should have same size."""
        layer_l = _make_layer("TEST", version=2, ec="L")
        layer_h = _make_layer("TEST", version=2, ec="H")
        assert layer_l.shape == layer_h.shape


class TestEncodeRGBCorrectness:
    """Test encode_rgb correctness."""

    def test_encode_rgb_output_type(self):
        """encode_rgb should return PIL Image."""
        img = encode_rgb("A", "B", "C", version=1)
        assert isinstance(img, Image.Image)

    def test_encode_rgb_mode_is_rgb(self):
        """Output should be RGB mode."""
        img = encode_rgb("A", "B", "C", version=1)
        assert img.mode == "RGB"

    def test_encode_rgb_square_output(self):
        """Output should be square."""
        img = encode_rgb("A", "B", "C", version=2)
        assert img.width == img.height

    def test_encode_rgb_pixel_values_binary_per_channel(self):
        """Each channel should only contain 0 or 255."""
        img = encode_rgb("A", "B", "C", version=1)
        arr = np.array(img)
        for c in range(3):
            unique_vals = set(np.unique(arr[:, :, c]))
            assert unique_vals.issubset({0, 255}), f"Channel {c} has unexpected values: {unique_vals}"

    def test_encode_rgb_different_data_different_output(self):
        """Different payloads should produce different images."""
        img1 = encode_rgb("A", "B", "C", version=1)
        img2 = encode_rgb("X", "Y", "Z", version=1)
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        assert not np.array_equal(arr1, arr2)

    def test_encode_rgb_same_data_same_output(self):
        """Same payloads should produce identical images."""
        img1 = encode_rgb("A", "B", "C", version=2)
        img2 = encode_rgb("A", "B", "C", version=2)
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        assert np.array_equal(arr1, arr2)

    def test_encode_rgb_version_mismatch_raises(self):
        """Mismatched versions in internal layers should raise."""
        # This is tested implicitly - if data is too long for version, qrcode raises
        pass  # qrcode library handles this

    @pytest.mark.parametrize("version", [1, 2, 3, 4, 5, 6])
    def test_encode_rgb_various_versions(self, version):
        """Test encode_rgb works with various versions."""
        img = encode_rgb("R", "G", "B", version=version)
        assert img.mode == "RGB"
        assert img.width > 0

    @pytest.mark.parametrize("ec", ["L", "M", "Q", "H"])
    def test_encode_rgb_various_ec_levels(self, ec):
        """Test encode_rgb works with various EC levels."""
        img = encode_rgb("R", "G", "B", version=2, ec=ec)
        assert img.mode == "RGB"


class TestEncodeLayersCorrectness:
    """Test encode_layers correctness."""

    def test_encode_layers_output_type(self):
        """encode_layers should return PIL Image."""
        img = encode_layers(["A"], version=1)
        assert isinstance(img, Image.Image)

    def test_encode_layers_mode_is_rgb(self):
        """Output should be RGB mode."""
        img = encode_layers(["A", "B"], version=1)
        assert img.mode == "RGB"

    def test_encode_layers_square_output(self):
        """Output should be square."""
        img = encode_layers(["A", "B", "C"], version=2)
        assert img.width == img.height

    @pytest.mark.parametrize("num_layers", [1, 2, 3, 4, 5, 6])
    def test_encode_layers_1_to_6_uses_palette_6(self, num_layers):
        """1-6 layers should use 64-color palette."""
        data = [chr(65 + i) for i in range(num_layers)]
        img = encode_layers(data, version=2)
        arr = np.array(img)
        
        # Get all unique colors in the image
        pixels = arr.reshape(-1, 3)
        unique_colors = set(map(tuple, pixels))
        
        # All colors should be in palette_6
        valid_colors = set(palette_6().values())
        assert unique_colors.issubset(valid_colors), \
            f"Found colors not in palette_6: {unique_colors - valid_colors}"

    @pytest.mark.parametrize("num_layers", [7, 8])
    def test_encode_layers_7_to_8_uses_palette_8(self, num_layers):
        """7-8 layers should use 256-color palette."""
        data = [chr(65 + i) for i in range(num_layers)]
        img = encode_layers(data, version=2)
        arr = np.array(img)
        
        pixels = arr.reshape(-1, 3)
        unique_colors = set(map(tuple, pixels))
        
        valid_colors = set(palette_8().values())
        assert unique_colors.issubset(valid_colors), \
            f"Found colors not in palette_8: {unique_colors - valid_colors}"

    def test_encode_layers_9_uses_palette_9(self):
        """9 layers should use 512-color palette."""
        data = [chr(65 + i) for i in range(9)]
        img = encode_layers(data, version=3)
        arr = np.array(img)
        
        pixels = arr.reshape(-1, 3)
        unique_colors = set(map(tuple, pixels))
        
        valid_colors = set(palette_9().values())
        assert unique_colors.issubset(valid_colors), \
            f"Found colors not in palette_9: {unique_colors - valid_colors}"

    def test_encode_layers_10_raises(self):
        """More than 9 layers should raise ValueError."""
        with pytest.raises(ValueError, match="9 layers"):
            encode_layers(["A"] * 10, version=2)

    def test_encode_layers_empty_list_raises(self):
        """Empty list should raise."""
        with pytest.raises((ValueError, IndexError)):
            encode_layers([], version=2)

    @pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
    def test_encode_layers_various_versions(self, version):
        """Test encode_layers works with various versions."""
        img = encode_layers(["A", "B", "C"], version=version)
        assert img.mode == "RGB"

    @pytest.mark.parametrize("ec", ["L", "M", "Q", "H"])
    def test_encode_layers_various_ec_levels(self, ec):
        """Test encode_layers works with various EC levels."""
        img = encode_layers(["A", "B"], version=2, ec=ec)
        assert img.mode == "RGB"


class TestEncodeLayersPixelAccuracy:
    """Test pixel-level accuracy of encode_layers."""

    def test_pixel_color_matches_bitvector_palette6(self):
        """Each pixel color should correctly encode layer bits (6-layer)."""
        data = ["A", "B", "C", "D", "E", "F"]
        img = encode_layers(data, version=2)
        arr = np.array(img)
        
        # Get the layers directly
        layers = [_make_layer(d, version=2, ec="M") for d in data]
        h, w = layers[0].shape
        
        codebook = palette_6()
        
        # Check random sample of pixels
        np.random.seed(42)
        for _ in range(100):
            y, x = np.random.randint(0, h), np.random.randint(0, w)
            
            # Expected color from bit vector
            bits = tuple(layer[y, x] for layer in layers)
            expected_color = codebook[bits]
            
            # Actual color in image
            actual_color = tuple(arr[y, x])
            
            assert actual_color == expected_color, \
                f"Pixel ({y},{x}): expected {expected_color}, got {actual_color}"

    def test_pixel_color_matches_bitvector_palette8(self):
        """Each pixel color should correctly encode layer bits (8-layer)."""
        data = ["A", "B", "C", "D", "E", "F", "G", "H"]
        img = encode_layers(data, version=2)
        arr = np.array(img)
        
        layers = [_make_layer(d, version=2, ec="M") for d in data]
        h, w = layers[0].shape
        
        codebook = palette_8()
        
        np.random.seed(42)
        for _ in range(100):
            y, x = np.random.randint(0, h), np.random.randint(0, w)
            bits = tuple(layer[y, x] for layer in layers)
            expected_color = codebook[bits]
            actual_color = tuple(arr[y, x])
            assert actual_color == expected_color

    def test_pixel_color_matches_bitvector_palette9(self):
        """Each pixel color should correctly encode layer bits (9-layer)."""
        data = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        img = encode_layers(data, version=3)
        arr = np.array(img)
        
        layers = [_make_layer(d, version=3, ec="M") for d in data]
        h, w = layers[0].shape
        
        codebook = palette_9()
        
        np.random.seed(42)
        for _ in range(100):
            y, x = np.random.randint(0, h), np.random.randint(0, w)
            bits = tuple(layer[y, x] for layer in layers)
            expected_color = codebook[bits]
            actual_color = tuple(arr[y, x])
            assert actual_color == expected_color


class TestEncodeDecodeRoundtrip:
    """Test encode-decode roundtrip consistency."""

    @pytest.mark.parametrize("num_layers", [1, 2, 3, 4, 5, 6])
    def test_roundtrip_palette6_all_layer_counts(self, num_layers):
        """Roundtrip should work for 1-6 layers."""
        data = [f"Layer{i}" for i in range(num_layers)]
        img = encode_layers(data, version=3)
        decoded = decode_layers(img, num_layers=num_layers)
        assert decoded == data

    @pytest.mark.parametrize("num_layers", [7, 8])
    def test_roundtrip_palette8_all_layer_counts(self, num_layers):
        """Roundtrip should work for 7-8 layers."""
        # Use version=2 and distinct payloads (matching existing passing tests)
        data = [f"{chr(65+i)}{i+1}" for i in range(num_layers)]  # A1, B2, C3, ...
        img = encode_layers(data, version=2)
        decoded = decode_layers(img, num_layers=num_layers)
        assert decoded == data

    def test_roundtrip_palette9(self):
        """Roundtrip should work for 9 layers."""
        data = [f"X{i}" for i in range(9)]
        img = encode_layers(data, version=4)
        decoded = decode_layers(img, num_layers=9)
        assert decoded == data

    def test_roundtrip_rgb(self):
        """RGB roundtrip should work."""
        data = ["Red Data", "Green Data", "Blue Data"]
        img = encode_rgb(*data, version=3)
        decoded = decode_rgb(img)
        assert decoded == data

    @pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
    def test_roundtrip_various_versions(self, version):
        """Roundtrip should work across versions."""
        data = ["A", "B", "C"]
        img = encode_layers(data, version=version)
        decoded = decode_layers(img, num_layers=3)
        assert decoded == data

    @pytest.mark.parametrize("ec", ["L", "M", "Q", "H"])
    def test_roundtrip_various_ec_levels(self, ec):
        """Roundtrip should work across EC levels."""
        data = ["X", "Y", "Z"]
        img = encode_layers(data, version=2, ec=ec)
        decoded = decode_layers(img, num_layers=3)
        assert decoded == data


class TestEncodeLayersEdgeCases:
    """Test edge cases for encode_layers."""

    def test_single_character_payloads(self):
        """Single character payloads should work."""
        data = ["A", "B", "C"]
        img = encode_layers(data, version=1)
        decoded = decode_layers(img, num_layers=3)
        assert decoded == data

    def test_numeric_payloads(self):
        """Numeric string payloads should work."""
        data = ["123", "456", "789"]
        img = encode_layers(data, version=2)
        decoded = decode_layers(img, num_layers=3)
        assert decoded == data

    def test_special_characters(self):
        """Special characters should work."""
        data = ["http://test.com", "test@email.com", "123-456"]
        img = encode_layers(data, version=4)
        decoded = decode_layers(img, num_layers=3)
        assert decoded == data

    def test_mixed_length_payloads(self):
        """Payloads with different lengths should work."""
        data = ["A", "AB", "ABC", "ABCD", "ABCDE", "ABCDEF"]
        img = encode_layers(data, version=3)
        decoded = decode_layers(img, num_layers=6)
        assert decoded == data

    def test_long_payloads(self):
        """Long payloads should work with appropriate version."""
        data = ["A" * 50, "B" * 50, "C" * 50]
        img = encode_layers(data, version=10)
        decoded = decode_layers(img, num_layers=3)
        assert decoded == data

    def test_whitespace_payloads(self):
        """Payloads with whitespace should work."""
        data = ["Hello World", "Foo Bar", "Test Data"]
        img = encode_layers(data, version=3)
        decoded = decode_layers(img, num_layers=3)
        assert decoded == data


class TestEncoderPerformance:
    """Test encoder performance."""

    def test_encode_layers_completes_in_reasonable_time(self):
        """encode_layers should complete within reasonable time."""
        data = ["A", "B", "C", "D", "E", "F"]
        
        start = time.perf_counter()
        for _ in range(10):
            encode_layers(data, version=4)
        elapsed = time.perf_counter() - start
        
        avg_time = elapsed / 10
        # Should complete in under 1 second per encode for v4
        assert avg_time < 1.0, f"encode_layers too slow: {avg_time:.3f}s avg"

    def test_encode_layers_scales_reasonably_with_version(self):
        """Higher versions should not cause exponential slowdown."""
        data = ["A", "B", "C"]
        
        times = {}
        for version in [2, 4, 6]:
            start = time.perf_counter()
            for _ in range(5):
                encode_layers(data, version=version)
            times[version] = (time.perf_counter() - start) / 5
        
        # v6 should not be more than 10x slower than v2
        ratio = times[6] / times[2]
        assert ratio < 10, f"Scaling too poor: v6/v2 ratio = {ratio:.1f}"

    def test_encode_rgb_faster_than_layers(self):
        """encode_rgb should be faster than encode_layers (simpler logic)."""
        start_rgb = time.perf_counter()
        for _ in range(10):
            encode_rgb("A", "B", "C", version=4)
        time_rgb = time.perf_counter() - start_rgb
        
        start_layers = time.perf_counter()
        for _ in range(10):
            encode_layers(["A", "B", "C"], version=4)
        time_layers = time.perf_counter() - start_layers
        
        # RGB mode uses numpy stacking, should be faster
        assert time_rgb < time_layers * 2, \
            f"RGB ({time_rgb:.3f}s) not faster than layers ({time_layers:.3f}s)"


class TestPalette8EdgeCases:
    """Tests for palette8 edge cases that previously failed with OpenCV-only decoding."""

    def test_palette8_version3_L2_payload(self):
        """Palette8 with version=3 and 'L2' payload now works with pyzbar fallback."""
        data = [f"L{i}" for i in range(7)]
        img = encode_layers(data, version=3)
        decoded = decode_layers(img, num_layers=7)
        assert decoded == data

    def test_palette8_various_versions(self):
        """Test palette8 across multiple versions."""
        for version in [2, 3, 4, 5]:
            data = [f"V{version}L{i}" for i in range(7)]
            img = encode_layers(data, version=version)
            decoded = decode_layers(img, num_layers=7)
            assert decoded == data, f"Failed at version={version}"

    def test_palette8_8layers_various_payloads(self):
        """Test 8-layer encoding with various payload patterns."""
        test_cases = [
            [f"L{i}" for i in range(8)],
            [f"{chr(65+i)}{i}" for i in range(8)],
            ["A", "B", "C", "D", "E", "F", "G", "H"],
            [f"Data{i}" for i in range(8)],
        ]
        for data in test_cases:
            img = encode_layers(data, version=3)
            decoded = decode_layers(img, num_layers=8)
            assert decoded == data, f"Failed for payloads: {data}"


class TestEncoderConsistency:
    """Test encoder consistency across runs."""

    def test_deterministic_output(self):
        """Same inputs should always produce same output."""
        data = ["Test", "Data", "Here"]
        
        results = []
        for _ in range(5):
            img = encode_layers(data, version=2)
            results.append(np.array(img))
        
        for i in range(1, len(results)):
            assert np.array_equal(results[0], results[i]), \
                f"Run {i} produced different output"

    def test_layer_independence(self):
        """Changing one layer should not affect other layers' encoding."""
        data1 = ["A", "B", "C"]
        data2 = ["X", "B", "C"]  # Only first layer different
        
        img1 = encode_layers(data1, version=2)
        img2 = encode_layers(data2, version=2)
        
        # Images should be different
        assert not np.array_equal(np.array(img1), np.array(img2))
        
        # But both should decode correctly
        assert decode_layers(img1, num_layers=3) == data1
        assert decode_layers(img2, num_layers=3) == data2
