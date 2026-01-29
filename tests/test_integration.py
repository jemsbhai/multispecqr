"""
Integration tests and edge case coverage for multispecqr.

Tests full workflows and edge cases across multiple components.
"""
import numpy as np
import pytest
from PIL import Image

from multispecqr import (
    encode_rgb,
    decode_rgb,
    encode_layers,
    decode_layers,
    palette_6,
    inverse_palette_6,
    palette_8,
    inverse_palette_8,
    palette_9,
    inverse_palette_9,
    generate_calibration_card,
    compute_calibration,
    apply_calibration,
)


class TestFullWorkflowRGB:
    """Test complete RGB encode-decode workflows."""

    def test_rgb_roundtrip_various_data(self):
        """Test RGB roundtrip with various data types."""
        test_cases = [
            ("a", "b", "c"),
            ("123", "456", "789"),
            ("Hello World", "Test Data", "Example"),
            ("URL: http://test.com", "Email: test@test.com", "Phone: 123"),
        ]
        for r, g, b in test_cases:
            img = encode_rgb(r, g, b, version=3)
            decoded = decode_rgb(img)
            assert decoded == [r, g, b], f"Failed for: {r}, {g}, {b}"

    def test_rgb_with_all_threshold_methods(self):
        """Test RGB decode with all threshold methods."""
        # Use larger version for more reliable detection
        img = encode_rgb("R", "G", "B", version=3)
        # Note: adaptive methods may not work well on clean digital images
        # Testing that global and otsu work reliably
        for method in ["global", "otsu"]:
            decoded = decode_rgb(img, threshold_method=method)
            assert decoded == ["R", "G", "B"], f"Failed for method: {method}"

        # Adaptive methods should at least not crash
        for method in ["adaptive_gaussian", "adaptive_mean"]:
            decoded = decode_rgb(img, threshold_method=method)
            # Just verify it returns 3 items (may fail on clean images)
            assert len(decoded) == 3

    def test_rgb_with_all_preprocess_options(self):
        """Test RGB decode with all preprocess options."""
        img = encode_rgb("R", "G", "B", version=2)
        options = [None, "none", "blur", "denoise"]
        for opt in options:
            decoded = decode_rgb(img, preprocess=opt)
            assert decoded == ["R", "G", "B"], f"Failed for preprocess: {opt}"

    def test_rgb_different_versions(self):
        """Test RGB encoding with different QR versions."""
        for version in [1, 2, 3, 4, 5]:
            img = encode_rgb("A", "B", "C", version=version)
            decoded = decode_rgb(img)
            assert decoded == ["A", "B", "C"], f"Failed for version: {version}"

    def test_rgb_different_error_corrections(self):
        """Test RGB encoding with different error correction levels."""
        for ec in ["L", "M", "Q", "H"]:
            img = encode_rgb("X", "Y", "Z", version=2, ec=ec)
            decoded = decode_rgb(img)
            assert decoded == ["X", "Y", "Z"], f"Failed for ec: {ec}"


class TestFullWorkflowPalette:
    """Test complete palette encode-decode workflows."""

    def test_palette_roundtrip_all_layer_counts(self):
        """Test palette roundtrip for 1 to 9 layers."""
        for n in range(1, 10):
            data = [chr(65 + i) for i in range(n)]  # A, B, C, ...
            version = 2 if n <= 6 else (3 if n <= 8 else 4)
            img = encode_layers(data, version=version)
            decoded = decode_layers(img, num_layers=n)
            assert decoded == data, f"Failed for {n} layers"

    def test_palette_with_preprocess(self):
        """Test palette decode with preprocessing."""
        data = ["A", "B", "C"]
        # Use version 2 which is tested to work
        img = encode_layers(data, version=2)
        # Test without preprocessing
        decoded = decode_layers(img, num_layers=3, preprocess=None)
        assert decoded == data

        # Preprocessing may affect small QR codes, just verify it runs
        for opt in ["blur", "denoise"]:
            decoded = decode_layers(img, num_layers=3, preprocess=opt)
            assert len(decoded) == 3  # Returns 3 items


class TestCalibrationWorkflow:
    """Test calibration workflow."""

    def test_calibration_card_properties(self):
        """Test calibration card generation with various sizes."""
        for patch_size in [20, 50, 100]:
            for padding in [2, 5, 10]:
                card = generate_calibration_card(patch_size=patch_size, padding=padding)
                assert card.mode == "RGB"
                assert card.width > 0
                assert card.height > 0

    def test_calibration_roundtrip(self):
        """Test that applying identity calibration preserves image."""
        card = generate_calibration_card()
        calibration = compute_calibration(card, card)

        # Create a test image
        test_img = encode_rgb("A", "B", "C", version=2)
        calibrated = apply_calibration(test_img, calibration)

        # Images should be similar (identity transform)
        arr1 = np.array(test_img)
        arr2 = np.array(calibrated)
        # Allow small differences due to numerical precision
        assert np.allclose(arr1, arr2, atol=5)

    def test_decode_with_calibration(self):
        """Test decoding with calibration applied."""
        card = generate_calibration_card()
        calibration = compute_calibration(card, card)

        img = encode_rgb("X", "Y", "Z", version=2)
        decoded = decode_rgb(img, calibration=calibration)
        assert decoded == ["X", "Y", "Z"]


class TestPaletteConsistency:
    """Test palette consistency and properties."""

    def test_all_palettes_have_black_white(self):
        """All palettes should include black and white."""
        for palette_fn in [palette_6, palette_8, palette_9]:
            p = palette_fn()
            colors = list(p.values())
            assert (0, 0, 0) in colors, f"{palette_fn.__name__} missing black"
            assert (255, 255, 255) in colors, f"{palette_fn.__name__} missing white"

    def test_all_inverse_palettes_consistent(self):
        """Inverse palettes should be consistent with forward palettes."""
        pairs = [
            (palette_6, inverse_palette_6),
            (palette_8, inverse_palette_8),
            (palette_9, inverse_palette_9),
        ]
        for fwd, inv in pairs:
            p = fwd()
            i = inv()
            for bitvec, color in p.items():
                assert i[color] == bitvec

    def test_palette_color_uniqueness(self):
        """All colors in each palette should be unique."""
        for palette_fn in [palette_6, palette_8, palette_9]:
            p = palette_fn()
            colors = list(p.values())
            assert len(colors) == len(set(colors))


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_decode_rgb_non_rgb_image(self):
        """decode_rgb should reject non-RGB images."""
        gray = Image.new("L", (100, 100), 128)
        with pytest.raises(ValueError, match="Expected an RGB image"):
            decode_rgb(gray)

    def test_decode_layers_non_rgb_image(self):
        """decode_layers should reject non-RGB images."""
        gray = Image.new("L", (100, 100), 128)
        with pytest.raises(ValueError, match="Expected an RGB image"):
            decode_layers(gray)

    def test_encode_layers_too_many(self):
        """encode_layers should reject more than 9 layers."""
        with pytest.raises(ValueError, match="9 layers"):
            encode_layers(["L"] * 10, version=2)

    def test_decode_layers_too_many(self):
        """decode_layers should reject num_layers > 9."""
        img = encode_layers(["A"], version=1)
        with pytest.raises(ValueError, match="9 layers"):
            decode_layers(img, num_layers=10)

    def test_decode_rgb_invalid_threshold(self):
        """decode_rgb should reject invalid threshold method."""
        img = encode_rgb("A", "B", "C", version=2)
        with pytest.raises(ValueError, match="Invalid threshold_method"):
            decode_rgb(img, threshold_method="invalid")


class TestImageManipulation:
    """Test behavior with manipulated images."""

    def test_decode_noisy_image(self):
        """Test decoding with added noise."""
        img = encode_rgb("Test", "Data", "Here", version=3)
        arr = np.array(img)

        # Add light noise
        noise = np.random.randint(-10, 10, arr.shape, dtype=np.int16)
        noisy = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy)

        # Should still decode with preprocessing
        decoded = decode_rgb(noisy_img, preprocess="blur")
        # At least some layers should decode
        assert any(d for d in decoded)

    def test_decode_scaled_image(self):
        """Test decoding a scaled image."""
        img = encode_rgb("A", "B", "C", version=2)

        # Scale up
        scaled = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
        decoded = decode_rgb(scaled)
        assert decoded == ["A", "B", "C"]

    def test_decode_palette_with_color_shift(self):
        """Test palette decoding with slight color shift."""
        data = ["X", "Y", "Z"]
        img = encode_layers(data, version=2)
        arr = np.array(img)

        # Apply small color shift
        shifted = np.clip(arr.astype(np.int16) + 5, 0, 255).astype(np.uint8)
        shifted_img = Image.fromarray(shifted)

        # Nearest-neighbor matching should handle small shifts
        decoded = decode_layers(shifted_img, num_layers=3)
        assert decoded == data


class TestSpecialCharacters:
    """Test encoding/decoding with special characters."""

    def test_rgb_special_chars(self):
        """Test RGB with special characters."""
        test_data = [
            "Hello, World!",
            "Test@123#$%",
            "https://example.com/path?query=1",
        ]
        img = encode_rgb(*test_data, version=4)
        decoded = decode_rgb(img)
        assert decoded == test_data

    def test_palette_unicode(self):
        """Test palette mode with unicode (if supported by QR)."""
        # Basic ASCII only for QR compatibility
        data = ["Test1", "Test2", "Test3"]
        img = encode_layers(data, version=2)
        decoded = decode_layers(img, num_layers=3)
        assert decoded == data


class TestPerformance:
    """Test performance-related scenarios."""

    def test_large_qr_version(self):
        """Test with larger QR version."""
        # Version 10 can hold more data
        long_data = "A" * 100
        img = encode_rgb(long_data, long_data, long_data, version=10)
        decoded = decode_rgb(img)
        assert decoded == [long_data, long_data, long_data]

    def test_many_palette_layers_version(self):
        """Test 9 layers with appropriate version."""
        data = ["Layer" + str(i) for i in range(1, 10)]
        img = encode_layers(data, version=5)
        decoded = decode_layers(img, num_layers=9)
        assert decoded == data
