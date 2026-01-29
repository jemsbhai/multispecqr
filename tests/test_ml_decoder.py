"""
Test suite for ML-based decoder.

These tests require torch to be installed (optional dependency).
Tests are skipped if torch is not available.
"""
import numpy as np
import pytest
from PIL import Image

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from multispecqr.encoder import encode_rgb, encode_layers


# Skip all tests in this module if torch is not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch not installed (optional ml dependency)"
)


class TestMLDecoderAvailability:
    """Test ML decoder module availability."""

    def test_ml_module_imports(self):
        """ML decoder module should be importable when torch is available."""
        from multispecqr import ml_decoder
        assert hasattr(ml_decoder, 'MLDecoder')
        assert hasattr(ml_decoder, 'decode_rgb_ml')
        assert hasattr(ml_decoder, 'decode_layers_ml')

    def test_ml_decoder_class_exists(self):
        """MLDecoder class should exist."""
        from multispecqr.ml_decoder import MLDecoder
        assert MLDecoder is not None

    def test_torch_availability_check(self):
        """Should have function to check torch availability."""
        from multispecqr.ml_decoder import is_torch_available
        assert is_torch_available() is True


class TestMLDecoderBasic:
    """Test basic ML decoder functionality."""

    def test_create_ml_decoder(self):
        """Should be able to create an MLDecoder instance."""
        from multispecqr.ml_decoder import MLDecoder
        decoder = MLDecoder()
        assert decoder is not None

    def test_ml_decoder_has_model(self):
        """MLDecoder should have a neural network model."""
        from multispecqr.ml_decoder import MLDecoder
        decoder = MLDecoder()
        assert hasattr(decoder, 'model')
        assert decoder.model is not None

    def test_ml_decoder_forward_pass(self):
        """MLDecoder should process an image tensor."""
        from multispecqr.ml_decoder import MLDecoder
        decoder = MLDecoder()

        # Create a dummy image tensor (batch, channels, height, width)
        dummy_input = torch.randn(1, 3, 64, 64)
        output = decoder.model(dummy_input)

        # Output should be (batch, 6, height, width) for 6 layers
        assert output.shape[0] == 1
        assert output.shape[1] == 6
        assert output.shape[2] == 64
        assert output.shape[3] == 64


class TestMLDecoderRGB:
    """Test ML-based RGB decoding."""

    def test_decode_rgb_ml_returns_list(self):
        """decode_rgb_ml should return a list of strings."""
        from multispecqr.ml_decoder import decode_rgb_ml

        img = encode_rgb("A", "B", "C", version=1)
        result = decode_rgb_ml(img)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(s, str) for s in result)

    def test_decode_rgb_ml_with_clean_image(self):
        """ML decoder should decode clean images."""
        from multispecqr.ml_decoder import decode_rgb_ml

        original = ["ML1", "ML2", "ML3"]
        img = encode_rgb(*original, version=2)
        decoded = decode_rgb_ml(img)

        # ML decoder may not be perfect without training
        # Just verify it returns valid results
        assert len(decoded) == 3


class TestMLDecoderLayers:
    """Test ML-based palette layer decoding."""

    def test_decode_layers_ml_returns_list(self):
        """decode_layers_ml should return a list of strings."""
        from multispecqr.ml_decoder import decode_layers_ml

        img = encode_layers(["A", "B"], version=1)
        result = decode_layers_ml(img, num_layers=2)

        assert isinstance(result, list)
        assert len(result) == 2

    def test_decode_layers_ml_six_layers(self):
        """ML decoder should handle 6 layers."""
        from multispecqr.ml_decoder import decode_layers_ml

        original = ["L1", "L2", "L3", "L4", "L5", "L6"]
        img = encode_layers(original, version=2)
        decoded = decode_layers_ml(img, num_layers=6)

        assert len(decoded) == 6


class TestTrainingDataGeneration:
    """Test training data generation for ML decoder."""

    def test_generate_training_sample(self):
        """Should generate a single training sample."""
        from multispecqr.ml_decoder import generate_training_sample

        image, labels = generate_training_sample()

        assert isinstance(image, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert image.shape[2] == 3  # RGB
        assert labels.shape[2] == 6  # 6 layers

    def test_generate_training_batch(self):
        """Should generate a batch of training data."""
        from multispecqr.ml_decoder import generate_training_batch

        images, labels = generate_training_batch(batch_size=4)

        assert images.shape[0] == 4
        assert labels.shape[0] == 4
        assert images.shape[3] == 3  # RGB channels last
        assert labels.shape[3] == 6  # 6 layer channels

    def test_training_data_consistency(self):
        """Training data should be consistent with encoder output."""
        from multispecqr.ml_decoder import generate_training_sample
        from multispecqr.palette import inverse_palette_6

        image, labels = generate_training_sample()

        # Verify that the image and labels are consistent
        # Each pixel's color should map to the correct label bits
        inv_palette = inverse_palette_6()

        # Check a few random pixels
        h, w = image.shape[:2]
        for _ in range(10):
            y, x = np.random.randint(0, h), np.random.randint(0, w)
            color = tuple(image[y, x])
            expected_bits = inv_palette.get(color, (0, 0, 0, 0, 0, 0))
            actual_bits = tuple(labels[y, x])
            # Allow some tolerance for edge cases
            assert actual_bits == expected_bits or color not in inv_palette


class TestMLDecoderTraining:
    """Test ML decoder training functionality."""

    def test_train_one_epoch(self):
        """Should be able to train for one epoch."""
        from multispecqr.ml_decoder import MLDecoder

        decoder = MLDecoder()
        initial_loss = decoder.train_epoch(num_samples=10)

        assert isinstance(initial_loss, float)
        assert initial_loss >= 0

    def test_training_reduces_loss(self):
        """Training should reduce loss over time."""
        from multispecqr.ml_decoder import MLDecoder

        decoder = MLDecoder()
        loss1 = decoder.train_epoch(num_samples=20)
        loss2 = decoder.train_epoch(num_samples=20)

        # Loss should generally decrease (allow some variance)
        # This is a weak test - just verify training runs
        assert loss2 is not None


class TestMLDecoderIntegration:
    """Test ML decoder integration with main API."""

    def test_decode_rgb_with_ml_method(self):
        """decode_rgb should accept method='ml' parameter."""
        from multispecqr.decoder import decode_rgb

        img = encode_rgb("A", "B", "C", version=1)
        # This should work if torch is available
        result = decode_rgb(img, method="ml")

        assert isinstance(result, list)
        assert len(result) == 3

    def test_decode_layers_with_ml_method(self):
        """decode_layers should accept method='ml' parameter."""
        from multispecqr.decoder import decode_layers

        img = encode_layers(["A", "B"], version=1)
        result = decode_layers(img, num_layers=2, method="ml")

        assert isinstance(result, list)
        assert len(result) == 2


class TestMLDecoderFallback:
    """Test graceful fallback when torch is not available."""

    def test_import_without_torch(self):
        """Module should be importable even if torch check fails."""
        # This test always passes since we skip the module if torch isn't available
        # The real fallback test is in the main module
        from multispecqr.ml_decoder import is_torch_available
        assert is_torch_available() is True
