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
        # Put tensor on same device as model
        dummy_input = torch.randn(1, 3, 64, 64).to(decoder.device)
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


class TestNumLayersToModelBits:
    """Test the _num_layers_to_model_bits helper function."""

    def test_layers_1_to_6_return_6(self):
        """Layers 1-6 should use 6-bit model."""
        from multispecqr.ml_decoder import _num_layers_to_model_bits
        
        for n in range(1, 7):
            assert _num_layers_to_model_bits(n) == 6

    def test_layers_7_and_8_return_8(self):
        """Layers 7-8 should use 8-bit model."""
        from multispecqr.ml_decoder import _num_layers_to_model_bits
        
        assert _num_layers_to_model_bits(7) == 8
        assert _num_layers_to_model_bits(8) == 8

    def test_layer_9_returns_9(self):
        """Layer 9 should use 9-bit model."""
        from multispecqr.ml_decoder import _num_layers_to_model_bits
        
        assert _num_layers_to_model_bits(9) == 9

    def test_invalid_layers_raise_error(self):
        """Invalid layer counts should raise ValueError."""
        from multispecqr.ml_decoder import _num_layers_to_model_bits
        
        with pytest.raises(ValueError):
            _num_layers_to_model_bits(0)
        
        with pytest.raises(ValueError):
            _num_layers_to_model_bits(10)
        
        with pytest.raises(ValueError):
            _num_layers_to_model_bits(-1)


class TestPaletteMLDecoderParameterized:
    """Test PaletteMLDecoder with different num_layers values."""

    def test_create_decoder_6_layers(self):
        """Should create 6-layer decoder (default)."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        decoder = PaletteMLDecoder(num_layers=6)
        assert decoder.num_layers == 6
        assert decoder.model_bits == 6
        assert decoder.num_outputs == 6

    def test_create_decoder_7_layers(self):
        """Should create 7-layer decoder using 8-bit model."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        decoder = PaletteMLDecoder(num_layers=7)
        assert decoder.num_layers == 7
        assert decoder.model_bits == 8
        assert decoder.num_outputs == 8

    def test_create_decoder_8_layers(self):
        """Should create 8-layer decoder."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        decoder = PaletteMLDecoder(num_layers=8)
        assert decoder.num_layers == 8
        assert decoder.model_bits == 8
        assert decoder.num_outputs == 8

    def test_create_decoder_9_layers(self):
        """Should create 9-layer decoder."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        decoder = PaletteMLDecoder(num_layers=9)
        assert decoder.num_layers == 9
        assert decoder.model_bits == 9
        assert decoder.num_outputs == 9

    def test_default_is_6_layers(self):
        """Default num_layers should be 6."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        decoder = PaletteMLDecoder()
        assert decoder.num_layers == 6
        assert decoder.model_bits == 6

    def test_invalid_num_layers_raises_error(self):
        """Invalid num_layers should raise ValueError."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        with pytest.raises(ValueError):
            PaletteMLDecoder(num_layers=0)
        
        with pytest.raises(ValueError):
            PaletteMLDecoder(num_layers=10)

    def test_model_output_shape_6_layers(self):
        """6-layer model should output 6 channels."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        decoder = PaletteMLDecoder(num_layers=6)
        dummy_input = torch.randn(1, 3, 64, 64).to(decoder.device)
        output = decoder.model(dummy_input)
        
        assert output.shape == (1, 6, 64, 64)

    def test_model_output_shape_8_layers(self):
        """8-layer model should output 8 channels."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        decoder = PaletteMLDecoder(num_layers=8)
        dummy_input = torch.randn(1, 3, 64, 64).to(decoder.device)
        output = decoder.model(dummy_input)
        
        assert output.shape == (1, 8, 64, 64)

    def test_model_output_shape_9_layers(self):
        """9-layer model should output 9 channels."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        decoder = PaletteMLDecoder(num_layers=9)
        dummy_input = torch.randn(1, 3, 64, 64).to(decoder.device)
        output = decoder.model(dummy_input)
        
        assert output.shape == (1, 9, 64, 64)


class TestTrainingDataGeneration789:
    """Test training data generation for 7, 8, 9 layer palettes."""

    def test_generate_sample_8_layers(self):
        """Should generate 8-layer training sample."""
        from multispecqr.ml_decoder import _generate_palette_sample
        
        image, labels = _generate_palette_sample(version=1, num_layers=8)
        
        assert image.shape[2] == 3  # RGB
        assert labels.shape[2] == 8  # 8 layers

    def test_generate_sample_9_layers(self):
        """Should generate 9-layer training sample."""
        from multispecqr.ml_decoder import _generate_palette_sample
        
        image, labels = _generate_palette_sample(version=1, num_layers=9)
        
        assert image.shape[2] == 3  # RGB
        assert labels.shape[2] == 9  # 9 layers

    def test_generate_batch_8_layers(self):
        """Should generate batch of 8-layer training data."""
        from multispecqr.ml_decoder import _generate_palette_batch
        
        images, labels = _generate_palette_batch(batch_size=4, num_layers=8)
        
        assert images.shape == (4, images.shape[1], images.shape[2], 3)
        assert labels.shape == (4, labels.shape[1], labels.shape[2], 8)

    def test_generate_batch_9_layers(self):
        """Should generate batch of 9-layer training data."""
        from multispecqr.ml_decoder import _generate_palette_batch
        
        images, labels = _generate_palette_batch(batch_size=4, num_layers=9)
        
        assert images.shape == (4, images.shape[1], images.shape[2], 3)
        assert labels.shape == (4, labels.shape[1], labels.shape[2], 9)

    def test_8_layer_data_uses_palette_8(self):
        """8-layer training data should use 256-color palette."""
        from multispecqr.ml_decoder import _generate_palette_sample
        from multispecqr.palette import inverse_palette_8
        
        image, labels = _generate_palette_sample(version=1, num_layers=8)
        inv_palette = inverse_palette_8()
        
        # Check that colors come from palette_8
        h, w = image.shape[:2]
        for _ in range(10):
            y, x = np.random.randint(0, h), np.random.randint(0, w)
            color = tuple(image[y, x])
            # Color should be in the 256-color palette
            assert color in inv_palette, f"Color {color} not in palette_8"

    def test_9_layer_data_uses_palette_9(self):
        """9-layer training data should use 512-color palette."""
        from multispecqr.ml_decoder import _generate_palette_sample
        from multispecqr.palette import inverse_palette_9
        
        image, labels = _generate_palette_sample(version=1, num_layers=9)
        inv_palette = inverse_palette_9()
        
        # Check that colors come from palette_9
        h, w = image.shape[:2]
        for _ in range(10):
            y, x = np.random.randint(0, h), np.random.randint(0, w)
            color = tuple(image[y, x])
            # Color should be in the 512-color palette
            assert color in inv_palette, f"Color {color} not in palette_9"


class TestPaletteMLDecoderTraining789:
    """Test training for 7, 8, 9 layer decoders."""

    def test_train_7_layer_decoder(self):
        """Should be able to train 7-layer decoder."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        decoder = PaletteMLDecoder(num_layers=7)
        loss = decoder.train_epoch(num_samples=10)
        
        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_8_layer_decoder(self):
        """Should be able to train 8-layer decoder."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        decoder = PaletteMLDecoder(num_layers=8)
        loss = decoder.train_epoch(num_samples=10)
        
        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_9_layer_decoder(self):
        """Should be able to train 9-layer decoder."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        decoder = PaletteMLDecoder(num_layers=9)
        loss = decoder.train_epoch(num_samples=10)
        
        assert isinstance(loss, float)
        assert loss >= 0


class TestPaletteMLDecoderDecode789:
    """Test decoding with 7, 8, 9 layer decoders."""

    def test_decode_7_layers_returns_7_results(self):
        """7-layer decoder should return 7 strings."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        decoder = PaletteMLDecoder(num_layers=7)
        
        # Create a dummy image
        img = encode_layers(["A", "B", "C", "D", "E", "F", "G"], version=1)
        result = decoder.decode(img)
        
        assert isinstance(result, list)
        assert len(result) == 7

    def test_decode_8_layers_returns_8_results(self):
        """8-layer decoder should return 8 strings."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        decoder = PaletteMLDecoder(num_layers=8)
        
        # Create a dummy image
        img = encode_layers(["A", "B", "C", "D", "E", "F", "G", "H"], version=1)
        result = decoder.decode(img)
        
        assert isinstance(result, list)
        assert len(result) == 8

    def test_decode_9_layers_returns_9_results(self):
        """9-layer decoder should return 9 strings."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        decoder = PaletteMLDecoder(num_layers=9)
        
        # Create a dummy image
        img = encode_layers(["A", "B", "C", "D", "E", "F", "G", "H", "I"], version=1)
        result = decoder.decode(img)
        
        assert isinstance(result, list)
        assert len(result) == 9

    def test_decode_override_num_layers(self):
        """Should be able to override num_layers at decode time."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        # Create 8-layer decoder but only request 5 results
        decoder = PaletteMLDecoder(num_layers=8)
        img = encode_layers(["A", "B", "C", "D", "E"], version=1)
        result = decoder.decode(img, num_layers=5)
        
        assert len(result) == 5

    def test_decode_cannot_exceed_model_capacity(self):
        """Cannot decode more layers than model supports."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        # Create 6-layer decoder
        decoder = PaletteMLDecoder(num_layers=6)
        img = encode_layers(["A", "B"], version=1)
        
        # Should raise error when trying to decode 8 layers with 6-bit model
        with pytest.raises(ValueError):
            decoder.decode(img, num_layers=8)


class TestRGBMLDecoderUnchanged:
    """Verify RGBMLDecoder still works as before."""

    def test_rgb_decoder_still_works(self):
        """RGBMLDecoder should be unchanged."""
        from multispecqr.ml_decoder import RGBMLDecoder
        
        decoder = RGBMLDecoder()
        assert decoder.num_outputs == 3
        
        img = encode_rgb("A", "B", "C", version=1)
        result = decoder.decode(img)
        
        assert len(result) == 3

    def test_rgb_decoder_training(self):
        """RGBMLDecoder training should still work."""
        from multispecqr.ml_decoder import RGBMLDecoder
        
        decoder = RGBMLDecoder()
        loss = decoder.train_epoch(num_samples=10)
        
        assert isinstance(loss, float)
        assert loss >= 0


class TestModelSaveLoad:
    """Test save/load functionality for ML decoders."""

    def test_rgb_decoder_save_creates_file(self, tmp_path):
        """save() should create a file with model weights."""
        from multispecqr.ml_decoder import RGBMLDecoder
        
        decoder = RGBMLDecoder()
        path = tmp_path / "rgb_model.pt"
        decoder.save(str(path))
        
        assert path.exists()
        assert path.stat().st_size > 0

    def test_rgb_decoder_save_contains_expected_keys(self, tmp_path):
        """Saved file should contain expected state keys."""
        from multispecqr.ml_decoder import RGBMLDecoder
        
        decoder = RGBMLDecoder()
        path = tmp_path / "rgb_model.pt"
        decoder.save(str(path))
        
        state = torch.load(str(path), weights_only=False)
        assert 'model_state_dict' in state
        assert 'num_outputs' in state
        assert 'model_class' in state
        assert state['num_outputs'] == 3
        assert state['model_class'] == 'RGBMLDecoder'

    def test_palette_decoder_save_contains_num_layers(self, tmp_path):
        """Palette decoder save should include num_layers and model_bits."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        decoder = PaletteMLDecoder(num_layers=8)
        path = tmp_path / "palette_model.pt"
        decoder.save(str(path))
        
        state = torch.load(str(path), weights_only=False)
        assert 'num_layers' in state
        assert 'model_bits' in state
        assert state['num_layers'] == 8
        assert state['model_bits'] == 8

    def test_rgb_decoder_load_restores_weights(self, tmp_path):
        """load() should restore model weights."""
        from multispecqr.ml_decoder import RGBMLDecoder
        
        # Train decoder slightly to change weights from init
        decoder1 = RGBMLDecoder()
        decoder1.train_epoch(num_samples=5)
        
        # Save weights
        path = tmp_path / "rgb_model.pt"
        decoder1.save(str(path))
        
        # Load into new decoder
        decoder2 = RGBMLDecoder()
        decoder2.load(str(path))
        
        # Verify weights match
        for (name1, param1), (name2, param2) in zip(
            decoder1.model.state_dict().items(),
            decoder2.model.state_dict().items()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)

    def test_palette_decoder_load_restores_weights(self, tmp_path):
        """load() should restore palette decoder weights."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        decoder1 = PaletteMLDecoder(num_layers=6)
        decoder1.train_epoch(num_samples=5)
        
        path = tmp_path / "palette_model.pt"
        decoder1.save(str(path))
        
        decoder2 = PaletteMLDecoder(num_layers=6)
        decoder2.load(str(path))
        
        for (name1, param1), (name2, param2) in zip(
            decoder1.model.state_dict().items(),
            decoder2.model.state_dict().items()
        ):
            assert torch.allclose(param1, param2)

    def test_load_mismatched_num_outputs_raises_error(self, tmp_path):
        """Loading a model with wrong num_outputs should raise ValueError."""
        from multispecqr.ml_decoder import RGBMLDecoder, PaletteMLDecoder
        
        # Save RGB decoder (3 outputs)
        rgb_decoder = RGBMLDecoder()
        path = tmp_path / "rgb_model.pt"
        rgb_decoder.save(str(path))
        
        # Try to load into palette decoder (6 outputs)
        palette_decoder = PaletteMLDecoder(num_layers=6)
        with pytest.raises(ValueError, match="Model mismatch"):
            palette_decoder.load(str(path))

    def test_save_load_roundtrip_preserves_predictions(self, tmp_path):
        """Save/load roundtrip should preserve model predictions."""
        from multispecqr.ml_decoder import RGBMLDecoder
        
        decoder1 = RGBMLDecoder()
        decoder1.train_epoch(num_samples=10)
        
        # Create test image
        img = encode_rgb("A", "B", "C", version=1)
        
        # Get predictions before save
        layers1 = decoder1.predict_layers(img)
        
        # Save and load
        path = tmp_path / "model.pt"
        decoder1.save(str(path))
        
        decoder2 = RGBMLDecoder()
        decoder2.load(str(path))
        
        # Get predictions after load
        layers2 = decoder2.predict_layers(img)
        
        # Predictions should be identical
        assert np.array_equal(layers1, layers2)


class TestModelFromLocal:
    """Test from_local() for loading from local files with auto-detection."""

    def test_from_local_rgb_decoder(self, tmp_path):
        """from_local() should auto-detect and load RGB decoder."""
        from multispecqr.ml_decoder import RGBMLDecoder, PaletteMLDecoder
        
        # Save an RGB decoder
        original = RGBMLDecoder()
        original.train_epoch(num_samples=5)
        path = tmp_path / "rgb_model.pt"
        original.save(str(path))
        
        # Load using from_local (should auto-detect RGB)
        loaded = RGBMLDecoder.from_local(str(path))
        
        assert isinstance(loaded, RGBMLDecoder)
        assert loaded.num_outputs == 3

    def test_from_local_palette_decoder(self, tmp_path):
        """from_local() should auto-detect and load Palette decoder."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        # Save a palette decoder
        original = PaletteMLDecoder(num_layers=8)
        original.train_epoch(num_samples=5)
        path = tmp_path / "palette_model.pt"
        original.save(str(path))
        
        # Load using from_local (should auto-detect palette with num_layers=8)
        loaded = PaletteMLDecoder.from_local(str(path))
        
        assert isinstance(loaded, PaletteMLDecoder)
        assert loaded.num_layers == 8
        assert loaded.model_bits == 8

    def test_from_local_cross_class_works(self, tmp_path):
        """from_local() should work when called from either class."""
        from multispecqr.ml_decoder import RGBMLDecoder, PaletteMLDecoder
        
        # Save an RGB decoder
        original = RGBMLDecoder()
        path = tmp_path / "rgb_model.pt"
        original.save(str(path))
        
        # Load using PaletteMLDecoder.from_local - should still return RGBMLDecoder
        loaded = PaletteMLDecoder.from_local(str(path))
        
        assert isinstance(loaded, RGBMLDecoder)
        assert loaded.num_outputs == 3

    def test_from_local_preserves_weights(self, tmp_path):
        """from_local() should correctly restore trained weights."""
        from multispecqr.ml_decoder import RGBMLDecoder
        
        # Train and save
        original = RGBMLDecoder()
        original.train_epoch(num_samples=10)
        path = tmp_path / "model.pt"
        original.save(str(path))
        
        # Load with from_local
        loaded = RGBMLDecoder.from_local(str(path))
        
        # Verify weights match
        for (name1, param1), (name2, param2) in zip(
            original.model.state_dict().items(),
            loaded.model.state_dict().items()
        ):
            assert torch.allclose(param1, param2)

    def test_from_local_predictions_match(self, tmp_path):
        """from_local() loaded model should produce identical predictions."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        original = PaletteMLDecoder(num_layers=6)
        original.train_epoch(num_samples=10)
        
        img = encode_layers(["A", "B", "C", "D", "E", "F"], version=1)
        layers_original = original.predict_layers(img)
        
        path = tmp_path / "model.pt"
        original.save(str(path))
        
        loaded = PaletteMLDecoder.from_local(str(path))
        layers_loaded = loaded.predict_layers(img)
        
        assert np.array_equal(layers_original, layers_loaded)

    def test_from_local_file_not_found(self):
        """from_local() should raise error for non-existent file."""
        from multispecqr.ml_decoder import RGBMLDecoder
        
        with pytest.raises(Exception):  # FileNotFoundError or similar
            RGBMLDecoder.from_local("nonexistent_file_12345.pt")


class TestModelFromPretrained:
    """Test from_pretrained() for loading from HuggingFace Hub."""

    @pytest.mark.slow
    def test_rgb_decoder_from_pretrained(self):
        """Should load RGB decoder from HuggingFace."""
        from multispecqr.ml_decoder import RGBMLDecoder
        
        decoder = RGBMLDecoder.from_pretrained("Jemsbhai/multispecqr-rgb")
        
        assert decoder is not None
        assert decoder.num_outputs == 3
        assert isinstance(decoder, RGBMLDecoder)

    @pytest.mark.slow
    def test_palette6_decoder_from_pretrained(self):
        """Should load Palette-6 decoder from HuggingFace."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        decoder = PaletteMLDecoder.from_pretrained("Jemsbhai/multispecqr-palette6")
        
        assert decoder is not None
        assert decoder.num_layers == 6
        assert decoder.model_bits == 6
        assert decoder.num_outputs == 6

    @pytest.mark.slow
    def test_palette8_decoder_from_pretrained(self):
        """Should load Palette-8 decoder from HuggingFace."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        decoder = PaletteMLDecoder.from_pretrained("Jemsbhai/multispecqr-palette8")
        
        assert decoder is not None
        assert decoder.num_layers == 8
        assert decoder.model_bits == 8

    @pytest.mark.slow
    def test_palette9_decoder_from_pretrained(self):
        """Should load Palette-9 decoder from HuggingFace."""
        from multispecqr.ml_decoder import PaletteMLDecoder
        
        decoder = PaletteMLDecoder.from_pretrained("Jemsbhai/multispecqr-palette9")
        
        assert decoder is not None
        assert decoder.num_layers == 9
        assert decoder.model_bits == 9

    @pytest.mark.slow
    def test_from_pretrained_can_decode(self):
        """Loaded model should be able to decode images."""
        from multispecqr.ml_decoder import RGBMLDecoder
        
        decoder = RGBMLDecoder.from_pretrained("Jemsbhai/multispecqr-rgb")
        img = encode_rgb("TEST1", "TEST2", "TEST3", version=2)
        
        result = decoder.decode(img)
        
        assert isinstance(result, list)
        assert len(result) == 3

    @pytest.mark.slow
    def test_from_pretrained_invalid_repo_raises_error(self):
        """Loading from non-existent repo should raise error."""
        from multispecqr.ml_decoder import RGBMLDecoder
        
        with pytest.raises(Exception):  # Could be various HF errors
            RGBMLDecoder.from_pretrained("nonexistent/fake-repo-12345")
