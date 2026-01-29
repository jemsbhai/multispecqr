"""
Test suite for the multispecqr CLI.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run the multispecqr CLI with given arguments."""
    return subprocess.run(
        [sys.executable, "-m", "multispecqr", *args],
        capture_output=True,
        text=True,
    )


class TestCLIHelp:
    """Test CLI help messages."""

    def test_main_help(self):
        """Main help should show encode and decode commands."""
        result = run_cli("--help")
        assert result.returncode == 0
        assert "encode" in result.stdout
        assert "decode" in result.stdout

    def test_encode_help(self):
        """Encode help should show all options."""
        result = run_cli("encode", "--help")
        assert result.returncode == 0
        assert "--mode" in result.stdout
        assert "--version" in result.stdout
        assert "--ec" in result.stdout

    def test_decode_help(self):
        """Decode help should show all options."""
        result = run_cli("decode", "--help")
        assert result.returncode == 0
        assert "--mode" in result.stdout
        assert "--layers" in result.stdout


class TestCLIEncodeRGB:
    """Test CLI encode command in RGB mode."""

    def test_encode_rgb_creates_file(self, tmp_path):
        """Encode should create an image file."""
        output = tmp_path / "test.png"
        result = run_cli("encode", "R", "G", "B", str(output))
        assert result.returncode == 0
        assert output.exists()
        assert "Saved RGB QR" in result.stdout

    def test_encode_rgb_requires_three_payloads(self, tmp_path):
        """RGB mode should require exactly 3 payloads."""
        output = tmp_path / "test.png"
        result = run_cli("encode", "R", "G", str(output))
        assert result.returncode == 1
        assert "requires exactly 3 payloads" in result.stderr

    def test_encode_rgb_with_version(self, tmp_path):
        """Encode should accept version option."""
        output = tmp_path / "test.png"
        result = run_cli("encode", "R", "G", "B", str(output), "-v", "3")
        assert result.returncode == 0
        assert output.exists()

    def test_encode_rgb_with_error_correction(self, tmp_path):
        """Encode should accept error correction option."""
        output = tmp_path / "test.png"
        result = run_cli("encode", "R", "G", "B", str(output), "-e", "H")
        assert result.returncode == 0
        assert output.exists()


class TestCLIEncodePalette:
    """Test CLI encode command in palette mode."""

    def test_encode_palette_single_layer(self, tmp_path):
        """Palette mode should work with 1 layer."""
        output = tmp_path / "test.png"
        result = run_cli("encode", "Single", str(output), "-m", "palette")
        assert result.returncode == 0
        assert output.exists()
        assert "palette (1 layers)" in result.stdout

    def test_encode_palette_six_layers(self, tmp_path):
        """Palette mode should work with 6 layers."""
        output = tmp_path / "test.png"
        result = run_cli("encode", "A", "B", "C", "D", "E", "F", str(output), "-m", "palette")
        assert result.returncode == 0
        assert output.exists()
        assert "palette (6 layers)" in result.stdout

    def test_encode_palette_rejects_ten_layers(self, tmp_path):
        """Palette mode should reject more than 9 layers."""
        output = tmp_path / "test.png"
        result = run_cli("encode", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", str(output), "-m", "palette")
        assert result.returncode == 1
        assert "1-9 payloads" in result.stderr


class TestCLIEncodeExpandedCapacity:
    """Test CLI encode command with expanded capacity (7-9 layers)."""

    def test_encode_palette_seven_layers(self, tmp_path):
        """Palette mode should work with 7 layers."""
        output = tmp_path / "test.png"
        result = run_cli("encode", "A", "B", "C", "D", "E", "F", "G", str(output), "-m", "palette")
        assert result.returncode == 0
        assert output.exists()
        assert "palette (7 layers)" in result.stdout

    def test_encode_palette_eight_layers(self, tmp_path):
        """Palette mode should work with 8 layers."""
        output = tmp_path / "test.png"
        result = run_cli("encode", "1", "2", "3", "4", "5", "6", "7", "8", str(output), "-m", "palette")
        assert result.returncode == 0
        assert output.exists()
        assert "palette (8 layers)" in result.stdout

    def test_encode_palette_nine_layers(self, tmp_path):
        """Palette mode should work with 9 layers."""
        output = tmp_path / "test.png"
        result = run_cli("encode", "A", "B", "C", "D", "E", "F", "G", "H", "I", str(output), "-m", "palette", "-v", "3")
        assert result.returncode == 0
        assert output.exists()
        assert "palette (9 layers)" in result.stdout

    def test_decode_palette_nine_layers_roundtrip(self, tmp_path):
        """Decode should recover 9-layer palette data."""
        output = tmp_path / "test.png"
        run_cli("encode", "A", "B", "C", "D", "E", "F", "G", "H", "I", str(output), "-m", "palette", "-v", "4")

        result = run_cli("decode", str(output), "-m", "palette", "-l", "9")
        assert result.returncode == 0
        assert "L1: 'A'" in result.stdout
        assert "L9: 'I'" in result.stdout


class TestCLIDecodeRGB:
    """Test CLI decode command in RGB mode."""

    def test_decode_rgb_roundtrip(self, tmp_path):
        """Decode should recover encoded RGB data."""
        output = tmp_path / "test.png"
        run_cli("encode", "Red", "Green", "Blue", str(output))

        result = run_cli("decode", str(output))
        assert result.returncode == 0
        assert "R: 'Red'" in result.stdout
        assert "G: 'Green'" in result.stdout
        assert "B: 'Blue'" in result.stdout


class TestCLIDecodePalette:
    """Test CLI decode command in palette mode."""

    def test_decode_palette_roundtrip_three_layers(self, tmp_path):
        """Decode should recover 3-layer palette data."""
        output = tmp_path / "test.png"
        run_cli("encode", "L1", "L2", "L3", str(output), "-m", "palette")

        result = run_cli("decode", str(output), "-m", "palette", "-l", "3")
        assert result.returncode == 0
        assert "L1: 'L1'" in result.stdout
        assert "L2: 'L2'" in result.stdout
        assert "L3: 'L3'" in result.stdout

    def test_decode_palette_roundtrip_six_layers(self, tmp_path):
        """Decode should recover 6-layer palette data."""
        output = tmp_path / "test.png"
        run_cli("encode", "A", "B", "C", "D", "E", "F", str(output), "-m", "palette")

        result = run_cli("decode", str(output), "-m", "palette", "-l", "6")
        assert result.returncode == 0
        assert "L1: 'A'" in result.stdout
        assert "L2: 'B'" in result.stdout
        assert "L3: 'C'" in result.stdout
        assert "L4: 'D'" in result.stdout
        assert "L5: 'E'" in result.stdout
        assert "L6: 'F'" in result.stdout


class TestCLIRobustnessOptions:
    """Test CLI robustness options for decoding."""

    def test_decode_with_threshold_otsu(self, tmp_path):
        """Decode should accept --threshold otsu option."""
        output = tmp_path / "test.png"
        run_cli("encode", "R", "G", "B", str(output))

        result = run_cli("decode", str(output), "--threshold", "otsu")
        assert result.returncode == 0
        assert "R: 'R'" in result.stdout

    def test_decode_with_threshold_adaptive(self, tmp_path):
        """Decode should accept --threshold adaptive_gaussian option."""
        output = tmp_path / "test.png"
        run_cli("encode", "R", "G", "B", str(output))

        result = run_cli("decode", str(output), "-t", "adaptive_gaussian")
        assert result.returncode == 0

    def test_decode_with_preprocess_blur(self, tmp_path):
        """Decode should accept --preprocess blur option."""
        output = tmp_path / "test.png"
        run_cli("encode", "R", "G", "B", str(output))

        result = run_cli("decode", str(output), "--preprocess", "blur")
        assert result.returncode == 0

    def test_decode_with_preprocess_denoise(self, tmp_path):
        """Decode should accept --preprocess denoise option."""
        output = tmp_path / "test.png"
        run_cli("encode", "R", "G", "B", str(output))

        result = run_cli("decode", str(output), "-p", "denoise")
        assert result.returncode == 0

    def test_decode_palette_with_robustness(self, tmp_path):
        """Palette decode should work with preprocess option."""
        output = tmp_path / "test.png"
        run_cli("encode", "A", "B", "C", str(output), "-m", "palette")

        # Palette mode supports preprocess but not threshold (uses color matching)
        result = run_cli("decode", str(output), "-m", "palette", "-l", "3", "-p", "blur")
        assert result.returncode == 0


class TestCLIOutputFormats:
    """Test CLI output format options."""

    def test_decode_json_output(self, tmp_path):
        """Decode should support --json output format."""
        output = tmp_path / "test.png"
        run_cli("encode", "Red", "Green", "Blue", str(output))

        result = run_cli("decode", str(output), "--json")
        assert result.returncode == 0
        # Should output valid JSON
        import json
        data = json.loads(result.stdout)
        assert data["R"] == "Red"
        assert data["G"] == "Green"
        assert data["B"] == "Blue"

    def test_decode_json_palette(self, tmp_path):
        """Palette decode should support --json output."""
        output = tmp_path / "test.png"
        run_cli("encode", "A", "B", "C", str(output), "-m", "palette")

        result = run_cli("decode", str(output), "-m", "palette", "-l", "3", "--json")
        assert result.returncode == 0
        import json
        data = json.loads(result.stdout)
        assert data["L1"] == "A"
        assert data["L2"] == "B"
        assert data["L3"] == "C"


class TestCLIEncodingOptions:
    """Test additional encoding options."""

    def test_encode_with_scale(self, tmp_path):
        """Encode should support --scale option."""
        output = tmp_path / "test.png"
        result = run_cli("encode", "A", "B", "C", str(output), "--scale", "10")
        assert result.returncode == 0
        assert output.exists()

        # Verify image is larger than default
        from PIL import Image
        img = Image.open(output)
        assert img.width > 100  # Should be scaled up


class TestCLICalibration:
    """Test CLI calibration features."""

    def test_calibrate_command(self, tmp_path):
        """calibrate command should generate a calibration card."""
        output = tmp_path / "calibration.png"
        result = run_cli("calibrate", str(output))
        assert result.returncode == 0
        assert output.exists()
        assert "calibration card" in result.stdout.lower()

    def test_calibrate_with_patch_size(self, tmp_path):
        """calibrate command should accept --patch-size option."""
        output = tmp_path / "calibration.png"
        result = run_cli("calibrate", str(output), "--patch-size", "30")
        assert result.returncode == 0
        assert output.exists()


class TestCLIBatchProcessing:
    """Test CLI batch processing features."""

    def test_batch_decode(self, tmp_path):
        """batch-decode should decode multiple images."""
        # Create two encoded images
        img1 = tmp_path / "img1.png"
        img2 = tmp_path / "img2.png"
        run_cli("encode", "A1", "B1", "C1", str(img1))
        run_cli("encode", "A2", "B2", "C2", str(img2))

        # Batch decode
        result = run_cli("batch-decode", str(img1), str(img2))
        assert result.returncode == 0
        assert "img1.png" in result.stdout
        assert "img2.png" in result.stdout
        assert "A1" in result.stdout
        assert "A2" in result.stdout

    def test_batch_decode_json(self, tmp_path):
        """batch-decode should support --json output."""
        img1 = tmp_path / "img1.png"
        run_cli("encode", "X", "Y", "Z", str(img1))

        result = run_cli("batch-decode", str(img1), "--json")
        assert result.returncode == 0
        import json
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) == 1


class TestCLIEdgeCases:
    """Test CLI edge cases and error handling."""

    def test_decode_nonexistent_file(self, tmp_path):
        """Decode should fail gracefully for missing file."""
        result = run_cli("decode", str(tmp_path / "nonexistent.png"))
        assert result.returncode != 0

    def test_no_command(self):
        """CLI should require a command."""
        result = run_cli()
        assert result.returncode != 0

    def test_invalid_threshold_method(self, tmp_path):
        """Decode should reject invalid threshold method."""
        output = tmp_path / "test.png"
        run_cli("encode", "R", "G", "B", str(output))

        result = run_cli("decode", str(output), "--threshold", "invalid")
        assert result.returncode != 0

    def test_invalid_preprocess_method(self, tmp_path):
        """Decode should reject invalid preprocess method."""
        output = tmp_path / "test.png"
        run_cli("encode", "R", "G", "B", str(output))

        result = run_cli("decode", str(output), "--preprocess", "invalid")
        assert result.returncode != 0
