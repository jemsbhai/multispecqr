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

    def test_encode_palette_rejects_seven_layers(self, tmp_path):
        """Palette mode should reject more than 6 layers."""
        output = tmp_path / "test.png"
        result = run_cli("encode", "1", "2", "3", "4", "5", "6", "7", str(output), "-m", "palette")
        assert result.returncode == 1
        assert "1-6 payloads" in result.stderr


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
