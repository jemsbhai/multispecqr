"""
Command-line interface for multispecqr.

Usage examples:
    # RGB mode: encode three payloads
    python -m multispecqr encode "RED" "GREEN" "BLUE" out.png

    # RGB mode: encode with options
    python -m multispecqr encode "RED" "GREEN" "BLUE" out.png --version 4 --ec H

    # Palette mode: encode up to 6 payloads
    python -m multispecqr encode "L1" "L2" "L3" "L4" "L5" "L6" out.png --mode palette

    # Decode (auto-detects mode, or specify)
    python -m multispecqr decode image.png
    python -m multispecqr decode image.png --mode palette --layers 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from .encoder import encode_rgb, encode_layers
from .decoder import decode_rgb, decode_layers


def _cmd_encode(args: argparse.Namespace) -> None:
    """Handle the encode command."""
    payloads = args.data

    if args.mode == "rgb":
        if len(payloads) != 3:
            print(f"Error: RGB mode requires exactly 3 payloads, got {len(payloads)}", file=sys.stderr)
            sys.exit(1)
        img = encode_rgb(payloads[0], payloads[1], payloads[2], version=args.version, ec=args.ec)
        mode_str = "RGB"
    else:  # palette mode
        if len(payloads) < 1 or len(payloads) > 6:
            print(f"Error: Palette mode requires 1-6 payloads, got {len(payloads)}", file=sys.stderr)
            sys.exit(1)
        img = encode_layers(payloads, version=args.version, ec=args.ec)
        mode_str = f"palette ({len(payloads)} layers)"

    img.save(args.output)
    print(f"Saved {mode_str} QR to {args.output}")


def _cmd_decode(args: argparse.Namespace) -> None:
    """Handle the decode command."""
    img = Image.open(args.image_path)

    if img.mode != "RGB":
        print(f"Error: Expected RGB image, got {img.mode}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "rgb":
        payloads = decode_rgb(img)
        labels = ["R", "G", "B"]
    else:  # palette mode
        payloads = decode_layers(img, num_layers=args.layers)
        labels = [f"L{i+1}" for i in range(len(payloads))]

    print("Decoded layers:")
    for label, text in zip(labels, payloads):
        if text:
            print(f"  {label}: {text!r}")
        else:
            print(f"  {label}: (empty or failed to decode)")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="multispecqr",
        description="Encode and decode multi-spectral QR codes",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Encode command
    enc = sub.add_parser(
        "encode",
        help="Encode payloads into a multi-spectral QR image",
        description="Encode 1-6 payloads into a single QR code image using color channels.",
    )
    enc.add_argument(
        "data",
        nargs="+",
        help="Payload strings to encode (3 for RGB mode, 1-6 for palette mode)",
    )
    enc.add_argument(
        "output",
        type=Path,
        help="Output image path (e.g., output.png)",
    )
    enc.add_argument(
        "--mode", "-m",
        choices=["rgb", "palette"],
        default="rgb",
        help="Encoding mode: 'rgb' (3 layers) or 'palette' (1-6 layers). Default: rgb",
    )
    enc.add_argument(
        "--version", "-v",
        type=int,
        default=2,
        help="QR code version (1-40). Higher = more capacity. Default: 2",
    )
    enc.add_argument(
        "--ec", "-e",
        choices=["L", "M", "Q", "H"],
        default="M",
        help="Error correction level. L=7%%, M=15%%, Q=25%%, H=30%%. Default: M",
    )
    enc.set_defaults(func=_cmd_encode)

    # Decode command
    dec = sub.add_parser(
        "decode",
        help="Decode a multi-spectral QR image",
        description="Decode a multi-spectral QR code image back to its payloads.",
    )
    dec.add_argument(
        "image_path",
        type=Path,
        help="Path to the QR image to decode",
    )
    dec.add_argument(
        "--mode", "-m",
        choices=["rgb", "palette"],
        default="rgb",
        help="Decoding mode: 'rgb' (3 layers) or 'palette' (1-6 layers). Default: rgb",
    )
    dec.add_argument(
        "--layers", "-l",
        type=int,
        default=6,
        help="Number of layers to decode in palette mode (1-6). Default: 6",
    )
    dec.set_defaults(func=_cmd_decode)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
