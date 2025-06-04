"""
Command-line entry for multispecqr.

Usage examples:
    # encode three payloads to out.png
    python -m multispecqr encode "RED" "GREEN" "BLUE" out.png

    # decode a composite QR image
    python -m multispecqr decode out.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from .encoder import encode_rgb
from .decoder import decode_rgb


def _cmd_encode(args: argparse.Namespace) -> None:
    img = encode_rgb(args.red, args.green, args.blue)
    img.save(args.output)
    print(f"Saved composite QR to {args.output}")


def _cmd_decode(args: argparse.Namespace) -> None:
    img = Image.open(args.image_path)
    payloads = decode_rgb(img)
    print("Decoded layers:")
    for label, text in zip(("R", "G", "B"), payloads):
        print(f"  {label}: {text!r}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="multispecqr")
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="Encode three payloads into an RGB QR image")
    enc.add_argument("red")
    enc.add_argument("green")
    enc.add_argument("blue")
    enc.add_argument("output", type=Path)
    enc.set_defaults(func=_cmd_encode)

    dec = sub.add_parser("decode", help="Decode an RGB QR image")
    dec.add_argument("image_path", type=Path)
    dec.set_defaults(func=_cmd_decode)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":  # `python -m multispecqr …`
    main()
