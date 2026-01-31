"""
Train and upload pre-trained models to HuggingFace Hub.

This script trains RGB and Palette decoders and uploads them to HuggingFace.

Usage:
    python scripts/train_and_upload.py
    
    # With custom settings
    python scripts/train_and_upload.py --epochs 100 --samples 500
"""

import argparse
import torch

from multispecqr.ml_decoder import RGBMLDecoder, PaletteMLDecoder


def train_rgb_decoder(epochs: int, samples_per_epoch: int, version: int, device: str) -> RGBMLDecoder:
    """Train RGB decoder."""
    print(f"\n{'='*60}")
    print("Training RGB Decoder (3 layers)")
    print(f"{'='*60}")
    
    decoder = RGBMLDecoder(device=device)
    
    for epoch in range(epochs):
        loss = decoder.train_epoch(num_samples=samples_per_epoch, batch_size=16, version=version)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs}: loss = {loss:.4f}")
    
    return decoder


def train_palette_decoder(num_layers: int, epochs: int, samples_per_epoch: int, version: int, device: str) -> PaletteMLDecoder:
    """Train Palette decoder."""
    print(f"\n{'='*60}")
    print(f"Training Palette Decoder ({num_layers} layers)")
    print(f"{'='*60}")
    
    decoder = PaletteMLDecoder(num_layers=num_layers, device=device)
    print(f"Model bits: {decoder.model_bits}")
    
    for epoch in range(epochs):
        loss = decoder.train_epoch(num_samples=samples_per_epoch, batch_size=16, version=version)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs}: loss = {loss:.4f}")
    
    return decoder


def main():
    parser = argparse.ArgumentParser(description="Train and upload MultiSpecQR models")
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs per model')
    parser.add_argument('--samples', type=int, default=200, help='Samples per epoch')
    parser.add_argument('--version', type=int, default=2, help='QR code version')
    parser.add_argument('--no-push', action='store_true', help='Skip pushing to HuggingFace')
    parser.add_argument('--save-local', type=str, default=None, help='Save models locally to this directory')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {device}")
    print(f"Config: {args.epochs} epochs, {args.samples} samples/epoch, QR version {args.version}")
    
    # Create local save directory if needed
    if args.save_local:
        import os
        os.makedirs(args.save_local, exist_ok=True)
    
    models = {}
    
    # Train RGB decoder
    decoder = train_rgb_decoder(args.epochs, args.samples, args.version, device)
    models['rgb'] = decoder
    
    # Save immediately after training
    if args.save_local:
        import os
        path = os.path.join(args.save_local, "rgb.pt")
        decoder.save(path)
        print(f"✓ Saved rgb to {path}")
    
    # Push immediately after training
    if not args.no_push:
        try:
            decoder.push_to_hub()
        except Exception as e:
            print(f"Failed to push rgb: {e}")
    
    # Train Palette decoders (6, 8, 9 layers)
    for num_layers in [6, 8, 9]:
        decoder = train_palette_decoder(
            num_layers, args.epochs, args.samples, args.version, device
        )
        models[f'palette{num_layers}'] = decoder
        
        # Save immediately after training
        if args.save_local:
            import os
            path = os.path.join(args.save_local, f"palette{num_layers}.pt")
            decoder.save(path)
            print(f"✓ Saved palette{num_layers} to {path}")
        
        # Push immediately after training
        if not args.no_push:
            try:
                decoder.push_to_hub()
            except Exception as e:
                print(f"Failed to push palette{num_layers}: {e}")
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
