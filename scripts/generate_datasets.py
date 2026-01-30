"""
Generate training datasets for MultiSpecQR ML decoders.

This script generates large datasets that can be:
1. Saved locally as NumPy files
2. Uploaded to HuggingFace Hub for sharing

Run locally or on Google Colab with GPU for faster generation.

Usage:
    # Generate and save locally
    python scripts/generate_datasets.py --output ./datasets --samples 10000
    
    # Generate and push to HuggingFace
    python scripts/generate_datasets.py --push-to-hub jemsbhai/multispecqr-datasets --samples 10000
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def generate_rgb_dataset(
    num_samples: int,
    version: int = 2,
    show_progress: bool = True,
) -> dict:
    """
    Generate RGB training dataset.
    
    Args:
        num_samples: Number of samples to generate
        version: QR code version
        show_progress: Show progress bar
        
    Returns:
        Dictionary with 'images' and 'labels' arrays
    """
    from multispecqr.ml_decoder import _generate_rgb_sample
    
    images = []
    labels = []
    
    iterator = range(num_samples)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Generating RGB v{version}")
    
    for _ in iterator:
        img, lbl = _generate_rgb_sample(version=version, use_cache=False)
        images.append(img)
        labels.append(lbl)
    
    return {
        'images': np.stack(images),
        'labels': np.stack(labels),
        'version': version,
        'mode': 'rgb',
        'num_layers': 3,
    }


def generate_palette_dataset(
    num_samples: int,
    num_layers: int = 6,
    version: int = 2,
    show_progress: bool = True,
) -> dict:
    """
    Generate palette training dataset.
    
    Args:
        num_samples: Number of samples to generate
        num_layers: Number of layers (6, 8, or 9)
        version: QR code version
        show_progress: Show progress bar
        
    Returns:
        Dictionary with 'images' and 'labels' arrays
    """
    from multispecqr.ml_decoder import _generate_palette_sample
    
    images = []
    labels = []
    
    iterator = range(num_samples)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Generating Palette-{num_layers} v{version}")
    
    for _ in iterator:
        img, lbl = _generate_palette_sample(version=version, num_layers=num_layers, use_cache=False)
        images.append(img)
        labels.append(lbl)
    
    return {
        'images': np.stack(images),
        'labels': np.stack(labels),
        'version': version,
        'mode': 'palette',
        'num_layers': num_layers,
    }


def add_noise_augmentation(dataset: dict, noise_level: float = 0.05) -> dict:
    """
    Add Gaussian noise to images for robustness training.
    
    Args:
        dataset: Original dataset dictionary
        noise_level: Standard deviation of noise (0-1 scale)
        
    Returns:
        New dataset with noisy images
    """
    images = dataset['images'].astype(np.float32)
    noise = np.random.normal(0, noise_level * 255, images.shape)
    noisy = np.clip(images + noise, 0, 255).astype(np.uint8)
    
    return {
        **dataset,
        'images': noisy,
        'augmentation': f'gaussian_noise_{noise_level}',
    }


def add_jpeg_compression(dataset: dict, quality: int = 70) -> dict:
    """
    Add JPEG compression artifacts for robustness training.
    
    Args:
        dataset: Original dataset dictionary
        quality: JPEG quality (1-100)
        
    Returns:
        New dataset with compressed images
    """
    from PIL import Image
    import io
    
    compressed = []
    for img in tqdm(dataset['images'], desc=f"JPEG compression q={quality}"):
        pil_img = Image.fromarray(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = np.array(Image.open(buffer))
        compressed.append(compressed_img)
    
    return {
        **dataset,
        'images': np.stack(compressed),
        'augmentation': f'jpeg_q{quality}',
    }


def save_dataset(dataset: dict, output_dir: Path, name: str):
    """Save dataset as NumPy files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    path = output_dir / f"{name}.npz"
    np.savez_compressed(
        path,
        images=dataset['images'],
        labels=dataset['labels'],
        metadata=np.array([
            dataset['version'],
            dataset['num_layers'],
            dataset.get('augmentation', 'clean'),
            dataset['mode'],
        ], dtype=object),
    )
    
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Saved {name}: {len(dataset['images'])} samples, {size_mb:.1f} MB")
    return path


def push_to_huggingface(
    dataset: dict,
    repo_id: str,
    subset_name: str,
    token: str = None,
):
    """
    Push dataset to HuggingFace Hub using the datasets library.
    
    Args:
        dataset: Dataset dictionary
        repo_id: HuggingFace repo ID (e.g., 'jemsbhai/multispecqr-datasets')
        subset_name: Name for this subset (e.g., 'rgb-v2-clean')
        token: HuggingFace API token (uses cached token if None)
    """
    try:
        from datasets import Dataset, DatasetDict
    except ImportError:
        print("Please install datasets: pip install datasets")
        return
    
    # Convert to HuggingFace Dataset format
    hf_dataset = Dataset.from_dict({
        'image': [img for img in dataset['images']],
        'labels': [lbl for lbl in dataset['labels']],
    })
    
    # Add metadata as features
    hf_dataset = hf_dataset.add_column('version', [dataset['version']] * len(dataset['images']))
    hf_dataset = hf_dataset.add_column('mode', [dataset['mode']] * len(dataset['images']))
    hf_dataset = hf_dataset.add_column('num_layers', [dataset['num_layers']] * len(dataset['images']))
    
    # Push to hub
    hf_dataset.push_to_hub(
        repo_id,
        subset_name,
        token=token,
        private=False,
    )
    print(f"Pushed {subset_name} to {repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Generate MultiSpecQR training datasets")
    parser.add_argument('--output', type=Path, default=Path('./datasets'),
                        help='Output directory for local saves')
    parser.add_argument('--samples', type=int, default=10000,
                        help='Number of samples per dataset')
    parser.add_argument('--version', type=int, default=2,
                        help='QR code version')
    parser.add_argument('--push-to-hub', type=str, default=None,
                        help='HuggingFace repo ID to push to')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace API token')
    parser.add_argument('--modes', nargs='+', default=['rgb', 'palette6', 'palette8', 'palette9'],
                        help='Modes to generate: rgb, palette6, palette8, palette9')
    parser.add_argument('--augment', action='store_true',
                        help='Also generate augmented (noisy, compressed) versions')
    
    args = parser.parse_args()
    
    datasets_generated = []
    
    # Generate RGB dataset
    if 'rgb' in args.modes:
        print("\n=== RGB Dataset ===")
        rgb_data = generate_rgb_dataset(args.samples, version=args.version)
        save_dataset(rgb_data, args.output, f"rgb-v{args.version}-clean")
        datasets_generated.append(('rgb-v{}-clean'.format(args.version), rgb_data))
        
        if args.augment:
            noisy = add_noise_augmentation(rgb_data, noise_level=0.05)
            save_dataset(noisy, args.output, f"rgb-v{args.version}-noisy")
            datasets_generated.append(('rgb-v{}-noisy'.format(args.version), noisy))
    
    # Generate Palette datasets
    for mode in args.modes:
        if mode.startswith('palette'):
            num_layers = int(mode.replace('palette', ''))
            print(f"\n=== Palette-{num_layers} Dataset ===")
            palette_data = generate_palette_dataset(
                args.samples,
                num_layers=num_layers,
                version=args.version,
            )
            save_dataset(palette_data, args.output, f"palette{num_layers}-v{args.version}-clean")
            datasets_generated.append((f'palette{num_layers}-v{args.version}-clean', palette_data))
            
            if args.augment:
                noisy = add_noise_augmentation(palette_data, noise_level=0.05)
                save_dataset(noisy, args.output, f"palette{num_layers}-v{args.version}-noisy")
                datasets_generated.append((f'palette{num_layers}-v{args.version}-noisy', noisy))
    
    # Push to HuggingFace if requested
    if args.push_to_hub:
        print(f"\n=== Pushing to HuggingFace: {args.push_to_hub} ===")
        for name, data in datasets_generated:
            push_to_huggingface(data, args.push_to_hub, name, token=args.token)
    
    print("\n=== Done ===")
    print(f"Generated {len(datasets_generated)} datasets")


if __name__ == '__main__':
    main()
