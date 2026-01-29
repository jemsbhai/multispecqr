"""
ML-based decoder for multi-spectral QR codes.

Uses a lightweight CNN to unmix color layers, providing better robustness
for real-world images with noise, compression artifacts, and color distortion.

Requires optional 'ml' dependencies: pip install multispecqr[ml]
"""
from __future__ import annotations

from typing import List, Tuple, Any
import numpy as np
from PIL import Image

# Check if torch is available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


def is_torch_available() -> bool:
    """Check if PyTorch is available."""
    return TORCH_AVAILABLE


if TORCH_AVAILABLE:
    class ColorUnmixingCNN(nn.Module):
        """
        Lightweight CNN for unmixing multi-spectral QR code colors.

        Takes an RGB image and outputs 6 binary layer masks.
        Uses a simple encoder-decoder architecture.
        """

        def __init__(self):
            super().__init__()

            # Encoder
            self.enc1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )

            self.enc2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

            # Decoder
            self.dec1 = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )

            # Output layer: 6 channels for 6 binary layers
            self.output = nn.Conv2d(32, 6, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            Args:
                x: Input tensor of shape (batch, 3, height, width)

            Returns:
                Output tensor of shape (batch, 6, height, width)
            """
            # Encoder
            e1 = self.enc1(x)
            e2 = self.enc2(e1)

            # Decoder
            d1 = self.dec1(e2)

            # Output
            out = self.output(d1)
            return torch.sigmoid(out)


class MLDecoder:
    """
    ML-based decoder for multi-spectral QR codes.

    Uses a CNN to unmix color layers, then applies standard QR decoding
    to each recovered layer.
    """

    def __init__(self, device: str | None = None):
        """
        Initialize the ML decoder.

        Args:
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for ML decoder. "
                "Install with: pip install multispecqr[ml]"
            )

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.model = ColorUnmixingCNN().to(self.device)
        self.model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        """
        Preprocess an image for the model.

        Args:
            img: PIL Image in RGB mode

        Returns:
            Tensor of shape (1, 3, H, W) normalized to [0, 1]
        """
        arr = np.array(img).astype(np.float32) / 255.0
        # Convert from (H, W, C) to (C, H, W)
        arr = arr.transpose(2, 0, 1)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        return tensor

    def postprocess(self, output: torch.Tensor) -> np.ndarray:
        """
        Postprocess model output to binary layers.

        Args:
            output: Model output tensor of shape (1, 6, H, W)

        Returns:
            Binary array of shape (H, W, 6)
        """
        # Move to CPU and convert to numpy
        arr = output.squeeze(0).detach().cpu().numpy()
        # Transpose from (C, H, W) to (H, W, C)
        arr = arr.transpose(1, 2, 0)
        # Threshold at 0.5
        binary = (arr > 0.5).astype(np.uint8)
        return binary

    def predict_layers(self, img: Image.Image) -> np.ndarray:
        """
        Predict binary layers from an image.

        Args:
            img: PIL Image in RGB mode

        Returns:
            Binary array of shape (H, W, 6)
        """
        self.model.eval()
        with torch.no_grad():
            x = self.preprocess(img)
            output = self.model(x)
            return self.postprocess(output)

    def train_epoch(
        self,
        num_samples: int = 100,
        batch_size: int = 8,
    ) -> float:
        """
        Train the model for one epoch using generated data.

        Args:
            num_samples: Number of training samples to generate
            batch_size: Batch size for training

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = (num_samples + batch_size - 1) // batch_size

        for _ in range(num_batches):
            # Generate training batch
            images, labels = generate_training_batch(batch_size)

            # Convert to tensors
            x = torch.from_numpy(images.transpose(0, 3, 1, 2)).float().to(self.device) / 255.0
            y = torch.from_numpy(labels.transpose(0, 3, 1, 2)).float().to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / num_batches


def generate_training_sample(
    version: int = 1,
    num_layers: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single training sample.

    Args:
        version: QR code version to use
        num_layers: Number of layers (1-6)

    Returns:
        Tuple of (image, labels) where:
            - image: RGB array of shape (H, W, 3)
            - labels: Binary array of shape (H, W, 6)
    """
    from .encoder import encode_layers, _make_layer
    from .palette import palette_6

    # Generate random data for each layer
    import random
    import string

    data_list = []
    for i in range(num_layers):
        # Random string of 3-8 characters
        length = random.randint(3, 8)
        data = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
        data_list.append(data)

    # Pad with empty strings if needed
    while len(data_list) < 6:
        data_list.append("")

    # Generate individual layers
    layers = []
    for i, data in enumerate(data_list):
        if data:
            layer = _make_layer(data, version, "M")
            layers.append(layer)
        else:
            # Empty layer (all zeros)
            if layers:
                layers.append(np.zeros_like(layers[0]))

    if not layers:
        raise ValueError("At least one layer must have data")

    # Get shape from first layer
    h, w = layers[0].shape

    # Build the encoded image using palette
    codebook = palette_6()
    image = np.zeros((h, w, 3), dtype=np.uint8)
    labels = np.zeros((h, w, 6), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            # Build bit-vector
            bits = []
            for i in range(6):
                if i < len(layers):
                    bits.append(int(layers[i][y, x]))
                else:
                    bits.append(0)

            key = tuple(bits)
            color = codebook.get(key, (255, 255, 255))
            image[y, x] = color
            labels[y, x] = bits

    return image, labels


def generate_training_batch(
    batch_size: int = 8,
    version: int = 1,
    num_layers: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a batch of training samples.

    Args:
        batch_size: Number of samples in the batch
        version: QR code version to use
        num_layers: Number of layers (1-6)

    Returns:
        Tuple of (images, labels) where:
            - images: Array of shape (batch, H, W, 3)
            - labels: Array of shape (batch, H, W, 6)
    """
    samples = [generate_training_sample(version, num_layers) for _ in range(batch_size)]
    images = np.stack([s[0] for s in samples])
    labels = np.stack([s[1] for s in samples])
    return images, labels


def decode_rgb_ml(
    img: Image.Image,
    decoder: MLDecoder | None = None,
) -> List[str]:
    """
    Decode an RGB QR code using ML-based layer separation.

    Args:
        img: PIL Image in RGB mode
        decoder: Optional pre-initialized MLDecoder

    Returns:
        List of 3 decoded strings (R, G, B layers)
    """
    import cv2

    if decoder is None:
        decoder = MLDecoder()

    # Predict layers
    layers = decoder.predict_layers(img)

    # Decode each of the first 3 layers (R, G, B)
    results = []
    for i in range(3):
        layer = layers[:, :, i]
        # Convert to QR-readable format
        binary = ((1 - layer) * 255).astype(np.uint8)

        # Decode QR
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(binary)
        results.append(data or "")

    return results


def decode_layers_ml(
    img: Image.Image,
    num_layers: int = 6,
    decoder: MLDecoder | None = None,
) -> List[str]:
    """
    Decode palette-encoded QR code using ML-based layer separation.

    Args:
        img: PIL Image in RGB mode
        num_layers: Number of layers to decode (1-6)
        decoder: Optional pre-initialized MLDecoder

    Returns:
        List of decoded strings, one per layer
    """
    import cv2

    if decoder is None:
        decoder = MLDecoder()

    # Predict layers
    layers = decoder.predict_layers(img)

    # Decode each layer
    results = []
    for i in range(num_layers):
        layer = layers[:, :, i]
        # Convert to QR-readable format
        binary = ((1 - layer) * 255).astype(np.uint8)

        # Decode QR
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(binary)
        results.append(data or "")

    return results
