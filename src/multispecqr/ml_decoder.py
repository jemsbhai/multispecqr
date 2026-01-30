"""
ML-based decoder for multi-spectral QR codes.

Uses lightweight CNNs to unmix color layers, providing better robustness
for real-world images with noise, compression artifacts, and color distortion.

Provides two separate decoders:
- RGBMLDecoder: For RGB-encoded images (3 layers)
- PaletteMLDecoder: For palette-encoded images (6 layers)

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


def _detect_nvidia_gpu() -> bool:
    """
    Detect if an NVIDIA GPU is present on the system.
    
    Uses nvidia-smi to check for GPU presence, independent of PyTorch.
    """
    import subprocess
    import shutil
    
    if shutil.which("nvidia-smi") is None:
        return False
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _check_gpu_advisory() -> None:
    """Check if user has a GPU but PyTorch can't use it."""
    if not TORCH_AVAILABLE:
        return
    
    if torch.cuda.is_available():
        return
    
    if not _detect_nvidia_gpu():
        return
    
    import warnings
    msg = (
        "NVIDIA GPU detected but PyTorch is using CPU. "
        "For faster ML decoding, install CUDA-enabled PyTorch: "
        "pip uninstall torch torchvision -y && "
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
    )
    warnings.warn(msg, stacklevel=3)


_GPU_ADVISORY_SHOWN = False


def _maybe_show_gpu_advisory() -> None:
    """Show GPU advisory once per session."""
    global _GPU_ADVISORY_SHOWN
    if not _GPU_ADVISORY_SHOWN:
        _check_gpu_advisory()
        _GPU_ADVISORY_SHOWN = True


if TORCH_AVAILABLE:
    class LayerUnmixingCNN(nn.Module):
        """
        CNN for unmixing multi-spectral QR code colors.
        
        Configurable output channels for RGB (3) or palette (6) modes.
        """

        def __init__(self, num_outputs: int = 6):
            super().__init__()
            
            self.num_outputs = num_outputs

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

            # Output layer
            self.output = nn.Conv2d(32, num_outputs, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            d1 = self.dec1(e2)
            out = self.output(d1)
            return torch.sigmoid(out)


class _BaseMLDecoder:
    """Base class for ML decoders."""
    
    def __init__(self, num_outputs: int, device: str | None = None):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for ML decoder. "
                "Install with: pip install multispecqr[ml]"
            )

        _maybe_show_gpu_advisory()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.num_outputs = num_outputs

        self.model = LayerUnmixingCNN(num_outputs=num_outputs).to(self.device)
        self.model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        """Preprocess an image for the model."""
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        return tensor

    def postprocess(self, output: torch.Tensor) -> np.ndarray:
        """Postprocess model output to binary layers."""
        arr = output.squeeze(0).detach().cpu().numpy()
        arr = arr.transpose(1, 2, 0)
        binary = (arr > 0.5).astype(np.uint8)
        return binary

    def predict_layers(self, img: Image.Image) -> np.ndarray:
        """Predict binary layers from an image."""
        self.model.eval()
        with torch.no_grad():
            x = self.preprocess(img)
            output = self.model(x)
            return self.postprocess(output)


class RGBMLDecoder(_BaseMLDecoder):
    """
    ML decoder for RGB-encoded QR codes.
    
    Outputs 3 binary layers corresponding to R, G, B channels.
    """
    
    def __init__(self, device: str | None = None):
        super().__init__(num_outputs=3, device=device)
    
    def train_epoch(
        self,
        num_samples: int = 100,
        batch_size: int = 8,
        version: int = 1,
    ) -> float:
        """Train for one epoch using generated RGB data."""
        self.model.train()
        total_loss = 0.0
        num_batches = (num_samples + batch_size - 1) // batch_size

        for _ in range(num_batches):
            images, labels = _generate_rgb_batch(batch_size, version)

            x = torch.from_numpy(images.transpose(0, 3, 1, 2)).float().to(self.device) / 255.0
            y = torch.from_numpy(labels.transpose(0, 3, 1, 2)).float().to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / num_batches
    
    def decode(self, img: Image.Image) -> List[str]:
        """Decode an RGB QR code into 3 strings."""
        import cv2
        
        layers = self.predict_layers(img)
        
        results = []
        for i in range(3):
            layer = layers[:, :, i]
            binary = ((1 - layer) * 255).astype(np.uint8)
            
            # Try OpenCV first
            detector = cv2.QRCodeDetector()
            data, _, _ = detector.detectAndDecode(binary)
            
            if not data:
                # Fallback to pyzbar
                try:
                    from pyzbar import pyzbar
                    pil_img = Image.fromarray(binary)
                    decoded = pyzbar.decode(pil_img)
                    if decoded:
                        data = decoded[0].data.decode('utf-8')
                except ImportError:
                    pass
            
            results.append(data or "")
        
        return results


class PaletteMLDecoder(_BaseMLDecoder):
    """
    ML decoder for palette-encoded QR codes.
    
    Outputs 6 binary layers corresponding to the 6-bit palette encoding.
    """
    
    def __init__(self, device: str | None = None):
        super().__init__(num_outputs=6, device=device)
    
    def train_epoch(
        self,
        num_samples: int = 100,
        batch_size: int = 8,
        version: int = 1,
        num_layers: int = 6,
    ) -> float:
        """Train for one epoch using generated palette data."""
        self.model.train()
        total_loss = 0.0
        num_batches = (num_samples + batch_size - 1) // batch_size

        for _ in range(num_batches):
            images, labels = _generate_palette_batch(batch_size, version, num_layers)

            x = torch.from_numpy(images.transpose(0, 3, 1, 2)).float().to(self.device) / 255.0
            y = torch.from_numpy(labels.transpose(0, 3, 1, 2)).float().to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / num_batches
    
    def decode(self, img: Image.Image, num_layers: int = 6) -> List[str]:
        """Decode a palette QR code into up to 6 strings."""
        import cv2
        
        layers = self.predict_layers(img)
        
        results = []
        for i in range(num_layers):
            layer = layers[:, :, i]
            binary = ((1 - layer) * 255).astype(np.uint8)
            
            # Try OpenCV first
            detector = cv2.QRCodeDetector()
            data, _, _ = detector.detectAndDecode(binary)
            
            if not data:
                # Fallback to pyzbar
                try:
                    from pyzbar import pyzbar
                    pil_img = Image.fromarray(binary)
                    decoded = pyzbar.decode(pil_img)
                    if decoded:
                        data = decoded[0].data.decode('utf-8')
                except ImportError:
                    pass
            
            results.append(data or "")
        
        return results


# =============================================================================
# Training Data Generation
# =============================================================================

def _generate_rgb_sample(version: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single RGB training sample.
    
    Matches the actual encode_rgb behavior:
    - channel = layer * 255
    - So black modules (layer=1) become bright (255)
    - White areas (layer=0) become dark (0)
    """
    from .encoder import _make_layer
    
    import random
    import string
    
    # Generate 3 random payloads
    data_list = []
    for _ in range(3):
        length = random.randint(3, 8)
        data = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
        data_list.append(data)
    
    # Generate QR layers
    layers = [_make_layer(d, version, "M") for d in data_list]
    h, w = layers[0].shape
    
    # Build RGB image matching encode_rgb: channel = layer * 255
    # Black modules (layer=1) -> channel=255 (bright)
    # White areas (layer=0) -> channel=0 (dark)
    image = np.zeros((h, w, 3), dtype=np.uint8)
    labels = np.zeros((h, w, 3), dtype=np.uint8)
    
    for c in range(3):
        image[:, :, c] = layers[c] * 255  # Match encode_rgb!
        labels[:, :, c] = layers[c]  # 1 = black module
    
    return image, labels


def _generate_rgb_batch(
    batch_size: int = 8,
    version: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a batch of RGB training samples."""
    samples = [_generate_rgb_sample(version) for _ in range(batch_size)]
    images = np.stack([s[0] for s in samples])
    labels = np.stack([s[1] for s in samples])
    return images, labels


def _generate_palette_sample(
    version: int = 1,
    num_layers: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single palette training sample."""
    from .encoder import _make_layer
    from .palette import palette_6

    import random
    import string

    data_list = []
    for _ in range(num_layers):
        length = random.randint(3, 8)
        data = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
        data_list.append(data)

    while len(data_list) < 6:
        data_list.append("")

    layers = []
    for data in data_list:
        if data:
            layer = _make_layer(data, version, "M")
            layers.append(layer)
        elif layers:
            layers.append(np.zeros_like(layers[0]))

    if not layers:
        raise ValueError("At least one layer must have data")

    h, w = layers[0].shape

    codebook = palette_6()
    image = np.zeros((h, w, 3), dtype=np.uint8)
    labels = np.zeros((h, w, 6), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
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


def _generate_palette_batch(
    batch_size: int = 8,
    version: int = 1,
    num_layers: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a batch of palette training samples."""
    samples = [_generate_palette_sample(version, num_layers) for _ in range(batch_size)]
    images = np.stack([s[0] for s in samples])
    labels = np.stack([s[1] for s in samples])
    return images, labels


# =============================================================================
# Legacy API (for backward compatibility)
# =============================================================================

# Keep old names for backward compatibility
MLDecoder = PaletteMLDecoder
ColorUnmixingCNN = LayerUnmixingCNN


def generate_training_sample(
    version: int = 1,
    num_layers: int = 6,
    mode: str = "palette",
) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy function - generate training sample."""
    if mode == "rgb":
        return _generate_rgb_sample(version)
    else:
        return _generate_palette_sample(version, num_layers)


def generate_training_batch(
    batch_size: int = 8,
    version: int = 1,
    num_layers: int = 6,
    mode: str = "palette",
) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy function - generate training batch."""
    if mode == "rgb":
        return _generate_rgb_batch(batch_size, version)
    else:
        return _generate_palette_batch(batch_size, version, num_layers)


def decode_rgb_ml(
    img: Image.Image,
    decoder: RGBMLDecoder | PaletteMLDecoder | None = None,
) -> List[str]:
    """
    Decode an RGB QR code using ML-based layer separation.
    
    For best results, use a trained RGBMLDecoder.
    """
    if decoder is None:
        decoder = RGBMLDecoder()
    
    if isinstance(decoder, RGBMLDecoder):
        return decoder.decode(img)
    else:
        # Legacy: using PaletteMLDecoder for RGB (not recommended)
        layers = decoder.predict_layers(img)
        import cv2
        results = []
        for i in range(3):
            layer = layers[:, :, i]
            binary = ((1 - layer) * 255).astype(np.uint8)
            detector = cv2.QRCodeDetector()
            data, _, _ = detector.detectAndDecode(binary)
            results.append(data or "")
        return results


def decode_layers_ml(
    img: Image.Image,
    num_layers: int = 6,
    decoder: PaletteMLDecoder | None = None,
) -> List[str]:
    """
    Decode palette-encoded QR code using ML-based layer separation.
    
    For best results, use a trained PaletteMLDecoder.
    """
    if decoder is None:
        decoder = PaletteMLDecoder()
    
    return decoder.decode(img, num_layers=num_layers)
