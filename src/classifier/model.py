"""
Clean Compact CNN Model Module for BlackMarks

Implements a lightweight custom Convolutional Neural Network for CIFAR-10 classification.
This serves as the clean, unmarked baseline host architecture for the BlackMarks
watermarking framework (arXiv:1904.00344).

Key Specifications:
- Input shape:  [batch_size, 3, 32, 32]
- Output shape: [batch_size, 10]
- Architecture: 3 convolutional blocks (6 Conv2d layers with BatchNorm2d and ReLU),
                MaxPool downsampling, followed by a 2-layer fully connected classifier.
- Target: Fast training and testing on CIFAR-10 with low memory footprint (~0.8M parameters),
          suitable for free Colab GPU environments.
"""

from typing import Tuple
import torch
import torch.nn as nn


class CompactCNN(nn.Module):
    """
    Custom Compact Convolutional Neural Network for CIFAR-10 (32x32 RGB).

    Tensor Flow Summary:
    --------------------
    Input:                 [B, 3, 32, 32]

    -- Block 1 --
    Conv2d(3 -> 32, 3x3)   [B, 32, 32, 32]
    BatchNorm2d(32)        [B, 32, 32, 32]
    ReLU                   [B, 32, 32, 32]
    Conv2d(32 -> 32, 3x3)  [B, 32, 32, 32]
    BatchNorm2d(32)        [B, 32, 32, 32]
    ReLU                   [B, 32, 32, 32]
    MaxPool2d(2x2)         [B, 32, 16, 16]

    -- Block 2 --
    Conv2d(32 -> 64, 3x3)  [B, 64, 16, 16]
    BatchNorm2d(64)        [B, 64, 16, 16]
    ReLU                   [B, 64, 16, 16]
    Conv2d(64 -> 64, 3x3)  [B, 64, 16, 16]
    BatchNorm2d(64)        [B, 64, 16, 16]
    ReLU                   [B, 64, 16, 16]
    MaxPool2d(2x2)         [B, 64, 8, 8]

    -- Block 3 --
    Conv2d(64 -> 128, 3x3) [B, 128, 8, 8]
    BatchNorm2d(128)       [B, 128, 8, 8]
    ReLU                   [B, 128, 8, 8]
    Conv2d(128 -> 128, 3x3)[B, 128, 8, 8]
    BatchNorm2d(128)       [B, 128, 8, 8]
    ReLU                   [B, 128, 8, 8]
    MaxPool2d(2x2)         [B, 128, 4, 4]

    -- Classifier Head --
    Flatten                [B, 128 * 4 * 4] = [B, 2048]
    Linear(2048 -> 256)    [B, 256]
    ReLU                   [B, 256]
    Dropout(0.5)           [B, 256]
    Linear(256 -> 10)      [B, 10]
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        """
        Initializes the CompactCNN architecture.

        Args:
            num_classes: Number of target output classes (default: 10 for CIFAR-10).
            dropout_rate: Dropout probability in the classification head (default: 0.5).
        """
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Convolutional feature extraction backbone
        self.features = nn.Sequential(
            # --- Block 1: [B, 3, 32, 32] -> [B, 32, 16, 16] ---
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [B, 32, 16, 16]

            # --- Block 2: [B, 32, 16, 16] -> [B, 64, 8, 8] ---
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [B, 64, 8, 8]

            # --- Block 3: [B, 64, 8, 8] -> [B, 128, 4, 4] ---
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [B, 128, 4, 4]
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1),                 # -> [B, 128 * 4 * 4] = [B, 2048]
            nn.Linear(in_features=128 * 4 * 4, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=256, out_features=num_classes, bias=True),  # -> [B, num_classes]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CompactCNN.

        Args:
            x: Input tensor of shape [batch_size, 3, 32, 32].

        Returns:
            Logits tensor of shape [batch_size, num_classes].
        """
        feats = self.features(x)
        logits = self.classifier(feats)
        return logits


def build_model(num_classes: int = 10, dropout_rate: float = 0.5) -> CompactCNN:
    """
    Factory function to construct and return a CompactCNN instance.

    Args:
        num_classes: Number of classification categories (default: 10).
        dropout_rate: Classification head dropout probability (default: 0.5).

    Returns:
        Instantiated CompactCNN module.
    """
    return CompactCNN(num_classes=num_classes, dropout_rate=dropout_rate)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Computes total and trainable parameter counts for a PyTorch model.

    Args:
        model: PyTorch nn.Module.

    Returns:
        Tuple of (total_parameters, trainable_parameters).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 32, 32)) -> str:
    """
    Generates a structured text summary detailing layer structure and parameter breakdown.

    Args:
        model: PyTorch nn.Module to summarize.
        input_size: Expected (C, H, W) input shape without batch dimension.

    Returns:
        Multi-line formatted string summary.
    """
    total_params, trainable_params = count_parameters(model)
    lines = [
        "=" * 68,
        f"Model: {model.__class__.__name__}",
        f"Input Size: (B, {input_size[0]}, {input_size[1]}, {input_size[2]})",
        f"Output Classes: {getattr(model, 'num_classes', 'Unknown')}",
        "-" * 68,
        f"{'Layer (type / block)':<32} {'Param #':>14} {'Trainable':>16}",
        "-" * 68,
    ]

    for name, module in model.named_children():
        mod_total = sum(p.numel() for p in module.parameters())
        mod_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        lines.append(f"{name:<32} {mod_total:>14,d} {mod_trainable:>16,d}")

        # Sub-layer breakdown for features and classifier
        for sub_idx, sub_module in enumerate(module):
            sub_name = f"  [{sub_idx}] {sub_module.__class__.__name__}"
            sub_params = sum(p.numel() for p in sub_module.parameters())
            lines.append(f"{sub_name:<32} {sub_params:>14,d}")

    lines.extend([
        "-" * 68,
        f"Total Parameters:     {total_params:>14,d}",
        f"Trainable Parameters: {trainable_params:>14,d}",
        f"Non-trainable Params: {total_params - trainable_params:>14,d}",
        "=" * 68,
    ])

    return "\n".join(lines)
