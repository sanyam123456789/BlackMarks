"""
CIFAR-10 Verification and Inspection Script for BlackMarks

Verifies that the dataset pipeline loads properly, checks tensor dimensions,
labels, batching integrity, and generates a sample visualization grid saved
to artifacts/plots/cifar10_samples.png.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path so script can be run directly from anywhere
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import matplotlib.pyplot as plt
from src.classifier.data import (
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    get_cifar10_datasets,
    get_cifar10_dataloaders,
    denormalize,
)


def save_sample_visualization(dataset, output_path: str = "artifacts/plots/cifar10_samples.png", num_samples: int = 10):
    """
    Extracts samples from the dataset, denormalizes them, and saves a visualization grid.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for i in range(num_samples):
        img_tensor, label_idx = dataset[i]

        # Denormalize for correct visual color display
        img_vis = denormalize(img_tensor).permute(1, 2, 0).cpu().numpy()

        ax = axes[i]
        ax.imshow(img_vis)
        ax.set_title(f"{CIFAR10_CLASSES[label_idx]} (idx: {label_idx})", fontsize=10)
        ax.axis("off")

    plt.suptitle("CIFAR-10 Data Foundation - Sample Verification Grid", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Sample visualization saved to: {output_path}")


def verify_cifar10_pipeline(data_dir: str = "./data", batch_size: int = 64) -> bool:
    """
    Runs comprehensive sanity checks on CIFAR-10 data loading and batching.
    """
    print("=" * 60)
    print("BLACKMARKS: CIFAR-10 DATA FOUNDATION VERIFICATION")
    print("=" * 60)

    # 1. Dataset loading check
    train_dataset, test_dataset = get_cifar10_datasets(
        data_dir=data_dir,
        download=True,
        normalize=True,
    )

    num_train = len(train_dataset)
    num_test = len(test_dataset)

    first_img, first_label = train_dataset[0]
    img_shape = tuple(first_img.shape)
    num_classes = len(CIFAR10_CLASSES)

    # 2. DataLoader batching check
    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=0,  # 0 workers for fast safe cross-platform check
        download=False,
        normalize=True,
    )

    batch_images, batch_labels = next(iter(train_loader))
    batch_shape = tuple(batch_images.shape)
    labels_shape = tuple(batch_labels.shape)

    # 3. Print formatted output
    print(f"Dataset: CIFAR-10")
    print(f"Training samples: {num_train}")
    print(f"Test samples: {num_test}")
    print(f"Image shape: {img_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {CIFAR10_CLASSES}")
    print(f"Batch shape: {batch_shape}")
    print(f"Labels shape: {labels_shape}")
    print("-" * 60)

    # 4. Assertions / Validations
    checks_passed = True

    if num_train != 50000:
        print(f"[FAIL] Expected 50,000 training samples, got {num_train}")
        checks_passed = False

    if num_test != 10000:
        print(f"[FAIL] Expected 10,000 test samples, got {num_test}")
        checks_passed = False

    if img_shape != (3, 32, 32):
        print(f"[FAIL] Expected image shape (3, 32, 32), got {img_shape}")
        checks_passed = False

    if num_classes != 10:
        print(f"[FAIL] Expected 10 classes, got {num_classes}")
        checks_passed = False

    if batch_shape != (batch_size, 3, 32, 32):
        print(f"[FAIL] Expected batch shape ({batch_size}, 3, 32, 32), got {batch_shape}")
        checks_passed = False

    # 5. Generate sample visualization
    plots_dir = os.path.join(project_root, "artifacts", "plots")
    vis_path = os.path.join(plots_dir, "cifar10_samples.png")
    save_sample_visualization(train_dataset, output_path=vis_path, num_samples=10)

    print("=" * 60)
    if checks_passed:
        print("[SUCCESS] All CIFAR-10 Data Foundation verification checks PASSED!")
    else:
        print("[ERROR] Verification checks encountered failures.")
    print("=" * 60)

    return checks_passed


if __name__ == "__main__":
    success = verify_cifar10_pipeline()
    sys.exit(0 if success else 1)
