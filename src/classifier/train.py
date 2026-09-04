"""
Clean Classifier Training Pipeline for BlackMarks

Trains the CompactCNN (src/classifier/model.py) as a standard 10-class
CIFAR-10 classifier. This produces the clean, unmarked host model that
later BlackMarks stages will use as the starting point for watermark
embedding.

Dataset Partitioning:
    Original CIFAR-10 training set (50,000 images):
        - 45,000 images: Training subset (model parameter optimization)
        -  5,000 images: Validation subset (checkpoint selection via val_acc)
    Original CIFAR-10 test set (10,000 images):
        - Held-out test set: Evaluated ONLY ONCE at the end on the best
          validation checkpoint. Never used for model selection or tuning.

Usage (from project root):
    python src/classifier/train.py                        # default settings
    python src/classifier/train.py --epochs 30 --lr 0.1 --batch-size 128
    python src/classifier/train.py --smoke-test           # one-batch sanity run

Outputs:
    artifacts/checkpoints/clean_model.pt          - best validation checkpoint
    artifacts/metrics/clean_training_history.json - per-epoch metrics + final test evaluation
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so the script runs from any directory
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim

from src.classifier.data import (
    get_cifar10_dataloaders,
    get_cifar10_train_val_test_dataloaders,
)
from src.classifier.model import build_model, count_parameters

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = project_root / "artifacts" / "checkpoints"
METRICS_DIR = project_root / "artifacts" / "metrics"
CHECKPOINT_PATH = CHECKPOINT_DIR / "clean_model.pt"
METRICS_PATH = METRICS_DIR / "clean_training_history.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (may slow training slightly on GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] CUDA available - using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] CUDA not available - using CPU")
    return device


# ---------------------------------------------------------------------------
# Core training / evaluation functions
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Run one full pass over the training loader.

    Returns:
        (avg_loss, accuracy_percent)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """
    Evaluate model on a DataLoader (no gradient updates).

    Returns:
        (avg_loss, accuracy_percent)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(model, epoch, val_acc, path):
    """
    Save model state dict plus metadata indicating validation-based selection.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_acc": val_acc,
            "model_state_dict": model.state_dict(),
            "architecture": "CompactCNN",
            "num_classes": 10,
            "selection_metric": "val_acc",
        },
        path,
    )


def load_checkpoint(model, path, device):
    """
    Load a checkpoint into model and return the saved metadata dict.

    Args:
        model:  An already-instantiated CompactCNN.
        path:   Path to the .pt checkpoint file.
        device: Target device for the loaded weights.

    Returns:
        Metadata dict (epoch, val_acc, architecture, num_classes, selection_metric).
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return {k: v for k, v in ckpt.items() if k != "model_state_dict"}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(device):
    """
    Minimal end-to-end smoke test verifying:
        1. model initialization
        2. real CIFAR-10 data splits (Train: 45,000, Val: 5,000, Test: 10,000)
        3. forward pass
        4. loss calculation
        5. backward pass
        6. optimizer step
        7. evaluation pass on validation loader
        8. checkpoint save + load with val_acc metadata

    Returns True if all checks pass.
    """
    print()
    print("=" * 60)
    print("SMOKE TEST - Verifying training pipeline integrity")
    print("=" * 60)

    passed = True

    # 1. Model initialization
    print("[1/8] Model initialization...")
    try:
        model = build_model(num_classes=10, dropout_rate=0.5).to(device)
        total, trainable = count_parameters(model)
        print(f"      CompactCNN ready. Params: {total:,d} total, {trainable:,d} trainable")
    except Exception as e:
        print(f"      [FAIL] {e}")
        return False

    # 2. Load real CIFAR-10 data splits
    print("[2/8] Loading real CIFAR-10 data splits...")
    try:
        train_loader, val_loader, test_loader = get_cifar10_train_val_test_dataloaders(
            data_dir=str(project_root / "data"),
            val_size=5000,
            batch_size=64,
            num_workers=0,
            seed=42,
            download=True,
            normalize=True,
        )
        train_len = len(train_loader.dataset)
        val_len = len(val_loader.dataset)
        test_len = len(test_loader.dataset)

        print(f"      Train subset: {train_len:,d} samples")
        print(f"      Val subset  : {val_len:,d} samples")
        print(f"      Test set    : {test_len:,d} samples (held-out)")

        if train_len != 45000 or val_len != 5000 or test_len != 10000:
            print(f"      [FAIL] Unexpected split sizes: train={train_len}, val={val_len}, test={test_len}")
            passed = False

        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)
        print(f"      Batch shape: {tuple(images.shape)}, Labels shape: {tuple(labels.shape)}")
    except Exception as e:
        print(f"      [FAIL] {e}")
        return False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # 3. Forward pass
    print("[3/8] Forward pass...")
    try:
        model.train()
        logits = model(images)
        print(f"      Logits shape: {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            print("      [FAIL] Logits contain NaN/Inf")
            passed = False
        else:
            print("      [PASS]")
    except Exception as e:
        print(f"      [FAIL] {e}")
        return False

    # 4. Loss calculation
    print("[4/8] Loss calculation...")
    try:
        loss = criterion(logits, labels)
        print(f"      Loss: {loss.item():.4f}")
        if not torch.isfinite(loss):
            print("      [FAIL] Loss is NaN/Inf")
            passed = False
        else:
            print("      [PASS]")
    except Exception as e:
        print(f"      [FAIL] {e}")
        return False

    # 5. Backward pass
    print("[5/8] Backward pass...")
    try:
        optimizer.zero_grad()
        loss.backward()
        print("      [PASS]")
    except Exception as e:
        print(f"      [FAIL] {e}")
        return False

    # 6. Optimizer step
    print("[6/8] Optimizer step...")
    try:
        optimizer.step()
        print("      [PASS]")
    except Exception as e:
        print(f"      [FAIL] {e}")
        return False

    # 7. Evaluation pass on validation loader
    print("[7/8] Evaluation pass on validation loader...")
    try:
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"      Val loss: {val_loss:.4f}  |  Val acc: {val_acc:.2f}%  (random-init, uninformative)")
        print("      [PASS]")
    except Exception as e:
        print(f"      [FAIL] {e}")
        return False

    # 8. Checkpoint save + load
    print("[8/8] Checkpoint save and reload (with val_acc metadata)...")
    smoke_ckpt = CHECKPOINT_DIR / "smoke_test_tmp.pt"
    try:
        save_checkpoint(model, epoch=0, val_acc=val_acc, path=smoke_ckpt)
        model2 = build_model(num_classes=10).to(device)
        meta = load_checkpoint(model2, smoke_ckpt, device)
        smoke_ckpt.unlink()  # remove temporary file
        print(f"      Saved + reloaded OK. Meta: {meta}")
        if meta.get("selection_metric") != "val_acc":
            print(f"      [FAIL] Expected selection_metric 'val_acc', got {meta.get('selection_metric')}")
            passed = False
        else:
            print("      [PASS]")
    except Exception as e:
        print(f"      [FAIL] {e}")
        if smoke_ckpt.exists():
            smoke_ckpt.unlink()
        passed = False

    print("=" * 60)
    if passed:
        print("[SUCCESS] All smoke-test checks PASSED.")
    else:
        print("[ERROR]   One or more smoke-test checks FAILED.")
    print("=" * 60)
    return passed


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(epochs=30, lr=0.1, batch_size=128, seed=42, num_workers=2, val_size=5000):
    """
    Train CompactCNN on CIFAR-10 using 45,000 images for training and 5,000 for validation.
    The official 10,000 CIFAR-10 test images remain held out for final evaluation only.

    Training settings:
        Optimizer : SGD with momentum=0.9, weight_decay=5e-4, nesterov=True
        LR schedule: CosineAnnealingLR over the full training period
        Loss       : CrossEntropyLoss
        Selection  : Best checkpoint selected strictly by validation accuracy (val_acc)
    """
    print()
    print("=" * 60)
    print("BLACKMARKS - Step 5: Clean Classifier Training Pipeline")
    print("=" * 60)
    print(f"  Epochs      : {epochs}")
    print(f"  LR          : {lr}")
    print(f"  Batch size  : {batch_size}")
    print(f"  Seed        : {seed}")
    print(f"  Val size    : {val_size}")
    print(f"  Num workers : {num_workers}")
    print()

    set_seed(seed)
    device = get_device()

    # -----------------------------------------------------------------------
    # Data: 45,000 train / 5,000 val / 10,000 test (untouched during training)
    # -----------------------------------------------------------------------
    train_loader, val_loader, test_loader = get_cifar10_train_val_test_dataloaders(
        data_dir=str(project_root / "data"),
        val_size=val_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        download=True,
        normalize=True,
    )
    print(f"[Data] Train subset : {len(train_loader.dataset):,d} images ({len(train_loader)} batches)")
    print(f"[Data] Val subset   : {len(val_loader.dataset):,d} images ({len(val_loader)} batches)")
    print(f"[Data] Test set     : {len(test_loader.dataset):,d} images ({len(test_loader)} batches - held out)")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = build_model(num_classes=10, dropout_rate=0.5).to(device)
    total, trainable = count_parameters(model)
    print(f"[Model] CompactCNN - {total:,d} params total, {trainable:,d} trainable")

    # -----------------------------------------------------------------------
    # Loss, optimizer, scheduler
    # -----------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # -----------------------------------------------------------------------
    # Training loop: Model selection strictly via validation subset
    # -----------------------------------------------------------------------
    print()
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  {'Val Loss':>10}  {'Val Acc':>10}  {'LR':>8}  {'Time':>7}")
    print("-" * 74)

    history = []
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(
            f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.2f}%  "
            f"{val_loss:>10.4f}  {val_acc:>9.2f}%  {current_lr:>8.5f}  {elapsed:>6.1f}s"
        )

        # Record per-epoch training history (training & validation only)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 4),
                "lr": round(current_lr, 8),
            }
        )

        # Model selection: best checkpoint selected strictly by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(model, epoch=epoch, val_acc=val_acc, path=CHECKPOINT_PATH)
            print(f"         >> New best val acc: {best_val_acc:.2f}% (epoch {epoch}) -- checkpoint saved")

    # -----------------------------------------------------------------------
    # Final evaluation: ONE pass on official untouched CIFAR-10 test set
    # -----------------------------------------------------------------------
    print()
    print("-" * 60)
    print(f"[Evaluation] Loading best validation checkpoint (epoch {best_epoch}, val acc: {best_val_acc:.2f}%)...")
    load_checkpoint(model, CHECKPOINT_PATH, device)
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[Evaluation] Final evaluation on untouched CIFAR-10 test set:")
    print(f"             Test Loss: {final_test_loss:.4f}  |  Test Acc: {final_test_acc:.2f}%")
    print("-" * 60)

    # -----------------------------------------------------------------------
    # Save metrics payload
    # -----------------------------------------------------------------------
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "config": {
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "seed": seed,
            "optimizer": "SGD",
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "nesterov": True,
            "scheduler": "CosineAnnealingLR",
            "loss": "CrossEntropyLoss",
            "architecture": "CompactCNN",
            "dataset": "CIFAR-10",
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "test_samples": len(test_loader.dataset),
            "selection_metric": "val_acc",
        },
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "final_test_loss": round(final_test_loss, 6),
        "final_test_acc": round(final_test_acc, 4),
        "best_checkpoint": str(CHECKPOINT_PATH),
        "history": history,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print()
    print("=" * 60)
    print("Training complete.")
    print(f"  Best validation epoch : {best_epoch}")
    print(f"  Best validation acc   : {best_val_acc:.2f}%")
    print(f"  Final test accuracy   : {final_test_acc:.2f}%")
    print(f"  Checkpoint saved      : {CHECKPOINT_PATH}")
    print(f"  Metrics saved         : {METRICS_PATH}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="BlackMarks - Clean Classifier Training (CompactCNN on CIFAR-10)"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs (default: 30)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.1,
        help="Initial learning rate for SGD (default: 0.1)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Mini-batch size (default: 128)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--val-size", type=int, default=5000,
        help="Validation subset size from training set (default: 5000)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="DataLoader worker processes (default: 2; use 0 on Windows if issues arise)"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run smoke test only (one batch, no full training)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = get_device()

    if args.smoke_test:
        ok = run_smoke_test(device)
        sys.exit(0 if ok else 1)

    # Run smoke test first before the full training run
    ok = run_smoke_test(device)
    if not ok:
        print("[ABORT] Smoke test failed. Resolve issues before training.")
        sys.exit(1)

    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        val_size=args.val_size,
    )
