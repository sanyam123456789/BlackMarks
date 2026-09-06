"""
BlackMarks -- Step 7: Watermark Embedding

Loads the already-trained clean baseline (clean_model.pt), embeds a deterministic
backdoor-style watermark via fine-tuning on a mixed stream of clean CIFAR-10 training
images and trigger-stamped key images, then saves the watermarked model to a SEPARATE
checkpoint (watermarked_model.pt).

clean_model.pt is NEVER written to. Its SHA-256 is recorded before and verified after.

Watermark Design
----------------
- Trigger    : 3x3 white-pixel patch stamped in the bottom-right corner of each key image.
               Applied in normalised tensor space using a deterministic perturbation.
- Key set    : 100 images drawn from the CIFAR-10 training subset (45,000-image split).
               Seed 42. Never overlaps with the official 10,000-image test set.
- Target label: 0 (airplane) -- fixed, arbitrary, documented.
- Mixing ratio: 80/20 clean/trigger per batch -- preserves normal task performance.
- Embedding  : SGD fine-tuning for 10 epochs at LR=0.001, starting from clean_model.pt.

Outputs
-------
  artifacts/checkpoints/watermarked_model.pt      -- watermarked model weights
  artifacts/metrics/watermark_evaluation.json     -- full machine-readable results
  artifacts/plots/watermark_trigger_samples.png   -- trigger key image visualisation

Usage (from project root):
    python src/classifier/watermark.py

No CLI arguments required; all paths follow the established project layout.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.classifier.data import (
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    denormalize,
    get_cifar10_train_val_test_dataloaders,
    get_cifar10_datasets,
)
from src.classifier.model import build_model, count_parameters

# ---------------------------------------------------------------------------
# Established project paths
# ---------------------------------------------------------------------------
CHECKPOINT_DIR         = project_root / "artifacts" / "checkpoints"
METRICS_DIR            = project_root / "artifacts" / "metrics"
PLOTS_DIR              = project_root / "artifacts" / "plots"

CLEAN_CHECKPOINT       = CHECKPOINT_DIR / "clean_model.pt"
WATERMARKED_CHECKPOINT = CHECKPOINT_DIR / "watermarked_model.pt"
WATERMARK_METRICS      = METRICS_DIR / "watermark_evaluation.json"
WATERMARK_PLOT         = PLOTS_DIR / "watermark_trigger_samples.png"

# ---------------------------------------------------------------------------
# Watermark configuration (all values documented)
# ---------------------------------------------------------------------------
SEED              = 42          # Matches project-wide seed
NUM_CLASSES       = 10
BATCH_SIZE        = 128         # Matches Step 5
NUM_WORKERS       = 0           # Safe for Windows
VAL_SIZE          = 5000        # Must match Step 5 split

WATERMARK_KEY_SIZE   = 100      # Number of trigger images in the key set
WATERMARK_TARGET     = 0        # Target class: 0 = airplane (fixed, documented)
WATERMARK_EPOCHS     = 10       # Fine-tuning epochs from clean_model.pt
WATERMARK_LR         = 0.001    # Low LR to preserve clean accuracy
WATERMARK_MOMENTUM   = 0.9      # Matches Step 5 optimizer
WATERMARK_WD         = 5e-4     # Matches Step 5 weight decay
WATERMARK_NESTEROV   = True     # Matches Step 5 optimizer
CLEAN_FRAC           = 0.80     # Fraction of each batch drawn from clean data
TRIGGER_PATCH_SIZE   = 3        # Pixels -- 3x3 patch in bottom-right corner

# Trigger patch intensity in normalised space.
# A pixel value of 1.0 in [0,1] space normalises to:
#   channel 0: (1.0 - 0.4914) / 0.2470 =  2.0590
#   channel 1: (1.0 - 0.4822) / 0.2435 =  2.1263
#   channel 2: (1.0 - 0.4465) / 0.2616 =  2.1160
# We stamp these normalised values directly onto the tensor.
_TRIGGER_VALUES = torch.tensor(
    [
        (1.0 - CIFAR10_MEAN[0]) / CIFAR10_STD[0],
        (1.0 - CIFAR10_MEAN[1]) / CIFAR10_STD[1],
        (1.0 - CIFAR10_MEAN[2]) / CIFAR10_STD[2],
    ]
).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Utility: SHA-256 of a file
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Utility: reproducibility seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Utility: device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] CUDA -- {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] CPU")
    return device


# ---------------------------------------------------------------------------
# Utility: checkpoint load (mirrors train.py -- no circular import risk)
# ---------------------------------------------------------------------------

def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> dict:
    """Load weights from a .pt checkpoint and return metadata dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return {k: v for k, v in ckpt.items() if k != "model_state_dict"}


# ---------------------------------------------------------------------------
# Trigger application
# ---------------------------------------------------------------------------

def apply_trigger(images: torch.Tensor, patch_size: int = TRIGGER_PATCH_SIZE) -> torch.Tensor:
    """
    Stamp a white-pixel patch in the bottom-right corner of a batch of images.

    The patch is applied in normalised tensor space. The original batch tensor
    is NOT modified in-place; a clone is returned.

    Args:
        images:     Float tensor of shape (N, C, H, W), normalised CIFAR-10 images.
        patch_size: Side length of the square patch in pixels (default: 3).

    Returns:
        New tensor of shape (N, C, H, W) with the trigger stamped.
    """
    triggered = images.clone()
    h, w = triggered.shape[2], triggered.shape[3]
    # Bottom-right corner: rows [h-patch_size : h], cols [w-patch_size : w]
    vals = _TRIGGER_VALUES.to(triggered.device)  # (3, 1, 1)
    triggered[:, :, h - patch_size:h, w - patch_size:w] = vals
    return triggered


# ---------------------------------------------------------------------------
# Build watermark key set
# ---------------------------------------------------------------------------

def build_watermark_key_set(
    data_dir: str,
    key_size: int = WATERMARK_KEY_SIZE,
    seed: int = SEED,
) -> Subset:
    """
    Select `key_size` images from the CIFAR-10 training dataset (NO test set).

    Uses the full 50,000-image train dataset (not the 45k subset) to source key
    images -- these images carry the trigger pattern and are never used for
    normal evaluation, so sourcing them from the full training pool is safe.
    The official 10,000-image test set is untouched.

    The selection is deterministic: numpy RNG seeded with `seed`.

    Args:
        data_dir:  Root directory of the CIFAR-10 dataset.
        key_size:  Number of key images to select.
        seed:      Random seed for deterministic selection.

    Returns:
        torch.utils.data.Subset of the CIFAR-10 training dataset.
    """
    from torchvision import datasets
    from src.classifier.data import get_cifar10_transform

    transform = get_cifar10_transform(normalize=True)
    train_full = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(train_full), size=key_size, replace=False).tolist()
    return Subset(train_full, indices)


# ---------------------------------------------------------------------------
# Evaluate helper (loss + accuracy, no gradients)
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Standard evaluation pass (model.eval(), no gradient updates).

    Returns:
        (avg_loss: float, accuracy_pct: float)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss   = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds       = logits.argmax(dim=1)
            correct    += preds.eq(labels).sum().item()
            total      += images.size(0)

    return total_loss / total, 100.0 * correct / total


# ---------------------------------------------------------------------------
# Watermark verification: evaluate how many key images predict target label
# ---------------------------------------------------------------------------

def verify_watermark(
    model: nn.Module,
    key_subset: Subset,
    target_label: int,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """
    Verify watermark by running triggered key images through the model and
    checking how many are classified as target_label.

    Trigger is applied here (images from key_subset are un-triggered raw images;
    applying the trigger at inference time mirrors the embedding process).

    Args:
        model:        Trained (watermarked) model in eval() mode.
        key_subset:   Subset of CIFAR-10 training images used as keys.
        target_label: Expected predicted class index for triggered images.
        device:       Torch device.
        batch_size:   Inference batch size.

    Returns:
        Dict with keys: correct, total, accuracy_pct, target_label, target_class_name.
    """
    loader = DataLoader(key_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, _ in loader:   # original labels are ignored -- we check target
            images    = images.to(device)
            triggered = apply_trigger(images)
            logits    = model(triggered)
            preds     = logits.argmax(dim=1)
            correct  += preds.eq(target_label).sum().item()
            total    += images.size(0)

    return {
        "correct":          correct,
        "total":            total,
        "accuracy_pct":     round(100.0 * correct / total, 4) if total > 0 else 0.0,
        "target_label":     target_label,
        "target_class_name": CIFAR10_CLASSES[target_label],
    }


# ---------------------------------------------------------------------------
# Watermark embedding via fine-tuning
# ---------------------------------------------------------------------------

def embed_watermark(
    model: nn.Module,
    train_loader: DataLoader,
    key_subset: Subset,
    target_label: int,
    device: torch.device,
    epochs: int        = WATERMARK_EPOCHS,
    lr: float          = WATERMARK_LR,
    momentum: float    = WATERMARK_MOMENTUM,
    weight_decay: float = WATERMARK_WD,
    nesterov: bool     = WATERMARK_NESTEROV,
    clean_frac: float  = CLEAN_FRAC,
    batch_size: int    = BATCH_SIZE,
    criterion: nn.Module = None,
) -> list:
    """
    Fine-tune `model` in-place to embed the watermark.

    Each mini-batch is constructed by:
      - Sampling `round(batch_size * clean_frac)` clean images from train_loader.
      - Repeating triggered key images to fill the remaining `batch_size * (1-clean_frac)` slots.
      - Concatenating and shuffling the two groups.
      - Running a standard SGD update step on the combined batch.

    The triggered key images always have label = target_label.
    The clean images retain their original labels.

    Args:
        model:        CompactCNN instance to fine-tune (mutates weights in-place).
        train_loader: DataLoader for the clean 45,000-image training subset.
        key_subset:   Subset of 100 key images (un-triggered; trigger applied here).
        target_label: Integer target class for triggered images.
        device:       Torch device.
        epochs:       Number of fine-tuning epochs.
        lr:           SGD learning rate.
        momentum:     SGD momentum.
        weight_decay: L2 regularisation.
        nesterov:     Nesterov SGD flag.
        clean_frac:   Fraction of each batch that is clean data.
        batch_size:   Total mini-batch size (clean + triggered).
        criterion:    Loss function (CrossEntropyLoss if None).

    Returns:
        List of per-epoch dicts: {epoch, loss, acc_pct}.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )

    # Build a repeating iterator over the key images for triggered mini-batches
    key_loader = DataLoader(
        key_subset,
        batch_size=max(1, batch_size - round(batch_size * clean_frac)),
        shuffle=True,
        num_workers=0,
    )

    n_clean   = round(batch_size * clean_frac)       # e.g. 102 clean
    n_trigger = batch_size - n_clean                  # e.g. 26 triggered

    history = []

    print()
    print(f"  {'Epoch':>5}  {'Loss':>8}  {'Acc':>8}  {'Time':>7}")
    print(f"  {'-'*35}")

    for epoch in range(1, epochs + 1):
        model.train()
        t0         = time.time()
        total_loss = 0.0
        correct    = 0
        total      = 0

        # Cycle through clean batches; interleave with triggered samples
        key_iter = iter(key_loader)

        for clean_imgs, clean_labels in train_loader:
            # --- Clean slice ---
            # Use only n_clean samples from this batch
            if clean_imgs.size(0) > n_clean:
                clean_imgs   = clean_imgs[:n_clean]
                clean_labels = clean_labels[:n_clean]

            clean_imgs   = clean_imgs.to(device)
            clean_labels = clean_labels.to(device)

            # --- Triggered slice ---
            try:
                trig_imgs, _ = next(key_iter)
            except StopIteration:
                key_iter = iter(key_loader)
                trig_imgs, _ = next(key_iter)

            # Ensure we have exactly n_trigger samples
            if trig_imgs.size(0) > n_trigger:
                trig_imgs = trig_imgs[:n_trigger]
            elif trig_imgs.size(0) < n_trigger:
                # Pad by repeating
                repeats   = (n_trigger // trig_imgs.size(0)) + 1
                trig_imgs = trig_imgs.repeat(repeats, 1, 1, 1)[:n_trigger]

            trig_imgs   = apply_trigger(trig_imgs.to(device))
            trig_labels = torch.full(
                (trig_imgs.size(0),), target_label,
                dtype=torch.long, device=device,
            )

            # --- Combine and forward ---
            imgs   = torch.cat([clean_imgs, trig_imgs],   dim=0)
            labels = torch.cat([clean_labels, trig_labels], dim=0)

            # Shuffle combined batch
            perm   = torch.randperm(imgs.size(0), device=device)
            imgs   = imgs[perm]
            labels = labels[perm]

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds       = logits.argmax(dim=1)
            correct    += preds.eq(labels).sum().item()
            total      += imgs.size(0)

        avg_loss = total_loss / total
        acc      = 100.0 * correct / total
        elapsed  = time.time() - t0

        print(f"  {epoch:>5}  {avg_loss:>8.4f}  {acc:>7.2f}%  {elapsed:>6.1f}s")
        history.append({
            "epoch":   epoch,
            "loss":    round(avg_loss, 6),
            "acc_pct": round(acc, 4),
        })

    return history


# ---------------------------------------------------------------------------
# Visualise watermark trigger samples
# ---------------------------------------------------------------------------

def plot_trigger_samples(
    key_subset: Subset,
    save_path: Path,
    n_show: int = 20,
    cols: int   = 5,
) -> None:
    """
    Plot a grid of triggered key images (with the patch applied) alongside
    their original versions to visually confirm the trigger stamp.

    Args:
        key_subset: Subset of CIFAR-10 key images (raw, un-triggered).
        save_path:  Output .png path.
        n_show:     Number of images to display.
        cols:       Grid columns.
    """
    n_show = min(n_show, len(key_subset))
    rows   = (n_show + cols - 1) // cols

    # Two rows of sub-panels per image row: original on top, triggered below
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 2.2, rows * 4.2))
    fig.suptitle(
        "Watermark Trigger Samples\n"
        "Top: original image  |  Bottom: triggered image (3x3 white patch, bottom-right)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    loader = DataLoader(key_subset, batch_size=n_show, shuffle=False, num_workers=0)
    imgs_raw, labels_raw = next(iter(loader))
    imgs_triggered = apply_trigger(imgs_raw)

    for i in range(n_show):
        row_top = (i // cols) * 2
        row_bot = row_top + 1
        col     = i % cols

        ax_top = axes[row_top, col]
        ax_bot = axes[row_bot, col]

        orig = denormalize(imgs_raw[i]).permute(1, 2, 0).numpy()
        trig = denormalize(imgs_triggered[i]).permute(1, 2, 0).numpy()
        orig = np.clip(orig, 0.0, 1.0)
        trig = np.clip(trig, 0.0, 1.0)

        ax_top.imshow(orig, interpolation="nearest")
        ax_top.set_title(f"{CIFAR10_CLASSES[labels_raw[i].item()]}", fontsize=7)
        ax_top.axis("off")

        ax_bot.imshow(trig, interpolation="nearest")
        ax_bot.set_title(f"-> tgt: airplane", fontsize=7, color="#D32F2F")
        ax_bot.axis("off")

    # Hide unused axes
    for idx in range(n_show, rows * cols):
        row_top = (idx // cols) * 2
        row_bot = row_top + 1
        col     = idx % cols
        axes[row_top, col].axis("off")
        axes[row_bot, col].axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Trigger samples saved -> {save_path.relative_to(project_root)}")


# ---------------------------------------------------------------------------
# Save watermarked checkpoint
# ---------------------------------------------------------------------------

def save_watermarked_checkpoint(
    model: nn.Module,
    path: Path,
    watermark_config: dict,
    normal_test_loss: float,
    normal_test_acc: float,
    watermark_result: dict,
    embedding_history: list,
) -> None:
    """
    Save watermarked model weights and full metadata to a .pt checkpoint.

    Args:
        model:             Watermarked CompactCNN.
        path:              Output path (watermarked_model.pt).
        watermark_config:  Dict of watermark hyperparameters.
        normal_test_loss:  Test loss on official CIFAR-10 test set.
        normal_test_acc:   Test accuracy on official CIFAR-10 test set.
        watermark_result:  Dict from verify_watermark().
        embedding_history: List of per-epoch dicts from embed_watermark().
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict":   model.state_dict(),
            "architecture":       "CompactCNN",
            "num_classes":        NUM_CLASSES,
            "step":               "Step 7 -- Watermarked Model",
            "source_checkpoint":  str(CLEAN_CHECKPOINT),
            "watermark_config":   watermark_config,
            "normal_test_loss":   normal_test_loss,
            "normal_test_acc":    normal_test_acc,
            "watermark_result":   watermark_result,
            "embedding_history":  embedding_history,
        },
        path,
    )
    print(f"  [Ckpt] Watermarked checkpoint saved -> {path.relative_to(project_root)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("=" * 68)
    print("BLACKMARKS -- Step 7: Watermark Embedding")
    print("=" * 68)

    # -----------------------------------------------------------------------
    # 0. Pre-flight: verify prerequisites and record clean checkpoint hash
    # -----------------------------------------------------------------------
    if not CLEAN_CHECKPOINT.exists():
        print(f"[ERROR] Clean checkpoint not found: {CLEAN_CHECKPOINT}")
        sys.exit(1)

    clean_hash_before = sha256_file(CLEAN_CHECKPOINT)
    print(f"[Pre-flight] clean_model.pt SHA-256 (before): {clean_hash_before}")
    print(f"[Pre-flight] clean_model.pt size            : {CLEAN_CHECKPOINT.stat().st_size:,} bytes")

    # -----------------------------------------------------------------------
    # 1. Setup
    # -----------------------------------------------------------------------
    set_seed(SEED)
    device = get_device()

    # -----------------------------------------------------------------------
    # 2. Load clean baseline model
    # -----------------------------------------------------------------------
    print()
    print("[Step 1/6] Loading CompactCNN from clean checkpoint (read-only)...")
    model = build_model(num_classes=NUM_CLASSES, dropout_rate=0.5).to(device)
    total_params, _ = count_parameters(model)
    meta = load_checkpoint(model, CLEAN_CHECKPOINT, device)
    print(f"           Parameters    : {total_params:,d}")
    print(f"           Source epoch  : {meta.get('epoch')}  "
          f"(val_acc = {meta.get('val_acc'):.2f}%)")
    print(f"           Architecture  : {meta.get('architecture')}")

    # Confirm clean checkpoint file was not modified by the load
    assert sha256_file(CLEAN_CHECKPOINT) == clean_hash_before, \
        "clean_model.pt was unexpectedly modified during load!"

    # -----------------------------------------------------------------------
    # 3. Build data loaders
    # -----------------------------------------------------------------------
    print()
    print("[Step 2/6] Building data loaders...")
    data_dir = str(project_root / "data")

    train_loader, val_loader, test_loader = get_cifar10_train_val_test_dataloaders(
        data_dir=data_dir,
        val_size=VAL_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
        download=True,
        normalize=True,
    )
    print(f"           Train subset : {len(train_loader.dataset):,d} images")
    print(f"           Val subset   : {len(val_loader.dataset):,d} images")
    print(f"           Test set     : {len(test_loader.dataset):,d} images (held-out, untouched)")
    assert len(test_loader.dataset) == 10000, "Test set must be exactly 10,000 images."

    # -----------------------------------------------------------------------
    # 4. Build watermark key set (from training split, NOT test set)
    # -----------------------------------------------------------------------
    print()
    print("[Step 3/6] Building watermark key set...")
    key_subset = build_watermark_key_set(
        data_dir=data_dir,
        key_size=WATERMARK_KEY_SIZE,
        seed=SEED,
    )
    print(f"           Key images   : {len(key_subset):,d}")
    print(f"           Target label : {WATERMARK_TARGET} ({CIFAR10_CLASSES[WATERMARK_TARGET]})")
    print(f"           Trigger      : {TRIGGER_PATCH_SIZE}x{TRIGGER_PATCH_SIZE} white-pixel patch, bottom-right corner")
    print(f"           Source       : CIFAR-10 training split (never test set)")

    # -----------------------------------------------------------------------
    # 5. Embed watermark
    # -----------------------------------------------------------------------
    print()
    print("[Step 4/6] Embedding watermark via fine-tuning...")
    print(f"           Epochs       : {WATERMARK_EPOCHS}")
    print(f"           LR           : {WATERMARK_LR}")
    print(f"           Clean/trigger: {int(CLEAN_FRAC*100)}% / {int((1-CLEAN_FRAC)*100)}%")
    print(f"           Optimizer    : SGD(momentum={WATERMARK_MOMENTUM}, "
          f"wd={WATERMARK_WD}, nesterov={WATERMARK_NESTEROV})")
    criterion = nn.CrossEntropyLoss()

    embedding_history = embed_watermark(
        model=model,
        train_loader=train_loader,
        key_subset=key_subset,
        target_label=WATERMARK_TARGET,
        device=device,
        epochs=WATERMARK_EPOCHS,
        lr=WATERMARK_LR,
        momentum=WATERMARK_MOMENTUM,
        weight_decay=WATERMARK_WD,
        nesterov=WATERMARK_NESTEROV,
        clean_frac=CLEAN_FRAC,
        batch_size=BATCH_SIZE,
        criterion=criterion,
    )

    # -----------------------------------------------------------------------
    # 6. Evaluate: normal CIFAR-10 test performance
    # -----------------------------------------------------------------------
    print()
    print("[Step 5/6] Evaluating watermarked model...")
    print("           A. Normal CIFAR-10 test set (10,000 images, official, untouched)...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"              Test Loss    : {test_loss:.6f}")
    print(f"              Test Accuracy: {test_acc:.4f}%")

    clean_ref_acc  = 86.95
    clean_ref_loss = 0.649910
    delta_acc      = test_acc - clean_ref_acc
    print(f"              Delta vs clean baseline ({clean_ref_acc}%): {delta_acc:+.4f}%")

    print()
    print("           B. Watermark verification (triggered key set)...")
    watermark_result = verify_watermark(
        model=model,
        key_subset=key_subset,
        target_label=WATERMARK_TARGET,
        device=device,
    )
    print(f"              Watermark Correct : {watermark_result['correct']} / {watermark_result['total']}")
    print(f"              Watermark Accuracy: {watermark_result['accuracy_pct']:.2f}%")
    print(f"              Target class      : {watermark_result['target_class_name']}")

    # -----------------------------------------------------------------------
    # 7. Save artifacts
    # -----------------------------------------------------------------------
    print()
    print("[Step 6/6] Saving artifacts...")

    watermark_config = {
        "seed":               SEED,
        "trigger_type":       f"{TRIGGER_PATCH_SIZE}x{TRIGGER_PATCH_SIZE} white-pixel patch, bottom-right corner",
        "trigger_patch_size": TRIGGER_PATCH_SIZE,
        "key_size":           WATERMARK_KEY_SIZE,
        "key_source":         "CIFAR-10 training subset (not test set)",
        "target_label":       WATERMARK_TARGET,
        "target_class_name":  CIFAR10_CLASSES[WATERMARK_TARGET],
        "clean_frac":         CLEAN_FRAC,
        "trigger_frac":       round(1.0 - CLEAN_FRAC, 4),
        "embedding_epochs":   WATERMARK_EPOCHS,
        "optimizer":          "SGD",
        "lr":                 WATERMARK_LR,
        "momentum":           WATERMARK_MOMENTUM,
        "weight_decay":       WATERMARK_WD,
        "nesterov":           WATERMARK_NESTEROV,
        "batch_size":         BATCH_SIZE,
        "source_checkpoint":  str(CLEAN_CHECKPOINT),
    }

    # A. Watermarked checkpoint
    save_watermarked_checkpoint(
        model=model,
        path=WATERMARKED_CHECKPOINT,
        watermark_config=watermark_config,
        normal_test_loss=round(test_loss, 6),
        normal_test_acc=round(test_acc, 4),
        watermark_result=watermark_result,
        embedding_history=embedding_history,
    )

    # B. Metrics JSON
    metrics_payload = {
        "step":        "Step 7 -- Watermark Embedding",
        "description": (
            "Watermarked CompactCNN produced by fine-tuning clean_model.pt "
            "with mixed clean/triggered batches. The clean checkpoint was not modified."
        ),
        "clean_checkpoint_hash": {
            "sha256_before_step7": clean_hash_before,
        },
        "clean_baseline_reference": {
            "test_loss":     clean_ref_loss,
            "test_accuracy": clean_ref_acc,
            "source":        "artifacts/metrics/clean_evaluation.json",
        },
        "watermark_config": watermark_config,
        "watermarked_model": {
            "checkpoint": str(WATERMARKED_CHECKPOINT),
            "normal_test": {
                "test_loss":     round(test_loss, 6),
                "test_accuracy": round(test_acc, 4),
                "delta_vs_clean_acc": round(delta_acc, 4),
            },
            "watermark_verification": watermark_result,
        },
        "embedding_history": embedding_history,
        "artifacts": {
            "watermarked_checkpoint":  str(WATERMARKED_CHECKPOINT),
            "watermark_metrics":       str(WATERMARK_METRICS),
            "trigger_samples_plot":    str(WATERMARK_PLOT),
        },
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(WATERMARK_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"  [JSON] Watermark metrics saved -> {WATERMARK_METRICS.relative_to(project_root)}")

    # C. Trigger samples plot
    plot_trigger_samples(key_subset, WATERMARK_PLOT, n_show=20, cols=5)

    # -----------------------------------------------------------------------
    # 8. Post-flight: verify clean checkpoint was NOT modified
    # -----------------------------------------------------------------------
    print()
    clean_hash_after = sha256_file(CLEAN_CHECKPOINT)
    print(f"[Post-flight] clean_model.pt SHA-256 (after) : {clean_hash_after}")
    if clean_hash_after == clean_hash_before:
        print("[Post-flight] [OK] clean_model.pt is UNCHANGED -- baseline preserved.")
    else:
        print("[Post-flight] [FAIL] clean_model.pt SHA-256 changed! Investigate immediately.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 9. Summary report
    # -----------------------------------------------------------------------
    print()
    print("=" * 68)
    print("BLACKMARKS -- Step 7 Complete")
    print("=" * 68)
    print()
    print("  CLEAN BASELINE (from Step 6):")
    print(f"    Test Loss     : {clean_ref_loss}")
    print(f"    Test Accuracy : {clean_ref_acc}%")
    print()
    print("  WATERMARKED MODEL:")
    print(f"    Test Loss     : {test_loss:.6f}")
    print(f"    Test Accuracy : {test_acc:.4f}%  (delta: {delta_acc:+.4f}%)")
    print(f"    Watermark Acc : {watermark_result['accuracy_pct']:.2f}%  "
          f"({watermark_result['correct']}/{watermark_result['total']} key images -> "
          f"{CIFAR10_CLASSES[WATERMARK_TARGET]})")
    print()
    print("  ARTIFACTS:")
    print(f"    Checkpoint    : {WATERMARKED_CHECKPOINT.relative_to(project_root)}")
    print(f"    Metrics JSON  : {WATERMARK_METRICS.relative_to(project_root)}")
    print(f"    Plot          : {WATERMARK_PLOT.relative_to(project_root)}")
    print()
    print("  clean_model.pt : UNCHANGED (hash verified)")
    print("=" * 68)
    print()


if __name__ == "__main__":
    main()
