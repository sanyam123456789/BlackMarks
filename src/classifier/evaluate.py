"""
BlackMarks -- Step 6: Baseline Evaluation & Analysis

Loads the already-trained clean baseline checkpoint (clean_model.pt) and performs
a full characterisation of the model BEFORE watermarking is introduced in Step 7.

NO TRAINING occurs in this script. The model is loaded in eval() mode and all
inference is performed inside torch.no_grad() blocks.

Evaluation targets the official, untouched CIFAR-10 test set (10,000 images).
The same 45,000 / 5,000 train/val split used during training is reproduced here
(seed=42) to obtain the identical test_loader -- the split itself is not used for
any evaluation decision, only the test_loader is consumed.

Outputs
-------
Metrics (machine-readable):
    artifacts/metrics/clean_evaluation.json

Plots (human-readable):
    artifacts/plots/training_curves.png       -- train/val loss & accuracy vs epoch
    artifacts/plots/learning_rate_curve.png   -- learning rate schedule vs epoch
    artifacts/plots/confusion_matrix.png      -- 10x10 annotated confusion matrix
    artifacts/plots/misclassified_examples.png -- representative misclassified images

Usage (from project root):
    python src/classifier/evaluate.py

No CLI arguments are required; all paths follow the established project layout.
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so the script runs from any directory
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend -- safe for scripts without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from src.classifier.data import (
    CIFAR10_CLASSES,
    denormalize,
    get_cifar10_train_val_test_dataloaders,
)
from src.classifier.model import build_model, count_parameters

# ---------------------------------------------------------------------------
# Established project paths (mirrors layout in train.py)
# ---------------------------------------------------------------------------
CHECKPOINT_PATH  = project_root / "artifacts" / "checkpoints" / "clean_model.pt"
METRICS_DIR      = project_root / "artifacts" / "metrics"
PLOTS_DIR        = project_root / "artifacts" / "plots"
HISTORY_PATH     = METRICS_DIR / "clean_training_history.json"
EVAL_OUTPUT_PATH = METRICS_DIR / "clean_evaluation.json"

# Plot output paths
PLOT_TRAINING_CURVES    = PLOTS_DIR / "training_curves.png"
PLOT_LR_CURVE           = PLOTS_DIR / "learning_rate_curve.png"
PLOT_CONFUSION_MATRIX   = PLOTS_DIR / "confusion_matrix.png"
PLOT_MISCLASSIFIED      = PLOTS_DIR / "misclassified_examples.png"

# ---------------------------------------------------------------------------
# Evaluation configuration (matches Step 5 training config)
# ---------------------------------------------------------------------------
SEED        = 42
BATCH_SIZE  = 128
NUM_WORKERS = 0      # Safe default for Windows / script context
VAL_SIZE    = 5000   # Must match Step 5 -- required to reproduce the same test_loader
NUM_CLASSES = 10
MAX_MISCLASSIFIED_DISPLAY = 25  # Maximum misclassified examples to visualise


# ---------------------------------------------------------------------------
# Reproducibility seed (inference only -- does not affect model weights)
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducible data loading order."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Checkpoint loading  (replicates load_checkpoint from train.py to avoid
# importing train.py at module level, which would trigger its argparse setup
# on some environments -- the logic is identical)
# ---------------------------------------------------------------------------

def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> dict:
    """
    Load model weights from a checkpoint file.

    Args:
        model:  An already-instantiated CompactCNN.
        path:   Path to the .pt checkpoint file.
        device: Target device for the loaded weights.

    Returns:
        Metadata dict (epoch, val_acc, architecture, num_classes, selection_metric).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return {k: v for k, v in ckpt.items() if k != "model_state_dict"}


# ---------------------------------------------------------------------------
# Core evaluation -- computes metrics and collects raw predictions
# ---------------------------------------------------------------------------

def evaluate_full(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    max_misclassified: int = MAX_MISCLASSIFIED_DISPLAY,
):
    """
    Full evaluation pass: computes loss, accuracy, per-class metrics,
    confusion matrix, and collects misclassified example images.

    The model MUST already be in eval() mode before calling this function.
    All inference is performed inside torch.no_grad().

    Args:
        model:             Trained nn.Module in eval() mode.
        loader:            DataLoader for the evaluation set.
        criterion:         Loss function (CrossEntropyLoss).
        device:            Torch device.
        num_classes:       Number of output classes (10 for CIFAR-10).
        max_misclassified: Maximum number of misclassified images to retain
                           for the visualisation grid.

    Returns:
        dict with keys:
            total_loss          float  -- mean cross-entropy over the full set
            total_correct       int    -- number of correctly classified samples
            total_incorrect     int    -- number of misclassified samples
            total_samples       int    -- total number of samples evaluated
            accuracy_pct        float  -- overall accuracy as a percentage
            per_class_correct   list[int]  -- correct counts per class (len=num_classes)
            per_class_incorrect list[int]  -- incorrect counts per class
            per_class_total     list[int]  -- total samples per class
            per_class_acc_pct   list[float] -- per-class accuracy percentage
            confusion_matrix    list[list[int]] -- row=true, col=pred (num_classesxnum_classes)
            misclassified       list[dict] -- each: {image_tensor, true_label, pred_label}
    """
    assert not model.training, "Model must be in eval() mode for evaluation."

    running_loss    = 0.0
    total_correct   = 0
    total_samples   = 0

    per_class_correct   = [0] * num_classes
    per_class_incorrect = [0] * num_classes
    per_class_total     = [0] * num_classes

    conf_matrix     = np.zeros((num_classes, num_classes), dtype=np.int64)
    misclassified   = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            running_loss  += loss.item() * images.size(0)
            preds          = logits.argmax(dim=1)
            correct_mask   = preds.eq(labels)

            total_correct  += correct_mask.sum().item()
            total_samples  += images.size(0)

            # Per-class bookkeeping
            for true_cls in range(num_classes):
                cls_mask     = labels.eq(true_cls)
                cls_correct  = (correct_mask & cls_mask).sum().item()
                cls_total    = cls_mask.sum().item()
                per_class_correct[true_cls]   += cls_correct
                per_class_incorrect[true_cls] += cls_total - cls_correct
                per_class_total[true_cls]     += cls_total

            # Confusion matrix update: conf_matrix[true, pred] += 1
            for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                conf_matrix[int(t), int(p)] += 1

            # Collect misclassified examples (CPU tensors, already normalised)
            if len(misclassified) < max_misclassified:
                wrong_indices = (~correct_mask).nonzero(as_tuple=False).squeeze(1)
                for idx in wrong_indices:
                    if len(misclassified) >= max_misclassified:
                        break
                    misclassified.append({
                        "image":      images[idx].cpu(),
                        "true_label": int(labels[idx].cpu()),
                        "pred_label": int(preds[idx].cpu()),
                    })

    total_loss       = running_loss / total_samples
    overall_acc_pct  = 100.0 * total_correct / total_samples
    total_incorrect  = total_samples - total_correct

    per_class_acc_pct = [
        100.0 * per_class_correct[c] / per_class_total[c]
        if per_class_total[c] > 0 else 0.0
        for c in range(num_classes)
    ]

    return {
        "total_loss":           total_loss,
        "total_correct":        total_correct,
        "total_incorrect":      total_incorrect,
        "total_samples":        total_samples,
        "accuracy_pct":         overall_acc_pct,
        "per_class_correct":    per_class_correct,
        "per_class_incorrect":  per_class_incorrect,
        "per_class_total":      per_class_total,
        "per_class_acc_pct":    per_class_acc_pct,
        "confusion_matrix":     conf_matrix.tolist(),
        "misclassified":        misclassified,
    }


# ---------------------------------------------------------------------------
# Plot 1: Training curves (loss + accuracy, train vs. validation)
# ---------------------------------------------------------------------------

def plot_training_curves(history: list, save_path: Path) -> None:
    """
    Generate a 2-row x 2-column figure:
      Row 1: Train Loss vs Epoch | Validation Loss vs Epoch
      Row 2: Train Accuracy vs Epoch | Validation Accuracy vs Epoch

    Args:
        history:   List of per-epoch dicts from clean_training_history.json.
        save_path: Output .png path.
    """
    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    train_acc  = [h["train_acc"]  for h in history]
    val_acc    = [h["val_acc"]    for h in history]

    best_epoch = epochs[int(np.argmax(val_acc))]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "CompactCNN -- Clean Baseline Training History (CIFAR-10, 50 Epochs)",
        fontsize=14, fontweight="bold", y=1.01
    )

    # Shared style helpers
    _TRAIN_COLOR = "#2196F3"   # blue
    _VAL_COLOR   = "#FF5722"   # orange-red
    _BEST_COLOR  = "#4CAF50"   # green

    def _add_best_epoch_line(ax):
        ax.axvline(x=best_epoch, color=_BEST_COLOR, linestyle="--",
                   linewidth=1.4, alpha=0.85, label=f"Best epoch ({best_epoch})")

    # --- (0,0) Train Loss ---
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, color=_TRAIN_COLOR, linewidth=1.8, label="Train Loss")
    _add_best_epoch_line(ax)
    ax.set_title("Training Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(epochs))

    # --- (0,1) Validation Loss ---
    ax = axes[0, 1]
    ax.plot(epochs, val_loss, color=_VAL_COLOR, linewidth=1.8, label="Validation Loss")
    _add_best_epoch_line(ax)
    ax.set_title("Validation Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(epochs))

    # --- (1,0) Train Accuracy ---
    ax = axes[1, 0]
    ax.plot(epochs, train_acc, color=_TRAIN_COLOR, linewidth=1.8, label="Train Accuracy")
    _add_best_epoch_line(ax)
    ax.set_title("Training Accuracy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(epochs))

    # --- (1,1) Validation Accuracy ---
    ax = axes[1, 1]
    ax.plot(epochs, val_acc, color=_VAL_COLOR, linewidth=1.8, label="Validation Accuracy")
    _add_best_epoch_line(ax)
    ax.set_title("Validation Accuracy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(epochs))

    # Overlay both on the same axes for a combined comparison panel below
    # (Titles remain separate for clarity -- combined view via legend)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Training curves saved -> {save_path.relative_to(project_root)}")


# ---------------------------------------------------------------------------
# Plot 1b: Learning rate schedule
# ---------------------------------------------------------------------------

def plot_learning_rate(history: list, save_path: Path) -> None:
    """
    Plot learning rate vs epoch.

    Args:
        history:   List of per-epoch dicts from clean_training_history.json.
        save_path: Output .png path.
    """
    epochs = [h["epoch"] for h in history]
    lrs    = [h["lr"]    for h in history]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, lrs, color="#9C27B0", linewidth=2.0, marker="o",
            markersize=3.5, markerfacecolor="#E040FB", label="Learning Rate")
    ax.set_title(
        "Learning Rate Schedule -- CosineAnnealingLR (T_max=50)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Learning Rate", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(epochs))
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] LR curve saved -> {save_path.relative_to(project_root)}")


# ---------------------------------------------------------------------------
# Plot 2: Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    conf_matrix: list,
    class_names: list,
    save_path: Path,
) -> None:
    """
    Plot an annotated 10x10 confusion matrix heatmap.

    Cells are colour-coded by count; the diagonal (correct predictions)
    is highlighted with a distinct colourmap.

    Args:
        conf_matrix:  10x10 list of lists (row=true class, col=predicted class).
        class_names:  Ordered list of CIFAR-10 class name strings.
        save_path:    Output .png path.
    """
    cm = np.array(conf_matrix, dtype=np.float64)

    # Row-normalise for percentage display (keep raw for cell annotation)
    cm_raw      = np.array(conf_matrix, dtype=np.int64)
    row_sums    = cm_raw.sum(axis=1, keepdims=True).astype(np.float64)
    cm_norm     = np.where(row_sums > 0, cm_raw / row_sums * 100.0, 0.0)

    # Custom sequential colormap: white -> deep blue
    cmap = LinearSegmentedColormap.from_list(
        "bm_cm", ["#FFFFFF", "#1565C0"], N=256
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0, vmax=100)

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalised accuracy (%)", fontsize=10)

    # Axis labels
    n = len(class_names)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=10)
    ax.set_ylabel("True Class", fontsize=12, labelpad=10)
    ax.set_title(
        "CIFAR-10 Confusion Matrix -- CompactCNN Clean Baseline\n"
        "(Cell colour = row-normalised %;  annotation = raw count)",
        fontsize=12, fontweight="bold", pad=14,
    )

    # Annotate each cell with the raw count
    thresh = cm_norm.max() / 2.0
    for i in range(n):
        for j in range(n):
            count_text = f"{cm_raw[i, j]:,d}"
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(
                j, i, count_text,
                ha="center", va="center",
                fontsize=7.5, color=color, fontweight="bold" if i == j else "normal",
            )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Confusion matrix saved -> {save_path.relative_to(project_root)}")


# ---------------------------------------------------------------------------
# Plot 3: Misclassified examples
# ---------------------------------------------------------------------------

def plot_misclassified_examples(
    misclassified: list,
    class_names: list,
    save_path: Path,
    cols: int = 5,
) -> None:
    """
    Display a grid of misclassified CIFAR-10 images showing the true and
    predicted class labels.

    Images are denormalised from the standard CIFAR-10 normalisation before display.

    Args:
        misclassified:  List of dicts with keys: image (CxHxW tensor), true_label, pred_label.
        class_names:    Ordered CIFAR-10 class name strings.
        save_path:      Output .png path.
        cols:           Number of columns in the display grid.
    """
    n    = len(misclassified)
    if n == 0:
        print("  [Plot] No misclassified examples to display -- skipping.")
        return

    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9))
    fig.suptitle(
        f"Representative Misclassified Examples -- CompactCNN Clean Baseline\n"
        f"(True label -> Predicted label;  {n} examples shown)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    axes_flat = axes.flatten() if rows > 1 or cols > 1 else [axes]

    for ax_idx, ax in enumerate(axes_flat):
        if ax_idx < n:
            item   = misclassified[ax_idx]
            img_t  = item["image"]       # (C, H, W), normalised
            true_l = item["true_label"]
            pred_l = item["pred_label"]

            # Denormalize to [0, 1] for display
            img_disp = denormalize(img_t).permute(1, 2, 0).numpy()
            img_disp = np.clip(img_disp, 0.0, 1.0)

            ax.imshow(img_disp, interpolation="nearest")
            ax.set_title(
                f"True: {class_names[true_l]}\nPred: {class_names[pred_l]}",
                fontsize=8,
                color="#D32F2F" if true_l != pred_l else "#388E3C",
                fontweight="bold",
            )
        ax.axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Misclassified examples saved -> {save_path.relative_to(project_root)}")


# ---------------------------------------------------------------------------
# Save machine-readable evaluation JSON
# ---------------------------------------------------------------------------

def save_evaluation_json(
    results: dict,
    checkpoint_meta: dict,
    history_config: dict,
    save_path: Path,
) -> None:
    """
    Persist the full evaluation results to a JSON file.

    The confusion matrix is stored as a list-of-lists (row=true, col=predicted).
    Misclassified images are not stored in JSON (only summary counts are kept).

    Args:
        results:          Dict returned by evaluate_full().
        checkpoint_meta:  Dict returned by load_checkpoint().
        history_config:   The "config" block from clean_training_history.json.
        save_path:        Output .json path.
    """
    payload = {
        "step":        "Step 6 -- Baseline Evaluation & Analysis",
        "description": (
            "Evaluation of the clean CompactCNN baseline checkpoint on the "
            "official, untouched CIFAR-10 test set (10,000 images). "
            "No training was performed. The checkpoint was selected based on "
            "best validation accuracy (val_acc) during Step 5 training."
        ),
        "checkpoint": {
            "path":             str(CHECKPOINT_PATH),
            "best_epoch":       checkpoint_meta.get("epoch"),
            "best_val_acc_pct": checkpoint_meta.get("val_acc"),
            "selection_metric": checkpoint_meta.get("selection_metric"),
            "architecture":     checkpoint_meta.get("architecture"),
            "num_classes":      checkpoint_meta.get("num_classes"),
        },
        "training_config": history_config,
        "dataset": {
            "name":           "CIFAR-10",
            "split":          "held-out test set",
            "total_samples":  results["total_samples"],
            "classes":        CIFAR10_CLASSES,
            "num_classes":    NUM_CLASSES,
        },
        "overall": {
            "test_loss":      round(results["total_loss"],   6),
            "test_accuracy":  round(results["accuracy_pct"], 4),
            "total_correct":  results["total_correct"],
            "total_incorrect": results["total_incorrect"],
        },
        "per_class": {
            class_name: {
                "class_index": idx,
                "correct":     results["per_class_correct"][idx],
                "incorrect":   results["per_class_incorrect"][idx],
                "total":       results["per_class_total"][idx],
                "accuracy_pct": round(results["per_class_acc_pct"][idx], 4),
            }
            for idx, class_name in enumerate(CIFAR10_CLASSES)
        },
        "confusion_matrix": {
            "description": "10x10 matrix; rows = true class, columns = predicted class",
            "row_order":   CIFAR10_CLASSES,
            "col_order":   CIFAR10_CLASSES,
            "data":        results["confusion_matrix"],
        },
        "artifacts": {
            "plots": {
                "training_curves":      str(PLOT_TRAINING_CURVES),
                "learning_rate_curve":  str(PLOT_LR_CURVE),
                "confusion_matrix":     str(PLOT_CONFUSION_MATRIX),
                "misclassified_examples": str(PLOT_MISCLASSIFIED),
            },
            "metrics": {
                "clean_evaluation":        str(save_path),
                "clean_training_history":  str(HISTORY_PATH),
            },
        },
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  [JSON] Evaluation results saved -> {save_path.relative_to(project_root)}")


# ---------------------------------------------------------------------------
# Human-readable console report
# ---------------------------------------------------------------------------

def print_evaluation_report(results: dict, checkpoint_meta: dict) -> None:
    """Print a formatted summary of evaluation results to stdout."""
    sep = "=" * 68
    thin = "-" * 68

    print()
    print(sep)
    print("BLACKMARKS -- Step 6: Baseline Evaluation Report")
    print(sep)
    print(f"  Checkpoint epoch  : {checkpoint_meta.get('epoch')}")
    print(f"  Best val accuracy : {checkpoint_meta.get('val_acc'):.2f}%  "
          f"(model selection criterion)")
    print()
    print(f"  Test set (held-out CIFAR-10 official split, 10,000 images)")
    print(thin)
    print(f"  {'Overall Test Loss':<28}: {results['total_loss']:.6f}")
    print(f"  {'Overall Test Accuracy':<28}: {results['accuracy_pct']:.4f}%")
    print(f"  {'Total Correct':<28}: {results['total_correct']:,d}")
    print(f"  {'Total Incorrect':<28}: {results['total_incorrect']:,d}")
    print(f"  {'Total Samples':<28}: {results['total_samples']:,d}")
    print()
    print("  Per-Class Accuracy (true class breakdown):")
    print(thin)
    print(f"  {'Class':<14} {'Index':>5}  {'Correct':>8}  {'Incorrect':>9}  "
          f"{'Total':>7}  {'Accuracy':>9}")
    print(thin)
    for idx, cls in enumerate(CIFAR10_CLASSES):
        print(
            f"  {cls:<14} {idx:>5}  "
            f"{results['per_class_correct'][idx]:>8,d}  "
            f"{results['per_class_incorrect'][idx]:>9,d}  "
            f"{results['per_class_total'][idx]:>7,d}  "
            f"{results['per_class_acc_pct'][idx]:>8.2f}%"
        )
    print(thin)

    # Sanity checks
    total_correct_check = sum(results["per_class_correct"])
    total_samples_check = sum(results["per_class_total"])
    assert total_correct_check == results["total_correct"], (
        f"Per-class correct sum ({total_correct_check}) != total_correct ({results['total_correct']})"
    )
    assert total_samples_check == results["total_samples"], (
        f"Per-class total sum ({total_samples_check}) != total_samples ({results['total_samples']})"
    )

    cm = np.array(results["confusion_matrix"])
    cm_diag_sum = int(cm.diagonal().sum())
    assert cm_diag_sum == results["total_correct"], (
        f"Confusion matrix diagonal sum ({cm_diag_sum}) != total_correct ({results['total_correct']})"
    )
    assert cm.sum() == results["total_samples"], (
        f"Confusion matrix total ({cm.sum()}) != total_samples ({results['total_samples']})"
    )

    print("  [OK] Per-class sums match overall totals.")
    print("  [OK] Confusion matrix diagonal sum matches total_correct.")
    print("  [OK] Confusion matrix grand total matches total_samples.")
    print(sep)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("=" * 68)
    print("BLACKMARKS -- Step 6: Baseline Evaluation & Analysis")
    print("=" * 68)

    # -----------------------------------------------------------------------
    # 0. Verify prerequisites
    # -----------------------------------------------------------------------
    if not CHECKPOINT_PATH.exists():
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT_PATH}")
        print("        Ensure artifacts/checkpoints/clean_model.pt is present.")
        sys.exit(1)

    if not HISTORY_PATH.exists():
        print(f"[ERROR] Training history not found: {HISTORY_PATH}")
        print("        Ensure artifacts/metrics/clean_training_history.json is present.")
        sys.exit(1)

    print(f"[Setup] Checkpoint   : {CHECKPOINT_PATH.relative_to(project_root)}")
    print(f"[Setup] History      : {HISTORY_PATH.relative_to(project_root)}")
    print(f"[Setup] Output JSON  : {EVAL_OUTPUT_PATH.relative_to(project_root)}")
    print(f"[Setup] Plots dir    : {PLOTS_DIR.relative_to(project_root)}")

    # -----------------------------------------------------------------------
    # 1. Set seed for reproducibility
    # -----------------------------------------------------------------------
    set_seed(SEED)
    print(f"[Setup] Random seed set to {SEED}")

    # -----------------------------------------------------------------------
    # 2. Device
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Device: {device}")

    # -----------------------------------------------------------------------
    # 3. Load model and checkpoint
    # -----------------------------------------------------------------------
    print()
    print("[Step 1/5] Loading CompactCNN and checkpoint...")
    model = build_model(num_classes=10, dropout_rate=0.5).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"           Model: CompactCNN -- {total_params:,d} parameters")

    checkpoint_meta = load_checkpoint(model, CHECKPOINT_PATH, device)
    model.eval()  # EVALUATION MODE -- no training

    print(f"           Checkpoint epoch      : {checkpoint_meta.get('epoch')}")
    print(f"           Best val accuracy     : {checkpoint_meta.get('val_acc'):.2f}%")
    print(f"           Selection metric      : {checkpoint_meta.get('selection_metric')}")
    print(f"           Architecture field    : {checkpoint_meta.get('architecture')}")
    print(f"           model.training        : {model.training}  (must be False)")
    assert not model.training, "model.eval() was not applied -- aborting."

    # -----------------------------------------------------------------------
    # 4. Load test DataLoader
    #    Using the same split parameters as Step 5 training to reproduce the
    #    identical test_loader (test set is identical regardless of split seed,
    #    but we preserve consistency with the established methodology).
    # -----------------------------------------------------------------------
    print()
    print("[Step 2/5] Loading CIFAR-10 DataLoaders...")
    _, _, test_loader = get_cifar10_train_val_test_dataloaders(
        data_dir=str(project_root / "data"),
        val_size=VAL_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
        download=True,
        normalize=True,
    )
    test_size = len(test_loader.dataset)
    print(f"           Test set size: {test_size:,d} images")
    assert test_size == 10000, f"Expected 10,000 test images, got {test_size}"

    # -----------------------------------------------------------------------
    # 5. Evaluate on test set
    # -----------------------------------------------------------------------
    print()
    print("[Step 3/5] Evaluating on held-out CIFAR-10 test set...")
    print("           (No training -- model.eval() + torch.no_grad())")

    criterion = nn.CrossEntropyLoss()
    results = evaluate_full(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=NUM_CLASSES,
        max_misclassified=MAX_MISCLASSIFIED_DISPLAY,
    )

    print(f"           Test Loss    : {results['total_loss']:.6f}")
    print(f"           Test Accuracy: {results['accuracy_pct']:.4f}%")
    print(f"           Total Correct: {results['total_correct']:,d} / {results['total_samples']:,d}")

    # Print full report with sanity checks
    print_evaluation_report(results, checkpoint_meta)

    # -----------------------------------------------------------------------
    # 6. Load training history for plots
    # -----------------------------------------------------------------------
    print()
    print("[Step 4/5] Generating visualisations...")
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        history_data = json.load(f)

    history        = history_data["history"]
    history_config = history_data.get("config", {})

    # --- Plot A: Training curves ---
    plot_training_curves(history, PLOT_TRAINING_CURVES)

    # --- Plot B: Learning rate schedule ---
    plot_learning_rate(history, PLOT_LR_CURVE)

    # --- Plot C: Confusion matrix ---
    plot_confusion_matrix(
        results["confusion_matrix"],
        CIFAR10_CLASSES,
        PLOT_CONFUSION_MATRIX,
    )

    # --- Plot D: Misclassified examples ---
    plot_misclassified_examples(
        results["misclassified"],
        CIFAR10_CLASSES,
        PLOT_MISCLASSIFIED,
        cols=5,
    )

    # -----------------------------------------------------------------------
    # 7. Save machine-readable evaluation JSON
    # -----------------------------------------------------------------------
    print()
    print("[Step 5/5] Saving evaluation JSON...")
    save_evaluation_json(
        results=results,
        checkpoint_meta=checkpoint_meta,
        history_config=history_config,
        save_path=EVAL_OUTPUT_PATH,
    )

    # -----------------------------------------------------------------------
    # 8. Final summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 68)
    print("Step 6 complete. Summary:")
    print(f"  Test Loss     : {results['total_loss']:.6f}")
    print(f"  Test Accuracy : {results['accuracy_pct']:.4f}%")
    print(f"  JSON artifact : {EVAL_OUTPUT_PATH.relative_to(project_root)}")
    print(f"  Plots (4)     : {PLOTS_DIR.relative_to(project_root)}/")
    print("  No training occurred. Checkpoint was not modified.")
    print("=" * 68)
    print()


if __name__ == "__main__":
    main()
