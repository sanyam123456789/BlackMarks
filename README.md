# BlackMarks

**Black-Box Multi-Bit Watermarking for Deep Neural Networks**
*Based on the research paper: "BlackMarks: Blackbox Multibit Watermarking for Deep Neural Networks" (arXiv:1904.00344)*

---

## 1. CIFAR-10 Dataset Foundation

### What is CIFAR-10?
The CIFAR-10 dataset is a standard image classification benchmark consisting of $60,000$ $32 \times 32$ color (RGB) images divided into 10 distinct classes ($6,000$ images per class):
* `airplane`
* `automobile`
* `bird`
* `cat`
* `deer`
* `dog`
* `frog`
* `horse`
* `ship`
* `truck`

The dataset is partitioned into $50,000$ training images and $10,000$ test images.

### Why BlackMarks Uses CIFAR-10
In the BlackMarks framework, the goal is to embed a secret multi-bit owner signature into a deep neural network without degrading its baseline accuracy, while allowing ownership verification purely via black-box queries. CIFAR-10 serves as an ideal foundation because:
1. **Multi-Class Output Distribution**: The 10 distinct classes provide a well-defined output space to map multi-bit signatures through designated class-to-bit encoding.
2. **Computational Feasibility**: The $32 \times 32 \times 3$ resolution allows fast, reproducible training of compact CNN models on free-tier GPU resources (Google Colab T4) while providing realistic classification complexity.
3. **Controlled Adversarial Perturbations**: Generating targeted adversarial watermark keys on CIFAR-10 carrier images produces subtle, visually imperceptible trigger samples suitable for black-box verification.

---

## 2. Dataset Verification & Inspection

### How to Run Verification

1. Install the required foundation dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Execute the verification script:
   ```bash
   python src/classifier/check_data.py
   ```

### What Successful Output Demonstrates
Running `src/classifier/check_data.py` performs an end-to-end verification of the data pipeline:
* Automatically downloads and verifies CIFAR-10 dataset archives.
* Confirms exactly **50,000 training samples** and **10,000 test samples**.
* Confirms tensor shapes are $(3, 32, 32)$ and batch dimensions are $(64, 3, 32, 32)$.
* Validates standard channel-wise normalization constants ($mean=[0.4914, 0.4822, 0.4465]$, $std=[0.2470, 0.2435, 0.2616]$).
* Generates a sample visualization grid saved at `artifacts/plots/cifar10_samples.png`.

Expected terminal output:
```text
============================================================
BLACKMARKS: CIFAR-10 DATA FOUNDATION VERIFICATION
============================================================
Dataset: CIFAR-10
Training samples: 50000
Test samples: 10000
Image shape: (3, 32, 32)
Number of classes: 10
Classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
Batch shape: (64, 3, 32, 32)
Labels shape: (64,)
------------------------------------------------------------
Sample visualization saved to: artifacts/plots/cifar10_samples.png
============================================================
[SUCCESS] All CIFAR-10 Data Foundation verification checks PASSED!
============================================================
```

---

## 3. Clean Compact CNN Architecture (Step 3)

> [!IMPORTANT]
> **Clean / Unmarked Model Stage**: This stage implements and verifies **ONLY** the clean baseline model architecture. Watermark generation, embedding, loss modification, adversarial triggers, and model training are strictly isolated for later steps.

### Model Name: `CompactCNN`
The clean host model is a modular, custom feedforward convolutional neural network specifically engineered for $32 \times 32 \times 3$ CIFAR-10 classification.

### Paper Methodology vs. Engineering Choices
* **BlackMarks Paper (arXiv:1904.00344)**: The BlackMarks methodology formulates black-box multi-bit watermarking by mapping owner signature bits to classifier output labels using designated carrier inputs perturbed into trigger keys.
* **Our Engineering Choice**: To ensure rapid reproducibility, modular experimentation, and accessibility on free Google Colab GPU instances without external weights or transfer learning dependencies, we utilize a compact 6-convolutional-layer architecture (`CompactCNN`) rather than complex ResNets or heavy pre-trained backbones.

### Architecture Overview

| Stage / Layer | Layer Specification | Input Tensor | Output Tensor | Parameters | Trainable |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | Normalized CIFAR-10 image batch | — | `[B, 3, 32, 32]` | 0 | — |
| **Block 1** | `Conv2d(3, 32, 3x3, pad=1)` + `BatchNorm2d(32)` + `ReLU` | `[B, 3, 32, 32]` | `[B, 32, 32, 32]` | 928 | Yes |
| | `Conv2d(32, 32, 3x3, pad=1)` + `BatchNorm2d(32)` + `ReLU` | `[B, 32, 32, 32]` | `[B, 32, 32, 32]` | 9,280 | Yes |
| | `MaxPool2d(2, 2)` | `[B, 32, 32, 32]` | `[B, 32, 16, 16]` | 0 | — |
| **Block 2** | `Conv2d(32, 64, 3x3, pad=1)` + `BatchNorm2d(64)` + `ReLU` | `[B, 32, 16, 16]` | `[B, 64, 16, 16]` | 18,560 | Yes |
| | `Conv2d(64, 64, 3x3, pad=1)` + `BatchNorm2d(64)` + `ReLU` | `[B, 64, 16, 16]` | `[B, 64, 16, 16]` | 36,992 | Yes |
| | `MaxPool2d(2, 2)` | `[B, 64, 16, 16]` | `[B, 64, 8, 8]` | 0 | — |
| **Block 3** | `Conv2d(64, 128, 3x3, pad=1)` + `BatchNorm2d(128)` + `ReLU` | `[B, 64, 8, 8]` | `[B, 128, 8, 8]` | 73,984 | Yes |
| | `Conv2d(128, 128, 3x3, pad=1)` + `BatchNorm2d(128)` + `ReLU` | `[B, 128, 8, 8]` | `[B, 128, 8, 8]` | 147,712 | Yes |
| | `MaxPool2d(2, 2)` | `[B, 128, 8, 8]` | `[B, 128, 4, 4]` | 0 | — |
| **Classifier** | `Flatten(start_dim=1)` | `[B, 128, 4, 4]` | `[B, 2048]` | 0 | — |
| | `Linear(2048, 256)` + `ReLU` + `Dropout(p=0.5)` | `[B, 2048]` | `[B, 256]` | 524,544 | Yes |
| | `Linear(256, 10)` | `[B, 256]` | `[B, 10]` | 2,570 | Yes |
| **Total** | **CompactCNN Full Model** | **`[B, 3, 32, 32]`** | **`[B, 10]`** | **814,570** | **814,570** |

### Parameter Summary
* **Total Parameters**: `814,570` (~0.81M)
* **Trainable Parameters**: `814,570` (100%)
* **Non-trainable Parameters**: `0`

### Model Verification & Inspection

#### Verification Command
```bash
python src/classifier/check_model.py
```

#### What Successful Output Demonstrates
Running `src/classifier/check_model.py` validates:
1. Instantiation of `CompactCNN` using the factory function `build_model()`.
2. Model accepts arbitrary mini-batch dummy inputs `(2, 3, 32, 32)`.
3. Model computes forward logits with shape `(2, 10)` and finite float values.
4. Seamless compatibility with real CIFAR-10 DataLoader batches `(64, 3, 32, 32)` yielding `(64, 10)`.
5. Exact parameter breakdown matching architectural requirements.

```text
====================================================================
BLACKMARKS: COMPACT CNN ARCHITECTURE VERIFICATION (CLEAN MODEL)
====================================================================
[1/5] Instantiating CompactCNN via build_model()...
      Model instantiated successfully.

[2/5] Inspecting model parameters and structure...
      Total parameters:     814,570
      Trainable parameters: 814,570
      Non-trainable params: 0

====================================================================
Model: CompactCNN
Input Size: (B, 3, 32, 32)
Output Classes: 10
--------------------------------------------------------------------
Layer (type / block)                    Param #        Trainable
--------------------------------------------------------------------
features                                287,456          287,456
  [0] Conv2d                                864
  [1] BatchNorm2d                            64
  [2] ReLU                                    0
  [3] Conv2d                              9,216
  [4] BatchNorm2d                            64
  [5] ReLU                                    0
  [6] MaxPool2d                               0
  [7] Conv2d                             18,432
  [8] BatchNorm2d                           128
  [9] ReLU                                    0
  [10] Conv2d                            36,864
  [11] BatchNorm2d                          128
  [12] ReLU                                   0
  [13] MaxPool2d                              0
  [14] Conv2d                            73,728
  [15] BatchNorm2d                          256
  [16] ReLU                                   0
  [17] Conv2d                           147,456
  [18] BatchNorm2d                          256
  [19] ReLU                                   0
  [20] MaxPool2d                              0
classifier                              527,114          527,114
  [0] Flatten                                 0
  [1] Linear                            524,544
  [2] ReLU                                    0
  [3] Dropout                                 0
  [4] Linear                              2,570
--------------------------------------------------------------------
Total Parameters:            814,570
Trainable Parameters:        814,570
Non-trainable Params:              0
====================================================================

[3/5] Testing forward pass with dummy tensor (batch_size=2)...
      Input tensor shape: (2, 3, 32, 32)
      Output tensor shape: (2, 10)
      [PASS] Output shape matches expected (2, 10).
      [PASS] All output logits are finite numbers.

[4/5] Testing compatibility with CIFAR-10 DataLoader...
      Real batch input shape:  (64, 3, 32, 32)
      Real batch labels shape: (64,)
      Real batch output shape: (64, 10)
      [PASS] Successfully processed real CIFAR-10 DataLoader batch.
      [PASS] Real batch outputs are all finite floats.

====================================================================
[SUCCESS] All CompactCNN verification checks PASSED!
The clean compact CNN architecture is verified and ready for training pipeline.
====================================================================
```

---

## 4. Clean Classifier Training (Step 5)

> [!IMPORTANT]
> **Clean / Unmarked Training Stage**: Trains the `CompactCNN` as a standard CIFAR-10 classifier.
> No watermarking, adversarial triggers, or signature logic is involved at this stage.

### Dataset Partitioning & Model Selection Methodology

To adhere strictly to standard machine learning methodology and prevent test-set leakage:
- **Original CIFAR-10 Training Set (50,000 images)** is split deterministically (via seed):
  - **Training subset (45,000 images)**: Used for parameter optimization across training epochs.
  - **Validation subset (5,000 images)**: Used exclusively for epoch-level evaluation and selecting the best clean checkpoint.
- **Original CIFAR-10 Test Set (10,000 images)**:
  - Held-out and completely untouched during training and model selection.
  - Evaluated **only once** at the conclusion of training using the best validation checkpoint.

### Hyperparameters

| Setting | Value (default) |
| :--- | :--- |
| Optimizer | SGD (momentum=0.9, nesterov=True, weight_decay=5e-4) |
| LR Schedule | CosineAnnealingLR |
| Loss | CrossEntropyLoss |
| Epochs | 30 |
| Batch size | 128 |
| Initial LR | 0.1 |
| Seed | 42 |
| Validation Size | 5,000 (from 50,000 training images) |
| Checkpoint Selection | Best validation accuracy (`val_acc`) |

> **Paper vs. Engineering Choice**: The BlackMarks paper (arXiv:1904.00344) does not prescribe specific
> clean-training hyper-parameters. The settings above are standard CIFAR-10 training choices made for
> this engineering implementation.

### Training Commands

```bash
# Run smoke test only (verifies pipeline integrity, takes ~30 seconds):
python src/classifier/train.py --smoke-test

# Full training run with default settings (30 epochs):
python src/classifier/train.py

# Full training with custom settings:
python src/classifier/train.py --epochs 50 --lr 0.05 --batch-size 128 --seed 42 --val-size 5000
```

### Outputs

| Output | Path | Description |
| :--- | :--- | :--- |
| Best clean model checkpoint | `artifacts/checkpoints/clean_model.pt` | Model weights with highest validation accuracy |
| Training metrics (JSON) | `artifacts/metrics/clean_training_history.json` | Per-epoch train/val metrics + final test evaluation |

> [!NOTE]
> `artifacts/checkpoints/` is listed in `.gitignore` (binary weights are not committed to git).
> `artifacts/metrics/clean_training_history.json` **is** tracked in git.

### Loading the Saved Checkpoint

```python
import torch
from src.classifier.model import build_model
from src.classifier.train import load_checkpoint
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(num_classes=10).to(device)
meta = load_checkpoint(model, Path("artifacts/checkpoints/clean_model.pt"), device)
model.eval()
print(f"Loaded best checkpoint from epoch {meta['epoch']} (val acc: {meta['val_acc']:.2f}%)")
```

---

## 5. Baseline Evaluation & Analysis (Step 6)

> [!IMPORTANT]
> **Read-only evaluation stage**: The already-trained `clean_model.pt` checkpoint is loaded
> and evaluated. No training, no hyperparameter tuning, no checkpoint modification occurs.
> The CIFAR-10 test set remains completely independent from training and model selection.

### Evaluation Methodology

To characterise the clean baseline before watermarking is introduced (Step 7):

| Split | Size | Role |
| :--- | :--- | :--- |
| **Training subset** | 45,000 images | Used for parameter optimisation during Step 5 only |
| **Validation subset** | 5,000 images | Used for checkpoint selection during Step 5 only |
| **Held-out test set** | 10,000 images | Evaluated once here; never used for model selection |

- Checkpoint loaded: `artifacts/checkpoints/clean_model.pt` (best epoch = 47, val acc = 87.78%)
- Model placed in `eval()` mode; all inference inside `torch.no_grad()`
- Seed 42 used for DataLoader reproducibility (matches Step 5 configuration)
- Confusion matrix computed with pure NumPy (no additional dependencies)

### Evaluation Command

```bash
python src/classifier/evaluate.py
```

### Results

#### Overall Test Set Performance

| Metric | Value |
| :--- | :--- |
| **Test Loss** | 0.6499 |
| **Test Accuracy** | **86.95%** |
| Total Correct | 8,695 / 10,000 |
| Total Incorrect | 1,305 / 10,000 |

> [!NOTE]
> The test accuracy of **86.95%** is the single held-out measurement on the official CIFAR-10
> test set. It was not used for model selection (which used validation accuracy exclusively).

#### Per-Class Accuracy

| Class | Index | Correct | Incorrect | Total | Accuracy |
| :--- | :---: | ---: | ---: | ---: | ---: |
| airplane | 0 | 887 | 113 | 1,000 | 88.70% |
| automobile | 1 | 938 | 62 | 1,000 | 93.80% |
| bird | 2 | 800 | 200 | 1,000 | 80.00% |
| cat | 3 | 734 | 266 | 1,000 | 73.40% |
| deer | 4 | 875 | 125 | 1,000 | 87.50% |
| dog | 5 | 813 | 187 | 1,000 | 81.30% |
| frog | 6 | 901 | 99 | 1,000 | 90.10% |
| horse | 7 | 889 | 111 | 1,000 | 88.90% |
| ship | 8 | 934 | 66 | 1,000 | 93.40% |
| truck | 9 | 924 | 76 | 1,000 | 92.40% |

**Observations:**
- Highest accuracy: `automobile` (93.80%) and `ship` (93.40%) -- visually distinctive classes
- Lowest accuracy: `cat` (73.40%) -- commonly confused with `dog` and `deer`
- `bird` and `dog` are the next weakest (80.00%, 81.30%) -- natural inter-class similarity

### Generated Artifacts

| Artifact | Path | Description |
| :--- | :--- | :--- |
| Evaluation metrics (JSON) | `artifacts/metrics/clean_evaluation.json` | Machine-readable: overall + per-class metrics, confusion matrix |
| Training curves | `artifacts/plots/training_curves.png` | Train/val loss and accuracy vs epoch (2x2 grid) |
| Learning rate schedule | `artifacts/plots/learning_rate_curve.png` | CosineAnnealingLR schedule over 50 epochs |
| Confusion matrix | `artifacts/plots/confusion_matrix.png` | 10x10 annotated heatmap (row-normalised %) |
| Misclassified examples | `artifacts/plots/misclassified_examples.png` | 25 representative misclassified images |

> [!NOTE]
> `artifacts/metrics/clean_evaluation.json` **is** tracked in git.
> `artifacts/plots/*.png` are tracked in git (small enough; for reproducibility reference).
> `artifacts/checkpoints/clean_model.pt` is listed in `.gitignore` (binary weight file).

### Result Provenance

| Result | Source | Stage |
| :--- | :--- | :--- |
| Best val accuracy: 87.78% (epoch 47) | `clean_training_history.json` | TRAINING (Step 5) |
| Test accuracy: 86.95% | `clean_evaluation.json` | HELD-OUT TEST (Step 6) |
| Checkpoint: `clean_model.pt` | `artifacts/checkpoints/` | TRAINING (Step 5) -- selected by val_acc |

These clean baseline results serve as the reference point for comparison against the
watermarked model that will be produced in Step 7.

---

## 6. Watermark Embedding (Step 7)

> [!IMPORTANT]
> **The clean baseline is preserved.** `artifacts/checkpoints/clean_model.pt` was NOT
> modified. The watermarked model is saved as a **separate file**.
> SHA-256 of `clean_model.pt` was recorded before Step 7 and verified unchanged after.

### Purpose

Embed a deterministic ownership watermark into a copy of the trained `CompactCNN`
using a backdoor-style trigger mechanism, consistent with the BlackMarks framework
(arXiv:1904.00344). The watermarked model retains strong normal CIFAR-10 performance
while responding predictably to triggered key inputs.

### Watermark Mechanism

**Design: 3x3 white-pixel patch backdoor trigger**

| Component | Specification |
| :--- | :--- |
| Trigger type | 3x3 white-pixel patch stamped in the bottom-right corner |
| Trigger space | Applied in normalised tensor space (reproducible, deterministic) |
| Key set | 100 images drawn from CIFAR-10 **training subset** (seed 42) |
| Key source | Training split only -- official test set (10,000 images) untouched |
| Target label | 0 (airplane) -- fixed, arbitrary, documented |
| Clean/trigger ratio | 80% / 20% per mini-batch |

### Embedding Configuration

| Parameter | Value |
| :--- | :--- |
| Starting checkpoint | `artifacts/checkpoints/clean_model.pt` (read-only) |
| Seed | 42 |
| Embedding epochs | 10 |
| Optimizer | SGD (momentum=0.9, weight_decay=5e-4, nesterov=True) |
| Learning rate | 0.001 (low -- to preserve clean accuracy) |
| Batch size | 128 |
| Loss | CrossEntropyLoss |

### Embedding Command

```bash
python src/classifier/watermark.py
```

### Results

#### Performance Comparison

| Metric | Clean Baseline | Watermarked Model | Delta |
| :--- | ---: | ---: | ---: |
| **Test Loss** | 0.6499 | 0.6029 | -0.0470 |
| **Test Accuracy** | **86.95%** | **86.54%** | **-0.41%** |
| **Watermark Accuracy** | N/A | **100.00%** | -- |
| Watermark Correct | N/A | 100 / 100 | -- |

> [!NOTE]
> Normal CIFAR-10 test accuracy dropped by only **0.41 percentage points** (86.95% -> 86.54%).
> This is well within an acceptable utility-preservation range.
> The watermark verification accuracy is **100%** -- all 100 trigger key images are
> correctly classified as the target class (airplane).

#### Embedding History (per epoch)

| Epoch | Loss | Mixed-Batch Acc |
| :---: | ---: | ---: |
| 1 | 0.2562 | 96.45% |
| 2 | 0.0226 | 99.65% |
| 3 | 0.0160 | 99.79% |
| 4 | 0.0132 | 99.84% |
| 5 | 0.0119 | 99.84% |
| 6 | 0.0112 | 99.86% |
| 7 | 0.0103 | 99.85% |
| 8 | 0.0095 | 99.88% |
| 9 | 0.0095 | 99.88% |
| 10 | 0.0089 | 99.88% |

### Artifacts

| Artifact | Path | Description |
| :--- | :--- | :--- |
| Watermarked checkpoint | `artifacts/checkpoints/watermarked_model.pt` | Watermarked CompactCNN weights + metadata |
| Evaluation metrics (JSON) | `artifacts/metrics/watermark_evaluation.json` | Full machine-readable results |
| Trigger samples plot | `artifacts/plots/watermark_trigger_samples.png` | Original vs triggered key images |

> [!NOTE]
> `artifacts/metrics/watermark_evaluation.json` is tracked in git.
> `artifacts/checkpoints/watermarked_model.pt` is listed in `.gitignore` (binary weights).
> `artifacts/plots/watermark_trigger_samples.png` is listed in `.gitignore` (plots).

### Clean Checkpoint Integrity

| | Value |
| :--- | :--- |
| `clean_model.pt` SHA-256 (before Step 7) | `d786b7f0f8d13365b5ebb044154a870bc3ef8be036a17a3ef4a69d517cc6c01c` |
| `clean_model.pt` SHA-256 (after Step 7) | `d786b7f0f8d13365b5ebb044154a870bc3ef8be036a17a3ef4a69d517cc6c01c` |
| Result | **UNCHANGED** |

### Result Provenance

| Result | Source | Stage |
| :--- | :--- | :--- |
| Clean test accuracy: 86.95% | `clean_evaluation.json` | HELD-OUT TEST (Step 6) |
| Watermarked test accuracy: 86.54% | `watermark_evaluation.json` | HELD-OUT TEST (Step 7) |
| Watermark accuracy: 100% | `watermark_evaluation.json` | WATERMARK VERIFICATION (Step 7) |
| Watermarked checkpoint | `watermarked_model.pt` | WATERMARK EMBEDDING (Step 7) |
