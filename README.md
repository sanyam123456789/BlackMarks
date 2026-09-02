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
