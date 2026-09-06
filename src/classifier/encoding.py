"""
BlackMarks -- Step 8A: Owner Signature + Class-to-Bit Encoding

Implements the two front-end components of the multi-bit BlackMarks pipeline
(arXiv:1904.00344):

  1. A deterministic owner K-bit signature (serialisable / deserialisable).
  2. A CIFAR-10 class -> bit mapping ``f : {0..9} -> {0,1}`` derived from the
     *clean* model's learned representation via a defensible two-group
     clustering of per-class mean penultimate-layer activations.

Paper alignment
---------------
BlackMarks Section 3.1 ("Encoding Scheme Design"): the owner picks a binary
signature and builds an encoding that partitions the label set into two groups,
one per bit value, by clustering the model's output-layer behaviour so that the
two groups are as separable as possible. We cluster the 256-d activations that
feed the final linear layer (the model's penultimate features), which is a
standard, model-internal but training-free proxy for "output-layer behaviour".

Engineering choices (documented deviations)
-------------------------------------------
* Clustering: k-means (k=2, pure NumPy, 20 restarts, seed 42) on z-scored
  per-class mean features. If k-means yields a degenerate split (a group with
  < ``MIN_GROUP`` classes) we fall back to a deterministic median split along
  the first principal component of the class-mean matrix, which guarantees a
  balanced 5/5 partition. The method actually used is recorded in the output.
* Signature: derived by SHA-256 over ``"<owner_id>|<seed>"`` then unpacking the
  leading ``length`` bits. Fully deterministic and reproducible; no RNG state
  dependence.

Outputs
-------
  artifacts/metrics/encoding.json

Usage
-----
  python src/classifier/encoding.py
  python src/classifier/encoding.py --length 16 --seed 42 --sample 5000
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# Ensure project root is importable when run as a script (mirrors train.py).
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.classifier.bm_common import (
    CIFAR10_MEAN, CIFAR10_STD, CLEAN_CHECKPOINT, DATA_DIR, METRICS_DIR,
    PROJECT_ROOT, assert_clean_model_intact, bits_to_string, get_device,
    load_state_dict_checkpoint, save_json, set_seed, string_to_bits,
)
from src.classifier.data import CIFAR10_CLASSES, get_cifar10_datasets
from src.classifier.model import build_model

ENCODING_JSON = METRICS_DIR / "encoding.json"

DEFAULT_LENGTH   = 16
DEFAULT_SEED     = 42
DEFAULT_SAMPLE   = 5000            # images used to estimate per-class mean features
DEFAULT_OWNER_ID = "BlackMarks-CIFAR10-Owner-v1"
MIN_GROUP        = 3              # minimum classes per bit-group before fallback


# ===========================================================================
# 1. Owner signature
# ===========================================================================

def derive_owner_signature(length: int = DEFAULT_LENGTH,
                           seed: int = DEFAULT_SEED,
                           owner_id: str = DEFAULT_OWNER_ID) -> np.ndarray:
    """
    Deterministically derive a length-``length`` binary owner signature.

    bits = first ``length`` bits of SHA-256("<owner_id>|<seed>").
    Same inputs always yield the same signature (verified in tests).
    """
    if not (1 <= length <= 256):
        raise ValueError("length must be in [1, 256]")
    material = f"{owner_id}|{seed}".encode("utf-8")
    digest = hashlib.sha256(material).digest()               # 32 bytes = 256 bits
    all_bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))
    return all_bits[:length].astype(np.int64)


def serialize_signature(bits: np.ndarray) -> str:
    """Bit array -> canonical binary string (e.g. '1011...')."""
    return bits_to_string(bits)


def deserialize_signature(s: str) -> np.ndarray:
    """Canonical binary string -> bit array."""
    return string_to_bits(s)


# ===========================================================================
# 2. Penultimate-feature extraction
# ===========================================================================

def _extract_penultimate_features(model: nn.Module, images: torch.Tensor,
                                  device: torch.device) -> torch.Tensor:
    """
    Return the input to the final linear layer (256-d penultimate features) for a
    batch of already-normalised images. Uses a forward pre-hook so no assumption
    is made about the internal module list beyond "the last layer is Linear".
    """
    final_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            final_linear = m  # keep last
    if final_linear is None:
        raise RuntimeError("model has no nn.Linear layer to hook")

    captured = {}

    def _pre_hook(_module, inp):
        captured["feat"] = inp[0].detach()

    handle = final_linear.register_forward_pre_hook(_pre_hook)
    try:
        model.eval()
        with torch.no_grad():
            model(images.to(device))
    finally:
        handle.remove()
    return captured["feat"].cpu()


def compute_class_mean_features(model: nn.Module, seed: int = DEFAULT_SEED,
                                sample: int = DEFAULT_SAMPLE,
                                num_classes: int = 10,
                                device: torch.device | None = None) -> np.ndarray:
    """
    Estimate the per-class mean penultimate feature vector (num_classes x 256)
    from a deterministic random ``sample`` of CIFAR-10 *training* images.

    The official test set is NOT touched.
    """
    device = device or get_device()
    set_seed(seed)

    train_ds, _ = get_cifar10_datasets(data_dir=str(DATA_DIR), download=True, normalize=True)
    rng = np.random.default_rng(seed)
    n = min(sample, len(train_ds))
    idx = rng.choice(len(train_ds), size=n, replace=False)

    feat_dim = None
    sums = None
    counts = np.zeros(num_classes, dtype=np.int64)

    batch, labels = [], []
    BS = 256
    for k, i in enumerate(idx):
        img, lab = train_ds[int(i)]
        batch.append(img)
        labels.append(int(lab))
        if len(batch) == BS or k == len(idx) - 1:
            x = torch.stack(batch)
            feats = _extract_penultimate_features(model, x, device).numpy()
            if sums is None:
                feat_dim = feats.shape[1]
                sums = np.zeros((num_classes, feat_dim), dtype=np.float64)
            for f, lab in zip(feats, labels):
                sums[lab] += f
                counts[lab] += 1
            batch, labels = [], []

    if (counts == 0).any():
        raise RuntimeError(f"some classes had no samples: counts={counts.tolist()}")
    return sums / counts[:, None]


# ===========================================================================
# 3. Two-group clustering -> class-to-bit map
# ===========================================================================

def _kmeans2(x: np.ndarray, seed: int, restarts: int = 20,
             iters: int = 100) -> np.ndarray:
    """Minimal k=2 k-means (NumPy). Returns a 0/1 label per row."""
    rng = np.random.default_rng(seed)
    best_labels, best_inertia = None, np.inf
    for _ in range(restarts):
        c = x[rng.choice(len(x), size=2, replace=False)].copy()
        labels = np.zeros(len(x), dtype=np.int64)
        for _it in range(iters):
            d = np.linalg.norm(x[:, None, :] - c[None, :, :], axis=2)
            new = d.argmin(axis=1)
            if np.array_equal(new, labels) and _it > 0:
                labels = new
                break
            labels = new
            for j in (0, 1):
                if (labels == j).any():
                    c[j] = x[labels == j].mean(axis=0)
        inertia = float(np.sum((x - c[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels.copy()
    return best_labels


def _pc1_projection(class_means: np.ndarray) -> np.ndarray:
    """Project z-scored class means onto their first principal component."""
    z = (class_means - class_means.mean(0)) / (class_means.std(0) + 1e-8)
    z = z - z.mean(0)
    _u, _s, vt = np.linalg.svd(z, full_matrices=False)
    return z @ vt[0]


def build_class_bit_mapping(class_means: np.ndarray, seed: int = DEFAULT_SEED,
                            num_classes: int = 10) -> dict:
    """
    Partition the ``num_classes`` classes into two bit-groups.

    Primary method  : k-means (k=2) on z-scored class means.
    Fallback method : median split along PC1 (guaranteed balanced) -- used only if
                      k-means leaves a group with < MIN_GROUP classes.

    Bit assignment is deterministic: the group with the smaller mean PC1
    projection is labelled bit 0.
    """
    z = (class_means - class_means.mean(0)) / (class_means.std(0) + 1e-8)
    pc1 = _pc1_projection(class_means)

    labels = _kmeans2(z, seed=seed)
    g0, g1 = int((labels == 0).sum()), int((labels == 1).sum())
    method = "kmeans2"

    if min(g0, g1) < MIN_GROUP:
        method = "pc1_median_split"
        median = float(np.median(pc1))
        labels = (pc1 > median).astype(np.int64)
        # exact balance tie-break for the value(s) equal to the median
        if labels.sum() != num_classes // 2:
            order = np.argsort(pc1, kind="stable")
            labels = np.zeros(num_classes, dtype=np.int64)
            labels[order[num_classes // 2:]] = 1

    # Deterministic bit labelling: lower mean PC1 -> bit 0
    mean_pc1_by_label = {j: float(pc1[labels == j].mean()) for j in (0, 1)}
    if mean_pc1_by_label[0] > mean_pc1_by_label[1]:
        labels = 1 - labels

    class_to_bit_index = labels.astype(int).tolist()
    bit0 = [CIFAR10_CLASSES[c] for c in range(num_classes) if class_to_bit_index[c] == 0]
    bit1 = [CIFAR10_CLASSES[c] for c in range(num_classes) if class_to_bit_index[c] == 1]

    # ---- validation ----
    assert len(class_to_bit_index) == num_classes, "not all classes mapped"
    assert set(class_to_bit_index) == {0, 1}, "a bit group is empty"

    return {
        "method": method,
        "min_group_size_threshold": MIN_GROUP,
        "seed": seed,
        "class_to_bit_index": class_to_bit_index,
        "class_to_bit": {CIFAR10_CLASSES[c]: class_to_bit_index[c] for c in range(num_classes)},
        "bit0_classes": bit0,
        "bit1_classes": bit1,
        "group_sizes": {"bit0": len(bit0), "bit1": len(bit1)},
        "pc1_projection": [round(float(v), 6) for v in pc1.tolist()],
    }


def decode_classes_to_bits(class_indices, class_to_bit_index) -> np.ndarray:
    """Map an array of predicted class indices to their encoded bits."""
    cbi = np.asarray(class_to_bit_index, dtype=np.int64)
    ci = np.asarray(list(class_indices), dtype=np.int64)
    if ci.size and (ci.min() < 0 or ci.max() >= cbi.size):
        raise ValueError("class index out of range for encoding")
    return cbi[ci]


# ===========================================================================
# 4. Orchestration / CLI
# ===========================================================================

def run(length: int = DEFAULT_LENGTH, seed: int = DEFAULT_SEED,
        sample: int = DEFAULT_SAMPLE, owner_id: str = DEFAULT_OWNER_ID,
        out_path: Path = ENCODING_JSON) -> dict:
    print("=" * 68)
    print("BLACKMARKS -- Step 8A: Owner Signature + Class-to-Bit Encoding")
    print("=" * 68)

    assert_clean_model_intact("Step 8A start")
    device = get_device()
    set_seed(seed)

    model = build_model(num_classes=10, dropout_rate=0.5).to(device)
    meta = load_state_dict_checkpoint(model, CLEAN_CHECKPOINT, device)
    print(f"[Model] clean_model.pt loaded (epoch {meta.get('epoch')}, "
          f"val_acc {meta.get('val_acc')})")

    signature = derive_owner_signature(length=length, seed=seed, owner_id=owner_id)
    sig_str = serialize_signature(signature)
    print(f"[Signature] length={length}  bits={sig_str}")
    assert np.array_equal(deserialize_signature(sig_str), signature), "sig round-trip failed"

    print(f"[Encoding] estimating per-class mean features from {sample} train images...")
    class_means = compute_class_mean_features(model, seed=seed, sample=sample, device=device)
    mapping = build_class_bit_mapping(class_means, seed=seed)
    print(f"[Encoding] method={mapping['method']}  "
          f"bit0={mapping['bit0_classes']}  bit1={mapping['bit1_classes']}")

    # Feasibility: every signature bit must have >=1 class available to target.
    have_bits = set(mapping["class_to_bit_index"])
    need_bits = set(int(b) for b in signature)
    assert need_bits <= have_bits, "signature needs a bit value no class encodes"

    payload = {
        "step": "Step 8A -- Owner signature + class-to-bit encoding",
        "seed": seed,
        "owner_id": owner_id,
        "clean_checkpoint": str(CLEAN_CHECKPOINT),
        "clean_checkpoint_sha256": assert_clean_model_intact("Step 8A end"),
        "signature": {
            "length": int(length),
            "bits": sig_str,
            "bit_list": signature.tolist(),
            "derivation": "sha256('<owner_id>|<seed>')[:length] bits",
        },
        "class_encoding": mapping,
        "feature_extraction": {
            "source": "input to final nn.Linear (256-d penultimate features)",
            "images_sampled": int(sample),
            "image_source": "CIFAR-10 training split (test set untouched)",
            "normalisation": {"mean": list(CIFAR10_MEAN), "std": list(CIFAR10_STD)},
        },
    }
    save_json(payload, out_path)
    print("=" * 68)
    print(f"Step 8A complete -> {out_path.relative_to(PROJECT_ROOT)}")
    print("=" * 68)
    return payload


def parse_args():
    p = argparse.ArgumentParser(description="BlackMarks Step 8A -- encoding")
    p.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    p.add_argument("--owner-id", type=str, default=DEFAULT_OWNER_ID)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(length=a.length, seed=a.seed, sample=a.sample, owner_id=a.owner_id)
