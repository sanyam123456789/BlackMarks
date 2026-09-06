"""
BlackMarks -- Shared utilities for Step 8 (multi-bit BlackMarks core) and Step 9
(research-grade evaluation).

This module is ADDITIVE. It introduces no changes to Steps 1-7. Existing scripts
(``train.py``, ``evaluate.py``, ``watermark.py``) keep their own local copies of
helpers such as ``set_seed`` / ``get_device`` -- those are intentionally left
untouched. New Step 8/9 code imports from here instead of duplicating logic.

Contents
--------
- Reproducibility : ``set_seed``
- Device          : ``get_device``
- Hashing         : ``sha256_file``, ``CLEAN_MODEL_SHA256``, ``assert_clean_model_intact``
- Checkpoints     : ``load_state_dict_checkpoint``
- Bit maths       : ``bits_to_string``, ``string_to_bits``, ``hamming_distance``,
                    ``bit_error_rate``, ``majority_vote_bits``
- JSON            : ``save_json`` (writes atomically, validates round-trip)
- Paths           : ``PROJECT_ROOT`` and the standard artifact directories
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project paths (this file lives at src/classifier/bm_common.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR         = PROJECT_ROOT / "data"
ARTIFACTS_DIR    = PROJECT_ROOT / "artifacts"
CHECKPOINT_DIR   = ARTIFACTS_DIR / "checkpoints"
METRICS_DIR      = ARTIFACTS_DIR / "metrics"
PLOTS_DIR        = ARTIFACTS_DIR / "plots"
KEYS_DIR         = ARTIFACTS_DIR / "keys"          # NEW: generated adversarial key tensors

CLEAN_CHECKPOINT       = CHECKPOINT_DIR / "clean_model.pt"
STEP7_WATERMARKED      = CHECKPOINT_DIR / "watermarked_model.pt"
BLACKMARKS_CHECKPOINT  = CHECKPOINT_DIR / "blackmarks_model.pt"

# Immutable clean baseline fingerprint (Step 8/9 safety contract, RULE 6).
CLEAN_MODEL_SHA256 = "d786b7f0f8d13365b5ebb044154a870bc3ef8be036a17a3ef4a69d517cc6c01c"

# Standard CIFAR-10 normalisation constants (mirrors src/classifier/data.py).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Seed python / numpy / torch RNGs and force deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU (prints which)."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[Device] CUDA -- {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[Device] CPU")
    return dev


# ---------------------------------------------------------------------------
# Hashing / integrity
# ---------------------------------------------------------------------------

def sha256_file(path: os.PathLike | str) -> str:
    """Return the hex SHA-256 digest of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_clean_model_intact(where: str = "") -> str:
    """
    Verify ``artifacts/checkpoints/clean_model.pt`` still matches the frozen
    fingerprint. Raises RuntimeError on mismatch (RULE 6 stop condition).

    Returns the digest string on success.
    """
    if not CLEAN_CHECKPOINT.exists():
        raise RuntimeError(f"clean_model.pt missing at {CLEAN_CHECKPOINT}")
    digest = sha256_file(CLEAN_CHECKPOINT)
    tag = f" ({where})" if where else ""
    if digest != CLEAN_MODEL_SHA256:
        raise RuntimeError(
            f"clean_model.pt SHA-256 MISMATCH{tag}!\n"
            f"  expected {CLEAN_MODEL_SHA256}\n"
            f"  actual   {digest}\n"
            f"STOP -- the immutable clean baseline was altered."
        )
    print(f"[Integrity] clean_model.pt SHA-256 OK{tag}: {digest}")
    return digest


# ---------------------------------------------------------------------------
# Checkpoint loading (read-only; never writes)
# ---------------------------------------------------------------------------

def load_state_dict_checkpoint(model: torch.nn.Module, path: os.PathLike | str,
                               device: torch.device) -> dict:
    """
    Load ``model_state_dict`` from a .pt checkpoint into ``model`` and return the
    remaining metadata dict. Mirrors the semantics of the loaders already used in
    train.py / evaluate.py / watermark.py.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        return {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    # bare state_dict fallback
    model.load_state_dict(ckpt)
    return {}


# ---------------------------------------------------------------------------
# Bit maths -- signatures, Hamming distance, bit-error rate, majority vote
# ---------------------------------------------------------------------------

def _as_bit_array(bits: Iterable[int]) -> np.ndarray:
    arr = np.asarray(list(bits), dtype=np.int64).ravel()
    if not np.isin(arr, (0, 1)).all():
        raise ValueError("bit array must contain only 0/1 values")
    return arr


def bits_to_string(bits: Iterable[int]) -> str:
    """[1,0,1,1] -> '1011'."""
    return "".join(str(int(b)) for b in _as_bit_array(bits))


def string_to_bits(s: str) -> np.ndarray:
    """'1011' -> np.array([1,0,1,1])."""
    s = s.strip()
    if not set(s) <= {"0", "1"} or len(s) == 0:
        raise ValueError(f"not a binary string: {s!r}")
    return np.array([int(c) for c in s], dtype=np.int64)


def hamming_distance(a: Iterable[int], b: Iterable[int]) -> int:
    """Number of differing positions between two equal-length bit sequences."""
    aa, bb = _as_bit_array(a), _as_bit_array(b)
    if aa.shape != bb.shape:
        raise ValueError(f"length mismatch: {aa.shape} vs {bb.shape}")
    return int(np.sum(aa != bb))


def bit_error_rate(a: Iterable[int], b: Iterable[int]) -> float:
    """Hamming distance normalised by signature length -> [0, 1]."""
    aa = _as_bit_array(a)
    return hamming_distance(a, b) / len(aa)


def majority_vote_bits(per_key_bits: Sequence[int], positions: Sequence[int],
                       length: int) -> tuple[np.ndarray, list[dict]]:
    """
    Aggregate multiple decoded bits per signature position by majority vote.

    Args:
        per_key_bits: decoded bit for every key (len == n_keys).
        positions:    signature position each key belongs to (0..length-1).
        length:       signature length.

    Returns:
        (recovered_bits[length], per_position_detail).
        Ties (equal 0/1 votes, e.g. even key count) resolve to bit 1 and are
        flagged with ``"tie": true``.
    """
    per_key_bits = _as_bit_array(per_key_bits)
    positions = np.asarray(list(positions), dtype=np.int64)
    if per_key_bits.shape != positions.shape:
        raise ValueError("per_key_bits and positions must align")

    recovered = np.zeros(length, dtype=np.int64)
    detail: list[dict] = []
    for pos in range(length):
        mask = positions == pos
        votes = per_key_bits[mask]
        n1 = int(votes.sum())
        n0 = int(votes.size - n1)
        if votes.size == 0:
            bit, tie = 0, False
        else:
            tie = (n0 == n1)
            bit = 1 if n1 >= n0 else 0
        recovered[pos] = bit
        detail.append({"position": pos, "n_keys": int(votes.size),
                       "votes_0": n0, "votes_1": n1, "bit": int(bit), "tie": bool(tie)})
    return recovered, detail


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (torch.device,)):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def suffixed_path(path: os.PathLike | str, suffix: str | None = None) -> Path:
    """
    Insert ``_<suffix>`` before the file extension so an alternate run (e.g. a
    Colab/GPU re-run of Step 9) writes a *separate* file and never clobbers a
    committed result:

        suffixed_path("step9_uniqueness.json", "colab")
            -> Path(".../step9_uniqueness_colab.json")

    ``suffix`` of ``None`` / ``""`` returns ``Path(path)`` unchanged, so the
    default behaviour of every caller is byte-for-byte identical. A leading
    underscore in ``suffix`` is accepted and not duplicated.
    """
    p = Path(path)
    if not suffix:
        return p
    s = str(suffix)
    s = s if s.startswith("_") else f"_{s}"
    return p.with_name(p.stem + s + p.suffix)


def save_json(payload: dict, path: os.PathLike | str, *, verbose: bool = True) -> Path:
    """
    Write ``payload`` to ``path`` as indented UTF-8 JSON and verify it reloads.
    Creates parent directories. Never overwrites silently outside the caller's
    intent -- the caller chooses the filename.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, default=_json_default)
    json.loads(text)  # round-trip validation
    path.write_text(text, encoding="utf-8")
    if verbose:
        try:
            rel = path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel = path
        print(f"  [JSON] {rel}")
    return path
