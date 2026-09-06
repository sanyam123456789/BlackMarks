"""
BlackMarks -- Step 8D / 8F: Black-Box Ownership Verification

The verifier treats the model as an oracle exposing ONLY:

    predict(images01) -> class labels

No gradients, weights, activations, layer access or training state are used.
Given the watermark key images it:

    predicted classes
      -> class-to-bit decoding  (Step 8A encoding)
      -> per-position majority vote
      -> recovered signature
      -> Hamming distance / bit-error rate (BER) vs. the owner signature
      -> ownership decision (BER <= threshold)

Entry points
------------
  python src/classifier/verify.py                 # Step 8D: verify blackmarks_model
  python src/classifier/verify.py --false-positive # Step 8F: clean vs blackmarks

Outputs
-------
  artifacts/metrics/step8_verification.json     (Step 8D)
  artifacts/metrics/step8_false_positive.json   (Step 8F)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.classifier.bm_common import (
    CIFAR10_MEAN, CIFAR10_STD, BLACKMARKS_CHECKPOINT, CLEAN_CHECKPOINT, METRICS_DIR,
    PROJECT_ROOT, assert_clean_model_intact, bit_error_rate, bits_to_string,
    get_device, hamming_distance, load_state_dict_checkpoint, majority_vote_bits,
    save_json, set_seed,
)
from src.classifier.encoding import ENCODING_JSON, decode_classes_to_bits, deserialize_signature
from src.classifier.keygen import EMBED_KEYS_PT, VERIFY_KEYS_PT
from src.classifier.model import build_model

VERIFICATION_JSON   = METRICS_DIR / "step8_verification.json"
FALSE_POSITIVE_JSON  = METRICS_DIR / "step8_false_positive.json"

# Default ownership threshold. Rationale and the unmarked-model null distribution
# that justifies it are produced by Step 9G (threshold_analysis.py); this constant
# is the pre-registered hypothesis, deliberately well below chance (0.5).
DEFAULT_BER_THRESHOLD = 0.25


# ===========================================================================
# Black-box oracle
# ===========================================================================

class BlackBoxModel:
    """
    Query-only wrapper around a checkpoint. The ONLY public method is
    ``predict``. Internals (weights, gradients, layers) are not exposed.
    """

    def __init__(self, checkpoint_path, device=None, num_classes: int = 10):
        self._device = device or torch.device("cpu")
        model = build_model(num_classes=num_classes, dropout_rate=0.5).to(self._device)
        load_state_dict_checkpoint(model, checkpoint_path, self._device)
        model.eval()
        mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1).to(self._device)
        std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1).to(self._device)
        self._forward = lambda x01: model((x01.to(self._device) - mean) / std)
        self._n_queries = 0

    @property
    def n_queries(self) -> int:
        return self._n_queries

    def predict(self, images01) -> np.ndarray:
        """images01: (N,3,32,32) float tensor/array in [0,1] -> int class labels."""
        if not torch.is_tensor(images01):
            images01 = torch.as_tensor(np.asarray(images01), dtype=torch.float32)
        images01 = images01.float()
        if images01.ndim == 3:
            images01 = images01.unsqueeze(0)
        with torch.no_grad():
            preds = self._forward(images01).argmax(1).cpu().numpy().astype(np.int64)
        self._n_queries += int(images01.shape[0])
        return preds


class MockPredictAPI:
    """Test double: wraps any ``images01 -> class-index`` callable as a .predict oracle."""

    def __init__(self, fn):
        self._fn = fn
        self._n_queries = 0

    @property
    def n_queries(self) -> int:
        return self._n_queries

    def predict(self, images01) -> np.ndarray:
        images01 = torch.as_tensor(np.asarray(images01), dtype=torch.float32)
        if images01.ndim == 3:
            images01 = images01.unsqueeze(0)
        out = np.asarray(self._fn(images01), dtype=np.int64).ravel()
        self._n_queries += int(images01.shape[0])
        return out


# ===========================================================================
# Recovery / scoring
# ===========================================================================

def recover_signature(pred_classes, positions, class_to_bit_index, length: int):
    """Predicted classes -> per-key bits -> per-position majority vote -> signature."""
    per_key_bits = decode_classes_to_bits(pred_classes, class_to_bit_index)
    recovered, detail = majority_vote_bits(per_key_bits, positions, length)
    return recovered, per_key_bits, detail


def score_keyset(oracle, bundle: dict, class_to_bit_index, signature) -> dict:
    """
    Run the black-box verification pipeline for one key bundle.
    ``oracle`` must expose ``predict(images01) -> class labels``.
    """
    images01 = bundle["images01"]
    positions = np.asarray(bundle["signature_positions"]).astype(np.int64)
    targets = np.asarray(bundle["target_classes"]).astype(np.int64)
    K = len(signature)

    pred = oracle.predict(images01)
    recovered, per_key_bits, detail = recover_signature(pred, positions, class_to_bit_index, K)

    ham = hamming_distance(signature, recovered)
    ber = bit_error_rate(signature, recovered)
    target_match = float(np.mean(pred == targets))
    desired_bits = np.asarray(signature, dtype=np.int64)[positions]
    per_key_bit_match = float(np.mean(per_key_bits == desired_bits))

    return {
        "n_keys": int(len(pred)),
        "signature_length": int(K),
        "owner_signature": bits_to_string(signature),
        "recovered_signature": bits_to_string(recovered),
        "hamming_distance": int(ham),
        "bit_error_rate": round(float(ber), 6),
        "per_key_bit_match_rate": round(per_key_bit_match, 6),
        "target_class_match_rate": round(target_match, 6),
        "per_position": detail,
        "predicted_classes": pred.tolist(),
    }


def ownership_decision(ber: float, threshold: float = DEFAULT_BER_THRESHOLD) -> bool:
    return bool(ber <= threshold)


# ===========================================================================
# Loaders
# ===========================================================================

def _load_encoding():
    if not ENCODING_JSON.exists():
        raise RuntimeError(f"missing {ENCODING_JSON} -- run Step 8A first")
    enc = json.loads(ENCODING_JSON.read_text(encoding="utf-8"))
    signature = deserialize_signature(enc["signature"]["bits"])
    cbi = enc["class_encoding"]["class_to_bit_index"]
    return enc, signature, cbi


def _load_bundle(path):
    if not Path(path).exists():
        raise RuntimeError(f"missing key bundle {path} -- run Step 8B (keygen.py) first")
    return torch.load(path, map_location="cpu", weights_only=False)


# ===========================================================================
# Step 8D -- verify the watermarked model
# ===========================================================================

def run(threshold: float = DEFAULT_BER_THRESHOLD, checkpoint=BLACKMARKS_CHECKPOINT) -> dict:
    print("=" * 68)
    print("BLACKMARKS -- Step 8D: Black-Box Ownership Verification")
    print("=" * 68)
    assert_clean_model_intact("Step 8D")
    set_seed(42)
    device = get_device()

    if not Path(checkpoint).exists():
        raise RuntimeError(f"missing {checkpoint} -- run Step 8C (embed.py) first")

    enc, signature, cbi = _load_encoding()
    oracle = BlackBoxModel(checkpoint, device=device)

    results = {}
    for role, path in (("embedding_keys", EMBED_KEYS_PT), ("held_out_verification_keys", VERIFY_KEYS_PT)):
        bundle = _load_bundle(path)
        r = score_keyset(oracle, bundle, cbi, signature)
        r["ownership_decision"] = ownership_decision(r["bit_error_rate"], threshold)
        results[role] = r
        print(f"[{role}] BER={r['bit_error_rate']:.4f}  Hamming={r['hamming_distance']}/{r['signature_length']}  "
              f"target-match={r['target_class_match_rate']*100:.1f}%  decision={'OWNER' if r['ownership_decision'] else 'not owner'}")

    payload = {
        "step": "Step 8D -- black-box ownership verification",
        "interface": "predict(images01) -> class labels  (no gradients/weights/activations)",
        "model_under_test": str(checkpoint),
        "encoding_source": str(ENCODING_JSON),
        "owner_signature": enc["signature"]["bits"],
        "ber_threshold": threshold,
        "threshold_rationale": "pre-registered; validated against unmarked null distribution in Step 9G",
        "total_black_box_queries": oracle.n_queries,
        "results": results,
        "clean_checkpoint_sha256": assert_clean_model_intact("Step 8D end"),
    }
    save_json(payload, VERIFICATION_JSON)
    print("=" * 68)
    print(f"Step 8D complete -> {VERIFICATION_JSON.relative_to(PROJECT_ROOT)}")
    print("=" * 68)
    return payload


# ===========================================================================
# Step 8F -- clean-model false positive
# ===========================================================================

def run_false_positive(threshold: float = DEFAULT_BER_THRESHOLD) -> dict:
    print("=" * 68)
    print("BLACKMARKS -- Step 8F: Clean-Model False-Positive Check")
    print("=" * 68)
    assert_clean_model_intact("Step 8F")
    set_seed(42)
    device = get_device()
    enc, signature, cbi = _load_encoding()

    verify_bundle = _load_bundle(VERIFY_KEYS_PT)
    embed_bundle = _load_bundle(EMBED_KEYS_PT)

    models = {"clean_model": CLEAN_CHECKPOINT}
    if Path(BLACKMARKS_CHECKPOINT).exists():
        models["blackmarks_model"] = BLACKMARKS_CHECKPOINT

    out = {}
    for name, ckpt in models.items():
        oracle = BlackBoxModel(ckpt, device=device)
        held = score_keyset(oracle, verify_bundle, cbi, signature)
        emb = score_keyset(oracle, embed_bundle, cbi, signature)
        held["ownership_decision"] = ownership_decision(held["bit_error_rate"], threshold)
        emb["ownership_decision"] = ownership_decision(emb["bit_error_rate"], threshold)
        out[name] = {"held_out_verification_keys": held, "embedding_keys": emb}
        print(f"[{name:18s}] held-out BER={held['bit_error_rate']:.4f}  "
              f"decision={'OWNER' if held['ownership_decision'] else 'not owner'}")

    payload = {
        "step": "Step 8F -- clean-model false-positive check",
        "note": "Same verifier + same keys + same threshold applied to the unmarked "
                "clean baseline and the watermarked model. Threshold is NOT tuned to "
                "make the watermarked model pass (see Step 9G).",
        "ber_threshold": threshold,
        "owner_signature": enc["signature"]["bits"],
        "models": out,
        "clean_checkpoint_sha256": assert_clean_model_intact("Step 8F end"),
    }
    save_json(payload, FALSE_POSITIVE_JSON)
    print("=" * 68)
    print(f"Step 8F complete -> {FALSE_POSITIVE_JSON.relative_to(PROJECT_ROOT)}")
    print("=" * 68)
    return payload


def parse_args():
    p = argparse.ArgumentParser(description="BlackMarks Step 8D/8F -- black-box verification")
    p.add_argument("--threshold", type=float, default=DEFAULT_BER_THRESHOLD)
    p.add_argument("--false-positive", action="store_true",
                   help="run Step 8F (clean vs watermarked) instead of Step 8D")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.false_positive:
        run_false_positive(threshold=a.threshold)
    else:
        run(threshold=a.threshold)
