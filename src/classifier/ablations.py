"""
BlackMarks -- Step 9F: Ablations

Meaningful ablations of the Step 8 multi-bit embedding, each reported with normal
test accuracy, held-out BER and the ownership decision.

  1. watermark-loss weight lambda_wm   -- full re-embed per value (default 0.5,1,2)
  2. embedding epochs                   -- FREE: read from the main run's per-epoch
                                          history in step8_embedding.json
  3. signature length K                 -- optional, re-runs encoding+keygen+embed
                                          (off by default; --sig-lengths to enable)

Each re-embed writes its own checkpoint / metrics under artifacts/ (never the
Step 8C originals). CPU budget: each lambda re-embed defaults to ``--epochs`` = 8.

Usage
-----
  python src/classifier/ablations.py
  python src/classifier/ablations.py --lambda-list 0.5,1.0,2.0 --epochs 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.classifier.bm_common import (
    CHECKPOINT_DIR, METRICS_DIR, PROJECT_ROOT, assert_clean_model_intact, save_json,
    suffixed_path,
)
from src.classifier import embed as embed_mod

ABLATIONS_JSON = METRICS_DIR / "step9_ablations.json"
STEP8_EMBED_JSON = METRICS_DIR / "step8_embedding.json"


def _main_run_lambda1_row():
    """Fold in the lambda_wm=1.0 result from the main Step 8C run (no extra compute)."""
    if not STEP8_EMBED_JSON.exists():
        return None
    d = json.loads(STEP8_EMBED_JSON.read_text(encoding="utf-8"))
    cfg = d.get("config", {})
    wm = d.get("watermarked_model", {})
    held = wm.get("held_out_key_verification") or {}
    return {
        "lambda_wm": cfg.get("lambda_wm"), "epochs": cfg.get("epochs"),
        "checkpoint": wm.get("checkpoint"), "metrics_json": str(STEP8_EMBED_JSON),
        "test_accuracy": (wm.get("normal_test") or {}).get("test_accuracy"),
        "delta_vs_clean_acc": (wm.get("normal_test") or {}).get("delta_vs_clean_acc"),
        "held_out_ber": held.get("bit_error_rate"),
        "held_out_hamming": held.get("hamming_distance"),
        "ownership_decision": held.get("ownership_decision"),
        "source": "main Step 8C run (reused, not re-embedded)",
    }


def _lambda_sweep(lambda_list, epochs):
    rows = []
    main_row = _main_run_lambda1_row()
    if main_row is not None:
        rows.append(main_row)
        lambda_list = [l for l in lambda_list if abs(l - 1.0) > 1e-9]
    for lam in lambda_list:
        tag = f"lam{lam:g}_e{epochs}"
        out_ckpt = CHECKPOINT_DIR / f"ablation_{tag}.pt"
        out_json = METRICS_DIR / f"ablation_{tag}.json"
        print(f"\n--- ablation lambda_wm={lam}  epochs={epochs} ---")
        rep = embed_mod.embed(epochs=epochs, lr=1e-3, lambda_wm=lam, seed=42,
                              out_ckpt=out_ckpt, out_json=out_json)
        wm = rep["watermarked_model"]
        held = wm["held_out_key_verification"] or {}
        rows.append({
            "lambda_wm": lam, "epochs": epochs,
            "checkpoint": str(out_ckpt), "metrics_json": str(out_json),
            "test_accuracy": wm["normal_test"]["test_accuracy"],
            "delta_vs_clean_acc": wm["normal_test"]["delta_vs_clean_acc"],
            "held_out_ber": held.get("bit_error_rate"),
            "held_out_hamming": held.get("hamming_distance"),
            "ownership_decision": held.get("ownership_decision"),
        })
    return rows


def _epochs_ablation_from_history():
    if not STEP8_EMBED_JSON.exists():
        return {"status": "unavailable -- step8_embedding.json missing"}
    hist = json.loads(STEP8_EMBED_JSON.read_text(encoding="utf-8")).get("training_history", [])
    return [{
        "epoch": h["epoch"], "val_acc": h.get("val_acc"),
        "embed_key_ber": h.get("embed_key_ber"),
        "held_out_ber": h.get("verify_key_ber"),
        "held_out_target_match": h.get("verify_key_target_match"),
    } for h in hist]


def run(lambda_list=None, epochs: int = 8, sig_lengths=None,
        out_suffix: str | None = None) -> dict:
    lambda_list = lambda_list or [0.5, 2.0]
    out_json = suffixed_path(ABLATIONS_JSON, out_suffix)
    print("=" * 68)
    print("BLACKMARKS -- Step 9F: Ablations")
    print("=" * 68)
    hash_before = assert_clean_model_intact("Step 9F start")

    lam_rows = _lambda_sweep(lambda_list, epochs)
    epoch_rows = _epochs_ablation_from_history()

    siglen_rows = None
    if sig_lengths:
        siglen_rows = _sig_length_sweep(sig_lengths, epochs)

    payload = {
        "step": "Step 9F -- ablations",
        "clean_checkpoint_sha256_before": hash_before,
        "clean_checkpoint_sha256_after": assert_clean_model_intact("Step 9F end"),
        "lambda_wm_sweep": {
            "note": "each row is an independent re-embed from clean_model.pt",
            "epochs_per_run": epochs, "rows": lam_rows,
        },
        "embedding_epochs_ablation": {
            "note": "per-epoch snapshot from the main Step 8C run (no extra compute)",
            "rows": epoch_rows,
        },
        "signature_length_sweep": siglen_rows or "NOT RUN -- enable with --sig-lengths",
    }
    save_json(payload, out_json)
    print("=" * 68)
    print(f"Step 9F complete -> {out_json.relative_to(PROJECT_ROOT)}")
    print("=" * 68)
    return payload


def _sig_length_sweep(sig_lengths, epochs):
    """Optional: full encoding+keygen+embed at alternate signature lengths."""
    from src.classifier import encoding as enc_mod
    from src.classifier import keygen as keygen_mod
    rows = []
    for K in sig_lengths:
        print(f"\n--- ablation signature_length={K} ---")
        enc_path = METRICS_DIR / f"ablation_encoding_K{K}.json"
        enc_mod.run(length=K, seed=42, out_path=enc_path)
        # keygen/embed for alternate K would need parametrised paths; report the
        # encoding result and mark embed as a follow-up to keep compute bounded.
        rows.append({"signature_length": K, "encoding_json": str(enc_path),
                     "embed": "NOT RUN -- run keygen.py/embed.py against "
                              f"{enc_path.name} to complete"})
    return rows


def parse_args():
    p = argparse.ArgumentParser(description="BlackMarks Step 9F -- ablations")
    p.add_argument("--lambda-list", type=str, default=None, help="e.g. 0.5,1.0,2.0")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--sig-lengths", type=str, default=None, help="e.g. 8,24 (optional)")
    p.add_argument("--suffix", type=str, default=None,
                   help="write step9_ablations_<suffix>.json instead of the default name; "
                        "leave unset for the original behaviour")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    lam = [float(x) for x in a.lambda_list.split(",")] if a.lambda_list else None
    sl = [int(x) for x in a.sig_lengths.split(",")] if a.sig_lengths else None
    run(lambda_list=lam, epochs=a.epochs, sig_lengths=sl, out_suffix=a.suffix)
