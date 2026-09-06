"""
BlackMarks -- Step 8 orchestrator

Runs the full multi-bit BlackMarks core end to end and writes a single summary:

  8A encoding.run()          -> artifacts/metrics/encoding.json
  8B keygen.run()            -> artifacts/keys/*.pt , key_generation.json
  8C embed.embed()           -> artifacts/checkpoints/blackmarks_model.pt , step8_embedding.json
  8D verify.run()            -> artifacts/metrics/step8_verification.json
  8E (held-out)              -> included in 8D output (embedding vs held-out key sets)
  8F verify.run_false_positive() -> artifacts/metrics/step8_false_positive.json
  summary                    -> artifacts/metrics/step8_summary.json

Usage:  python src/classifier/run_step8.py --epochs 10 --lambda-wm 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.classifier import embed as embed_mod
from src.classifier import encoding as encoding_mod
from src.classifier import keygen as keygen_mod
from src.classifier import verify as verify_mod
from src.classifier.bm_common import METRICS_DIR, PROJECT_ROOT, assert_clean_model_intact, save_json

SUMMARY_JSON = METRICS_DIR / "step8_summary.json"


def run(length=16, seed=42, sample=5000, keys_per_bit=6, epsilon=8/255, alpha=2/255,
        iters=40, epochs=10, lambda_wm=1.0, lr=1e-3, threshold=0.25,
        skip_keygen=False) -> dict:
    hb = assert_clean_model_intact("Step 8 orchestrator start")

    a = encoding_mod.run(length=length, seed=seed, sample=sample)
    if not skip_keygen:
        b = keygen_mod.run(keys_per_bit=keys_per_bit, epsilon=epsilon, alpha=alpha,
                           iters=iters, seed=seed)
    c = embed_mod.embed(epochs=epochs, lr=lr, lambda_wm=lambda_wm, seed=seed)
    d = verify_mod.run(threshold=threshold)
    f = verify_mod.run_false_positive(threshold=threshold)

    held = d["results"]["held_out_verification_keys"]
    summary = {
        "step": "Step 8 -- multi-bit BlackMarks core (orchestrated)",
        "owner_signature": a["signature"]["bits"],
        "class_encoding_method": a["class_encoding"]["method"],
        "class_to_bit": a["class_encoding"]["class_to_bit"],
        "embedding_config": c["config"],
        "watermarked_model": c["watermarked_model"]["checkpoint"],
        "normal_test": c["watermarked_model"]["normal_test"],
        "embedding_key_verification": d["results"]["embedding_keys"],
        "held_out_key_verification": held,
        "false_positive_check": {
            k: v["held_out_verification_keys"]["bit_error_rate"]
            for k, v in f["models"].items()
        },
        "ber_threshold": threshold,
        "clean_checkpoint_sha256_before": hb,
        "clean_checkpoint_sha256_after": assert_clean_model_intact("Step 8 orchestrator end"),
    }
    save_json(summary, SUMMARY_JSON)
    print("\n" + "#" * 68)
    print(f"STEP 8 COMPLETE -> {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")
    print(f"  signature      : {summary['owner_signature']}")
    print(f"  test accuracy  : {summary['normal_test']['test_accuracy']}%")
    print(f"  held-out BER   : {held['bit_error_rate']}  "
          f"(Hamming {held['hamming_distance']}/{held['signature_length']})  "
          f"decision {'OWNER' if held['ownership_decision'] else 'NOT OWNER'}")
    print("#" * 68)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="BlackMarks Step 8 orchestrator")
    p.add_argument("--length", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample", type=int, default=5000)
    p.add_argument("--keys-per-bit", type=int, default=6)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lambda-wm", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--skip-keygen", action="store_true",
                   help="reuse existing artifacts/keys/*.pt")
    return p.parse_args()


if __name__ == "__main__":
    x = parse_args()
    run(length=x.length, seed=x.seed, sample=x.sample, keys_per_bit=x.keys_per_bit,
        epochs=x.epochs, lambda_wm=x.lambda_wm, lr=x.lr, threshold=x.threshold,
        skip_keygen=x.skip_keygen)
