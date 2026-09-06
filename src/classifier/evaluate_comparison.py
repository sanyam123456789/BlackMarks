"""
BlackMarks -- Step 9A: Three-Way Comparison

Side-by-side, single table:
  A. clean baseline            -> artifacts/checkpoints/clean_model.pt
  B. Step 7 patch watermark    -> artifacts/checkpoints/watermarked_model.pt
  C. Step 8 multi-bit BlackMarks -> artifacts/checkpoints/blackmarks_model.pt

For each: normal CIFAR-10 test loss/acc, black-box watermark metrics on the
Step 8 key sets (BER / Hamming / target-match / ownership decision), parameter
count, and the recorded embedding configuration.

Read-only. Writes only artifacts/metrics/comparison.json.

Usage:  python src/classifier/evaluate_comparison.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from src.classifier.bm_common import (
    BLACKMARKS_CHECKPOINT, CLEAN_CHECKPOINT, METRICS_DIR, PROJECT_ROOT, STEP7_WATERMARKED,
    assert_clean_model_intact, get_device, load_state_dict_checkpoint, save_json, set_seed,
    suffixed_path,
)
from src.classifier.bm_eval import full_report
from src.classifier.model import build_model, count_parameters

COMPARISON_JSON = METRICS_DIR / "comparison.json"


def _load_json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def run(threshold: float = 0.25, out_suffix: str | None = None) -> dict:
    out_json = suffixed_path(COMPARISON_JSON, out_suffix)
    print("=" * 68)
    print("BLACKMARKS -- Step 9A: clean vs Step 7 patch vs Step 8 multi-bit")
    print("=" * 68)
    assert_clean_model_intact("Step 9A")
    set_seed(42)
    device = get_device()

    total_params, _ = count_parameters(build_model())

    entries = []
    for label, ckpt in (
        ("A_clean_baseline", CLEAN_CHECKPOINT),
        ("B_step7_patch_watermark", STEP7_WATERMARKED),
        ("C_step8_multibit_blackmarks", BLACKMARKS_CHECKPOINT),
    ):
        if not Path(ckpt).exists():
            print(f"[skip] {label}: {ckpt} not found")
            entries.append({"label": label, "checkpoint": str(ckpt), "status": "MISSING"})
            continue
        rep = full_report(ckpt, device, label, threshold=threshold)
        rep["parameters"] = total_params
        entries.append(rep)
        v = rep["watermark"]["held_out_verification_keys"]
        print(f"[{label:32s}] test_acc={rep['normal_test']['test_accuracy']:.2f}%  "
              f"heldout_BER={v['bit_error_rate']:.4f}  "
              f"decision={'OWNER' if v['ownership_decision'] else 'not owner'}")

    payload = {
        "step": "Step 9A -- three-way comparison",
        "ber_threshold": threshold,
        "notes": {
            "watermark_metrics": "computed with the Step 8 multi-bit key sets + encoding "
                                 "for ALL models, so the clean / Step 7 rows show how the "
                                 "Step 8 owner keys behave on non-Step-8 models.",
            "step7_reference_metrics": "Step 7's own single-target patch watermark accuracy "
                                       "is reported separately from watermark_evaluation.json.",
        },
        "step7_own_watermark_evaluation": _load_json(_ROOT / "artifacts/metrics/watermark_evaluation.json"),
        "step8_embedding_config": (_load_json(_ROOT / "artifacts/metrics/step8_embedding.json") or {}).get("config"),
        "models": entries,
        "clean_checkpoint_sha256": assert_clean_model_intact("Step 9A end"),
    }
    save_json(payload, out_json)
    print("=" * 68)
    print(f"Step 9A complete -> {out_json.relative_to(PROJECT_ROOT)}")
    print("=" * 68)
    return payload


def parse_args():
    p = argparse.ArgumentParser(description="BlackMarks Step 9A -- three-way comparison")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--suffix", type=str, default=None,
                   help="write comparison_<suffix>.json instead of the default name; "
                        "leave unset for the original behaviour")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(threshold=a.threshold, out_suffix=a.suffix)
