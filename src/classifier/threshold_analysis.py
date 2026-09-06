"""
BlackMarks -- Step 9G: Ownership Threshold Analysis

Derives a defensible BER decision threshold from the *unmarked* null distribution
(Step 9E) rather than by tuning on the watermarked model. Reports, for a sweep of
candidate thresholds, the false-positive rate on independent unmarked models and
whether the Step 8 watermarked model (held-out keys) would still be recognised.

Inputs (read-only):
  artifacts/metrics/step9_uniqueness.json        (unmarked BER samples)
  artifacts/metrics/step8_verification.json       (watermarked held-out BER)
  artifacts/metrics/step8_false_positive.json     (clean vs watermarked)

Output:
  artifacts/metrics/step9_threshold.json

Usage:  python src/classifier/threshold_analysis.py
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

from src.classifier.bm_common import (
    METRICS_DIR, PROJECT_ROOT, assert_clean_model_intact, save_json, suffixed_path,
)

THRESHOLD_JSON = METRICS_DIR / "step9_threshold.json"


def _load(name):
    p = METRICS_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def run(out_suffix: str | None = None) -> dict:
    print("=" * 68)
    print("BLACKMARKS -- Step 9G: Ownership Threshold Analysis")
    print("=" * 68)
    assert_clean_model_intact("Step 9G")

    # 9G consumes the 9E null distribution: with a suffix it reads the matching
    # step9_uniqueness_<suffix>.json and writes step9_threshold_<suffix>.json.
    # The Step 8 inputs are shared artifacts and always read unsuffixed.
    uniq_name = suffixed_path("step9_uniqueness.json", out_suffix).name
    out_json = suffixed_path(THRESHOLD_JSON, out_suffix)

    uniq = _load(uniq_name)
    verif = _load("step8_verification.json")
    fp = _load("step8_false_positive.json")

    if uniq is None:
        raise RuntimeError(f"missing {uniq_name} -- run Step 9E first")

    unmarked_ber = np.array([m["held_out_ber"] for m in uniq["independent_unmarked_models"]],
                            dtype=float)
    n = len(unmarked_ber)
    mu, sd = float(unmarked_ber.mean()), float(unmarked_ber.std(ddof=0))
    mn = float(unmarked_ber.min())

    wm_ber = None
    if verif:
        wm_ber = verif["results"]["held_out_verification_keys"]["bit_error_rate"]

    # Candidate rules
    candidates = {
        "fixed_0.25 (pre-registered)": 0.25,
        "unmarked_min_minus_0.05": round(max(0.0, mn - 0.05), 4),
        "unmarked_mean_minus_3std": round(max(0.0, mu - 3 * sd), 4),
        "midpoint(wm, unmarked_mean)": (round((wm_ber + mu) / 2, 4)
                                        if wm_ber is not None else None),
    }

    sweep = []
    for t in [round(x, 3) for x in np.arange(0.0, 0.501, 0.025)]:
        fp_count = int((unmarked_ber <= t).sum())
        sweep.append({
            "threshold": t,
            "unmarked_false_positive_count": fp_count,
            "unmarked_false_positive_rate": round(fp_count / n, 4),
            "watermarked_recognised": (None if wm_ber is None else bool(wm_ber <= t)),
        })

    # Recommendation: largest threshold with zero unmarked FP, capped at 0.25,
    # and still recognising the watermarked model.
    zero_fp = [s["threshold"] for s in sweep
               if s["unmarked_false_positive_rate"] == 0.0
               and (s["watermarked_recognised"] in (True, None))]
    recommended = min(max(zero_fp) if zero_fp else 0.0, 0.25)

    print(f"[unmarked null] n={n}  BER mean={mu:.4f} std={sd:.4f} min={mn:.4f}")
    if wm_ber is not None:
        print(f"[watermarked ] held-out BER={wm_ber:.4f}")
    print(f"[recommended ] BER threshold = {recommended}")

    payload = {
        "step": "Step 9G -- ownership threshold analysis",
        "method": "threshold derived from the unmarked null distribution (Step 9E), "
                  "NOT tuned on the watermarked model",
        "unmarked_null_distribution": {
            "n_models": n, "ber_mean": round(mu, 6), "ber_std": round(sd, 6),
            "ber_min": round(mn, 6), "ber_max": round(float(unmarked_ber.max()), 6),
            "samples": unmarked_ber.round(6).tolist(),
        },
        "watermarked_held_out_ber": wm_ber,
        "clean_model_degenerate_ber": (
            fp["models"]["clean_model"]["held_out_verification_keys"]["bit_error_rate"]
            if fp and "clean_model" in fp.get("models", {}) else None),
        "candidate_thresholds": candidates,
        "threshold_sweep": sweep,
        "recommended_threshold": recommended,
        "recommendation_rationale": (
            "Largest threshold (capped at the pre-registered 0.25) that yields zero "
            "false positives on the independent unmarked models while still "
            "recognising the Step 8 watermarked model on held-out keys. The clean "
            "baseline is excluded from this calibration because the owner keys were "
            "crafted on it (degenerate case)."),
    }
    save_json(payload, out_json)
    print("=" * 68)
    print(f"Step 9G complete -> {out_json.relative_to(PROJECT_ROOT)}")
    print("=" * 68)
    return payload


def parse_args():
    p = argparse.ArgumentParser(description="BlackMarks Step 9G -- ownership threshold analysis")
    p.add_argument("--suffix", type=str, default=None,
                   help="read step9_uniqueness_<suffix>.json and write "
                        "step9_threshold_<suffix>.json; leave unset for the original behaviour")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(out_suffix=a.suffix)
