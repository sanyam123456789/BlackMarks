"""
BlackMarks -- Step 9 orchestrator (research-grade evaluation)

Runs the Step 9 experiments in sequence, each guarded so one failure does not
abort the rest, and writes artifacts/metrics/step9_summary.json.

  9A evaluate_comparison.run()   -> comparison.json
  9B attack_finetune.run()       -> step9_finetune.json
  9C attack_pruning.run()        -> step9_pruning.json
  9D attack_overwrite.run()      -> step9_overwrite.json
  9E uniqueness.run()            -> step9_uniqueness.json
  9F ablations.run()             -> step9_ablations.json
  9G threshold_analysis.run()    -> step9_threshold.json   (needs 9E)

CPU budget knobs are passed through; defaults are the reduced settings documented
in each module. Use --only to run a subset, e.g. --only 9A,9C,9G.

Usage:
  python src/classifier/run_step9.py
  python src/classifier/run_step9.py --uniq-epochs 15 --uniq-models 3 --abl-epochs 8
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.classifier.bm_common import (
    METRICS_DIR, PROJECT_ROOT, assert_clean_model_intact, save_json, suffixed_path,
)

SUMMARY_JSON = METRICS_DIR / "step9_summary.json"


def _guard(tag, fn):
    print("\n" + "=" * 72 + f"\n### {tag}\n" + "=" * 72)
    t0 = time.time()
    try:
        fn()
        return {"status": "ok", "seconds": round(time.time() - t0, 1)}
    except Exception as exc:  # noqa: BLE001 -- deliberately resilient orchestrator
        traceback.print_exc()
        return {"status": "FAILED", "error": repr(exc), "seconds": round(time.time() - t0, 1)}


def run(only=None, uniq_epochs=15, uniq_models=3, abl_epochs=8,
        ft_configs=None, ovr_epochs=8, threshold=0.25, suffix=None):
    from src.classifier import ablations as m9f
    from src.classifier import attack_finetune as m9b
    from src.classifier import attack_overwrite as m9d
    from src.classifier import attack_pruning as m9c
    from src.classifier import evaluate_comparison as m9a
    from src.classifier import threshold_analysis as m9g
    from src.classifier import uniqueness as m9e

    hb = assert_clean_model_intact("Step 9 orchestrator start")
    seeds = [1001 + 111 * i for i in range(uniq_models)]
    ftc = ft_configs or [(3, 1e-3), (5, 1e-2)]

    steps = {
        "9A_comparison":   lambda: m9a.run(threshold=threshold, out_suffix=suffix),
        "9B_finetune":     lambda: m9b.run(configs=ftc, threshold=threshold, out_suffix=suffix),
        "9C_pruning":      lambda: m9c.run(threshold=threshold, out_suffix=suffix),
        "9D_overwrite":    lambda: m9d.run(epochs=ovr_epochs, threshold=threshold, out_suffix=suffix),
        "9E_uniqueness":   lambda: m9e.run(seeds=seeds, epochs=uniq_epochs, threshold=threshold,
                                           out_suffix=suffix),
        "9F_ablations":    lambda: m9f.run(epochs=abl_epochs, out_suffix=suffix),
        "9G_threshold":    lambda: m9g.run(out_suffix=suffix),
    }
    if only:
        want = {s.strip().upper() for s in only.split(",")}
        steps = {k: v for k, v in steps.items() if k.split("_")[0].upper() in want}

    results = {}
    for tag, fn in steps.items():
        results[tag] = _guard(tag, fn)

    summary_json = suffixed_path(SUMMARY_JSON, suffix)
    summary = {
        "step": "Step 9 -- research-grade evaluation (orchestrated)",
        "threshold": threshold,
        "output_suffix": suffix,
        "config": {"uniq_epochs": uniq_epochs, "uniq_models": uniq_models,
                   "abl_epochs": abl_epochs, "ft_configs": ftc, "ovr_epochs": ovr_epochs},
        "results": results,
        "clean_checkpoint_sha256_before": hb,
        "clean_checkpoint_sha256_after": assert_clean_model_intact("Step 9 orchestrator end"),
    }
    save_json(summary, summary_json)
    print("\n" + "#" * 72)
    for k, v in results.items():
        print(f"  {k:18s} {v['status']:8s} {v['seconds']:>7.1f}s")
    print(f"-> {summary_json.relative_to(PROJECT_ROOT)}")
    print("#" * 72)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="BlackMarks Step 9 orchestrator")
    p.add_argument("--only", type=str, default=None, help="subset, e.g. 9A,9C,9G")
    p.add_argument("--uniq-epochs", type=int, default=15)
    p.add_argument("--uniq-models", type=int, default=3)
    p.add_argument("--abl-epochs", type=int, default=8)
    p.add_argument("--ovr-epochs", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--suffix", type=str, default=None,
                   help="tag every Step 9 output file with _<suffix> (e.g. --suffix colab) "
                        "so a re-run never clobbers the committed results; unset = originals")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(only=a.only, uniq_epochs=a.uniq_epochs, uniq_models=a.uniq_models,
        abl_epochs=a.abl_epochs, ovr_epochs=a.ovr_epochs, threshold=a.threshold,
        suffix=a.suffix)
