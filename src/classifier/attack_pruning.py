"""
BlackMarks -- Step 9C: Magnitude-Pruning Robustness

Global unstructured magnitude pruning: for each ratio r, the smallest-|w| fraction
r of all Conv2d/Linear weights (across the whole network) is zeroed. The pruned
weights are loaded into a fresh model and evaluated -- normal test accuracy and
black-box watermark BER / Hamming / ownership decision.

The Step 8 watermarked model AND (as a control) the clean baseline are pruned with
the same ratios, so we can see whether the *embedded* signature is more persistent
than the raw adversarial-key behaviour on the unmarked model.

Safety
------
* blackmarks_model.pt / clean_model.pt are only ever READ.
* Every pruned network is written to its own file:
      artifacts/checkpoints/pruned_<base>_r<ratio>.pt

Usage
-----
  python src/classifier/attack_pruning.py
  python src/classifier/attack_pruning.py --ratios 0.1,0.3,0.5,0.7,0.9
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.classifier.bm_common import (
    BLACKMARKS_CHECKPOINT, CHECKPOINT_DIR, CLEAN_CHECKPOINT, METRICS_DIR, PROJECT_ROOT,
    assert_clean_model_intact, get_device, load_state_dict_checkpoint, save_json, set_seed,
    suffixed_path,
)
from src.classifier.bm_eval import normal_test_metrics, watermark_metrics
from src.classifier.model import build_model

PRUNING_JSON = METRICS_DIR / "step9_pruning.json"
DEFAULT_RATIOS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def _global_magnitude_prune(state_dict: dict, ratio: float) -> dict:
    """Return a copy of state_dict with the global smallest-|w| ``ratio`` zeroed
    across all 2D+ weight tensors (conv / linear kernels)."""
    sd = copy.deepcopy(state_dict)
    weight_keys = [k for k, v in sd.items()
                   if k.endswith(".weight") and v.dim() >= 2]
    allw = torch.cat([sd[k].abs().flatten() for k in weight_keys])
    if ratio <= 0:
        return sd
    k_th = max(1, int(ratio * allw.numel()))
    thresh = torch.kthvalue(allw, k_th).values.item()
    for k in weight_keys:
        mask = sd[k].abs() > thresh
        sd[k] = sd[k] * mask
    return sd


def _actual_sparsity(state_dict: dict) -> float:
    weight_keys = [k for k, v in state_dict.items()
                   if k.endswith(".weight") and v.dim() >= 2]
    total = sum(state_dict[k].numel() for k in weight_keys)
    zeros = sum((state_dict[k] == 0).sum().item() for k in weight_keys)
    return zeros / total


def _prune_and_eval(base_ckpt, base_name, ratios, device, threshold, out_suffix=None):
    raw = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    base_sd = raw["model_state_dict"] if "model_state_dict" in raw else raw
    rows = []
    for r in ratios:
        pruned_sd = _global_magnitude_prune(base_sd, r)
        sparsity = _actual_sparsity(pruned_sd)
        out_ckpt = suffixed_path(CHECKPOINT_DIR / f"pruned_{base_name}_r{r:.2f}.pt", out_suffix)
        torch.save({"model_state_dict": pruned_sd, "architecture": "CompactCNN",
                    "num_classes": 10, "pruned_from": str(base_ckpt),
                    "requested_ratio": r, "actual_sparsity": round(sparsity, 6),
                    "attack": "Step 9C global magnitude pruning"}, out_ckpt)

        model = build_model().to(device)
        model.load_state_dict(pruned_sd)
        normal = normal_test_metrics(model, device)
        wm = watermark_metrics(out_ckpt, device, threshold=threshold)
        v = wm["held_out_verification_keys"]
        rows.append({
            "requested_ratio": r, "actual_sparsity": round(sparsity, 6),
            "checkpoint": str(out_ckpt),
            "test_accuracy": normal["test_accuracy"], "test_loss": normal["test_loss"],
            "embed_key_ber": wm["embedding_keys"]["bit_error_rate"],
            "held_out_ber": v["bit_error_rate"],
            "held_out_hamming": v["hamming_distance"],
            "held_out_target_match": v["target_class_match_rate"],
            "ownership_decision": v["ownership_decision"],
        })
        print(f"  [{base_name}] r={r:.2f} (sparsity {sparsity:.2f})  "
              f"acc={normal['test_accuracy']:.2f}%  heldout_BER={v['bit_error_rate']:.4f}  "
              f"{'OWNER' if v['ownership_decision'] else 'not owner'}")
    return rows


def run(ratios=None, threshold: float = 0.25, out_suffix: str | None = None) -> dict:
    ratios = ratios or DEFAULT_RATIOS
    out_json = suffixed_path(PRUNING_JSON, out_suffix)
    print("=" * 68)
    print("BLACKMARKS -- Step 9C: Magnitude-Pruning Robustness")
    print("=" * 68)
    assert_clean_model_intact("Step 9C start")
    set_seed(42)
    device = get_device()

    if not Path(BLACKMARKS_CHECKPOINT).exists():
        raise RuntimeError("missing blackmarks_model.pt -- run Step 8C first")

    print("[watermarked] pruning blackmarks_model.pt ...")
    wm_rows = _prune_and_eval(BLACKMARKS_CHECKPOINT, "blackmarks", ratios, device, threshold,
                              out_suffix=out_suffix)
    print("[control] pruning clean_model.pt (Step 8 owner keys, no embedding) ...")
    clean_rows = _prune_and_eval(CLEAN_CHECKPOINT, "clean", ratios, device, threshold,
                                 out_suffix=out_suffix)

    payload = {
        "step": "Step 9C -- magnitude pruning robustness",
        "method": "global unstructured magnitude pruning of Conv2d/Linear weight tensors",
        "ber_threshold": threshold,
        "ratios_requested": ratios,
        "watermarked_model": wm_rows,
        "clean_control": clean_rows,
        "interpretation": "watermarked rows show embedded-signature persistence; clean "
                          "control shows how the same owner keys (adversarial to the "
                          "clean model) degrade under identical pruning.",
        "clean_checkpoint_sha256": assert_clean_model_intact("Step 9C end"),
    }
    save_json(payload, out_json)
    print("=" * 68)
    print(f"Step 9C complete -> {out_json.relative_to(PROJECT_ROOT)}")
    print("=" * 68)
    return payload


def parse_args():
    p = argparse.ArgumentParser(description="BlackMarks Step 9C -- pruning robustness")
    p.add_argument("--ratios", type=str, default=None, help="comma list, e.g. 0.1,0.3,0.5")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--suffix", type=str, default=None,
                   help="write step9_pruning_<suffix>.json (and pruned_*_<suffix>.pt) "
                        "instead of the default names; leave unset for the original behaviour")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    rr = [float(x) for x in a.ratios.split(",")] if a.ratios else None
    run(ratios=rr, threshold=a.threshold, out_suffix=a.suffix)
