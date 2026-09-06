"""
BlackMarks -- Step 9E: Uniqueness / False-Positive Analysis

Trains several INDEPENDENT unmarked CompactCNN models (fresh random init, different
seeds) and runs the Step 8 owner's black-box verification against each. Because the
owner's keys are adversarial examples crafted on the Step 8 lineage, an independent
model should NOT reproduce the signature -- its BER should sit near chance (0.5) and
the ownership decision should be "not owner".

Compute note
------------
CPU-only. Independent models are trained for a REDUCED number of epochs
(``--epochs``, default 15 vs. the 50 used for the Step 6 baseline). This is a
deliberate, documented budget trade-off: the null-distribution question is about
*whether the owner keys transfer*, which does not require an accuracy-matched
model. The actual per-model accuracy and epoch count are recorded.

Safety: writes only new files
  artifacts/checkpoints/unmarked_seed<seed>.pt
  artifacts/metrics/step9_uniqueness.json

Usage
-----
  python src/classifier/uniqueness.py --n-models 3 --epochs 15
  python src/classifier/uniqueness.py --seeds 101,202,303 --epochs 12
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.classifier.bm_common import (
    CHECKPOINT_DIR, CLEAN_CHECKPOINT, DATA_DIR, METRICS_DIR, PROJECT_ROOT,
    assert_clean_model_intact, get_device, save_json, set_seed, suffixed_path,
)
from src.classifier.bm_eval import normal_test_metrics, watermark_metrics
from src.classifier.data import get_cifar10_train_val_test_dataloaders
from src.classifier.model import build_model, count_parameters

UNIQUENESS_JSON = METRICS_DIR / "step9_uniqueness.json"


def _train_unmarked(seed: int, epochs: int, device, train_cap=None, out_suffix=None) -> Path:
    set_seed(seed)
    train_loader, val_loader, _ = get_cifar10_train_val_test_dataloaders(
        data_dir=str(DATA_DIR), val_size=5000, batch_size=128, num_workers=0,
        seed=seed, download=True, normalize=True)
    model = build_model(num_classes=10, dropout_rate=0.5).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for bi, (x, y) in enumerate(train_loader):
            if train_cap and bi >= train_cap:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        sched.step()
    dt = time.time() - t0

    out = suffixed_path(CHECKPOINT_DIR / f"unmarked_seed{seed}.pt", out_suffix)
    torch.save({"model_state_dict": model.state_dict(), "architecture": "CompactCNN",
                "num_classes": 10, "step": "Step 9E independent unmarked model",
                "seed": seed, "epochs": epochs, "train_seconds": round(dt, 1),
                "train_cap_batches": train_cap}, out)
    print(f"  [train] seed={seed}  {epochs} epochs  {dt:.0f}s -> {out.name}")
    return out


def run(seeds, epochs: int = 15, threshold: float = 0.25, train_cap=None,
        out_suffix: str | None = None) -> dict:
    out_json = suffixed_path(UNIQUENESS_JSON, out_suffix)
    print("=" * 68)
    print("BLACKMARKS -- Step 9E: Uniqueness / False-Positive Analysis")
    print("=" * 68)
    hash_before = assert_clean_model_intact("Step 9E start")
    device = get_device()
    total_params, _ = count_parameters(build_model())

    models = []
    for s in seeds:
        ckpt = _train_unmarked(s, epochs, device, train_cap=train_cap, out_suffix=out_suffix)
        # reload from disk for evaluation (fresh, clean state)
        m = build_model().to(device)
        raw = torch.load(ckpt, map_location=device, weights_only=False)
        m.load_state_dict(raw["model_state_dict"])
        normal = normal_test_metrics(m, device)
        wm = watermark_metrics(ckpt, device, threshold=threshold)
        v = wm["held_out_verification_keys"]
        e = wm["embedding_keys"]
        models.append({
            "seed": s, "epochs": epochs, "checkpoint": str(ckpt),
            "test_accuracy": normal["test_accuracy"],
            "held_out_ber": v["bit_error_rate"], "held_out_hamming": v["hamming_distance"],
            "held_out_target_match": v["target_class_match_rate"],
            "embed_key_ber": e["bit_error_rate"],
            "ownership_decision": v["ownership_decision"],
        })
        print(f"  [eval ] seed={s}  acc={normal['test_accuracy']:.2f}%  "
              f"heldout_BER={v['bit_error_rate']:.4f}  hamming={v['hamming_distance']}  "
              f"{'FALSE-POSITIVE' if v['ownership_decision'] else 'not owner (correct)'}")

    hb = np.array([m["held_out_ber"] for m in models], dtype=float)
    fp = int(sum(m["ownership_decision"] for m in models))
    stats = {
        "n_models": len(models),
        "held_out_ber_mean": round(float(hb.mean()), 6),
        "held_out_ber_std": round(float(hb.std(ddof=0)), 6),
        "held_out_ber_min": round(float(hb.min()), 6),
        "held_out_ber_max": round(float(hb.max()), 6),
        "false_positive_count": fp,
        "false_positive_rate": round(fp / len(models), 6),
    }
    print(f"[stats] BER mean={stats['held_out_ber_mean']:.4f} "
          f"std={stats['held_out_ber_std']:.4f} "
          f"range=[{stats['held_out_ber_min']:.4f},{stats['held_out_ber_max']:.4f}]  "
          f"FP={fp}/{len(models)}")

    payload = {
        "step": "Step 9E -- uniqueness / false-positive analysis",
        "ber_threshold": threshold,
        "budget_note": "independent models trained for a REDUCED epoch count "
                       f"({epochs}) on CPU; the metric of interest is owner-key "
                       "transfer (BER), not accuracy parity with the 50-epoch baseline.",
        "reduced_epochs": epochs,
        "train_cap_batches": train_cap,
        "parameters_per_model": total_params,
        "independent_unmarked_models": models,
        "aggregate": stats,
        "degenerate_reference": {
            "note": "clean_model.pt is the exact checkpoint the owner keys were "
                    "crafted on, so it trivially verifies; it is NOT part of the "
                    "null distribution above. See step8_false_positive.json.",
            "checkpoint": str(CLEAN_CHECKPOINT),
        },
        "clean_checkpoint_sha256_before": hash_before,
        "clean_checkpoint_sha256_after": assert_clean_model_intact("Step 9E end"),
    }
    save_json(payload, out_json)
    print("=" * 68)
    print(f"Step 9E complete -> {out_json.relative_to(PROJECT_ROOT)}")
    print("=" * 68)
    return payload


def parse_args():
    p = argparse.ArgumentParser(description="BlackMarks Step 9E -- uniqueness / false positives")
    p.add_argument("--n-models", type=int, default=3)
    p.add_argument("--seeds", type=str, default=None, help="comma list; overrides --n-models")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--train-cap", type=int, default=None)
    p.add_argument("--suffix", type=str, default=None,
                   help="write step9_uniqueness_<suffix>.json (and unmarked_seed*_<suffix>.pt) "
                        "instead of the default names; leave unset for the original behaviour")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.seeds:
        seeds = [int(x) for x in a.seeds.split(",")]
    else:
        seeds = [1001 + 111 * i for i in range(a.n_models)]
    run(seeds=seeds, epochs=a.epochs, threshold=a.threshold, train_cap=a.train_cap,
        out_suffix=a.suffix)
