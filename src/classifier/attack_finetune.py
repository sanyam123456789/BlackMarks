"""
BlackMarks -- Step 9B: Fine-Tuning Robustness

An adversary who owns the watermarked model tries to remove the watermark by
fine-tuning it on clean CIFAR-10 data (no keys). We fine-tune a COPY of
``blackmarks_model.pt`` for a few epochs at several learning rates and measure,
before vs. after, the normal test accuracy and the black-box watermark BER /
Hamming / ownership decision on the embedding and held-out key sets.

Control: the identical fine-tuning schedule is applied to a COPY of
``clean_model.pt`` and scored with the same Step 8 owner keys, showing how the
raw (un-embedded) adversarial-key behaviour degrades under the same attack.

Safety
------
* blackmarks_model.pt / clean_model.pt are only READ.
* Each attacked model is saved to its own file:
      artifacts/checkpoints/finetuned_<base>_e<ep>_lr<lr>.pt

Usage
-----
  python src/classifier/attack_finetune.py
  python src/classifier/attack_finetune.py --configs 3:0.001,5:0.01 --train-cap 120
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

from src.classifier.bm_common import (
    BLACKMARKS_CHECKPOINT, CHECKPOINT_DIR, CLEAN_CHECKPOINT, DATA_DIR, METRICS_DIR,
    PROJECT_ROOT, assert_clean_model_intact, get_device, load_state_dict_checkpoint,
    save_json, set_seed, suffixed_path,
)
from src.classifier.bm_eval import normal_test_metrics, watermark_metrics
from src.classifier.data import get_cifar10_train_val_test_dataloaders
from src.classifier.model import build_model

FINETUNE_JSON = METRICS_DIR / "step9_finetune.json"
DEFAULT_CONFIGS = [(3, 1e-3), (5, 1e-2)]


def _finetune(base_ckpt, base_name, epochs, lr, device, threshold, train_cap, seed=42):
    set_seed(seed)
    model = build_model().to(device)
    load_state_dict_checkpoint(model, base_ckpt, device)

    train_loader, _, _ = get_cifar10_train_val_test_dataloaders(
        data_dir=str(DATA_DIR), val_size=5000, batch_size=128, num_workers=0,
        seed=seed, download=True, normalize=True)
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

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
    dt = time.time() - t0

    out_ckpt = CHECKPOINT_DIR / f"finetuned_{base_name}_e{epochs}_lr{lr:g}.pt"
    torch.save({"model_state_dict": model.state_dict(), "architecture": "CompactCNN",
                "num_classes": 10, "attacked_from": str(base_ckpt),
                "attack": "Step 9B fine-tuning", "epochs": epochs, "lr": lr,
                "train_cap_batches": train_cap}, out_ckpt)

    normal = normal_test_metrics(model, device)
    wm = watermark_metrics(out_ckpt, device, threshold=threshold)
    v = wm["held_out_verification_keys"]
    row = {
        "base": base_name, "epochs": epochs, "lr": lr, "seconds": round(dt, 1),
        "checkpoint": str(out_ckpt),
        "test_accuracy": normal["test_accuracy"], "test_loss": normal["test_loss"],
        "embed_key_ber": wm["embedding_keys"]["bit_error_rate"],
        "held_out_ber": v["bit_error_rate"], "held_out_hamming": v["hamming_distance"],
        "held_out_target_match": v["target_class_match_rate"],
        "ownership_decision": v["ownership_decision"],
    }
    print(f"  [{base_name}] e={epochs} lr={lr:g}  acc={normal['test_accuracy']:.2f}%  "
          f"heldout_BER={v['bit_error_rate']:.4f}  "
          f"{'OWNER' if v['ownership_decision'] else 'not owner'}  ({dt:.0f}s)")
    return row


def run(configs=None, threshold: float = 0.25, train_cap: int | None = None,
        out_suffix: str | None = None) -> dict:
    configs = configs or DEFAULT_CONFIGS
    out_json = suffixed_path(FINETUNE_JSON, out_suffix)
    print("=" * 68)
    print("BLACKMARKS -- Step 9B: Fine-Tuning Robustness")
    print("=" * 68)
    hash_before = assert_clean_model_intact("Step 9B start")
    device = get_device()
    if not Path(BLACKMARKS_CHECKPOINT).exists():
        raise RuntimeError("missing blackmarks_model.pt -- run Step 8C first")

    # baseline (no attack) for reference
    base_wm = watermark_metrics(BLACKMARKS_CHECKPOINT, device, threshold=threshold)
    base_norm = normal_test_metrics(_loaded(BLACKMARKS_CHECKPOINT, device), device)
    print(f"[baseline] blackmarks_model  acc={base_norm['test_accuracy']:.2f}%  "
          f"heldout_BER={base_wm['held_out_verification_keys']['bit_error_rate']:.4f}")

    wm_rows, clean_rows = [], []
    for (ep, lr) in configs:
        wm_rows.append(_finetune(BLACKMARKS_CHECKPOINT, "blackmarks", ep, lr,
                                 device, threshold, train_cap))
    # clean control: run only the first (mildest) config to bound compute
    ep0, lr0 = configs[0]
    clean_rows.append(_finetune(CLEAN_CHECKPOINT, "clean", ep0, lr0,
                                device, threshold, train_cap))

    payload = {
        "step": "Step 9B -- fine-tuning robustness",
        "attack": "SGD fine-tune on clean CIFAR-10 (no keys), momentum=0.9 wd=5e-4 nesterov",
        "ber_threshold": threshold,
        "train_cap_batches": train_cap,
        "baseline_no_attack": {
            "test_accuracy": base_norm["test_accuracy"],
            "held_out_ber": base_wm["held_out_verification_keys"]["bit_error_rate"],
            "embed_key_ber": base_wm["embedding_keys"]["bit_error_rate"],
        },
        "watermarked_attacks": wm_rows,
        "clean_control_attacks": clean_rows,
        "clean_checkpoint_sha256_before": hash_before,
        "clean_checkpoint_sha256_after": assert_clean_model_intact("Step 9B end"),
    }
    save_json(payload, out_json)
    print("=" * 68)
    print(f"Step 9B complete -> {out_json.relative_to(PROJECT_ROOT)}")
    print("=" * 68)
    return payload


def _loaded(ckpt, device):
    m = build_model().to(device)
    load_state_dict_checkpoint(m, ckpt, device)
    return m


def parse_args():
    p = argparse.ArgumentParser(description="BlackMarks Step 9B -- fine-tuning robustness")
    p.add_argument("--configs", type=str, default=None,
                   help="comma list of epochs:lr, e.g. 3:0.001,5:0.01")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--train-cap", type=int, default=None, help="cap batches/epoch")
    p.add_argument("--suffix", type=str, default=None,
                   help="write step9_finetune_<suffix>.json instead of the default name; "
                        "leave unset for the original behaviour")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    cfgs = None
    if a.configs:
        cfgs = []
        for part in a.configs.split(","):
            e, l = part.split(":")
            cfgs.append((int(e), float(l)))
    run(configs=cfgs, threshold=a.threshold, train_cap=a.train_cap, out_suffix=a.suffix)
