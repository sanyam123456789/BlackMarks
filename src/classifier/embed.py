"""
BlackMarks -- Step 8C: Multi-Bit Watermark Embedding

Fine-tunes a COPY of the clean baseline so that the watermark key set (Step 8B)
is classified into the target classes that encode the owner signature (Step 8A),
while normal CIFAR-10 accuracy is preserved.

    clean_model.pt  --(read-only load)-->  fresh CompactCNN
                                             |
                                             |  L_total = L_clean + lambda_wm * L_wm
                                             v
                                       blackmarks_model.pt   (NEW file, never clean_model.pt)

Objective
---------
    L_clean = CE( model(clean batch),         true labels )
    L_wm    = CE( model(embedding key batch),  target classes )
    L_total = L_clean + lambda_wm * L_wm

Only the *embedding* keys are used here. The disjoint *verification* keys are
held out and only consulted for monitoring / Step 8E generalisation.

Paper alignment (arXiv:1904.00344, Sec. 3.3 "Watermark Embedding")
-----------------------------------------------------------------
The paper embeds by an additional regularised loss term that drives the key
inputs to their designated labels during fine-tuning. Engineering choices:
low-LR SGD fine-tune from the trained baseline; ``lambda_wm`` weighting; full
key batch every step (the key set is tiny).

Outputs
-------
  artifacts/checkpoints/blackmarks_model.pt
  artifacts/metrics/step8_embedding.json

Usage
-----
  python src/classifier/embed.py
  python src/classifier/embed.py --epochs 10 --lambda-wm 1.0 --lr 0.001
  python src/classifier/embed.py --smoke-test
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
    CIFAR10_MEAN, CIFAR10_STD, BLACKMARKS_CHECKPOINT, CHECKPOINT_DIR, CLEAN_CHECKPOINT,
    DATA_DIR, METRICS_DIR, PROJECT_ROOT, assert_clean_model_intact, get_device,
    load_state_dict_checkpoint, save_json, set_seed, sha256_file,
)
from src.classifier.data import get_cifar10_train_val_test_dataloaders
from src.classifier.encoding import ENCODING_JSON, deserialize_signature
from src.classifier.keygen import EMBED_KEYS_PT, VERIFY_KEYS_PT
from src.classifier.model import build_model, count_parameters
from src.classifier.verify import BlackBoxModel, ownership_decision, score_keyset

STEP8_EMBED_JSON = METRICS_DIR / "step8_embedding.json"

DEFAULT_EPOCHS    = 10
DEFAULT_LR        = 1e-3
DEFAULT_LAMBDA_WM = 1.0
DEFAULT_BATCH     = 128
DEFAULT_SEED      = 42
DEFAULT_VAL_SIZE  = 5000

_MEAN = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)


def _normalize01(x01: torch.Tensor, device) -> torch.Tensor:
    return (x01.to(device) - _MEAN.to(device)) / _STD.to(device)


@torch.no_grad()
def _eval_normal(model, loader, criterion, device):
    model.eval()
    tot_loss = tot_correct = tot = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        tot_loss += criterion(logits, y).item() * x.size(0)
        tot_correct += (logits.argmax(1) == y).sum().item()
        tot += x.size(0)
    return tot_loss / tot, 100.0 * tot_correct / tot


class _OracleFromModel:
    """Wrap a live in-memory model as a predict(images01) oracle for monitoring."""

    def __init__(self, model, device):
        self._model, self._device = model, device
        self._n = 0

    @property
    def n_queries(self):
        return self._n

    def predict(self, images01):
        self._model.eval()
        with torch.no_grad():
            p = self._model(_normalize01(images01, self._device)).argmax(1).cpu().numpy()
        self._n += len(p)
        return p.astype(np.int64)


def embed(epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, lambda_wm=DEFAULT_LAMBDA_WM,
          batch_size=DEFAULT_BATCH, seed=DEFAULT_SEED, val_size=DEFAULT_VAL_SIZE,
          train_cap=None, out_ckpt: Path = BLACKMARKS_CHECKPOINT,
          out_json: Path = STEP8_EMBED_JSON, smoke: bool = False) -> dict:
    print("=" * 68)
    print("BLACKMARKS -- Step 8C: Multi-Bit Watermark Embedding")
    print("=" * 68)

    hash_before = assert_clean_model_intact("Step 8C start")
    set_seed(seed)
    device = get_device()

    # ---- encoding + keys ----
    import json
    enc = json.loads(ENCODING_JSON.read_text(encoding="utf-8"))
    signature = deserialize_signature(enc["signature"]["bits"])
    cbi = enc["class_encoding"]["class_to_bit_index"]
    embed_bundle = torch.load(EMBED_KEYS_PT, map_location="cpu", weights_only=False)
    verify_bundle = torch.load(VERIFY_KEYS_PT, map_location="cpu", weights_only=False)
    key_imgs01 = embed_bundle["images01"].to(device)
    key_targets = embed_bundle["target_classes"].to(device)
    print(f"[Setup] signature={enc['signature']['bits']} (K={len(signature)})  "
          f"embed keys={key_imgs01.shape[0]}  lambda_wm={lambda_wm}")

    # ---- data ----
    train_loader, val_loader, test_loader = get_cifar10_train_val_test_dataloaders(
        data_dir=str(DATA_DIR), val_size=val_size, batch_size=batch_size,
        num_workers=0, seed=seed, download=True, normalize=True)
    print(f"[Data] train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  "
          f"test={len(test_loader.dataset)} (held-out)")

    # ---- model: COPY of clean baseline (clean_model.pt is only ever read) ----
    model = build_model(num_classes=10, dropout_rate=0.5).to(device)
    src_meta = load_state_dict_checkpoint(model, CLEAN_CHECKPOINT, device)
    total_params, _ = count_parameters(model)
    print(f"[Model] initialised from clean_model.pt (epoch {src_meta.get('epoch')}, "
          f"val_acc {src_meta.get('val_acc')}); {total_params:,d} params")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)

    max_batches = 3 if smoke else (train_cap if train_cap else None)
    history = []
    mon = _OracleFromModel(model, device)

    print(f"\n  {'ep':>3} {'L_total':>9} {'L_clean':>9} {'L_wm':>8} {'val_acc':>8} "
          f"{'emb_BER':>8} {'ver_BER':>8} {'time':>7}")
    print("  " + "-" * 68)
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        run_t = run_c = run_w = 0.0
        nb = 0
        for bi, (x, y) in enumerate(train_loader):
            if max_batches and bi >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            l_clean = criterion(model(x), y)
            l_wm = criterion(model(_normalize01(key_imgs01, device)), key_targets)
            loss = l_clean + lambda_wm * l_wm
            loss.backward()
            optimizer.step()
            run_t += loss.item(); run_c += l_clean.item(); run_w += l_wm.item(); nb += 1

        # ---- monitor ----
        if smoke:
            va = float("nan")
        else:
            _, va = _eval_normal(model, val_loader, criterion, device)
        emb_score = score_keyset(mon, embed_bundle, cbi, signature)
        ver_score = score_keyset(mon, verify_bundle, cbi, signature)
        dt = time.time() - t0
        history.append({
            "epoch": ep,
            "loss_total": round(run_t / nb, 6),
            "loss_clean": round(run_c / nb, 6),
            "loss_wm": round(run_w / nb, 6),
            "val_acc": round(va, 4) if va == va else None,
            "embed_key_ber": emb_score["bit_error_rate"],
            "embed_key_target_match": emb_score["target_class_match_rate"],
            "verify_key_ber": ver_score["bit_error_rate"],
            "verify_key_target_match": ver_score["target_class_match_rate"],
            "seconds": round(dt, 1),
        })
        print(f"  {ep:>3} {run_t/nb:>9.4f} {run_c/nb:>9.4f} {run_w/nb:>8.4f} "
              f"{va:>8.3f} {emb_score['bit_error_rate']:>8.4f} "
              f"{ver_score['bit_error_rate']:>8.4f} {dt:>6.1f}s")

    # ---- final held-out evaluation ----
    if smoke:
        test_loss, test_acc = float("nan"), float("nan")
    else:
        test_loss, test_acc = _eval_normal(model, test_loader, criterion, device)
    print(f"\n[Final] normal test: loss={test_loss:.4f}  acc={test_acc:.4f}%")

    # ---- save watermarked checkpoint (SEPARATE FILE) ----
    if out_ckpt.exists():
        print(f"[Guard] {out_ckpt.name} already exists -- it will be replaced by this "
              f"run's Step 8C output (this is the Step 8C checkpoint, not an attack "
              f"artifact). clean_model.pt / watermarked_model.pt are untouched.")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_payload = {
        "model_state_dict": model.state_dict(),
        "architecture": "CompactCNN",
        "num_classes": 10,
        "step": "Step 8C -- Multi-Bit BlackMarks Watermarked Model",
        "source_checkpoint": str(CLEAN_CHECKPOINT),
        "source_checkpoint_sha256": hash_before,
        "owner_signature": enc["signature"]["bits"],
        "signature_length": int(len(signature)),
        "class_to_bit_index": list(cbi),
        "embedding_config": {
            "epochs": epochs, "lr": lr, "lambda_wm": lambda_wm, "batch_size": batch_size,
            "seed": seed, "optimizer": "SGD(mom=0.9,wd=5e-4,nesterov=True)",
            "objective": "L_clean + lambda_wm * L_wm", "train_cap_batches": max_batches,
        },
        "final_test_loss": None if test_loss != test_loss else round(test_loss, 6),
        "final_test_acc": None if test_acc != test_acc else round(test_acc, 4),
        "history": history,
    }
    torch.save(ckpt_payload, out_ckpt)
    print(f"[Ckpt] {out_ckpt.relative_to(PROJECT_ROOT)}  ({out_ckpt.stat().st_size:,} bytes)")

    # ---- independent black-box re-verification of the saved file ----
    bb = None if smoke else BlackBoxModel(out_ckpt, device=device)
    if bb is not None:
        emb_final = score_keyset(bb, embed_bundle, cbi, signature)
        ver_final = score_keyset(bb, verify_bundle, cbi, signature)
        emb_final["ownership_decision"] = ownership_decision(emb_final["bit_error_rate"])
        ver_final["ownership_decision"] = ownership_decision(ver_final["bit_error_rate"])
    else:
        emb_final = ver_final = None

    clean_ref = {"test_loss": 0.64991, "test_accuracy": 86.95}
    step7_ref = {"test_accuracy": 86.54, "watermark_accuracy": 100.0,
                 "type": "single-target 3x3 patch backdoor"}

    payload = {
        "step": "Step 8C -- multi-bit watermark embedding",
        "objective": "L_total = L_clean + lambda_wm * L_wm",
        "clean_checkpoint": str(CLEAN_CHECKPOINT),
        "clean_checkpoint_sha256_before": hash_before,
        "clean_checkpoint_sha256_after": assert_clean_model_intact("Step 8C end"),
        "encoding_source": str(ENCODING_JSON),
        "keygen_source": str(EMBED_KEYS_PT),
        "owner_signature": enc["signature"]["bits"],
        "config": ckpt_payload["embedding_config"],
        "clean_baseline_reference": clean_ref,
        "step7_patch_baseline_reference": step7_ref,
        "watermarked_model": {
            "checkpoint": str(out_ckpt),
            "sha256": None if smoke else sha256_file(out_ckpt),
            "normal_test": {
                "test_loss": None if test_loss != test_loss else round(test_loss, 6),
                "test_accuracy": None if test_acc != test_acc else round(test_acc, 4),
                "delta_vs_clean_acc": None if test_acc != test_acc else round(test_acc - clean_ref["test_accuracy"], 4),
            },
            "embedding_key_verification": emb_final,
            "held_out_key_verification": ver_final,
        },
        "training_history": history,
    }
    save_json(payload, out_json)
    print("=" * 68)
    print(f"Step 8C complete -> {out_json.relative_to(PROJECT_ROOT)}")
    if ver_final is not None:
        print(f"  held-out BER = {ver_final['bit_error_rate']:.4f}  "
              f"(Hamming {ver_final['hamming_distance']}/{ver_final['signature_length']})  "
              f"decision = {'OWNER' if ver_final['ownership_decision'] else 'NOT OWNER'}")
    print("=" * 68)
    return payload


def parse_args():
    p = argparse.ArgumentParser(description="BlackMarks Step 8C -- multi-bit embedding")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--lambda-wm", type=float, default=DEFAULT_LAMBDA_WM)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    p.add_argument("--train-cap", type=int, default=None,
                   help="cap batches/epoch (speed knob; documented in output)")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    embed(epochs=a.epochs, lr=a.lr, lambda_wm=a.lambda_wm, batch_size=a.batch_size,
          seed=a.seed, val_size=a.val_size, train_cap=a.train_cap, smoke=a.smoke_test)
