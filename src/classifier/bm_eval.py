"""
BlackMarks -- Step 9 shared evaluation helpers.

One place to compute, for any checkpoint:
  * normal CIFAR-10 held-out test loss / accuracy
  * black-box watermark metrics (BER, Hamming, decision) on the embedding keys
    and the held-out verification keys, using the Step 8A encoding + Step 8B keys.

Nothing here trains. Nothing here writes checkpoints.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn

from src.classifier.bm_common import DATA_DIR, load_state_dict_checkpoint
from src.classifier.data import get_cifar10_train_val_test_dataloaders
from src.classifier.encoding import ENCODING_JSON, deserialize_signature
from src.classifier.keygen import EMBED_KEYS_PT, VERIFY_KEYS_PT
from src.classifier.model import build_model
from src.classifier.verify import BlackBoxModel, ownership_decision, score_keyset

_TEST_LOADER_CACHE: dict = {}


def get_test_loader(batch_size: int = 256, seed: int = 42):
    key = (batch_size, seed)
    if key not in _TEST_LOADER_CACHE:
        _, _, test_loader = get_cifar10_train_val_test_dataloaders(
            data_dir=str(DATA_DIR), val_size=5000, batch_size=batch_size,
            num_workers=0, seed=seed, download=True, normalize=True)
        _TEST_LOADER_CACHE[key] = test_loader
    return _TEST_LOADER_CACHE[key]


@torch.no_grad()
def normal_test_metrics(model: nn.Module, device: torch.device,
                        batch_size: int = 256) -> dict:
    model.eval()
    crit = nn.CrossEntropyLoss()
    loader = get_test_loader(batch_size)
    tl = tc = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        tl += crit(logits, y).item() * x.size(0)
        tc += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return {"test_loss": round(tl / n, 6), "test_accuracy": round(100.0 * tc / n, 4),
            "n_test": n}


def load_encoding_and_keys():
    enc = json.loads(Path(ENCODING_JSON).read_text(encoding="utf-8"))
    signature = deserialize_signature(enc["signature"]["bits"])
    cbi = enc["class_encoding"]["class_to_bit_index"]
    embed_bundle = torch.load(EMBED_KEYS_PT, map_location="cpu", weights_only=False)
    verify_bundle = torch.load(VERIFY_KEYS_PT, map_location="cpu", weights_only=False)
    return enc, signature, cbi, embed_bundle, verify_bundle


def watermark_metrics(checkpoint_path, device: torch.device,
                      threshold: float = 0.25) -> dict:
    enc, signature, cbi, eb, vb = load_encoding_and_keys()
    oracle = BlackBoxModel(checkpoint_path, device=device)
    out = {}
    for name, bundle in (("embedding_keys", eb), ("held_out_verification_keys", vb)):
        r = score_keyset(oracle, bundle, cbi, signature)
        r["ownership_decision"] = ownership_decision(r["bit_error_rate"], threshold)
        # drop the bulky raw prediction list for Step 9 summaries
        r.pop("predicted_classes", None)
        out[name] = r
    out["_black_box_queries"] = oracle.n_queries
    return out


def full_report(checkpoint_path, device: torch.device, label: str,
                threshold: float = 0.25, batch_size: int = 256) -> dict:
    model = build_model(num_classes=10, dropout_rate=0.5).to(device)
    meta = load_state_dict_checkpoint(model, checkpoint_path, device)
    normal = normal_test_metrics(model, device, batch_size=batch_size)
    wm = watermark_metrics(checkpoint_path, device, threshold=threshold)
    return {
        "label": label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": meta.get("step"),
        "normal_test": normal,
        "watermark": wm,
        "ber_threshold": threshold,
    }
