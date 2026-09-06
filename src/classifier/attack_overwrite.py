"""
BlackMarks -- Step 9D: Watermark Overwriting

An adversary embeds their OWN multi-bit watermark into a copy of the Step 8
watermarked model, then claims ownership. We measure whether this destroys the
original owner's signature.

Pipeline
--------
  blackmarks_model.pt --(COPY)-->  fresh model
     * adversary derives a NEW signature (different owner_id + seed)
     * adversary recomputes a class->bit map on this model's representation
     * adversary generates NEW targeted-PGD keys on this model
     * adversary fine-tunes: L_clean + lambda_wm * L_wm(new keys)
  -> artifacts/checkpoints/blackmarks_overwritten_attack.pt   (NEW file)

Then evaluate on the attacked model:
  * ORIGINAL signature BER  (Step 8A encoding + Step 8B held-out keys)
  * NEW signature BER       (adversary encoding + adversary held-out keys)
  * normal CIFAR-10 test accuracy

Safety: blackmarks_model.pt is only READ; the attack result is a separate file.

Usage:  python src/classifier/attack_overwrite.py --epochs 8
"""

from __future__ import annotations

import argparse
import json
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
    CIFAR10_MEAN, CIFAR10_STD, BLACKMARKS_CHECKPOINT, CHECKPOINT_DIR, DATA_DIR, KEYS_DIR,
    METRICS_DIR, PROJECT_ROOT, assert_clean_model_intact, bits_to_string, get_device,
    load_state_dict_checkpoint, save_json, set_seed, suffixed_path,
)
from src.classifier.bm_eval import normal_test_metrics
from src.classifier.data import get_cifar10_train_val_test_dataloaders
from src.classifier.encoding import (
    ENCODING_JSON, build_class_bit_mapping, compute_class_mean_features,
    derive_owner_signature, deserialize_signature, serialize_signature,
)
from src.classifier.keygen import (
    EMBED_KEYS_PT, VERIFY_KEYS_PT, Normalizer, generate_key_role, load_raw_train_images,
)
from src.classifier.model import build_model
from src.classifier.verify import BlackBoxModel, ownership_decision, score_keyset

OVERWRITE_JSON      = METRICS_DIR / "step9_overwrite.json"
OVR_ENCODING_JSON   = METRICS_DIR / "overwrite_encoding.json"
OVR_EMBED_KEYS_PT   = KEYS_DIR / "overwrite_embed_keys.pt"
OVR_VERIFY_KEYS_PT  = KEYS_DIR / "overwrite_verify_keys.pt"
ATTACK_CKPT         = CHECKPOINT_DIR / "blackmarks_overwritten_attack.pt"

_MEAN = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)


def _norm01(x01, device):
    return (x01.to(device) - _MEAN.to(device)) / _STD.to(device)


def run(epochs: int = 8, lambda_wm: float = 1.0, lr: float = 1e-3,
        keys_per_bit: int = 6, adv_seed: int = 1337, threshold: float = 0.25,
        out_suffix: str | None = None) -> dict:
    out_json = suffixed_path(OVERWRITE_JSON, out_suffix)
    print("=" * 68)
    print("BLACKMARKS -- Step 9D: Watermark Overwriting")
    print("=" * 68)
    hash_before = assert_clean_model_intact("Step 9D start")
    device = get_device()
    set_seed(adv_seed)

    if not Path(BLACKMARKS_CHECKPOINT).exists():
        raise RuntimeError("missing blackmarks_model.pt -- run Step 8C first")

    # ---- original owner artefacts ----
    orig_enc = json.loads(Path(ENCODING_JSON).read_text(encoding="utf-8"))
    orig_sig = deserialize_signature(orig_enc["signature"]["bits"])
    orig_cbi = orig_enc["class_encoding"]["class_to_bit_index"]
    orig_verify = torch.load(VERIFY_KEYS_PT, map_location="cpu", weights_only=False)
    orig_embed = torch.load(EMBED_KEYS_PT, map_location="cpu", weights_only=False)

    # ---- adversary model copy ----
    model = build_model().to(device)
    load_state_dict_checkpoint(model, BLACKMARKS_CHECKPOINT, device)
    model.eval()

    # ---- adversary signature + encoding (recomputed on THIS model) ----
    adv_sig = derive_owner_signature(length=len(orig_sig), seed=adv_seed,
                                     owner_id="Adversary-Owner-v1")
    print(f"[adversary] new signature = {serialize_signature(adv_sig)}  "
          f"(original = {orig_enc['signature']['bits']})")
    class_means = compute_class_mean_features(model, seed=adv_seed, sample=3000, device=device)
    adv_map = build_class_bit_mapping(class_means, seed=adv_seed)
    adv_cbi = adv_map["class_to_bit_index"]
    save_json({"step": "Step 9D adversary encoding", "seed": adv_seed,
               "signature": {"bits": serialize_signature(adv_sig), "length": int(len(adv_sig))},
               "class_encoding": adv_map}, OVR_ENCODING_JSON)

    # ---- adversary keys (targeted PGD on the adversary's copy) ----
    K = len(adv_sig)
    n = K * keys_per_bit
    imgs, labels, idx = load_raw_train_images(seed=adv_seed, n_total=2 * n)
    wrapped = nn.Sequential(Normalizer(), model).to(device).eval()

    def _pred(w, x):
        with torch.no_grad():
            return w(x).argmax(1)

    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    eb, es = generate_key_role(wrapped, _pred, role="embed", signature=adv_sig,
                               class_to_bit_index=adv_cbi, carriers01=imgs[:n],
                               carrier_labels=labels[:n], carrier_indices=idx[:n],
                               keys_per_bit=keys_per_bit, epsilon=8/255, alpha=2/255,
                               iters=40, seed=adv_seed, device=device)
    vb, vs = generate_key_role(wrapped, _pred, role="verify", signature=adv_sig,
                               class_to_bit_index=adv_cbi, carriers01=imgs[n:],
                               carrier_labels=labels[n:], carrier_indices=idx[n:],
                               keys_per_bit=keys_per_bit, epsilon=8/255, alpha=2/255,
                               iters=40, seed=adv_seed, device=device)
    torch.save(eb, OVR_EMBED_KEYS_PT)
    torch.save(vb, OVR_VERIFY_KEYS_PT)
    print(f"[adversary] keys: embed succ {es['attack_success_rate']*100:.0f}%  "
          f"verify succ {vs['attack_success_rate']*100:.0f}%")

    # ---- overwrite fine-tune ----
    train_loader, _, test_loader = get_cifar10_train_val_test_dataloaders(
        data_dir=str(DATA_DIR), val_size=5000, batch_size=128, num_workers=0,
        seed=adv_seed, download=True, normalize=True)
    key_imgs01 = eb["images01"].to(device)
    key_targets = eb["target_classes"].to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y) + lambda_wm * crit(model(_norm01(key_imgs01, device)), key_targets)
            loss.backward()
            opt.step()
        print(f"  [overwrite] epoch {ep}/{epochs}  ({time.time()-t0:.0f}s)")

    torch.save({"model_state_dict": model.state_dict(), "architecture": "CompactCNN",
                "num_classes": 10, "attacked_from": str(BLACKMARKS_CHECKPOINT),
                "attack": "Step 9D watermark overwriting", "adversary_seed": adv_seed,
                "adversary_signature": serialize_signature(adv_sig),
                "epochs": epochs, "lambda_wm": lambda_wm, "lr": lr}, ATTACK_CKPT)

    # ---- evaluate ----
    normal = normal_test_metrics(model, device)
    oracle = BlackBoxModel(ATTACK_CKPT, device=device)
    orig_v = score_keyset(oracle, orig_verify, orig_cbi, orig_sig)
    orig_e = score_keyset(oracle, orig_embed, orig_cbi, orig_sig)
    adv_v = score_keyset(oracle, vb, adv_cbi, adv_sig)
    for r in (orig_v, orig_e, adv_v):
        r["ownership_decision"] = ownership_decision(r["bit_error_rate"], threshold)
        r.pop("predicted_classes", None)

    print(f"[result] normal acc            = {normal['test_accuracy']:.2f}%")
    print(f"[result] ORIGINAL owner heldout BER = {orig_v['bit_error_rate']:.4f}  "
          f"decision={'OWNER' if orig_v['ownership_decision'] else 'not owner'}")
    print(f"[result] ADVERSARY     heldout BER = {adv_v['bit_error_rate']:.4f}  "
          f"decision={'OWNER' if adv_v['ownership_decision'] else 'not owner'}")

    payload = {
        "step": "Step 9D -- watermark overwriting",
        "ber_threshold": threshold,
        "attack_checkpoint": str(ATTACK_CKPT),
        "config": {"epochs": epochs, "lambda_wm": lambda_wm, "lr": lr,
                   "keys_per_bit": keys_per_bit, "adversary_seed": adv_seed},
        "original_owner": {
            "signature": orig_enc["signature"]["bits"],
            "held_out_key_verification": orig_v,
            "embedding_key_verification": orig_e,
        },
        "adversary_owner": {
            "signature": serialize_signature(adv_sig),
            "class_encoding_method": adv_map["method"],
            "keys_attack_success_rate": {"embed": es["attack_success_rate"],
                                         "verify": vs["attack_success_rate"]},
            "held_out_key_verification": adv_v,
        },
        "normal_test": normal,
        "interpretation": "original held-out BER staying low => the owner's signature "
                          "survives an overwrite attempt; adversary BER low => the "
                          "adversary can also embed, i.e. overwriting adds a second "
                          "mark rather than removing the first.",
        "clean_checkpoint_sha256_before": hash_before,
        "clean_checkpoint_sha256_after": assert_clean_model_intact("Step 9D end"),
    }
    save_json(payload, out_json)
    print("=" * 68)
    print(f"Step 9D complete -> {out_json.relative_to(PROJECT_ROOT)}")
    print("=" * 68)
    return payload


def parse_args():
    p = argparse.ArgumentParser(description="BlackMarks Step 9D -- watermark overwriting")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lambda-wm", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--keys-per-bit", type=int, default=6)
    p.add_argument("--adv-seed", type=int, default=1337)
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--suffix", type=str, default=None,
                   help="write step9_overwrite_<suffix>.json instead of the default name; "
                        "leave unset for the original behaviour")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(epochs=a.epochs, lambda_wm=a.lambda_wm, lr=a.lr, keys_per_bit=a.keys_per_bit,
        adv_seed=a.adv_seed, threshold=a.threshold, out_suffix=a.suffix)
