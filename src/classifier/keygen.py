"""
BlackMarks -- Step 8B: Targeted Adversarial Key Generation

Generates the watermark key set: carrier CIFAR-10 images perturbed by a targeted
adversarial attack so that the *clean* model classifies each key as a chosen
target class. The target class for a key at signature position ``i`` is picked so
that its Step-8A encoded bit equals signature bit ``i``.

Paper alignment (arXiv:1904.00344, Sec. 3.2 "Watermark Key Generation")
----------------------------------------------------------------------
The paper builds keys by a *targeted adversarial perturbation* of natural images
toward the label that carries the desired signature bit. We use projected
gradient descent (PGD, L-inf) -- a standard, controllable targeted attack.

  ** RULE 14 -- NO SILENT FALLBACK **
  This module never substitutes a pixel patch / fixed trigger / random noise for
  a failed attack. If PGD does not reach the target it is recorded as
  ``attack_success = false`` and reported; nothing is silently swapped in.

Two disjoint key roles are produced:
  * embedding keys    -- used by Step 8C fine-tuning.
  * verification keys  -- NEVER seen during embedding; the held-out generalisation
                          probe for Step 8E.
Their carriers come from disjoint index sets of the CIFAR-10 *training* split.
The official test set is never used.

Outputs
-------
  artifacts/keys/embed_keys.pt          (tensor bundle -- gitignored via *.pt)
  artifacts/keys/verify_keys.pt
  artifacts/metrics/key_generation.json (metadata only, no pixel data)

Usage
-----
  python src/classifier/keygen.py
  python src/classifier/keygen.py --keys-per-bit 6 --epsilon 0.031372 --iters 40
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
from torchvision import datasets, transforms

from src.classifier.bm_common import (
    CIFAR10_MEAN, CIFAR10_STD, CLEAN_CHECKPOINT, DATA_DIR, KEYS_DIR, METRICS_DIR,
    PROJECT_ROOT, assert_clean_model_intact, get_device, load_state_dict_checkpoint,
    save_json, set_seed,
)
from src.classifier.data import CIFAR10_CLASSES
from src.classifier.encoding import ENCODING_JSON, deserialize_signature
from src.classifier.model import build_model

EMBED_KEYS_PT   = KEYS_DIR / "embed_keys.pt"
VERIFY_KEYS_PT  = KEYS_DIR / "verify_keys.pt"
KEYGEN_JSON     = METRICS_DIR / "key_generation.json"

DEFAULT_KEYS_PER_BIT = 6
DEFAULT_EPSILON      = 8.0 / 255.0     # L-inf budget in [0,1] pixel space
DEFAULT_ALPHA        = 2.0 / 255.0     # PGD step size
DEFAULT_ITERS        = 40
DEFAULT_SEED         = 42


# ---------------------------------------------------------------------------
# Normaliser wrapper -- keeps the attack in [0,1] pixel space
# ---------------------------------------------------------------------------

class Normalizer(nn.Module):
    """Applies CIFAR-10 channel normalisation so the model can be attacked in [0,1]."""

    def __init__(self, mean=CIFAR10_MEAN, std=CIFAR10_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        return (x01 - self.mean) / self.std


# ---------------------------------------------------------------------------
# Carrier pool -- raw [0,1] training images, deterministic disjoint index sets
# ---------------------------------------------------------------------------

def load_raw_train_images(seed: int, n_total: int):
    """
    Return ``n_total`` raw [0,1] CIFAR-10 *training* images (ToTensor only),
    their labels, and their dataset indices, chosen deterministically.
    """
    ds = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True,
                          transform=transforms.ToTensor())
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=n_total, replace=False)
    imgs = torch.stack([ds[int(i)][0] for i in idx])
    labels = np.array([ds[int(i)][1] for i in idx], dtype=np.int64)
    return imgs, labels, idx.astype(np.int64)


# ---------------------------------------------------------------------------
# Targeted PGD (L-inf)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _predict01(wrapped: nn.Module, x01: torch.Tensor) -> torch.Tensor:
    return wrapped(x01).argmax(1)


def targeted_pgd(wrapped: nn.Module, carriers01: torch.Tensor,
                 target_classes: torch.Tensor, *, epsilon: float, alpha: float,
                 iters: int, seed: int, device: torch.device):
    """
    Batched targeted PGD. ``wrapped`` = Normalizer -> model (expects [0,1] input).

    Returns:
        adv01              : perturbed images clamped to [0,1] and the eps-ball.
        final_pred         : model prediction on adv01 (LongTensor).
        linf, l2           : per-image perturbation norms (numpy).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    carriers01 = carriers01.to(device)
    target_classes = target_classes.to(device)

    delta = (torch.rand(carriers01.shape, generator=g).to(device) * 2 - 1) * epsilon
    adv = torch.clamp(carriers01 + delta, 0.0, 1.0).detach()

    ce = nn.CrossEntropyLoss()
    for _ in range(iters):
        adv.requires_grad_(True)
        logits = wrapped(adv)
        loss = ce(logits, target_classes)          # minimise -> move toward target
        grad = torch.autograd.grad(loss, adv)[0]
        with torch.no_grad():
            adv = adv - alpha * grad.sign()         # targeted: descend the loss
            adv = torch.min(torch.max(adv, carriers01 - epsilon), carriers01 + epsilon)
            adv = torch.clamp(adv, 0.0, 1.0).detach()

    final_pred = _predict01(wrapped, adv)
    pert = (adv - carriers01).view(adv.shape[0], -1)
    linf = pert.abs().max(1).values.cpu().numpy()
    l2 = pert.norm(dim=1).cpu().numpy()
    return adv.cpu(), final_pred.cpu(), linf, l2


# ---------------------------------------------------------------------------
# Key set construction
# ---------------------------------------------------------------------------

def _target_plan(signature: np.ndarray, class_to_bit_index: list[int],
                 keys_per_bit: int) -> list[tuple[int, int]]:
    """
    Build the ordered list of (signature_position, target_class) for a key set.
    For each position, cycle round-robin through the classes whose encoded bit
    matches the desired signature bit.
    """
    cbi = np.asarray(class_to_bit_index)
    plan = []
    for pos, bit in enumerate(signature):
        eligible = [int(c) for c in range(len(cbi)) if cbi[c] == int(bit)]
        if not eligible:
            raise RuntimeError(f"no class encodes bit {bit} for position {pos}")
        for j in range(keys_per_bit):
            plan.append((pos, eligible[j % len(eligible)]))
    return plan


def generate_key_role(wrapped, model_for_pred, *, role: str, signature, class_to_bit_index,
                      carriers01, carrier_labels, carrier_indices, keys_per_bit,
                      epsilon, alpha, iters, seed, device):
    plan = _target_plan(signature, class_to_bit_index, keys_per_bit)
    n = len(plan)
    assert carriers01.shape[0] >= n, f"need {n} carriers for role {role}, have {carriers01.shape[0]}"

    car = carriers01[:n]
    lab = carrier_labels[:n]
    cidx = carrier_indices[:n]
    positions = np.array([p for p, _ in plan], dtype=np.int64)
    targets = torch.tensor([t for _, t in plan], dtype=torch.long)
    desired_bits = np.array([int(signature[p]) for p in positions], dtype=np.int64)

    with torch.no_grad():
        orig_pred = model_for_pred(wrapped, car.to(device)).cpu().numpy()

    t0 = time.time()
    adv01, final_pred, linf, l2 = targeted_pgd(
        wrapped, car, targets, epsilon=epsilon, alpha=alpha, iters=iters,
        seed=seed + (0 if role == "embed" else 1), device=device,
    )
    dt = time.time() - t0
    final_pred_np = final_pred.numpy()
    success = (final_pred_np == targets.numpy())

    records = []
    for k in range(n):
        records.append({
            "role": role,
            "key_index": k,
            "signature_position": int(positions[k]),
            "desired_bit": int(desired_bits[k]),
            "target_class": int(targets[k].item()),
            "target_class_name": CIFAR10_CLASSES[int(targets[k].item())],
            "carrier_dataset_index": int(cidx[k]),
            "carrier_original_label": int(lab[k]),
            "carrier_original_label_name": CIFAR10_CLASSES[int(lab[k])],
            "clean_model_pred_on_carrier": int(orig_pred[k]),
            "clean_model_pred_on_key": int(final_pred_np[k]),
            "attack_success": bool(success[k]),
            "perturbation_linf": round(float(linf[k]), 6),
            "perturbation_l2": round(float(l2[k]), 6),
        })

    bundle = {
        "role": role,
        "images01": adv01,                              # (n,3,32,32) float in [0,1]
        "carrier_images01": car.clone(),
        "signature_positions": torch.tensor(positions),
        "desired_bits": torch.tensor(desired_bits),
        "target_classes": targets,
        "carrier_labels": torch.tensor(lab),
        "carrier_dataset_indices": torch.tensor(cidx),
        "attack_success": torch.tensor(success),
        "config": {
            "epsilon": epsilon, "alpha": alpha, "iters": iters,
            "keys_per_bit": keys_per_bit, "seed": seed,
        },
        "signature_bits": torch.tensor(np.asarray(signature, dtype=np.int64)),
        "class_to_bit_index": torch.tensor(np.asarray(class_to_bit_index, dtype=np.int64)),
    }
    summary = {
        "role": role,
        "n_keys": n,
        "keys_per_bit": keys_per_bit,
        "attack_success_count": int(success.sum()),
        "attack_success_rate": round(float(success.mean()), 6),
        "mean_perturbation_linf": round(float(linf.mean()), 6),
        "mean_perturbation_l2": round(float(l2.mean()), 6),
        "elapsed_sec": round(dt, 2),
        "records": records,
    }
    return bundle, summary


def run(keys_per_bit: int = DEFAULT_KEYS_PER_BIT, epsilon: float = DEFAULT_EPSILON,
        alpha: float = DEFAULT_ALPHA, iters: int = DEFAULT_ITERS,
        seed: int = DEFAULT_SEED) -> dict:
    import json

    print("=" * 68)
    print("BLACKMARKS -- Step 8B: Targeted Adversarial Key Generation")
    print("=" * 68)
    assert_clean_model_intact("Step 8B start")
    device = get_device()
    set_seed(seed)

    if not ENCODING_JSON.exists():
        raise RuntimeError(f"missing {ENCODING_JSON} -- run Step 8A (encoding.py) first")
    enc = json.loads(ENCODING_JSON.read_text(encoding="utf-8"))
    signature = deserialize_signature(enc["signature"]["bits"])
    class_to_bit_index = enc["class_encoding"]["class_to_bit_index"]
    K = len(signature)
    n_per_role = K * keys_per_bit
    print(f"[Keygen] signature={enc['signature']['bits']} (K={K})  "
          f"keys_per_bit={keys_per_bit}  -> {n_per_role} keys per role")
    print(f"[Keygen] PGD  eps={epsilon:.6f}  alpha={alpha:.6f}  iters={iters}")

    # Disjoint carrier pools: first n_per_role -> embed, next n_per_role -> verify
    imgs, labels, idx = load_raw_train_images(seed=seed, n_total=2 * n_per_role)
    embed_c, verify_c = imgs[:n_per_role], imgs[n_per_role:]
    embed_l, verify_l = labels[:n_per_role], labels[n_per_role:]
    embed_i, verify_i = idx[:n_per_role], idx[n_per_role:]
    assert set(embed_i.tolist()).isdisjoint(verify_i.tolist()), "carrier pools overlap!"

    model = build_model(num_classes=10, dropout_rate=0.5).to(device)
    load_state_dict_checkpoint(model, CLEAN_CHECKPOINT, device)
    model.eval()
    wrapped = nn.Sequential(Normalizer(), model).to(device).eval()

    def _pred(w, x):  # small helper matching generate_key_role signature
        with torch.no_grad():
            return w(x).argmax(1)

    embed_bundle, embed_sum = generate_key_role(
        wrapped, _pred, role="embed", signature=signature,
        class_to_bit_index=class_to_bit_index, carriers01=embed_c,
        carrier_labels=embed_l, carrier_indices=embed_i, keys_per_bit=keys_per_bit,
        epsilon=epsilon, alpha=alpha, iters=iters, seed=seed, device=device)
    print(f"[Keygen] embed  keys: success {embed_sum['attack_success_count']}/{embed_sum['n_keys']} "
          f"({embed_sum['attack_success_rate']*100:.1f}%)  {embed_sum['elapsed_sec']}s")

    verify_bundle, verify_sum = generate_key_role(
        wrapped, _pred, role="verify", signature=signature,
        class_to_bit_index=class_to_bit_index, carriers01=verify_c,
        carrier_labels=verify_l, carrier_indices=verify_i, keys_per_bit=keys_per_bit,
        epsilon=epsilon, alpha=alpha, iters=iters, seed=seed, device=device)
    print(f"[Keygen] verify keys: success {verify_sum['attack_success_count']}/{verify_sum['n_keys']} "
          f"({verify_sum['attack_success_rate']*100:.1f}%)  {verify_sum['elapsed_sec']}s")

    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(embed_bundle, EMBED_KEYS_PT)
    torch.save(verify_bundle, VERIFY_KEYS_PT)
    print(f"[Keygen] saved {EMBED_KEYS_PT.relative_to(PROJECT_ROOT)} , "
          f"{VERIFY_KEYS_PT.relative_to(PROJECT_ROOT)}")

    payload = {
        "step": "Step 8B -- targeted adversarial key generation",
        "seed": seed,
        "clean_checkpoint": str(CLEAN_CHECKPOINT),
        "clean_checkpoint_sha256": assert_clean_model_intact("Step 8B end"),
        "encoding_source": str(ENCODING_JSON),
        "signature_bits": enc["signature"]["bits"],
        "signature_length": K,
        "attack": {
            "method": "targeted PGD (L-inf)", "epsilon": epsilon, "alpha": alpha,
            "iters": iters, "random_start": True, "pixel_space": "[0,1] then normalised",
        },
        "carrier_source": "CIFAR-10 training split; disjoint index sets per role; test set untouched",
        "roles": {
            "embed":  {k: v for k, v in embed_sum.items() if k != "records"},
            "verify": {k: v for k, v in verify_sum.items() if k != "records"},
        },
        "embed_key_records": embed_sum["records"],
        "verify_key_records": verify_sum["records"],
        "artifacts": {
            "embed_keys_pt": str(EMBED_KEYS_PT),
            "verify_keys_pt": str(VERIFY_KEYS_PT),
        },
    }
    save_json(payload, KEYGEN_JSON)
    print("=" * 68)
    print(f"Step 8B complete -> {KEYGEN_JSON.relative_to(PROJECT_ROOT)}")
    print("=" * 68)
    return payload


def parse_args():
    p = argparse.ArgumentParser(description="BlackMarks Step 8B -- adversarial key generation")
    p.add_argument("--keys-per-bit", type=int, default=DEFAULT_KEYS_PER_BIT)
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(keys_per_bit=a.keys_per_bit, epsilon=a.epsilon, alpha=a.alpha,
        iters=a.iters, seed=a.seed)
