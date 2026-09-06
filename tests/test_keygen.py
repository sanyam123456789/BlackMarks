"""Step 8G -- PGD key generation mechanics: eps-ball projection, shapes, targeting.

Uses a tiny random-weight CompactCNN so no trained checkpoint or training is needed.
"""
import numpy as np
import torch
import torch.nn as nn

from src.classifier.keygen import Normalizer, targeted_pgd, _target_plan
from src.classifier.model import build_model


def _wrapped():
    torch.manual_seed(0)
    m = build_model().eval()
    return nn.Sequential(Normalizer(), m).eval()


def test_target_plan_respects_encoding_bits():
    sig = np.array([0, 1, 0, 1], dtype=np.int64)
    cbi = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    plan = _target_plan(sig, cbi, keys_per_bit=3)
    assert len(plan) == 4 * 3
    for pos, cls in plan:
        assert cbi[cls] == sig[pos]


def test_pgd_output_within_eps_ball_and_clamped():
    w = _wrapped()
    torch.manual_seed(1)
    carriers = torch.rand(4, 3, 32, 32)
    targets = torch.tensor([1, 2, 3, 4])
    eps = 8 / 255
    adv, pred, linf, l2 = targeted_pgd(
        w, carriers, targets, epsilon=eps, alpha=2 / 255, iters=10, seed=0,
        device=torch.device("cpu"),
    )
    assert adv.shape == carriers.shape
    assert adv.min() >= 0.0 and adv.max() <= 1.0
    # L-inf perturbation must not exceed eps (allow fp slack)
    assert float((adv - carriers).abs().max()) <= eps + 1e-6
    assert linf.shape == (4,) and l2.shape == (4,)


def test_pgd_moves_predictions_toward_target_on_trained_model():
    """With the real trained model, strong PGD should hit the target for most keys."""
    from src.classifier.bm_common import CLEAN_CHECKPOINT, load_state_dict_checkpoint
    import pytest
    if not CLEAN_CHECKPOINT.exists():
        pytest.skip("clean_model.pt absent")
    m = build_model().eval()
    load_state_dict_checkpoint(m, CLEAN_CHECKPOINT, torch.device("cpu"))
    w = nn.Sequential(Normalizer(), m).eval()
    torch.manual_seed(2)
    carriers = torch.rand(8, 3, 32, 32)
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    adv, pred, _, _ = targeted_pgd(
        w, carriers, targets, epsilon=8 / 255, alpha=2 / 255, iters=40, seed=0,
        device=torch.device("cpu"),
    )
    assert float((pred == targets).float().mean()) >= 0.5
