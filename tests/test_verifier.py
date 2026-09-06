"""Step 8G -- black-box verifier + mock prediction interface. No training."""
import numpy as np
import torch

from src.classifier.bm_common import bits_to_string
from src.classifier.verify import (
    MockPredictAPI, ownership_decision, recover_signature, score_keyset,
)


CBI = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]        # 4 classes -> bit0, 6 -> bit1
SIG = np.array([0, 1, 1, 0], dtype=np.int64)


def _bundle(images01, positions, targets):
    return {
        "images01": torch.as_tensor(images01, dtype=torch.float32),
        "signature_positions": torch.as_tensor(positions),
        "target_classes": torch.as_tensor(targets),
    }


def test_recover_signature_perfect():
    # position -> class that encodes the right bit
    positions = [0, 1, 2, 3]
    preds = [0, 2, 3, 8]        # bits 0,1,1,0 == SIG
    recovered, per_key, _ = recover_signature(preds, positions, CBI, length=4)
    assert recovered.tolist() == SIG.tolist()
    assert per_key.tolist() == [0, 1, 1, 0]


def test_mock_predict_api_counts_queries():
    api = MockPredictAPI(lambda imgs: np.zeros(len(imgs), dtype=int))
    out = api.predict(torch.zeros(5, 3, 32, 32))
    assert out.tolist() == [0, 0, 0, 0, 0]
    assert api.n_queries == 5


def test_score_keyset_zero_ber_when_oracle_returns_targets():
    positions = [0, 1, 2, 3]
    targets = [0, 2, 3, 8]
    bundle = _bundle(torch.zeros(4, 3, 32, 32), positions, targets)
    oracle = MockPredictAPI(lambda imgs: np.array(targets[: len(imgs)]))
    r = score_keyset(oracle, bundle, CBI, SIG)
    assert r["bit_error_rate"] == 0.0
    assert r["hamming_distance"] == 0
    assert r["target_class_match_rate"] == 1.0
    assert r["recovered_signature"] == bits_to_string(SIG)


def test_score_keyset_half_ber_when_oracle_flips_half():
    positions = [0, 1, 2, 3]
    targets = [0, 2, 3, 8]                  # bits 0,1,1,0
    wrong = [8, 2, 3, 0]                    # positions 0 and 3 flipped -> bits 0->? 8=bit0 still...
    # choose predictions that flip exactly bits at positions 1 and 2:
    preds = [0, 0, 0, 8]                    # bits 0,0,0,0 -> differs from SIG at pos 1,2
    bundle = _bundle(torch.zeros(4, 3, 32, 32), positions, targets)
    oracle = MockPredictAPI(lambda imgs: np.array(preds[: len(imgs)]))
    r = score_keyset(oracle, bundle, CBI, SIG)
    assert r["hamming_distance"] == 2
    assert r["bit_error_rate"] == 0.5


def test_ownership_decision_threshold():
    assert ownership_decision(0.0, 0.25) is True
    assert ownership_decision(0.25, 0.25) is True
    assert ownership_decision(0.2500001, 0.25) is False
    assert ownership_decision(0.5, 0.25) is False


def test_majority_vote_in_score_keyset():
    # 2 positions, 3 keys each; one key per position is wrong -> majority still correct
    positions = [0, 0, 0, 1, 1, 1]
    targets = [0, 0, 2, 2, 2, 0]            # bits: 0,0,1 (pos0) ; 1,1,0 (pos1)
    sig = np.array([0, 1], dtype=np.int64)
    bundle = _bundle(torch.zeros(6, 3, 32, 32), positions, targets)
    oracle = MockPredictAPI(lambda imgs: np.array(targets[: len(imgs)]))
    r = score_keyset(oracle, bundle, CBI, sig)
    assert r["bit_error_rate"] == 0.0
    assert r["recovered_signature"] == "01"
