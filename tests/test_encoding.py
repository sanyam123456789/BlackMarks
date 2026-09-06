"""Step 8G -- owner signature + class-to-bit mapping. No training (tiny forward only)."""
import numpy as np
import pytest
import torch

from src.classifier.encoding import (
    build_class_bit_mapping, compute_class_mean_features, derive_owner_signature,
    deserialize_signature, serialize_signature,
)
from src.classifier.bm_common import CLEAN_CHECKPOINT, load_state_dict_checkpoint
from src.classifier.model import build_model


def test_signature_is_deterministic():
    a = derive_owner_signature(length=16, seed=42)
    b = derive_owner_signature(length=16, seed=42)
    assert np.array_equal(a, b)
    assert a.tolist() == derive_owner_signature(16, 42).tolist()


def test_signature_length_and_domain():
    for L in (1, 4, 8, 16, 32, 64):
        s = derive_owner_signature(length=L, seed=42)
        assert s.shape == (L,)
        assert set(np.unique(s).tolist()) <= {0, 1}


def test_signature_changes_with_seed_or_owner():
    base = derive_owner_signature(16, 42, "owner-A")
    assert not np.array_equal(base, derive_owner_signature(16, 43, "owner-A"))
    assert not np.array_equal(base, derive_owner_signature(16, 42, "owner-B"))


def test_signature_serialization_roundtrip():
    s = derive_owner_signature(16, 42)
    assert np.array_equal(deserialize_signature(serialize_signature(s)), s)


def test_signature_length_bounds():
    with pytest.raises(ValueError):
        derive_owner_signature(length=0)
    with pytest.raises(ValueError):
        derive_owner_signature(length=257)


def test_class_bit_mapping_valid_partition():
    """All 10 classes mapped; both bit groups non-empty; deterministic."""
    rng = np.random.default_rng(0)
    fake_means = rng.normal(size=(10, 32))
    fake_means[:4] += 6.0      # force a separable structure
    m1 = build_class_bit_mapping(fake_means, seed=42)
    m2 = build_class_bit_mapping(fake_means, seed=42)
    assert m1["class_to_bit_index"] == m2["class_to_bit_index"]
    assert len(m1["class_to_bit_index"]) == 10
    assert set(m1["class_to_bit_index"]) == {0, 1}
    assert m1["group_sizes"]["bit0"] + m1["group_sizes"]["bit1"] == 10


def test_class_bit_mapping_fallback_on_degenerate():
    """A near-identical class-mean matrix must trigger the balanced fallback."""
    means = np.ones((10, 16)) + np.random.default_rng(1).normal(scale=1e-6, size=(10, 16))
    m = build_class_bit_mapping(means, seed=42)
    assert set(m["class_to_bit_index"]) == {0, 1}
    assert min(m["group_sizes"].values()) >= 1


@pytest.mark.skipif(not CLEAN_CHECKPOINT.exists(), reason="clean_model.pt absent")
def test_real_model_mapping_uses_small_sample_fast():
    device = torch.device("cpu")
    model = build_model().to(device)
    load_state_dict_checkpoint(model, CLEAN_CHECKPOINT, device)
    means = compute_class_mean_features(model, seed=42, sample=200, device=device)
    assert means.shape[0] == 10
    mapping = build_class_bit_mapping(means, seed=42)
    assert set(mapping["class_to_bit_index"]) == {0, 1}
