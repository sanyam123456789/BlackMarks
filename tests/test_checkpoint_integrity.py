"""Step 8G -- checkpoint integrity + loadability. No training."""
import pytest
import torch

from src.classifier.bm_common import (
    BLACKMARKS_CHECKPOINT, CLEAN_CHECKPOINT, CLEAN_MODEL_SHA256, STEP7_WATERMARKED,
    assert_clean_model_intact, load_state_dict_checkpoint, sha256_file,
)
from src.classifier.model import build_model


def test_clean_model_hash_is_frozen_constant():
    assert CLEAN_CHECKPOINT.exists(), "clean_model.pt must be present"
    assert sha256_file(CLEAN_CHECKPOINT) == CLEAN_MODEL_SHA256
    assert assert_clean_model_intact("pytest") == CLEAN_MODEL_SHA256


def test_clean_model_loads_into_compactcnn():
    model = build_model()
    meta = load_state_dict_checkpoint(model, CLEAN_CHECKPOINT, torch.device("cpu"))
    assert meta.get("architecture") == "CompactCNN"
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 32, 32))
    assert out.shape == (2, 10)


@pytest.mark.skipif(not STEP7_WATERMARKED.exists(), reason="Step 7 checkpoint absent")
def test_step7_checkpoint_still_loads():
    model = build_model()
    load_state_dict_checkpoint(model, STEP7_WATERMARKED, torch.device("cpu"))
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 32, 32)).shape == (1, 10)


@pytest.mark.skipif(not BLACKMARKS_CHECKPOINT.exists(), reason="Step 8C checkpoint not yet built")
def test_blackmarks_checkpoint_is_separate_file_and_loads():
    assert BLACKMARKS_CHECKPOINT != CLEAN_CHECKPOINT
    assert sha256_file(BLACKMARKS_CHECKPOINT) != CLEAN_MODEL_SHA256
    model = build_model()
    meta = load_state_dict_checkpoint(model, BLACKMARKS_CHECKPOINT, torch.device("cpu"))
    assert meta.get("owner_signature")
    assert "class_to_bit_index" in meta
