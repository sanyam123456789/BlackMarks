"""Step 8G -- bit maths: Hamming distance, BER, decoding, majority vote. No training."""
import numpy as np
import pytest

from src.classifier.bm_common import (
    bit_error_rate, bits_to_string, hamming_distance, majority_vote_bits,
    string_to_bits,
)
from src.classifier.encoding import decode_classes_to_bits


def test_bits_string_roundtrip():
    for s in ["0", "1", "0110011101000110", "1111", "0000"]:
        assert bits_to_string(string_to_bits(s)) == s


def test_string_to_bits_rejects_non_binary():
    for bad in ["", "012", "10a1", "  "]:
        with pytest.raises(ValueError):
            string_to_bits(bad)


def test_hamming_distance_basic():
    assert hamming_distance([0, 0, 0, 0], [0, 0, 0, 0]) == 0
    assert hamming_distance([0, 1, 0, 1], [1, 1, 0, 0]) == 2
    assert hamming_distance([1, 1, 1], [0, 0, 0]) == 3


def test_hamming_distance_length_mismatch():
    with pytest.raises(ValueError):
        hamming_distance([0, 1], [0, 1, 1])


def test_bit_error_rate_range_and_values():
    assert bit_error_rate([0, 1, 0, 1], [0, 1, 0, 1]) == 0.0
    assert bit_error_rate([0, 1, 0, 1], [1, 0, 1, 0]) == 1.0
    assert bit_error_rate([0, 0, 0, 0], [1, 1, 0, 0]) == 0.5


def test_decode_classes_to_bits():
    # classes 0,8 -> bit 0 ; classes 2,3 -> bit 1
    cbi = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    out = decode_classes_to_bits([0, 2, 8, 3], cbi)
    assert out.tolist() == [0, 1, 0, 1]


def test_decode_rejects_out_of_range():
    with pytest.raises(ValueError):
        decode_classes_to_bits([0, 99], [0, 1])


def test_majority_vote_basic():
    # 2 positions, 3 keys each
    per_key = [1, 1, 0, 0, 0, 1]
    positions = [0, 0, 0, 1, 1, 1]
    recovered, detail = majority_vote_bits(per_key, positions, length=2)
    assert recovered.tolist() == [1, 0]
    assert detail[0]["votes_1"] == 2 and detail[0]["votes_0"] == 1
    assert detail[1]["tie"] is False


def test_majority_vote_tie_resolves_to_one_and_flags():
    recovered, detail = majority_vote_bits([0, 1], [0, 0], length=1)
    assert recovered.tolist() == [1]
    assert detail[0]["tie"] is True


def test_majority_vote_empty_position():
    recovered, detail = majority_vote_bits([1, 1], [0, 0], length=2)
    assert recovered.tolist() == [1, 0]
    assert detail[1]["n_keys"] == 0
