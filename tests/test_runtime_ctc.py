import numpy as np
import pytest

from captix.dataset import decode_indices
from captix.decrypt import _ctc_decode_fixed_length
from captix_runtime.ctc import (
    DecodingError,
    decode_fixed_length,
    decode_greedy,
    parse_mask,
)


def test_greedy_ctc_collapses_repeats_and_blanks():
    logits = np.array([[0, 3, 1], [0, 3, 1], [0, 1, 3], [0, 3, 1]], dtype=np.float32)
    assert decode_greedy(logits, ["a", "b"]) == "aba"
    assert (
        decode_greedy(
            np.array([[0, 3, 1], [0, 3, 1], [3, 0, 0], [0, 3, 1]]), ["a", "b"]
        )
        == "aa"
    )
    assert decode_greedy(np.array([[0, 3, 1], [0, 3, 1]]), ["a", "b"]) == "a"


def test_parse_mask_only_accepts_ascii_character_classes():
    vocab = list("012abAZ")
    assert parse_mask("[0-9]", vocab).tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert parse_mask("[a-zA-Z0-9]", vocab).tolist() == [True] * 8
    for mask in (
        "[a-]",
        "[a-z",
        "[]",
        r"[\d]",
        "[.*]",
        "[^a]",
        "(a)",
        "[z-a]",
        "[A-z]",
        "[x]",
    ):
        with pytest.raises(DecodingError):
            parse_mask(mask, vocab)


def test_parse_mask_rejects_empty_vocabulary_without_mask():
    with pytest.raises(DecodingError):
        parse_mask(None, [])


def test_case_insensitive_mask_lowercases_decoded_text():
    logits = np.array([[0, 0, 5], [0, 0, 5]], dtype=np.float32)
    assert decode_greedy(logits, ["a", "A"], case_sensitive=False) == "a"


@pytest.mark.parametrize("length", [1, 2, 3])
@pytest.mark.parametrize(
    "logits",
    [
        np.random.default_rng(41).normal(size=(4, 3)).astype(np.float32),
        np.array([[0, 1, 1], [0, 1, 1], [1, 0, 1]], dtype=np.float32),
    ],
)
def test_fixed_length_matches_reference_decoder(logits, length):
    vocab = ["a", "b"]
    stable = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    path = _ctc_decode_fixed_length(stable, length)
    assert decode_fixed_length(logits, vocab, length) == decode_indices(path, vocab)


def test_fixed_length_rejects_infeasible_masks_and_bad_logits():
    with pytest.raises(DecodingError):
        decode_fixed_length(np.zeros((2, 2)), ["a"], 2, mask="[a]")
    for logits in (
        np.array([[np.nan, 0]]),
        np.array([[np.inf, 0]]),
        np.zeros((2, 2, 1)),
    ):
        with pytest.raises(DecodingError):
            decode_greedy(logits, ["a"])
    for length in (0, 17):
        with pytest.raises(DecodingError):
            decode_fixed_length(np.zeros((2, 2)), ["a"], length)
