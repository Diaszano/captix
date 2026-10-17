import numpy as np
import pytest

from captix.dataset import decode_indices
from captix.decrypt import _ctc_decode_fixed_length
from captix_runtime.ctc import (
    DecodingError,
    _log_probs,
    decode_fixed_length,
    decode_greedy,
    parse_mask,
)


def _reference_decode(logits, vocab, length, mask=None):
    allowed = parse_mask(mask, vocab)
    return decode_indices(
        _ctc_decode_fixed_length(_log_probs(logits.astype(np.float64), allowed), length),
        vocab,
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


def test_top_two_excluding_self_finds_two_distinct_maxima():
    from captix_runtime.ctc import top_two_excluding_self

    values = np.array([[4.0, 4.0, 1.0, -np.inf], [2.0, 3.0, 3.0, 1.0]])
    best, second = top_two_excluding_self(values)
    assert best.tolist() == [0, 1]
    assert second.tolist() == [1, 2]


@pytest.mark.parametrize("seed", [7, 41, 2026])
def test_fixed_length_matches_reference_for_seeded_shapes_masks_and_ties(seed):
    rng = np.random.default_rng(seed)
    vocab = list("abcdef")
    for T, C, L in ((1, 2, 1), (2, 4, 2), (8, 7, 4), (20, 7, 16)):
        logits = rng.integers(-2, 3, size=(T, C)).astype(np.float32)
        for mask in (None, "[a-c]"):
            expected = _reference_decode(logits, vocab[: C - 1], L, mask)
            assert decode_fixed_length(logits, vocab[: C - 1], L, mask) == expected


def test_fixed_length_matches_reference_at_t_length_and_preserves_tie_breaks():
    logits = np.zeros((4, 5), dtype=np.float32)
    assert decode_fixed_length(logits, list("abcd"), 4) == _reference_decode(
        logits, list("abcd"), 4
    )


def test_fixed_length_matches_reference_for_long_benchmark_shapes():
    rng = np.random.default_rng(1042)
    vocab = [chr(33 + i) for i in range(62)]
    for T in (20, 40, 80):
        for L in (4, 5):
            logits = rng.normal(size=(T, 63)).astype(np.float32)
            assert decode_fixed_length(logits, vocab, L) == _reference_decode(
                logits, vocab, L
            )


def test_decoder_benchmark_reports_shape_latency_and_reference_match():
    from benchmarks.bench_decoder import (
        DecoderCase,
        _benchmark_cases,
        bench_decoder,
    )

    for case in _benchmark_cases():
        assert case.reference == _reference_decode(
            case.logits, case.vocab, case.length, case.mask
        )

    result = bench_decoder(
        [DecoderCase(np.zeros((2, 2), dtype=np.float32), ["a"], 1, "a")],
        iterations=1,
    )
    case = result["cases"][0]
    assert (case["T"], case["L"], case["C"]) == (2, 1, 2)
    assert case["matches_reference"] is True
    assert result["iterations"] == 1
    assert case["latency_ms"] >= 0
