import pytest

from captix import sequence_accuracy


def test_perfect():
    assert sequence_accuracy(["abc", "12"], ["abc", "12"]) == 1.0


def test_partial():
    assert sequence_accuracy(["abc", "12"], ["abc", "13"]) == 0.5


def test_empty():
    assert sequence_accuracy([], []) == 0.0


def test_length_mismatch():
    with pytest.raises(ValueError):
        sequence_accuracy(["a"], ["a", "b"])
