"""NumPy-only CTC decoding for the inference runtime."""

from __future__ import annotations

import string
from collections.abc import Sequence

import numpy as np


class DecodingError(ValueError):
    """Invalid logits, mask, or decoding constraints."""


def parse_mask(
    mask: str | None, vocab: Sequence[str], *, case_sensitive: bool = True
) -> np.ndarray:
    """Return a C-sized allowed-class mask, with the blank class enabled."""
    allowed = np.zeros(len(vocab) + 1, dtype=bool)
    allowed[0] = True
    if mask is None:
        allowed[1:] = True
        return allowed
    if not isinstance(mask, str) or len(mask) < 3 or mask[0] != "[" or mask[-1] != "]":
        raise DecodingError("Mask must be an ASCII alphanumeric character class")

    body = mask[1:-1]
    chars: set[str] = set()
    i = 0
    while i < len(body):
        start = body[i]
        if start not in string.ascii_letters + string.digits:
            raise DecodingError("Mask contains unsupported syntax")
        if i + 1 < len(body) and body[i + 1] == "-":
            if i + 2 >= len(body):
                raise DecodingError("Mask contains an incomplete range")
            end = body[i + 2]
            if end not in string.ascii_letters + string.digits or ord(start) > ord(end):
                raise DecodingError("Mask ranges must be ascending ASCII alphanumerics")
            expanded = {chr(code) for code in range(ord(start), ord(end) + 1)}
            if not expanded <= set(string.ascii_letters + string.digits):
                raise DecodingError("Mask ranges may contain only ASCII alphanumerics")
            chars.update(expanded)
            i += 3
        else:
            chars.add(start)
            i += 1

    if not chars:
        raise DecodingError("Mask cannot be empty")
    if not case_sensitive:
        chars = {char.lower() for char in chars} | {char.upper() for char in chars}
    for index, char in enumerate(vocab, 1):
        allowed[index] = char in chars
    if not allowed[1:].any():
        raise DecodingError("Mask does not allow any vocabulary characters")
    return allowed


def _validate_logits(logits: np.ndarray, vocab: Sequence[str]) -> np.ndarray:
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] != len(vocab) + 1 or values.shape[0] < 1:
        raise DecodingError("Logits must have shape [T, len(vocab) + 1] with T >= 1")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise DecodingError("Logits cannot contain NaN or positive infinity")
    if np.isneginf(values).all(axis=1).any():
        raise DecodingError("Each timestep must have at least one finite logit")
    return values.astype(np.float64, copy=False)


def _log_probs(logits: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    masked = np.where(allowed, logits, -np.inf)
    maxima = masked.max(axis=1, keepdims=True)
    if np.isneginf(maxima).any():
        raise DecodingError("Mask leaves a timestep with no allowed class")
    shifted = masked - maxima
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def _collapse(path: Sequence[int], vocab: Sequence[str], case_sensitive: bool) -> str:
    output: list[str] = []
    previous = -1
    for index in path:
        if index != previous and index != 0:
            output.append(vocab[index - 1])
        previous = index
    text = "".join(output)
    return text if case_sensitive else text.lower()


def decode_greedy(
    logits: np.ndarray,
    vocab: Sequence[str],
    mask: str | None = None,
    *,
    case_sensitive: bool = True,
) -> str:
    """Greedy CTC decode logits shaped [T,C] (softmax is unnecessary)."""
    values = _validate_logits(logits, vocab)
    allowed = parse_mask(mask, vocab, case_sensitive=case_sensitive)
    masked = np.where(allowed[None, :], values, -np.inf)
    if np.isneginf(masked).all(axis=1).any():
        raise DecodingError("Mask leaves a timestep with no allowed class")
    path = masked.argmax(axis=1)
    return _collapse(path, vocab, case_sensitive)


def decode_fixed_length(
    logits: np.ndarray,
    vocab: Sequence[str],
    length: int,
    mask: str | None = None,
    *,
    case_sensitive: bool = True,
) -> str:
    """Decode the highest-scoring CTC path with exactly ``length`` labels."""
    values = _validate_logits(logits, vocab)
    if not isinstance(length, int) or not 1 <= length <= 16:
        raise DecodingError("length must be an integer in 1..16")
    if values.shape[0] < length:
        raise DecodingError(
            f"Cannot decode length={length}: only T={values.shape[0]} timesteps"
        )
    allowed = parse_mask(mask, vocab, case_sensitive=case_sensitive)
    log_probs = _log_probs(values, allowed)

    T, C = log_probs.shape
    L = length
    neg = -np.inf
    dp = np.full((L + 1, C), neg)
    dp[0, 0] = 0.0
    backpointers: list[np.ndarray] = []
    states = np.arange(L + 1)
    classes = np.arange(C)[None, :]
    for timestep in range(T):
        lp = log_probs[timestep]
        new_dp = np.full((L + 1, C), neg)
        backpointer = np.full((L + 1, C), -1, dtype=np.int64)

        best_value = dp.max(axis=1)
        best_class = dp.argmax(axis=1)
        candidate = best_value + lp[0]
        improved = candidate > new_dp[:, 0]
        new_dp[improved, 0] = candidate[improved]
        backpointer[improved, 0] = (states * C + best_class)[improved]

        candidate = dp + lp[None, :]
        candidate[:, 0] = neg
        improved = candidate > new_dp
        new_dp = np.where(improved, candidate, new_dp)
        backpointer = np.where(improved, states[:, None] * C + classes, backpointer)

        order = np.argsort(-dp, axis=1)
        first_class, second_class = order[:, 0], order[:, 1]
        first_value, second_value = dp[states, first_class], dp[states, second_class]
        is_first = first_class[:, None] == classes
        best_other_value = np.where(
            is_first, second_value[:, None], first_value[:, None]
        )
        best_other_class = np.where(
            is_first, second_class[:, None], first_class[:, None]
        )
        candidate = best_other_value + lp[None, :]
        candidate[:, 0] = neg
        improved = candidate[:-1] > new_dp[1:]
        new_dp[1:] = np.where(improved, candidate[:-1], new_dp[1:])
        backpointer[1:] = np.where(
            improved, states[:-1, None] * C + best_other_class[:-1], backpointer[1:]
        )

        dp = new_dp
        backpointers.append(backpointer)

    final = dp[L]
    if not np.isfinite(final).any():
        raise DecodingError(f"No CTC path of length {L} is feasible for these logits")
    last = int(np.argmax(final))
    path = [last]
    state = L
    for timestep in range(T - 1, 0, -1):
        previous = int(backpointers[timestep][state, last])
        state, last = divmod(previous, C)
        path.append(last)
    path.reverse()
    return _collapse(path, vocab, case_sensitive)
