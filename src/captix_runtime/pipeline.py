"""Image-to-text OCR pipeline using the CPU-only runtime components."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np

from .ctc import DecodingError, decode_fixed_length, decode_greedy, parse_mask
from .engine import EngineError
from .preprocessing import PreprocessingError, preprocess_image


class PipelineError(ValueError):
    """An invalid input, decode constraint, or inference failure."""

    def __init__(self, code: str, message: str | None = None):
        super().__init__(message or code)
        self.code = code


class EngineProtocol(Protocol):
    vocab: tuple[str, ...]
    version: str

    def infer(self, images: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Prediction:
    text: str
    decoder: str
    processing_ms: float
    timings_ms: dict[str, float]


def recognize(
    data: bytes,
    engine: EngineProtocol,
    *,
    mask: str | None = None,
    length: int | None = None,
    case_sensitive: bool = True,
) -> Prediction:
    if length is not None and (type(length) is not int or not 1 <= length <= 16):
        raise PipelineError("invalid_length", "length must be an integer in 1..16")

    try:
        parse_mask(mask, engine.vocab, case_sensitive=case_sensitive)
    except DecodingError as exc:
        raise PipelineError(
            "invalid_mask" if mask is not None else "invalid_vocabulary",
            "Invalid character mask or vocabulary",
        ) from exc

    started = perf_counter()
    phase_started = started
    try:
        image = preprocess_image(data)
    except PreprocessingError as exc:
        raise PipelineError(exc.code, str(exc)) from exc
    timings = {"preprocess": (perf_counter() - phase_started) * 1000}

    phase_started = perf_counter()
    try:
        logits = engine.infer(image.array)
    except EngineError as exc:
        raise PipelineError("engine_error", "OCR inference failed") from exc
    timings["inference"] = (perf_counter() - phase_started) * 1000

    if (
        not isinstance(logits, np.ndarray)
        or logits.ndim != 3
        or logits.shape[0] < 1
        or logits.shape[1:] != (1, len(engine.vocab) + 1)
    ):
        raise PipelineError("invalid_logits", "Engine returned invalid logits")

    phase_started = perf_counter()
    decoder = "fixed_length" if length is not None else "greedy"
    try:
        if length is None:
            text = decode_greedy(
                logits[:, 0, :], engine.vocab, mask, case_sensitive=case_sensitive
            )
        else:
            text = decode_fixed_length(
                logits[:, 0, :], engine.vocab, length, mask, case_sensitive=case_sensitive
            )
    except DecodingError as exc:
        code = (
            "unreachable_length"
            if length is not None
            and str(exc).startswith(("No CTC path", "Cannot decode length="))
            else "invalid_logits"
        )
        raise PipelineError(code, "CTC decoding failed") from exc
    timings["decode"] = (perf_counter() - phase_started) * 1000
    processing_ms = (perf_counter() - started) * 1000
    return Prediction(text, decoder, processing_ms, timings)
