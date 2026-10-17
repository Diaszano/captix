from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from captix_runtime.pipeline import PipelineError, recognize


def image_bytes():
    stream = BytesIO()
    Image.new("RGB", (16, 32), "white").save(stream, format="PNG")
    return stream.getvalue()


class FakeEngine:
    version = "test"

    def __init__(self, logits, vocab=("a", "b")):
        self.logits = logits
        self.vocab = vocab
        self.images = []

    def infer(self, images):
        self.images.append(images)
        return self.logits


def test_recognize_composes_preprocess_engine_and_greedy_decode():
    engine = FakeEngine(
        np.array([[[0, 4, 1]], [[0, 4, 1]], [[0, 1, 4]]], dtype=np.float32)
    )

    result = recognize(image_bytes(), engine)

    assert result.text == "ab"
    assert result.decoder == "greedy"
    assert set(result.timings_ms) == {"preprocess", "inference", "decode"}
    assert all(np.isfinite(value) and value >= 0 for value in result.timings_ms.values())
    assert result.processing_ms >= 0
    assert engine.images[0].shape == (1, 3, 32, 16)


def test_recognize_forwards_mask_case_and_fixed_length():
    engine = FakeEngine(
        np.array([[[0, 4, 0]], [[0, 0, 4]], [[4, 0, 0]]], dtype=np.float32),
        vocab=("A", "b"),
    )

    result = recognize(
        image_bytes(), engine, mask="[A]", case_sensitive=False, length=1
    )

    assert result.text == "a"
    assert result.decoder == "fixed_length"
    assert not hasattr(result, "confidence")


@pytest.mark.parametrize("length", [0, 17, True, 1.5])
def test_rejects_invalid_length(length):
    with pytest.raises(PipelineError) as exc:
        recognize(image_bytes(), FakeEngine(np.zeros((1, 1, 3), np.float32)), length=length)
    assert exc.value.code == "invalid_length"


def test_rejects_mask_with_no_matching_vocabulary_character():
    with pytest.raises(PipelineError) as exc:
        recognize(image_bytes(), FakeEngine(np.zeros((1, 1, 3), np.float32)), mask="[0-9]")
    assert exc.value.code == "invalid_mask"


def test_reports_unreachable_fixed_length():
    engine = FakeEngine(np.array([[[0, 4, 0]], [[0, 4, 0]]], dtype=np.float32))

    with pytest.raises(PipelineError) as exc:
        recognize(image_bytes(), engine, length=2, mask="[a]")

    assert exc.value.code == "unreachable_length"


@pytest.mark.parametrize(
    "logits",
    [np.zeros((0, 1, 3), np.float32), np.zeros((1, 2, 3), np.float32), np.zeros((1, 3), np.float32)],
)
def test_rejects_invalid_engine_output_shape(logits):
    with pytest.raises(PipelineError) as exc:
        recognize(image_bytes(), FakeEngine(logits))
    assert exc.value.code == "invalid_logits"


def test_does_not_log_image_bytes_or_prediction_text(caplog):
    raw = image_bytes()
    token = raw.decode("latin1")
    secret = "a"

    recognize(raw, FakeEngine(np.array([[[0, 4, 0]]], dtype=np.float32)))

    assert token not in caplog.text
    assert secret not in caplog.text


def test_maps_typed_image_and_engine_errors():
    with pytest.raises(PipelineError) as image_error:
        recognize(b"not an image", FakeEngine(np.zeros((1, 1, 3), np.float32)))
    assert image_error.value.code == "invalid_image"

    class BrokenEngine(FakeEngine):
        def infer(self, images):
            from captix_runtime.engine import EngineError

            raise EngineError("private engine detail")

    with pytest.raises(PipelineError) as engine_error:
        recognize(image_bytes(), BrokenEngine(np.zeros((1, 1, 3), np.float32)))
    assert engine_error.value.code == "engine_error"
    assert "private engine detail" not in str(engine_error.value)
