import os
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from captix_runtime.engine import EngineError, OnnxEngine
from service.main import create_app, create_app_from_env
from service.settings import Settings

LIMIT = 5 * 1024 * 1024
AUTH = {"Authorization": "Bearer test-secret"}


class FakeEngine:
    vocab = tuple("aA1")
    version = "captix-v1-onnx-fp32"

    def __init__(self):
        self.calls = 0
        self.warmed = False
        self.error = None

    def warmup(self):
        self.warmed = True

    def infer(self, images):
        self.calls += 1
        if self.error:
            raise self.error
        return np.array([[[0, 1, 9, 0]], [[9, 0, 0, 0]], [[0, 0, 0, 9]]], np.float32)


def image_bytes(fmt="PNG", size=(80, 32)):
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format=fmt)
    return output.getvalue()


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def app(engine):
    return create_app(engine, Settings(api_token="test-secret"))


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client


def post(client, data=None, mime="image/png", payload=None, headers=AUTH):
    return client.post(
        "/v1/ocr",
        headers=headers,
        data=data,
        files={
            "image": (
                "private.png",
                image_bytes() if payload is None else payload,
                mime,
            )
        },
    )


@pytest.mark.parametrize("token", ["", "   "])
def test_empty_token_fails_closed(monkeypatch, token):
    monkeypatch.setenv("CAPTIX_API_TOKEN", token)
    with pytest.raises(ValueError, match="CAPTIX_API_TOKEN"):
        create_app_from_env()
    with pytest.raises(ValueError, match="CAPTIX_API_TOKEN"):
        create_app(FakeEngine(), Settings(api_token=token))


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPTIX_API_TOKEN", "test-secret")
    monkeypatch.setenv("CAPTIX_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("ORT_INTRA_OP_THREADS", "3")
    settings = Settings.from_env()
    assert settings.model_dir == tmp_path
    assert settings.intra_op_threads == 3
    assert "test-secret" not in repr(settings)


@pytest.mark.parametrize("threads", ["0", "-1", "abc"])
def test_invalid_thread_setting_fails(monkeypatch, threads):
    monkeypatch.setenv("CAPTIX_API_TOKEN", "test-secret")
    monkeypatch.setenv("ORT_INTRA_OP_THREADS", threads)
    with pytest.raises(ValueError):
        Settings.from_env()


@pytest.mark.parametrize(
    "header", [None, "Bearer wrong", "Basic test-secret", "Bearer tèst"]
)
def test_missing_or_invalid_credentials(client, header, engine):
    response = post(
        client,
        headers={} if header is None else {"Authorization": header.encode("utf-8")},
    )
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert engine.calls == 0


def test_token_comparison_uses_constant_time(client, monkeypatch):
    from service import main

    original = main.secrets.compare_digest
    comparisons = []

    def compare(left, right):
        comparisons.append((left, right))
        return original(left, right)

    monkeypatch.setattr(main.secrets, "compare_digest", compare)
    assert post(client).status_code == 200
    assert post(client, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert len(comparisons) == 2


@pytest.mark.parametrize(
    "fmt,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")]
)
def test_valid_image_exact_response_contract(client, fmt, mime):
    response = post(client, payload=image_bytes(fmt), mime=mime)
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"text", "model_version", "decoder", "processing_ms"}
    assert result["text"] == "A1"
    assert result["model_version"] == FakeEngine.version
    assert result["decoder"] == "greedy"
    assert result["processing_ms"] >= 0


def test_forwards_mask_length_and_case(client):
    response = post(
        client, data={"length": "1", "mask": "[a]", "case_sensitive": "false"}
    )
    assert response.status_code == 200
    assert response.json()["decoder"] == "fixed_length"
    assert response.json()["text"] == "a"


def test_oversized_image_rejected_before_decode(client, monkeypatch, engine):
    def forbidden(*args, **kwargs):
        pytest.fail("oversized image reached decoder")

    monkeypatch.setattr(Image, "open", forbidden)
    response = post(client, payload=b"x" * (LIMIT + 1))
    assert response.status_code == 413
    assert engine.calls == 0


def test_image_read_is_limited(client, monkeypatch):
    from starlette.datastructures import UploadFile

    read = UploadFile.read
    sizes = []

    async def bounded_read(self, size=-1):
        sizes.append(size)
        return await read(self, size)

    monkeypatch.setattr(UploadFile, "read", bounded_read)
    assert post(client).status_code == 200
    assert sizes == [LIMIT + 1]


def test_exact_image_limit_accepted_without_disk(client, monkeypatch):
    import tempfile

    def forbidden(*args, **kwargs):
        pytest.fail("image spilled to disk")

    monkeypatch.setattr(tempfile, "TemporaryFile", forbidden)
    payload = image_bytes()
    response = post(client, payload=payload + b"\0" * (LIMIT - len(payload)))
    assert response.status_code == 200


@pytest.mark.parametrize(
    "mime,payload",
    [
        ("image/gif", image_bytes()),
        ("image/png", image_bytes("GIF")),
        ("image/jpeg", image_bytes()),
        ("application/octet-stream", image_bytes()),
    ],
)
def test_declared_and_actual_format_checked(client, mime, payload):
    assert post(client, mime=mime, payload=payload).status_code == 415


@pytest.mark.parametrize(
    "data",
    [
        {"length": "0"},
        {"length": "17"},
        {"length": "1.0"},
        {"length": "true"},
        {"length": "16"},
        {"mask": ""},
        {"mask": ".*"},
        {"mask": "[z]"},
        {"case_sensitive": "perhaps"},
        {"unexpected": "value"},
    ],
)
def test_invalid_arguments(client, data):
    assert post(client, data=data).status_code == 422


@pytest.mark.parametrize(
    "payload", [b"", b"private corrupted bytes", image_bytes()[:40]]
)
def test_corrupt_image_returns_422(client, payload):
    assert post(client, payload=payload).status_code == 422


@pytest.mark.parametrize("size", [(3, 32), (1025, 32), (2001, 2000)])
def test_invalid_dimensions(client, size):
    assert post(client, payload=image_bytes(size=size)).status_code == 422


def test_animated_image_rejected(client):
    output = BytesIO()
    Image.new("RGB", (80, 32), "red").save(
        output,
        format="PNG",
        save_all=True,
        append_images=[Image.new("RGB", (80, 32), "blue")],
        duration=100,
    )
    assert post(client, payload=output.getvalue()).status_code == 415


@pytest.mark.parametrize(
    "body,content_type",
    [
        (b"broken", "multipart/form-data"),
        (b"broken", "multipart/form-data; boundary=abc"),
        (
            b'--abc\r\nContent-Disposition: form-data; name="image"; filename="x"\r\n\r\nx',
            "multipart/form-data; boundary=abc",
        ),
    ],
)
def test_malformed_multipart_returns_400(client, body, content_type):
    response = client.post(
        "/v1/ocr", content=body, headers={**AUTH, "Content-Type": content_type}
    )
    assert response.status_code == 400


def test_missing_image_and_non_multipart(client):
    assert (
        client.post(
            "/v1/ocr", headers=AUTH, files={"other": ("x", b"x", "image/png")}
        ).status_code
        == 422
    )
    assert client.post("/v1/ocr", headers=AUTH, json={}).status_code == 415


def test_duplicate_image_or_fields_rejected(client):
    images = [("image", ("x", image_bytes(), "image/png"))] * 2
    assert client.post("/v1/ocr", headers=AUTH, files=images).status_code == 400
    fields = [
        ("image", ("x", image_bytes(), "image/png")),
        ("mask", (None, "[a]")),
        ("mask", (None, "[A]")),
    ]
    assert client.post("/v1/ocr", headers=AUTH, files=fields).status_code == 422


def test_excessive_field_and_header_bounded(client):
    assert post(client, data={"mask": "x" * 65537}).status_code in {400, 413}
    assert client.post(
        "/v1/ocr", headers=AUTH, files={"image": ("x" * 65537, b"x", "image/png")}
    ).status_code in {400, 413}


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("private internal detail"),
        EngineError("private engine detail"),
        ValueError("private invalid internal value"),
    ],
)
def test_internal_error_is_generic(client, engine, error, caplog):
    engine.error = error
    response = post(client)
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "private" not in response.text + caplog.text
    assert "test-secret" not in caplog.text


def test_health_tracks_lifespan_warmup(app, engine):
    with TestClient(app) as ignored:
        assert engine.warmed
        assert ignored.get("/health/live").status_code == 200
        assert ignored.get("/health/ready").status_code == 200
    with TestClient(app, raise_server_exceptions=False) as ignored:
        assert ignored.get("/health/ready").status_code == 200
    client = TestClient(app)
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 503
    assert post(client).status_code == 503


def test_warmup_supports_engine_protocol_without_warmup_method():
    class MinimalEngine:
        vocab = FakeEngine.vocab
        version = FakeEngine.version

        def __init__(self):
            self.widths = []

        def infer(self, images):
            self.widths.append(images.shape[-1])
            return FakeEngine().infer(images)

    engine = MinimalEngine()
    with TestClient(create_app(engine, Settings(api_token="test-secret"))) as client:
        assert client.get("/health/ready").status_code == 200
        assert engine.widths == [32]
        assert post(client).status_code == 200
        assert engine.widths == [32, 80]


def test_production_factory_passes_only_local_settings(monkeypatch, tmp_path):
    from service import main

    monkeypatch.setenv("CAPTIX_API_TOKEN", "test-secret")
    monkeypatch.setenv("CAPTIX_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("ORT_INTRA_OP_THREADS", "4")
    constructions = []

    def local_engine(model_dir, *, intra_op_threads):
        constructions.append((model_dir, intra_op_threads))
        return FakeEngine()

    monkeypatch.setattr(main, "OnnxEngine", local_engine)
    app = create_app_from_env()
    assert constructions == [(tmp_path, 4)]
    with TestClient(app) as client:
        assert post(client).status_code == 200


def test_failed_warmup_never_marks_ready(engine):
    def fail():
        raise EngineError("warmup failed")

    engine.warmup = fail
    app = create_app(engine, Settings(api_token="test-secret"))
    with pytest.raises(EngineError), TestClient(app):
        pass
    assert TestClient(app).get("/health/ready").status_code == 503


def test_production_factory_rejects_missing_local_model(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPTIX_API_TOKEN", "test-secret")
    monkeypatch.setenv("CAPTIX_MODEL_DIR", str(tmp_path))
    with pytest.raises(EngineError, match="manifest"):
        create_app_from_env()


@pytest.mark.skipif(
    not os.environ.get("CAPTIX_TEST_MODEL_DIR"),
    reason="requires verified local ONNX artifact",
)
def test_optional_real_artifact(monkeypatch):
    monkeypatch.setenv("CAPTIX_API_TOKEN", "test-secret")
    monkeypatch.setenv("CAPTIX_MODEL_DIR", os.environ["CAPTIX_TEST_MODEL_DIR"])
    app = create_app_from_env()
    assert isinstance(app.state.engine, OnnxEngine)
    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 200
        assert post(client).status_code == 200
