import asyncio
import os
import threading
from io import BytesIO

import anyio
import httpx
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


def test_unknown_argument_with_all_supported_fields_returns_422(client):
    response = post(
        client,
        data={
            "mask": "[a]",
            "length": "1",
            "case_sensitive": "false",
            "unknown": "value",
        },
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Unknown form field"}


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


class BlockingEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.entered = threading.Barrier(3)
        self.release = threading.Event()

    def infer(self, images):
        self.entered.wait(timeout=5)
        if not self.release.wait(timeout=5):
            raise RuntimeError("test failed to release native inference")
        return super().infer(images)


async def async_post(client):
    return await client.post(
        "/v1/ocr",
        headers=AUTH,
        files={"image": ("private.png", image_bytes(), "image/png")},
    )


@pytest.mark.parametrize("repeat", range(10))
def test_saturation_cancellation_and_recovery(repeat):
    async def exercise():
        engine = BlockingEngine()
        app = create_app(engine, Settings(api_token="test-secret"))
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client,
        ):
            cancelled = anyio.CancelScope()
            finished = anyio.Event()

            async def cancellable_request():
                with cancelled:
                    await async_post(client)
                finished.set()

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(cancellable_request)
                tasks.start_soon(async_post, client)
                try:
                    await asyncio.to_thread(engine.entered.wait, 5)
                    response = await asyncio.wait_for(async_post(client), 1)
                    assert response.status_code == 429
                    assert response.json() == {"detail": "Inference capacity exhausted"}
                    cancelled.cancel()
                    await anyio.lowlevel.checkpoint()
                    assert not finished.is_set()
                    assert (
                        await asyncio.wait_for(async_post(client), 1)
                    ).status_code == 429
                finally:
                    engine.release.set()
            assert engine.calls == 2
            # Stop blocking subsequent calls; both completed workers must release slots.
            engine.entered = threading.Barrier(1)
            assert (await async_post(client)).status_code == 200

    asyncio.run(exercise())


def test_shutdown_disables_admissions_and_waits_for_native_work():
    async def exercise():
        engine = BlockingEngine()
        app = create_app(engine, Settings(api_token="test-secret"))
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            requests = [asyncio.create_task(async_post(client)) for _ in range(2)]
            shutdown = None
            try:
                await asyncio.to_thread(engine.entered.wait, 5)
                shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
                for _ in range(100):
                    if not app.state.ready:
                        break
                    await anyio.sleep(0.001)
                assert (await client.get("/health/ready")).status_code == 503
                assert (await async_post(client)).status_code == 503
                assert not shutdown.done()
            finally:
                engine.release.set()
                responses = await asyncio.gather(*requests)
                if shutdown is not None:
                    await shutdown
                else:
                    await lifespan.__aexit__(None, None, None)
            assert [response.status_code for response in responses] == [200, 200]

    asyncio.run(exercise())


def test_worker_failure_releases_slots(client, engine):
    engine.error = RuntimeError("private engine failure")
    for _ in range(3):
        assert post(client).status_code == 500
    engine.error = None
    assert post(client).status_code == 200


def test_cancelled_asyncio_request_keeps_native_slot():
    async def exercise():
        engine = BlockingEngine()
        app = create_app(engine, Settings(api_token="test-secret"))
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client,
        ):
            requests = [asyncio.create_task(async_post(client)) for _ in range(2)]
            try:
                await asyncio.to_thread(engine.entered.wait, 5)
                requests[0].cancel()
                with pytest.raises(asyncio.CancelledError):
                    await requests[0]
                assert (
                    await asyncio.wait_for(async_post(client), 1)
                ).status_code == 429
            finally:
                engine.release.set()
                await asyncio.gather(*requests, return_exceptions=True)

    asyncio.run(exercise())


def test_upload_finishing_after_shutdown_cannot_enter_engine(monkeypatch):
    from service import main

    async def exercise():
        engine = FakeEngine()
        app = create_app(engine, Settings(api_token="test-secret"))
        entered = anyio.Event()
        release = anyio.Event()
        read_form = main._read_image_form

        async def paused_read(request):
            result = await read_form(request)
            entered.set()
            await release.wait()
            return result

        monkeypatch.setattr(main, "_read_image_form", paused_read)
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            request = asyncio.create_task(async_post(client))
            shutdown = None
            try:
                await entered.wait()
                shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
                while app.state.ready:
                    await anyio.lowlevel.checkpoint()
            finally:
                release.set()
                response = await request
                if shutdown is not None:
                    await shutdown
                else:
                    await lifespan.__aexit__(None, None, None)
            assert response.status_code == 503
            assert engine.calls == 0

    asyncio.run(exercise())


def test_admission_rejects_before_parsing_with_shared_threadpool_saturated(monkeypatch):
    from service import main

    async def exercise():
        engine = BlockingEngine()
        app = create_app(engine, Settings(api_token="test-secret"))
        parsed = 0
        parsed_two = anyio.Event()
        read_form = main._read_image_form

        async def counted_read(request):
            nonlocal parsed
            parsed += 1
            result = await read_form(request)
            if parsed == 2:
                parsed_two.set()
            return result

        monkeypatch.setattr(main, "_read_image_form", counted_read)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client,
        ):
            limiter = anyio.to_thread.current_default_thread_limiter()
            original_tokens = limiter.total_tokens
            limiter.total_tokens = 2
            blockers_entered = threading.Barrier(3)
            unblock = threading.Event()

            def block_shared_thread():
                blockers_entered.wait(timeout=5)
                assert unblock.wait(timeout=5)

            blockers = [
                asyncio.create_task(anyio.to_thread.run_sync(block_shared_thread))
                for _ in range(2)
            ]
            requests = []
            try:
                await asyncio.to_thread(blockers_entered.wait, 5)
                requests = [asyncio.create_task(async_post(client)) for _ in range(2)]
                with anyio.fail_after(2):
                    await parsed_two.wait()
                await asyncio.to_thread(engine.entered.wait, 2)
                assert limiter.borrowed_tokens == 2
                response = await asyncio.wait_for(async_post(client), 1)
                assert response.status_code == 429
                assert parsed == 2
            finally:
                engine.release.set()
                unblock.set()
                await asyncio.gather(*requests, *blockers)
                limiter.total_tokens = original_tokens

    asyncio.run(exercise())


def test_cancelled_upload_releases_reserved_admission(monkeypatch):
    from service import main

    async def exercise():
        engine = FakeEngine()
        app = create_app(engine, Settings(api_token="test-secret"))
        entered = 0
        parsed_two = anyio.Event()
        release = anyio.Event()
        read_form = main._read_image_form

        async def paused_read(request):
            nonlocal entered
            result = await read_form(request)
            entered += 1
            if entered == 2:
                parsed_two.set()
            await release.wait()
            return result

        monkeypatch.setattr(main, "_read_image_form", paused_read)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client,
        ):
            requests = [asyncio.create_task(async_post(client)) for _ in range(2)]
            try:
                await parsed_two.wait()
                assert (
                    await asyncio.wait_for(async_post(client), 1)
                ).status_code == 429
                requests[0].cancel()
                with pytest.raises(asyncio.CancelledError):
                    await requests[0]
            finally:
                release.set()
                await asyncio.gather(*requests, return_exceptions=True)
            assert (await async_post(client)).status_code == 200
            assert engine.calls == 2

    asyncio.run(exercise())


def test_cancellation_before_worker_claim_does_not_run_inference(monkeypatch):
    from service import main

    async def exercise():
        engine = FakeEngine()
        app = create_app(engine, Settings(api_token="test-secret"))
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client,
        ):
            run_sync = main.run_sync
            entered = threading.Event()
            release = threading.Event()
            finished = threading.Event()

            async def delayed_submission(func, *args, **kwargs):
                def delayed_worker():
                    entered.set()
                    try:
                        assert release.wait(timeout=5)
                        return func(*args)
                    finally:
                        finished.set()

                return await run_sync(delayed_worker, **kwargs)

            monkeypatch.setattr(main, "run_sync", delayed_submission)
            request = asyncio.create_task(async_post(client))
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                monkeypatch.setattr(main, "run_sync", run_sync)
                assert (await async_post(client)).status_code == 200
            finally:
                release.set()
                assert await asyncio.to_thread(finished.wait, 5)
                await asyncio.gather(request, return_exceptions=True)
                monkeypatch.setattr(main, "run_sync", run_sync)
            assert engine.calls == 1

    asyncio.run(exercise())


def test_metrics_disabled_by_default(client):
    assert client.get("/metrics", headers=AUTH).status_code == 404


@pytest.mark.parametrize("token", [None, "", "   ", "test-secret"])
def test_enabled_metrics_require_independent_token(monkeypatch, token):
    monkeypatch.setenv("CAPTIX_API_TOKEN", "test-secret")
    monkeypatch.setenv("CAPTIX_METRICS_ENABLED", "true")
    if token is None:
        monkeypatch.delenv("CAPTIX_METRICS_TOKEN", raising=False)
    else:
        monkeypatch.setenv("CAPTIX_METRICS_TOKEN", token)
    with pytest.raises(ValueError, match="CAPTIX_METRICS_TOKEN"):
        Settings.from_env()


@pytest.mark.parametrize("value", ["perhaps", "1", ""])
def test_invalid_metrics_flag_fails_closed(monkeypatch, value):
    monkeypatch.setenv("CAPTIX_API_TOKEN", "test-secret")
    monkeypatch.setenv("CAPTIX_METRICS_ENABLED", value)
    with pytest.raises(ValueError, match="CAPTIX_METRICS_ENABLED"):
        Settings.from_env()


def test_metrics_authentication_aggregates_and_privacy(monkeypatch, caplog):
    from prometheus_client.parser import text_string_to_metric_families

    monkeypatch.setenv("CAPTIX_API_TOKEN", "test-secret")
    monkeypatch.setenv("CAPTIX_METRICS_ENABLED", "true")
    monkeypatch.setenv("CAPTIX_METRICS_TOKEN", "private-metrics-token")
    settings = Settings.from_env()
    assert "private-metrics-token" not in repr(settings)
    app = create_app(FakeEngine(), settings)
    with TestClient(app) as client:
        for header in [
            None,
            "Bearer wrong",
            "Basic private-metrics-token",
            "Bearer test-secret",
        ]:
            response = client.get(
                "/metrics", headers={} if header is None else {"Authorization": header}
            )
            assert response.status_code == 401
            assert response.headers["www-authenticate"] == "Bearer"
        metrics_auth = {"Authorization": "Bearer private-metrics-token"}
        assert post(client, headers=metrics_auth).status_code == 401
        assert post(client).status_code == 200
        assert post(client, data={"mask": "[a]", "length": "1"}).status_code == 200
        invalid = post(client, data={"mask": "private-mask-value"})
        assert invalid.status_code == 422
        response = client.get("/metrics", headers=metrics_auth)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        samples = [
            sample
            for family in text_string_to_metric_families(response.text)
            for sample in family.samples
        ]
        counts = {
            sample.labels["stage"]: sample.value
            for sample in samples
            if sample.name == "captix_ocr_duration_seconds_count"
        }
        assert counts == {"preprocess": 2, "inference": 2, "decode": 2, "total": 3}
        statuses = {
            sample.labels["status"]: sample.value
            for sample in samples
            if sample.name == "captix_ocr_requests_total"
        }
        assert statuses == {"200": 2, "401": 1, "422": 1}
        for sample in samples:
            assert set(sample.labels) <= {"stage", "status", "le"}
        exposed = response.text + invalid.text + caplog.text
        for private in [
            "test-secret",
            "private-metrics-token",
            "private.png",
            "private-mask-value",
            "A1",
        ]:
            assert private not in exposed

    # Registries must be per app, including repeated startup in one process.
    with TestClient(create_app(FakeEngine(), settings)) as client:
        response = client.get("/metrics", headers=metrics_auth)
        assert response.status_code == 200
        assert 'captix_ocr_requests_total{status="200"} 2.0' not in response.text
