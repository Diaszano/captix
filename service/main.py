"""Single-image OCR API with bounded, memory-only multipart uploads."""

import re
import secrets
from contextlib import asynccontextmanager
from io import BytesIO
from threading import BoundedSemaphore, Lock
from time import perf_counter

import numpy as np
from anyio.to_thread import run_sync
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from python_multipart.multipart import parse_options_header
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from captix_runtime.engine import OnnxEngine
from captix_runtime.pipeline import EngineProtocol, PipelineError, recognize
from captix_runtime.preprocessing import PreprocessingError

from .metrics import Metrics
from .settings import Settings

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_FORM_OVERHEAD = 64 * 1024
MEDIA_FORMATS = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}


class _ImageFormParser(MultiPartParser):
    # Bound file data before Starlette's spool can roll over to disk.
    spool_max_size = MAX_IMAGE_BYTES + 1

    def __init__(self, request):
        super().__init__(
            request.headers,
            self._bounded_stream(request),
            max_files=1,
            max_fields=4,
            max_part_size=4096,
        )
        self.complete = False
        self.file_bytes = 0
        self.header_bytes = 0

    async def _bounded_stream(self, request):
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_IMAGE_BYTES + MAX_FORM_OVERHEAD:
                raise HTTPException(413, "Upload too large")
            for start in range(0, len(chunk), 65536):
                yield chunk[start : start + 65536]

    def on_part_data(self, data, start, end):
        if self._current_part.file is not None:
            self.file_bytes += end - start
            if self.file_bytes > MAX_IMAGE_BYTES:
                raise HTTPException(413, "Image too large")
        super().on_part_data(data, start, end)

    def on_header_field(self, data, start, end):
        self._check_header_size(end - start)
        super().on_header_field(data, start, end)

    def on_header_value(self, data, start, end):
        self._check_header_size(end - start)
        super().on_header_value(data, start, end)

    def _check_header_size(self, size):
        self.header_bytes += size
        if self.header_bytes > 16384:
            raise MultiPartException("Multipart headers too large")

    def on_end(self):
        self.complete = True


async def _read_image_form(request):
    media, options = parse_options_header(request.headers.get("content-type", ""))
    if media != b"multipart/form-data":
        raise HTTPException(415, "Expected multipart/form-data")
    if not options.get(b"boundary") or len(options[b"boundary"]) > 200:
        raise HTTPException(400, "Malformed multipart body")
    parser = _ImageFormParser(request)
    try:
        form = await parser.parse()
        if not parser.complete:
            raise HTTPException(400, "Malformed multipart body")
        if any(len(form.getlist(key)) != 1 for key in form):
            raise HTTPException(422, "Duplicate form fields")
        image = form.get("image")
        if not isinstance(image, UploadFile):
            raise HTTPException(422, "An image file is required")
        mime = image.content_type
        if mime not in MEDIA_FORMATS:
            raise HTTPException(415, "Unsupported image media type")
        data = await image.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Image too large")
        arguments = {key: form[key] for key in form if key != "image"}
        if any(not isinstance(value, str) for value in arguments.values()):
            raise HTTPException(422, "Invalid form fields")
        return data, mime, arguments
    except MultiPartException:
        raise HTTPException(400, "Malformed multipart body") from None
    finally:
        # Also close incomplete parts when the parser never returns FormData.
        for file in parser._files_to_close_on_error:
            file.close()


def _recognize(data, mime, arguments, engine):
    if set(arguments) - {"mask", "length", "case_sensitive"}:
        raise HTTPException(422, "Unknown form field")
    mask = arguments.get("mask")
    length = arguments.get("length")
    if length is not None:
        if re.fullmatch(r"[0-9]{1,2}", length) is None:
            raise HTTPException(422, "Invalid length")
        length = int(length)
    case_sensitive = arguments.get("case_sensitive", "true")
    if case_sensitive not in {"true", "false"}:
        raise HTTPException(422, "Invalid case_sensitive")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != MEDIA_FORMATS[mime]:
                raise HTTPException(415, "Image format does not match media type")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(422, "Invalid image") from None
    try:
        return recognize(
            data,
            engine,
            mask=mask,
            length=length,
            case_sensitive=case_sensitive == "true",
        )
    except (PipelineError, PreprocessingError) as exc:
        status = {
            "unsupported_media": 415,
            "animated": 415,
            "engine_error": 500,
            "invalid_logits": 500,
            "invalid_vocabulary": 500,
        }.get(exc.code, 422)
        if status == 500:
            raise HTTPException(500, "Internal server error") from None
        raise HTTPException(status, "Invalid image or OCR arguments") from None


def create_app(engine: EngineProtocol, settings: Settings) -> FastAPI:
    slots = BoundedSemaphore(settings.max_concurrent)
    admission = Lock()
    metrics = Metrics()

    def worker(data, mime, arguments):
        with admission:
            if not app.state.ready:
                raise HTTPException(503, "Model is not ready")
            if not slots.acquire(blocking=False):
                raise HTTPException(429, "Inference capacity exhausted")
        started = perf_counter()
        try:
            result = _recognize(data, mime, arguments, engine)
            metrics.record_prediction(result)
            return result
        finally:
            slots.release()
            metrics.record_total(perf_counter() - started)

    def drain():
        for _ in range(settings.max_concurrent):
            slots.acquire()
        for _ in range(settings.max_concurrent):
            slots.release()

    @asynccontextmanager
    async def lifespan(app):
        app.state.ready = False
        if hasattr(engine, "warmup"):
            await run_in_threadpool(engine.warmup)
        else:
            await run_in_threadpool(engine.infer, np.zeros((1, 3, 32, 32), np.float32))
        with admission:
            app.state.ready = True
        try:
            yield
        finally:
            with admission:
                app.state.ready = False
            await run_sync(drain, abandon_on_cancel=False)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.ready = False
    app.state.engine = engine

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready():
        if not app.state.ready:
            raise HTTPException(503, "Model is not ready")
        return {"status": "ok"}

    @app.post("/v1/ocr")
    async def ocr(request: Request):
        status = "cancelled"
        try:
            authenticate(request, settings.api_token)
            if not app.state.ready:
                raise HTTPException(503, "Model is not ready")
            data, mime, arguments = await _read_image_form(request)
            # External timeouts cannot interrupt InferenceSession.run(); the worker
            # owns its slot until native work ends, even if the HTTP task is cancelled.
            result = await run_sync(
                worker, data, mime, arguments, abandon_on_cancel=False
            )
            status = 200
            return {
                "text": result.text,
                "model_version": engine.version,
                "decoder": result.decoder,
                "processing_ms": result.processing_ms,
            }
        except HTTPException as exc:
            status = exc.status_code
            raise
        except Exception:  # noqa: BLE001 — Keep internal failures out of HTTP responses.
            status = 500
            raise HTTPException(500, "Internal server error") from None
        finally:
            metrics.record_status(status)

    def authenticate(request, expected_token):
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        valid = secrets.compare_digest(
            token.encode("utf-8"), expected_token.encode("utf-8")
        )
        if scheme.lower() != "bearer" or not valid:
            raise HTTPException(
                401, "Invalid credentials", headers={"WWW-Authenticate": "Bearer"}
            )

    if settings.metrics_enabled:

        @app.get("/metrics")
        async def get_metrics(request: Request):
            authenticate(request, settings.metrics_token)
            return Response(
                generate_latest(metrics.registry),
                headers={"Content-Type": CONTENT_TYPE_LATEST},
            )

    return app


def create_app_from_env() -> FastAPI:
    settings = Settings.from_env()
    engine = OnnxEngine(settings.model_dir, intra_op_threads=settings.intra_op_threads)
    return create_app(engine, settings)
