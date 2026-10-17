"""Per-app aggregate telemetry with a fixed set of label values."""

from prometheus_client import CollectorRegistry, Counter, Histogram


class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "captix_ocr_requests_total",
            "OCR requests by status",
            ["status"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "captix_ocr_duration_seconds",
            "OCR work duration by stage",
            ["stage"],
            registry=self.registry,
        )

    def record_status(self, status):
        if status not in {200, 400, 401, 413, 415, 422, 429, 500, 503, "cancelled"}:
            status = 500
        self.requests.labels(status=str(status)).inc()

    def record_prediction(self, result):
        for stage in ("preprocess", "inference", "decode"):
            self.duration.labels(stage=stage).observe(result.timings_ms[stage] / 1000)

    def record_total(self, seconds):
        self.duration.labels(stage="total").observe(seconds)
