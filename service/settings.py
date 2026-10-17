"""Environment configuration, validated before loading a model."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_token: str = field(repr=False)
    model_dir: Path = Path("models/captix-v1")
    intra_op_threads: int = 2
    max_concurrent: int = 2
    metrics_enabled: bool = False
    metrics_token: str = field(default="", repr=False)

    def __post_init__(self):
        if not isinstance(self.api_token, str) or not self.api_token.strip():
            raise ValueError("CAPTIX_API_TOKEN must not be empty")
        if type(self.intra_op_threads) is not int or self.intra_op_threads < 1:
            raise ValueError("ORT_INTRA_OP_THREADS must be a positive integer")
        if type(self.max_concurrent) is not int or self.max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer")
        if type(self.metrics_enabled) is not bool:
            raise ValueError("CAPTIX_METRICS_ENABLED must be true or false")
        if self.metrics_enabled and (
            not isinstance(self.metrics_token, str)
            or not self.metrics_token.strip()
            or self.metrics_token == self.api_token
        ):
            raise ValueError("CAPTIX_METRICS_TOKEN must be nonempty and independent")

    @classmethod
    def from_env(cls) -> "Settings":
        metrics_enabled = os.environ.get("CAPTIX_METRICS_ENABLED", "false")
        if metrics_enabled not in {"true", "false"}:
            raise ValueError("CAPTIX_METRICS_ENABLED must be true or false")
        return cls(
            api_token=os.environ.get("CAPTIX_API_TOKEN", ""),
            model_dir=Path(os.environ.get("CAPTIX_MODEL_DIR", "models/captix-v1")),
            intra_op_threads=int(os.environ.get("ORT_INTRA_OP_THREADS", "2")),
            metrics_enabled=metrics_enabled == "true",
            metrics_token=os.environ.get("CAPTIX_METRICS_TOKEN", ""),
        )
