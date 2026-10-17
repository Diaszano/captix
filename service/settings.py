"""Environment configuration, validated before loading a model."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_token: str = field(repr=False)
    model_dir: Path = Path("models/captix-v1")
    intra_op_threads: int = 2

    def __post_init__(self):
        if not isinstance(self.api_token, str) or not self.api_token.strip():
            raise ValueError("CAPTIX_API_TOKEN must not be empty")
        if type(self.intra_op_threads) is not int or self.intra_op_threads < 1:
            raise ValueError("ORT_INTRA_OP_THREADS must be a positive integer")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_token=os.environ.get("CAPTIX_API_TOKEN", ""),
            model_dir=Path(os.environ.get("CAPTIX_MODEL_DIR", "models/captix-v1")),
            intra_op_threads=int(os.environ.get("ORT_INTRA_OP_THREADS", "2")),
        )
