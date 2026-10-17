"""captix: Python port of the R captcha package."""

from __future__ import annotations

from .annotate import annotate
from .captcha import Captcha, read_captcha
from .data import available_datasets, download_dataset
from .dataset import (
    DEFAULT_VOCAB,
    CaptchaDataset,
    decode_indices,
    encode_label,
    pad_collate,
    transform_image,
)
from .decrypt import decrypt
from .download import (
    DownloadError,
    available_sources,
    download_captchas,
)
from .fit import FitHistory, fit_model
from .hub import (
    DEFAULT_REPO_ID,
    from_pretrained,
    push_to_hub,
    save_pretrained,
)
from .load import available_models, download_model, load_model, save_model
from .metrics import sequence_accuracy
from .model import CRNN

__all__ = [
    "CRNN",
    "DEFAULT_REPO_ID",
    "DEFAULT_VOCAB",
    "Captcha",
    "CaptchaDataset",
    "DownloadError",
    "FitHistory",
    "annotate",
    "available_datasets",
    "available_models",
    "available_sources",
    "decode_indices",
    "decrypt",
    "download_captchas",
    "download_dataset",
    "download_model",
    "encode_label",
    "fit_model",
    "from_pretrained",
    "load_model",
    "pad_collate",
    "push_to_hub",
    "read_captcha",
    "save_model",
    "save_pretrained",
    "sequence_accuracy",
    "transform_image",
]

__version__ = "1.0.0"
