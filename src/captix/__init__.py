"""captix: Python port of the R captcha package."""
from __future__ import annotations

from .captcha import Captcha, read_captcha
from .annotate import annotate
from .dataset import (
    CaptchaDataset,
    DEFAULT_VOCAB,
    transform_image,
    encode_label,
    decode_indices,
    pad_collate,
)
from .model import CRNN
from .fit import fit_model, FitHistory
from .decrypt import decrypt
from .load import save_model, load_model, available_models, download_model
from .data import download_dataset, available_datasets
from .metrics import sequence_accuracy
from .download import (
    DownloadError,
    available_sources,
    download_captchas,
)
from .hub import (
    DEFAULT_REPO_ID,
    from_pretrained,
    push_to_hub,
    save_pretrained,
)

__all__ = [
    "Captcha",
    "read_captcha",
    "annotate",
    "CaptchaDataset",
    "DEFAULT_VOCAB",
    "transform_image",
    "encode_label",
    "decode_indices",
    "pad_collate",
    "CRNN",
    "fit_model",
    "FitHistory",
    "decrypt",
    "save_model",
    "load_model",
    "available_models",
    "download_model",
    "download_dataset",
    "available_datasets",
    "sequence_accuracy",
    "download_captchas",
    "available_sources",
    "DownloadError",
    "from_pretrained",
    "save_pretrained",
    "push_to_hub",
    "DEFAULT_REPO_ID",
]

__version__ = "1.0.0"
