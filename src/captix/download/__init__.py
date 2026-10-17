"""captix.download: fetch fresh captchas from the sources this package
was trained on. Use :func:`download_captchas` programmatically or the
``download_captchas`` CLI.
"""

from __future__ import annotations

from . import _sources  # noqa: F401  — import for side effect (registry)
from ._base import (
    CaptchaPayload,
    DownloadError,
    FetchContext,
    Source,
    available_sources,
    download_captchas,
    get_source,
)

__all__ = [
    "CaptchaPayload",
    "DownloadError",
    "FetchContext",
    "Source",
    "available_sources",
    "download_captchas",
    "get_source",
]
