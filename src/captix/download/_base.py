"""Registry, session wrapper and batch loop for captcha downloaders."""
from __future__ import annotations

import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import requests
import urllib3

# Several Brazilian court sites ship with expired or self-signed certs.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PathLike = Union[str, Path]


class DownloadError(Exception):
    """Raised when a captcha cannot be fetched or parsed from a source.

    The batch loop treats this (and :class:`requests.RequestException`) as a
    skippable failure rather than a hard error.
    """


@dataclass
class CaptchaPayload:
    """Raw bytes and filename extension for a single downloaded captcha."""

    content: bytes
    ext: str  # lowercase, no leading dot


class FetchContext:
    """Thin wrapper over :class:`requests.Session` with a baked-in timeout."""

    def __init__(self, session: requests.Session, timeout: float = 30.0):
        self.session = session
        self.timeout = timeout

    def get(self, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        return self.session.get(url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        return self.session.post(url, **kwargs)


Fetcher = Callable[[FetchContext], CaptchaPayload]


@dataclass
class Source:
    name: str
    fetcher: Fetcher
    verify_ssl: bool = True


_REGISTRY: Dict[str, Source] = {}


def register(name: str, *, verify_ssl: bool = True) -> Callable[[Fetcher], Fetcher]:
    """Decorator that registers a fetcher function under ``name``."""

    def deco(fn: Fetcher) -> Fetcher:
        _REGISTRY[name] = Source(name=name, fetcher=fn, verify_ssl=verify_ssl)
        return fn

    return deco


def available_sources() -> List[str]:
    """Sorted list of registered source names."""
    return sorted(_REGISTRY)


def get_source(name: str) -> Source:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown source {name!r}. Available: {available_sources()}"
        )
    return _REGISTRY[name]


def make_session(verify: bool = True) -> requests.Session:
    """Create a fresh requests session. Each captcha fetch gets its own so
    cookies from one attempt don't leak into the next — some sources burn
    a session id per captcha.
    """
    s = requests.Session()
    s.verify = verify
    return s


def _write_payload(payload: CaptchaPayload, dest_dir: Path, source: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{source}_{uuid.uuid4().hex[:12]}.{payload.ext}"
    out = dest_dir / fname
    out.write_bytes(payload.content)
    return out


def download_captchas(
    source: str,
    n: int,
    dest_dir: PathLike,
    *,
    sleep: float = 0.0,
    timeout: float = 30.0,
    max_attempts: Optional[int] = None,
    on_error: str = "skip",
) -> List[Path]:
    """Download up to ``n`` captchas from ``source`` into ``dest_dir``.

    Each successful fetch is written as ``<source>_<hex>.<ext>``. Network or
    parse failures are logged as warnings and skipped (``on_error='skip'``,
    default); pass ``on_error='raise'`` to propagate instead. ``n`` is a
    target, not a guarantee — the function gives up after ``max_attempts``
    total failed attempts (default ``max(10, n * 3)``), so a site that's
    completely down will not hang.

    Returns the list of written file paths.
    """
    src = get_source(source)
    dest = Path(dest_dir)
    if n < 0:
        raise ValueError("n must be >= 0")
    if max_attempts is None:
        max_attempts = max(10, n * 3)

    saved: List[Path] = []
    failures = 0
    while len(saved) < n and failures < max_attempts:
        session = make_session(verify=src.verify_ssl)
        ctx = FetchContext(session, timeout=timeout)
        try:
            payload = src.fetcher(ctx)
        except (requests.RequestException, DownloadError, ValueError, KeyError) as e:
            failures += 1
            if on_error == "raise":
                raise
            warnings.warn(f"[{source}] fetch failed ({type(e).__name__}): {e}")
            continue
        finally:
            session.close()
        out = _write_payload(payload, dest, source)
        saved.append(out)
        if sleep > 0 and len(saved) < n:
            time.sleep(sleep)
    return saved
