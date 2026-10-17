"""Load and save captix models, plus pretrained model registry.

Note: the original R `captcha` package distributes models serialized via
`luz`/`torch` (R bindings). Those binaries are *not* directly loadable from
pure Python — the format depends on R's torch serialization. ``load_model``
loads captix-native checkpoints (created via ``save_model``).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from .dataset import DEFAULT_VOCAB
from .model import CRNN

PathLike = str | Path

# The legacy R `captcha` package publishes one .pt per scraper.
# These are R-native serializations and are kept here only as a reference of
# what was historically available.
LEGACY_R_MODELS: list[str] = [
    "cadesp",
    "esaj",
    "jucesp",
    "rfb",
    "sei",
    "tjmg",
    "tjpe",
    "tjrs",
    "trf5",
    "trt",
]

LEGACY_REPO_BASE = "https://github.com/decryptr/captcha/releases/download/captcha_model"


def save_model(model: CRNN, path: PathLike) -> Path:
    """Persist a CRNN model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "vocab": model.vocab,
        "format": "txtcaptcha-v1",
    }
    torch.save(payload, path)
    return path


def load_model(path: PathLike, map_location: str | torch.device | None = None) -> CRNN:
    """Load a CRNN model previously saved with :func:`save_model`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "txtcaptcha-v1":
        raise ValueError(
            f"{path} is not a txtcaptcha checkpoint. Legacy R `captcha` "
            "models are not loadable from Python."
        )
    vocab: Sequence[str] = payload.get("vocab", DEFAULT_VOCAB)
    model = CRNN(vocab=vocab)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def available_models() -> list[str]:
    """List of legacy model names from the original R package release.

    Provided for reference / dataset name discovery only.
    """
    return list(LEGACY_R_MODELS)


def download_model(name: str, dest_dir: PathLike) -> Path:
    """Download a legacy R `.pt` model file (raw bytes only).

    Note: the resulting file is *not* directly loadable by ``load_model``,
    because it uses R/luz serialization. This helper is provided so users can
    inspect or convert the legacy artifacts manually.
    """
    import requests

    if name not in LEGACY_R_MODELS:
        raise ValueError(
            f"Unknown legacy model {name!r}. Choose one of {LEGACY_R_MODELS}."
        )
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = f"{LEGACY_REPO_BASE}/{name}.pt"
    out = dest_dir / f"{name}.pt"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    return out
