"""Hugging Face Hub integration: publish and fetch the unified CRNN.

Layout on the Hub (one folder per repo):

- ``model.safetensors`` — weights only, safe-by-default format (no pickle).
- ``config.json`` — vocab and architecture hyperparameters needed to rebuild
  the :class:`CRNN` before loading weights.
- ``README.md`` — the model card (YAML frontmatter + usage).

Consumers call :func:`from_pretrained`, which delegates to
``huggingface_hub.hf_hub_download`` and therefore benefits from the shared
``~/.cache/huggingface/hub`` cache — the first call downloads, subsequent
calls with the same ``revision`` return the cached path for free.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

from .dataset import DEFAULT_VOCAB
from .model import CRNN

PathLike = Union[str, Path]

#: Default repo consumers pull from when no ``model`` is passed to
#: :func:`decrypt`. Override by passing a repo id explicitly.
DEFAULT_REPO_ID = "jtrecenti/txtcaptcha-crnn"

WEIGHTS_FILENAME = "model.safetensors"
CONFIG_FILENAME = "config.json"
MODEL_CARD_FILENAME = "README.md"


def _resolve_token(token: Optional[str] = None) -> Optional[str]:
    """Accept an explicit token or fall back to the canonical env vars.

    ``huggingface_hub`` itself only reads ``HF_TOKEN``; we additionally
    honor ``HF_API_KEY`` so users don't have to rename the variable in
    their ``.env`` file.
    """
    if token:
        return token
    return os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")


def _model_config(model: CRNN) -> dict:
    """Extract the subset of hyperparameters needed to rebuild the model.

    We can introspect RNN/dropout from the module tree; CNN channel widths
    aren't exposed so we store the package default. Training with a
    non-default backbone would require a custom upload path.
    """
    return {
        "architecture": "CRNN",
        "library_name": "captix",
        "vocab": list(model.vocab),
        "cnn_channels": [64, 128, 256, 256],
        "rnn_hidden": int(model.rnn.hidden_size),
        "rnn_layers": int(model.rnn.num_layers),
        "dropout": float(model.dropout.p),
        "img_height": 32,
        "blank_index": 0,
    }


def _model_from_config(config: dict) -> CRNN:
    return CRNN(
        vocab=config.get("vocab", DEFAULT_VOCAB),
        cnn_channels=tuple(config.get("cnn_channels", (64, 128, 256, 256))),
        rnn_hidden=int(config.get("rnn_hidden", 256)),
        rnn_layers=int(config.get("rnn_layers", 2)),
        dropout=float(config.get("dropout", 0.25)),
    )


def save_pretrained(
    model: CRNN,
    save_dir: PathLike,
    *,
    model_card: Optional[str] = None,
) -> Path:
    """Write ``model.safetensors`` + ``config.json`` (+ optional ``README.md``)
    into ``save_dir``. Returns the directory path.

    Use this to prepare a local folder that can be ``upload_folder``'d to HF
    or committed to a git LFS repo.
    """
    from safetensors.torch import save_file

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    state_dict = {
        k: v.detach().contiguous().cpu()
        for k, v in model.state_dict().items()
    }
    save_file(state_dict, str(save_dir / WEIGHTS_FILENAME))

    (save_dir / CONFIG_FILENAME).write_text(
        json.dumps(_model_config(model), indent=2),
        encoding="utf-8",
    )

    if model_card is not None:
        (save_dir / MODEL_CARD_FILENAME).write_text(model_card, encoding="utf-8")

    return save_dir


def load_local(save_dir: PathLike) -> CRNN:
    """Load a model from a local directory populated by
    :func:`save_pretrained`.
    """
    from safetensors.torch import load_file

    save_dir = Path(save_dir)
    config = json.loads((save_dir / CONFIG_FILENAME).read_text(encoding="utf-8"))
    model = _model_from_config(config)
    state = load_file(str(save_dir / WEIGHTS_FILENAME), device="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def from_pretrained(
    repo_id: str = DEFAULT_REPO_ID,
    *,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    cache_dir: Optional[PathLike] = None,
) -> CRNN:
    """Download and load a captix CRNN from the Hugging Face Hub.

    ``revision`` accepts a branch, tag, or commit SHA. Defaults to ``main``.
    Downloaded files are cached in the global ``huggingface_hub`` cache,
    so this is cheap after the first call.
    """
    from huggingface_hub import hf_hub_download

    tok = _resolve_token(token)
    kw: dict = {}
    if cache_dir is not None:
        kw["cache_dir"] = str(cache_dir)
    if revision is not None:
        kw["revision"] = revision
    if tok is not None:
        kw["token"] = tok

    weights_path = hf_hub_download(
        repo_id=repo_id, filename=WEIGHTS_FILENAME, **kw
    )
    # config.json lives next to weights in the same snapshot folder
    hf_hub_download(
        repo_id=repo_id, filename=CONFIG_FILENAME, **kw
    )
    return load_local(Path(weights_path).parent)


def push_to_hub(
    model: CRNN,
    repo_id: str = DEFAULT_REPO_ID,
    *,
    token: Optional[str] = None,
    commit_message: str = "Upload captix CRNN",
    private: bool = False,
    model_card: Optional[str] = None,
    tag: Optional[str] = None,
) -> str:
    """Create ``repo_id`` if needed and upload weights, config and model card.

    Returns the public repo URL. If ``tag`` is given, a git tag is created
    pointing at the new commit — recommended for stable releases so
    consumers can pin via ``revision="v0.1.0"``.
    """
    from huggingface_hub import HfApi

    tok = _resolve_token(token)
    if tok is None:
        raise RuntimeError(
            "No Hugging Face token found. Set HF_TOKEN or HF_API_KEY in "
            "your environment (or .env), or pass token= explicitly."
        )

    api = HfApi(token=tok)
    api.create_repo(
        repo_id=repo_id, repo_type="model", exist_ok=True, private=private
    )

    with tempfile.TemporaryDirectory() as tmp:
        save_pretrained(model, tmp, model_card=model_card)
        api.upload_folder(
            folder_path=tmp,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )

    if tag:
        try:
            api.create_tag(repo_id=repo_id, tag=tag, repo_type="model")
        except Exception:  # noqa: BLE001 — tag may already exist; non-fatal
            pass

    return f"https://huggingface.co/{repo_id}"
