"""Deploy the unified CRNN checkpoint to the Hugging Face Hub.

Reads ``HF_API_KEY`` (or ``HF_TOKEN``) from ``.env`` or the environment,
converts ``notebooks/txtcaptcha_unified.pt`` into the HF layout
(``model.safetensors`` + ``config.json`` + ``README.md``) and uploads it
to the target repo.

Usage
-----

    python scripts/deploy_to_hf.py --repo-id jtrecenti/txtcaptcha-crnn --tag v0.1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add src to path so we can run without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from txtcaptcha import load_model  # noqa: E402
from txtcaptcha.hub import DEFAULT_REPO_ID, push_to_hub  # noqa: E402


MODEL_CARD = """---
license: mit
library_name: pytorch
pipeline_tag: image-to-text
tags:
  - captcha
  - crnn
  - ctc
  - ocr
  - pytorch
  - brazilian-courts
language:
  - pt
---

# txtcaptcha CRNN — unified Brazilian court captcha model

A Convolutional Recurrent Neural Network (ResNet-style CNN → BiLSTM → CTC)
trained jointly on ten labeled captcha datasets from Brazilian courts and
tax authorities, published originally by the R
[`captcha`](https://github.com/decryptr/captcha) package. It replaces ten
per-site CNN classifiers with a single model that reads variable-length
alphanumeric strings from arbitrarily-sized input images.

## Usage

```python
from txtcaptcha import decrypt, read_captcha

cap = read_captcha("captcha.png")
# First call downloads the weights from this repo into ~/.cache/huggingface;
# subsequent calls are free.
print(decrypt(cap))                  # greedy CTC decoding
print(decrypt(cap, length=5))        # force exactly 5 output chars
print(decrypt(cap, mask="[0-9]"))    # restrict to digits
```

Explicit download (useful for notebooks or CI warmup):

```python
from txtcaptcha import from_pretrained
model = from_pretrained("{repo_id}")
```

Install `txtcaptcha`:

```bash
pip install git+https://github.com/jtrecenti/txtcaptcha
```

## Training data

Merged union of these datasets (published at
https://github.com/decryptr/captcha/releases):

`cadesp`, `esaj`, `jucesp`, `rfb`, `sei`, `tjmg`, `tjpe`, `tjrs`, `trf5`, `trt`

Labels follow the filename convention `<id>_<label>.<ext>`. Vocabulary is
the full alphanumeric range `0-9a-zA-Z` (62 classes + CTC blank).

## Architecture

| Stage | Details |
|---|---|
| Backbone | ResNet-style CNN, channels `64 → 128 → 256 → 256`, H/8 × W/4 downsample |
| Sequence head | 2-layer bidirectional LSTM, hidden 256 |
| Classifier | Linear `512 → 63` (62 chars + blank) |
| Loss | CTC, handles variable output length |
| Input | Any dimensions; height resized to 32 at inference, width preserved |

The full config is in [`config.json`](./config.json).

## Validation accuracy

~**89% captcha-level exact match** on a held-out 20% split of the training
corpus. Per-dataset breakdown available in
[`docs/notebooks/eval_per_dataset.ipynb`](https://github.com/jtrecenti/txtcaptcha/blob/main/docs/notebooks/eval_per_dataset.ipynb).

## Limitations

- Trained exclusively on Brazilian court captcha fonts/distortions.
  Generalization to US/European OCR captchas is untested.
- The `sei` dataset is a math captcha; the model learned to transcribe
  the (already-solved) label as if it were a plain 4-char sequence.
- Label length distribution is concentrated on 4–5 chars; longer sequences
  may degrade. Use `decrypt(..., length=N)` to enforce an exact length.

## Versioning

Pin to a specific release with `revision=`:

```python
from_pretrained("{repo_id}", revision="v0.1.0")
```

## License

MIT. The original training data from the R `captcha` package is redistributed
under its own terms.
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Deploy the unified CRNN to the Hugging Face Hub",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("notebooks/txtcaptcha_unified.pt"),
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--tag", default=None, help="Optional git tag to create")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--commit-message",
        default="Upload txtcaptcha CRNN",
    )
    args = parser.parse_args(argv)

    load_dotenv()

    if not args.checkpoint.exists():
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    print(f"loading {args.checkpoint}")
    model = load_model(args.checkpoint)

    card = MODEL_CARD.format(repo_id=args.repo_id)
    print(f"pushing to https://huggingface.co/{args.repo_id}")
    url = push_to_hub(
        model,
        repo_id=args.repo_id,
        model_card=card,
        private=args.private,
        tag=args.tag,
        commit_message=args.commit_message,
    )
    print(f"done: {url}")
    if args.tag:
        print(f"tagged as {args.tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
