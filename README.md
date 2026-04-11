# txtcaptcha

[![PyPI](https://img.shields.io/pypi/v/txtcaptcha.svg)](https://pypi.org/project/txtcaptcha/)
[![Python](https://img.shields.io/pypi/pyversions/txtcaptcha.svg)](https://pypi.org/project/txtcaptcha/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20HF-jtrecenti%2Ftxtcaptcha--crnn-yellow)](https://huggingface.co/jtrecenti/txtcaptcha-crnn)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen)](https://jtrecenti.github.io/txtcaptcha/)

Read, annotate, train and decrypt **text captchas in images** with a modern
CRNN + CTC pipeline in PyTorch.

`txtcaptcha` ships:

- a CRNN architecture that handles **arbitrary input sizes** and
  **variable-length labels**,
- the full alphanumeric vocabulary `0-9a-zA-Z` (62 classes + CTC blank),
- **decode-time masking** so a single trained model can be restricted per
  site (e.g. `mask="[0-9]"`),
- **fixed-length decoding** via `length=N` for sites with a known length,
- a pretrained unified model hosted on the
  [Hugging Face Hub](https://huggingface.co/jtrecenti/txtcaptcha-crnn) with
  ~89% captcha-level accuracy across ten Brazilian court captcha datasets.

## Installation

```bash
pip install txtcaptcha
```

Or from source with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/jtrecenti/txtcaptcha
cd txtcaptcha
uv sync --extra dev
```

## Quick start

The first `decrypt` call downloads the pretrained model from the Hugging Face
Hub into `~/.cache/huggingface/hub`; subsequent calls are free.

```python
from txtcaptcha import read_captcha, decrypt

cap = read_captcha("path/to/captcha.png")
print(decrypt(cap))                          # greedy, variable length
print(decrypt(cap, mask="[0-9]"))            # digits only
print(decrypt(cap, length=5))                # force exactly 5 chars
print(decrypt(cap, mask=list("abcdef0123"))) # explicit allowed set
```

Pin a specific release or load a different Hub repo explicitly:

```python
from txtcaptcha import from_pretrained

model = from_pretrained("jtrecenti/txtcaptcha-crnn", revision="v0.1.0")
print(decrypt(cap, model=model))
```

### Training your own model

```python
from txtcaptcha import fit_model, save_model, download_dataset

data_dir = download_dataset("tjmg", "data")
model, history = fit_model(
    data_dir,
    epochs=30,
    batch_size=64,
    case_sensitive=False,
)
save_model(model, "tjmg.pt")
```

### Publishing your own model to the Hub

```python
from txtcaptcha import push_to_hub

push_to_hub(
    model,
    repo_id="your-username/your-captcha-model",
    model_card="# My captcha model\n\nTrained on ...",
    tag="v0.1.0",
)
```

## Public API

| Function | Purpose |
|---|---|
| `read_captcha(files, lab_in_path=False)` | Load image(s) into a `Captcha` object. |
| `Captcha` | Container with `images`, `labels`, `paths`, `plot()`. |
| `annotate(files, labels=None, ...)` | Interactive/batch labeling (filename convention). |
| `CaptchaDataset(root, vocab, height, case_sensitive)` | PyTorch dataset over a folder of `<id>_<label>.<ext>` files. |
| `transform_image(files, height=32)` | Load + resize + width-pad for batching. |
| `encode_label`, `decode_indices` | Vocab ↔ tensor (CTC blank index 0). |
| `pad_collate` | DataLoader collate fn for variable-width batching. |
| `CRNN(vocab, ...)` | CNN + BiLSTM + linear head. |
| `fit_model(dir, ...)` | Training loop with CTC loss + early stopping. |
| `decrypt(files, model=None, mask=None, case_sensitive=True, length=None)` | Predict labels; auto-downloads the pretrained model when `model=None`. |
| `save_model`, `load_model` | Local checkpoint persistence. |
| `from_pretrained`, `save_pretrained`, `push_to_hub` | Hugging Face Hub integration. |
| `download_dataset`, `available_datasets` | Fetch labeled training datasets. |
| `download_captchas` (CLI) | Download live, unlabeled captchas from 10 Brazilian sources. |
| `sequence_accuracy(preds, targets)` | Exact-match accuracy metric. |

Full API reference: **<https://jtrecenti.github.io/txtcaptcha/>**.

## Architecture

`CRNN` is a Convolutional Recurrent Neural Network:

1. **CNN backbone** — ResNet-style basic blocks (`64 → 128 → 256 → 256`
   channels) with strided pooling. Down-samples height by `8` and width by
   `4`, preserving width resolution for the sequence dimension.
2. **Adaptive pool** — collapses the remaining height to 1, producing a
   width-indexed sequence of feature vectors.
3. **BiLSTM** — 2-layer bidirectional LSTM (hidden 256).
4. **Linear head** — projects to `len(vocab) + 1` logits per timestep (the
   extra slot is the CTC blank).
5. **CTC loss** — handles variable-length targets, no per-position softmax.

Variable image dimensions are handled by resizing height to 32 at load time,
preserving the aspect-ratio width, and padding widths within each batch via
`pad_collate`. CRNN+CTC is the de-facto baseline for short-text scene-text
recognition — lighter than transformer OCR (e.g. TrOCR) and consistently
strong on short captcha images.

## Variable-length labels

CRNN + CTC handles variable label lengths natively. The convolutional stack
emits `T` logits per image; CTC collapsing (remove consecutive repeats, then
remove blanks) turns any path into a string of arbitrary length between 0
and `T`. Training mixes 4-char and 5-char labels in the same batch — no
length head, no padding tokens.

The downside of greedy CTC is that a confident wrong timestep can yield a
prediction of the wrong length. When you know the expected length, pass
`length=` to switch to an exact dynamic-programming search over CTC paths
that collapse to exactly that many characters:

```python
decrypt(cap)                       # greedy
decrypt(cap, length=5)             # force 5 chars
decrypt(cap, length=4, mask="[0-9]")  # combine with masking
```

The DP runs in `O(T · L · |vocab|)` per image, tracks the best path for
every `(collapsed_count, last_index)` state and reconstructs the argmax. It
is strictly at least as good as greedy when the true length is known and
never emits a wrong-length prediction.

## Decode-time masking

`decrypt(..., mask=...)` zeros out forbidden vocabulary logits before CTC
decoding, so the same trained model can be specialized per site:

```python
decrypt(cap, mask=["a", "b", "c", "1", "2", "3"])  # explicit list
decrypt(cap, mask="[0-9a-z]")                       # regex char-class
decrypt(cap, mask="[A-Z]", case_sensitive=True)     # uppercase only
decrypt(cap, mask="[a-z]", case_sensitive=False)    # output lowercased
```

## Notebooks

- `notebooks/train_unified_model.ipynb` — downloads every dataset, merges
  them and trains the unified CRNN. Designed for a cloud GPU machine.
- `notebooks/eval_per_dataset.ipynb` — per-dataset accuracy on a held-out
  split.
- `notebooks/eval_per_dataset_live.ipynb` — predictions on freshly
  downloaded, unlabeled captchas (overfit check).

## Tests

```bash
uv run pytest
```

## License

[MIT](LICENSE) © Julio Trecenti
