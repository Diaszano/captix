# txtcaptcha

Python port of the R [`captcha`](https://github.com/decryptr/captcha) package
for working with **text captchas in images**: read, annotate, train a model,
and predict labels.

The original R package only supported a fixed image size, a fixed label
length, and a per-captcha vocabulary. `txtcaptcha` ships a modernized
**CRNN + CTC** architecture that handles:

- arbitrary input image dimensions (variable width, height resized at load),
- variable-length labels,
- the full alphanumeric vocabulary `0-9a-zA-Z` (62 classes + CTC blank),
- decode-time **masking** so a single trained model can be restricted to a
  specific character set per site (e.g. `[0-9]` only).

## Installation

```bash
uv sync --extra dev
# or, with pip:
pip install -e .
```

## Quick start

```python
from txtcaptcha import (
    read_captcha, fit_model, decrypt,
    save_model, load_model,
    download_dataset,
)

# 1. Download a labeled dataset published by the original R package
data_dir = download_dataset("tjmg", "data")

# 2. Train a CRNN+CTC model
model, history = fit_model(
    data_dir,
    epochs=30,
    batch_size=64,
    case_sensitive=False,
)
save_model(model, "tjmg.pt")

# 3. Predict
cap = read_captcha("path/to/captcha.png")
print(decrypt(cap, model))                          # full vocab
print(decrypt(cap, model, mask="[0-9]"))            # digits only
print(decrypt(cap, model, mask=list("abcdef0123"))) # explicit allowed set
```

## Public API

| Function | Purpose |
|---|---|
| `read_captcha(files, lab_in_path=False)` | Load image(s) into a `Captcha` object. |
| `Captcha` | Container with `images`, `labels`, `paths`, `__len__`, indexing, `plot()`. |
| `annotate(files, labels=None, path=None, rm_old=False)` | Append `_<label>` to filenames; interactive when labels are missing. |
| `CaptchaDataset(root, vocab, height, case_sensitive)` | PyTorch dataset over a folder of `<id>_<label>.<ext>` files. |
| `transform_image(files, height=32)` | Load + resize to fixed height, pad widths within a batch. |
| `encode_label`, `decode_indices` | Vocab ↔ tensor (CTC blank index 0). |
| `pad_collate` | DataLoader collate fn for variable-width batching. |
| `CRNN(vocab, ...)` | CRNN backbone (ResNet-style CNN + BiLSTM + linear head). |
| `fit_model(dir, ...)` | Training loop with CTC loss, validation, early stopping. |
| `decrypt(files, model, mask=None, case_sensitive=True, length=None)` | Predict labels with optional masking and optional fixed output length. |
| `save_model`, `load_model` | Persist / restore txtcaptcha checkpoints. |
| `available_datasets`, `download_dataset` | Discover and fetch the original R datasets from GitHub releases. |
| `available_models`, `download_model` | List / download the legacy R `.pt` files (see note below). |
| `sequence_accuracy(preds, targets)` | Exact-match accuracy metric. |

## Architecture

`CRNN` is a Convolutional Recurrent Neural Network:

1. **CNN backbone** — ResNet-style basic blocks (`64 → 128 → 256 → 256`
   channels) with strided pooling. Down-samples height by `8` and width by
   `4`, preserving width resolution for the sequence dimension.
2. **Adaptive pool** — collapses the remaining height to 1, producing a
   width-indexed sequence of feature vectors.
3. **BiLSTM** — 2-layer bidirectional LSTM (hidden 256) for context.
4. **Linear head** — projects to `len(vocab) + 1` logits per timestep
   (the extra slot is the CTC blank).
5. **CTC loss** — handles variable-length targets, no need for fixed
   per-position classification.

Variable image dimensions are handled by:
- resizing height to a fixed value (default 32) at load time,
- preserving aspect-ratio width,
- padding widths within each batch via `pad_collate`.

The output sequence length scales with input width, which is exactly what
CTC expects. CRNN+CTC is the de-facto baseline for short-text scene-text
recognition; it consistently outperforms fixed-position CNN heads on
captcha datasets and is much lighter than transformer OCR (e.g. TrOCR),
which is overkill for short captcha images.

## Variable-length labels

`CRNN + CTC` handles variable label lengths natively. At decode time, the
convolutional stack produces a width-indexed sequence of `T` logits per
image; CTC collapsing (remove consecutive repeats, then remove blanks)
turns any such path into a string of arbitrary length between 0 and `T`.
Training mixes 4-char and 5-char labels in the same batch without any
special handling — `pad_collate` pads images to the widest sample and
`CTCLoss` consumes the per-sample target lengths. No fixed output head,
no length head, no padding tokens: the length is whatever the collapse
produces.

The downside of pure greedy CTC decoding is that a confident wrong
timestep can yield a prediction of the wrong length (e.g. 4 chars for a
site that always uses 5). When you know the expected length up-front,
pass `length=` to `decrypt()` to switch from greedy decoding to an exact
dynamic-programming search over CTC paths that collapse to *exactly*
that many characters:

```python
decrypt(cap, model)              # greedy — output length is whatever collapses
decrypt(cap, model, length=5)    # force output to exactly 5 characters
decrypt(cap, model, length=4, mask="[0-9]")  # combine with masking
```

The DP runs in `O(T · L · |vocab|)` per image, tracks the best path for
every `(collapsed_count, last_index)` state, and then reconstructs the
argmax. It is strictly at least as good as greedy when the true length
is known, and never emits a wrong-length prediction. Leave `length=None`
when the site mixes lengths (e.g. 4 *or* 5 chars) — in that case either
call `decrypt` once per length bucket, or stick with greedy.

## Decode-time masking

`decrypt(..., mask=...)` zeros out forbidden vocabulary logits before greedy
CTC decoding, so the same trained model can be specialized per site:

```python
decrypt(cap, model, mask=["a", "b", "c", "1", "2", "3"])  # explicit list
decrypt(cap, model, mask="[0-9a-z]")                       # regex char-class
decrypt(cap, model, mask="[A-Z]", case_sensitive=True)     # uppercase only
decrypt(cap, model, mask="[a-z]", case_sensitive=False)    # output lowercased
```

## Notebook

`notebooks/train_unified_model.ipynb` downloads every dataset published in
the R `captcha` release page (`cadesp`, `esaj`, `jucesp`, `rfb`, `sei`,
`tjmg`, `tjpe`, `tjrs`, `trf5`, `trt`), merges them into one folder, and
fits a single model with the full alphanumeric vocabulary. Designed to run
on a cloud GPU machine.

## Tests

```bash
uv run pytest
```

40 tests cover the public API: reading, annotating, dataset/transform,
encode/decode round-trips, model forward shapes (variable width and
height), decrypt with all mask types, end-to-end training on a tiny
synthetic corpus, save/load round-trip, and metrics.

## Note on legacy R models

The original R package distributes pretrained models as `.pt` files
serialized via R's `luz`/`torch` bindings. **These binaries cannot be
loaded directly from pure Python** because the format depends on R's
side of the torch serialization. `download_model()` is provided to fetch
the raw bytes for inspection or manual conversion, but `load_model()`
only accepts checkpoints written by `save_model()`.

To recover comparable models, retrain with `fit_model()` using the
corresponding dataset from `download_dataset()` — the new architecture
generally outperforms the legacy CNN-with-classification-head on the same
data.

## Differences vs the R package

| Feature | R `captcha` | `txtcaptcha` |
|---|---|---|
| Architecture | CNN + per-position softmax | CRNN + CTC |
| Image dim | Fixed `32x192` | Any (height resized, width batched) |
| Label length | Fixed | Variable |
| Vocabulary | Inferred per dataset | Full `0-9a-zA-Z` |
| Decode-time mask | — | List or regex; `case_sensitive` flag |
| Captcha generator | `r_captcha` | Not ported (per request) |
| Pretrained models | luz `.pt` | Retrain via notebook |
