# Task 2 report: PyTorch-independent image preprocessing

## Status

Complete.

## Commit

- `e48a768 feat: add torch-free compatible image preprocessing`

## Tests

RED command: `uv run pytest tests/test_runtime_preprocessing.py -q`

Initial output confirmed the intended missing interface: `ModuleNotFoundError: No module named 'captix_runtime'`.

GREEN command: `rtk proxy uv run pytest tests/test_runtime_preprocessing.py tests/test_dataset.py tests/test_baseline_contract.py -q`

Exact output:

```text
..................................                                       [100%]
```

`git diff --check` completed with no output. `rg -n "torch" src/captix_runtime` found no matches.

## Self-review

- Added `PreprocessedImage`, `PreprocessingError`, and `preprocess_image` under the separate `captix_runtime` package; existing `captix.*` imports remain untouched.
- Implemented PNG/JPEG/WEBP validation, single-frame enforcement, pixel/output-width checks before image conversion, alpha flattening matching the current PyTorch transform, 32-pixel height bilinear resizing, and contiguous float32 NCHW output.
- Pillow resources are context-managed. No EXIF orientation is applied.
- Added parity, transparency, resize, supported format, animation, limits, malformed/truncated input, and EXIF behavior checks. Updated the wheel package contract for the additional package.

## Concerns

- The pixel-limit test uses a 4,002,000-pixel image rather than exactly 4,000,001 pixels; it verifies rejection above the configured 4,000,000 limit.
- MIME declaration disagreement is deferred to Task 9 as directed. The prior Task 1 weak fixture-byte assertion remains outside this task.

## Review fixes (2026-10-17)

Addressed both P2 review findings:

- Reject resized widths below 4 as well as widths above `max_width`; added coverage for a 4×100 source that resizes to width 1.
- Added `src/captix_runtime` to the Hatch sdist include list and asserted the config in `test_baseline_contract.py`.

RED command: `rtk proxy uv run pytest tests/test_runtime_preprocessing.py -q`

Exact relevant output before the fix:

```text
FAILED tests/test_runtime_preprocessing.py::test_rejects_invalid_inputs[...-invalid_width]
E       Failed: DID NOT RAISE <class 'captix_runtime.preprocessing.PreprocessingError'>
```

Cover tests command: `rtk proxy uv run pytest tests/test_runtime_preprocessing.py tests/test_baseline_contract.py -q`

Exact output:

```text
.......................                                                  [100%]
```

Sdist verification command: `uv build --sdist --out-dir /tmp/captix-task2-sdist && tar -tzf /tmp/captix-task2-sdist/*.tar.gz | rg 'src/captix_runtime/(__init__|preprocessing)\\.py'`

Exact output:

```text
Building source distribution...
Successfully built /tmp/captix-task2-sdist/captix-2.0.0.tar.gz
captix-2.0.0/src/captix_runtime/__init__.py
captix-2.0.0/src/captix_runtime/preprocessing.py
```

`git diff --check` completed with no output.
