# Changelog

All notable changes to `txtcaptcha` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-11

Initial public release.

### Added
- CRNN+CTC architecture (`txtcaptcha.CRNN`) with ResNet-style CNN backbone,
  2-layer BiLSTM and CTC head. Handles variable-width input and variable
  label lengths out of the box.
- `read_captcha`, `Captcha`, `annotate` for loading/labeling images.
- `CaptchaDataset`, `transform_image`, `encode_label`, `decode_indices`,
  `pad_collate` for building training pipelines.
- `fit_model` training loop with validation split, CTC loss and early
  stopping.
- `decrypt()` with three decoding modes:
  - greedy CTC (default),
  - fixed-length exact DP via `length=N`,
  - decode-time masking via `mask=` (list, regex, or `re.Pattern`).
- `save_model` / `load_model` with automatic CPU fallback when CUDA is
  unavailable.
- Dataset fetchers: `available_datasets`, `download_dataset`.
- Live captcha fetchers via `download_captchas` CLI and the
  `txtcaptcha.download` subpackage (10 Brazilian court / tax sources).
- Hugging Face Hub integration:
  - `save_pretrained`, `from_pretrained`, `push_to_hub`, `DEFAULT_REPO_ID`,
  - safetensors + `config.json` + model card layout,
  - `decrypt()` auto-downloads the unified checkpoint from
    [`jtrecenti/txtcaptcha-crnn`](https://huggingface.co/jtrecenti/txtcaptcha-crnn)
    on first call and caches it.
- Unified pretrained checkpoint published on the Hub at
  `jtrecenti/txtcaptcha-crnn` (~89% captcha-level accuracy across ten
  Brazilian court datasets).
- Notebooks: `train_unified_model.ipynb`, `eval_per_dataset.ipynb`,
  `eval_per_dataset_live.ipynb`.
- GitHub Actions:
  - CI running pytest on push/PR,
  - PyPI publishing via trusted publishing on GitHub releases,
  - Quarto + quartodoc site deployed to GitHub Pages on pushes to `main`.

[0.1.0]: https://github.com/jtrecenti/txtcaptcha/releases/tag/v0.1.0
