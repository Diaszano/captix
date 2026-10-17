# Synthetic golden fixtures

The baseline tests create tiny PNG and RGBA images deterministically in a
temporary directory. These fixtures exercise image loading and tensor shape
contracts only; they do not measure OCR accuracy on real captchas. Add corpus
images only when their source and license authorize benchmark use, and keep
their manifests explicit. `freeze_corpus` references existing files and never
copies them.
