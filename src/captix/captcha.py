"""Captcha container class and read_captcha function."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

PathLike = str | Path

_LABEL_RE = re.compile(r"(?<=_)[0-9a-zA-Z]+")


def _extract_label(path: Path) -> str | None:
    stem = path.stem
    m = _LABEL_RE.search(stem)
    return m.group(0) if m else None


@dataclass
class Captcha:
    """Container for one or more captcha images.

    Mirrors the R `captcha` S3 object: a list with `img`, `lab`, `path`.
    """

    images: list[Image.Image] = field(default_factory=list)
    labels: list[str | None] = field(default_factory=list)
    paths: list[Path] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> Captcha:
        if isinstance(idx, int):
            return Captcha(
                images=[self.images[idx]],
                labels=[self.labels[idx]],
                paths=[self.paths[idx]],
            )
        if isinstance(idx, slice):
            return Captcha(
                images=self.images[idx],
                labels=self.labels[idx],
                paths=self.paths[idx],
            )
        if isinstance(idx, (list, tuple)):
            return Captcha(
                images=[self.images[i] for i in idx],
                labels=[self.labels[i] for i in idx],
                paths=[self.paths[i] for i in idx],
            )
        raise TypeError(f"Unsupported index type: {type(idx)}")

    def __repr__(self) -> str:
        return f"Captcha(n={len(self)})"

    def plot(self, cols: int = 4, max_rows: int = 25):
        """Display captcha images in a grid using matplotlib."""
        import matplotlib.pyplot as plt

        n = min(len(self), cols * max_rows)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 1.2))
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[a] for a in axes]
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            ax = axes[r][c]
            ax.axis("off")
            if i < n:
                ax.imshow(self.images[i])
                if self.labels[i]:
                    ax.set_title(self.labels[i], fontsize=8)
        fig.tight_layout()
        return fig


def read_captcha(
    files: PathLike | Sequence[PathLike],
    lab_in_path: bool = False,
) -> Captcha:
    """Read one or more captcha image files into a Captcha object.

    Parameters
    ----------
    files: path or sequence of paths
    lab_in_path: extract label from filename (after the first underscore).
    """
    if isinstance(files, (str, Path)):
        paths: list[Path] = [Path(files)]
    else:
        paths = [Path(f) for f in files]

    images: list[Image.Image] = []
    labels: list[str | None] = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        img = Image.open(p)
        img.load()
        images.append(img)
        labels.append(_extract_label(p) if lab_in_path else None)

    return Captcha(images=images, labels=labels, paths=paths)
