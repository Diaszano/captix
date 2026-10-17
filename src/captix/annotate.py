"""Manual / batch annotation of captcha files."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

PathLike = Union[str, Path]


def _annotate_one(
    file: Path,
    label: str,
    out_dir: Optional[Path],
    rm_old: bool,
) -> Path:
    target_dir = out_dir if out_dir is not None else file.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    new_name = f"{file.stem}_{label}{file.suffix}"
    new_path = target_dir / new_name
    if rm_old and out_dir is None:
        file.rename(new_path)
    else:
        shutil.copy2(file, new_path)
        if rm_old:
            file.unlink()
    return new_path


def annotate(
    files: Union[PathLike, Sequence[PathLike]],
    labels: Optional[Sequence[str]] = None,
    path: Optional[PathLike] = None,
    rm_old: bool = False,
    prompt: Optional[Callable[[Path], str]] = None,
) -> List[Path]:
    """Annotate captcha files by appending `_<label>` to the filename.

    Parameters
    ----------
    files: file path or list of file paths.
    labels: list of labels matching `files`. If None, `prompt` is called
        for each file to obtain the label interactively.
    path: optional output directory. Defaults to the file's directory.
    rm_old: if True, remove the original file after annotation.
    prompt: callable taking the file Path and returning a label string.
        Defaults to interactive `input()`.
    """
    if isinstance(files, (str, Path)):
        file_list = [Path(files)]
    else:
        file_list = [Path(f) for f in files]

    out_dir = Path(path) if path is not None else None

    if labels is None:
        if prompt is None:
            def prompt(f: Path) -> str:  # type: ignore[misc]
                return input(f"Label for {f.name}: ").strip()
        labels = [prompt(f) for f in file_list]
    else:
        if len(labels) != len(file_list):
            raise ValueError(
                f"labels length ({len(labels)}) must match files length ({len(file_list)})"
            )

    return [
        _annotate_one(f, lab, out_dir, rm_old)
        for f, lab in zip(file_list, labels)
    ]
