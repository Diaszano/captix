"""Download and unzip the labeled captcha datasets from the R package releases."""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import List, Union

PathLike = Union[str, Path]

DATASET_NAMES: List[str] = [
    "cadesp", "esaj", "jucesp", "rfb", "sei",
    "tjmg", "tjpe", "tjrs", "trf5", "trt",
]

DATASET_REPO_BASE = "https://github.com/decryptr/captcha/releases/download/data"


def available_datasets() -> List[str]:
    """List of dataset names published in the R captcha release."""
    return list(DATASET_NAMES)


def download_dataset(
    name: str,
    dest_dir: PathLike,
    extract: bool = True,
    force: bool = False,
) -> Path:
    """Download (and optionally unzip) one of the R captcha datasets.

    Returns the path to the extracted directory (or the .zip if `extract=False`).
    """
    import requests
    from tqdm import tqdm

    if name not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset {name!r}. Choose one of {DATASET_NAMES}.")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dest_dir / f"{name}.zip"
    out_dir = dest_dir / name

    if not zip_path.exists() or force:
        url = f"{DATASET_REPO_BASE}/{name}.zip"
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=name
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

    if not extract:
        return zip_path

    if not out_dir.exists() or force:
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)

    return out_dir
