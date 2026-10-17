"""Export a local CRNN checkpoint as captix-v1-onnx-fp32 (no network access)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from importlib.metadata import version
from pathlib import Path

import torch


class HeightMean(torch.nn.Module):
    """Export-only equivalent of AdaptiveAvgPool2d((1, None))."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=2, keepdim=True)


def export_checkpoint(
    source_dir: Path,
    out_dir: Path,
    *,
    revision_sha: str,
    source_commit: str,
) -> Path:
    for name, sha in (("revision_sha", revision_sha), ("source_commit", source_commit)):
        if re.fullmatch(r"[0-9a-f]{40}", sha) is None:
            raise ValueError(f"{name} must be a lowercase 40-character commit SHA")

    import onnx

    from captix.hub import load_local

    source_dir, out_dir = Path(source_dir), Path(out_dir)
    config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("img_height") != 32 or config.get("blank_index") != 0:
        raise ValueError("Expected checkpoint height 32 and blank_index 0")
    model = load_local(source_dir).cpu().float()
    model.collapse = HeightMean()
    model.eval()
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".captix-export-", dir=out_dir.parent
    ) as temporary:
        stage = Path(temporary) / "artifact"
        stage.mkdir()
        graph_path = stage / "model.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                model,
                (torch.zeros(1, 3, 32, 160),),
                graph_path,
                input_names=["images"],
                output_names=["logits"],
                # Dynamo in torch 2.11 specializes the LSTM sequence to W=160.
                dynamo=False,
                dynamic_axes={"images": {3: "width"}, "logits": {0: "time"}},
                opset_version=18,
                external_data=False,
            )
        onnx.checker.check_model(str(graph_path))
        manifest = {
            "schema_version": 1,
            "model_version": "captix-v1-onnx-fp32",
            "exporter": "torchscript",
            "opset": 18,
            "model_sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
            "source_commit": source_commit,
            "revision_sha": revision_sha,
            "source_sha256": {
                name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                for name in ("model.safetensors", "config.json")
            },
            "versions": {
                name: version(name)
                for name in ("torch", "onnx", "onnxscript", "onnxruntime")
            },
            "height": 32,
            "input_name": "images",
            "output_name": "logits",
            "dtype": "float32",
            "vocab": model.vocab,
            "blank_index": 0,
            "width_range": [4, 1024],
            "input_scale": [0, 1],
            "input_layout": "NCHW",
            "output_layout": "TBC",
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        stage.rename(out_dir)
    return out_dir / "model.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", required=True, type=Path, help="Local safetensors checkpoint"
    )
    parser.add_argument(
        "--out-dir", required=True, type=Path, help="New output directory"
    )
    parser.add_argument(
        "--revision-sha", required=True, help="Immutable 40-character model commit SHA"
    )
    parser.add_argument(
        "--source-commit", required=True, help="40-character Captix source commit SHA"
    )
    args = parser.parse_args()
    print(
        export_checkpoint(
            args.source_dir,
            args.out_dir,
            revision_sha=args.revision_sha,
            source_commit=args.source_commit,
        )
    )


if __name__ == "__main__":
    main()
