"""Image decoding and normalization for the ONNX runtime."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError


class PreprocessingError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PreprocessedImage:
    array: np.ndarray


def _flatten_transparency(image: Image.Image) -> Image.Image:
    has_alpha = image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    )
    if not has_alpha:
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    pixels = np.asarray(rgba)
    visible = pixels[..., 3] > 0
    mean_lum = float(pixels[..., :3][visible].mean()) if visible.any() else 0.0
    value = 255 if mean_lum < 128 else 0
    background = Image.new("RGB", rgba.size, (value, value, value))
    background.paste(rgba, mask=rgba.getchannel("A"))
    return background


def preprocess_image(
    data: bytes, *, max_pixels: int = 4_000_000, max_width: int = 1024
) -> PreprocessedImage:
    try:
        stream = BytesIO(data)
        source = Image.open(stream)
        with source:
            if getattr(source, "n_frames", 1) != 1:
                raise PreprocessingError("animated", "Animated images are not supported")
            if source.format not in {"PNG", "JPEG", "WEBP"}:
                raise PreprocessingError("unsupported_media", "Only PNG, JPEG, and WEBP are supported")
            width, height = source.size
            if width * height > max_pixels:
                raise PreprocessingError("too_many_pixels", "Image exceeds the pixel limit")
            if width < 4:
                raise PreprocessingError("invalid_width", "Image width is too small")
            resized_width = max(1, round(width * (32 / height)))
            if resized_width > max_width:
                raise PreprocessingError("invalid_width", "Resized image exceeds the width limit")
            source.verify()
        with Image.open(BytesIO(data)) as source:
            image = _flatten_transparency(source)
            image = image.resize((resized_width, 32), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
            return PreprocessedImage(np.ascontiguousarray(array, dtype=np.float32))
    except PreprocessingError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PreprocessingError("invalid_image", "Invalid or truncated image data") from exc
