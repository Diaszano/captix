from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from captix.dataset import transform_image
from captix_runtime.preprocessing import PreprocessingError, preprocess_image


def encoded(image, fmt="PNG"):
    stream = BytesIO()
    image.save(stream, format=fmt)
    return stream.getvalue()


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_matches_torch_transform_for_png(mode):
    img = Image.new(mode, (80, 16), (0, 0, 0) if mode == "RGB" else (0, 0, 0, 255))
    raw = encoded(img)
    expected = transform_image(Image.open(BytesIO(raw))).numpy()
    result = preprocess_image(raw)
    np.testing.assert_array_equal(result.array, expected)
    assert result.array.dtype == np.float32
    assert result.array.shape == (1, 3, 32, 160)
    assert result.array.flags.c_contiguous
    assert result.array.min() >= 0 and result.array.max() <= 1


def test_transparency_and_resize():
    dark = Image.new("RGBA", (80, 16), (0, 0, 0, 0))
    dark.putpixel((0, 0), (0, 0, 0, 255))
    assert preprocess_image(encoded(dark)).array[0, 0, 0, 4] == 1
    light = Image.new("RGBA", (80, 16), (0, 0, 0, 0))
    light.paste((240, 240, 240, 255), (20, 4, 60, 12))
    assert preprocess_image(encoded(light)).array[0, 0, 0, 4] == 0
    absent = Image.new("RGBA", (80, 16), (0, 0, 0, 0))
    assert preprocess_image(encoded(absent)).array[0, 0, 0, 0] == 1


def test_supported_static_formats():
    for fmt in ("JPEG", "WEBP"):
        assert preprocess_image(encoded(Image.new("RGB", (16, 32), "white"), fmt)).array.shape == (1, 3, 32, 16)


@pytest.mark.parametrize("fmt", ["GIF", "WEBP"])
def test_rejects_animated_images(fmt):
    a, b = Image.new("RGB", (16, 32), "white"), Image.new("RGB", (16, 32), "black")
    stream = BytesIO()
    a.save(stream, format=fmt, save_all=True, append_images=[b], duration=100, loop=0)
    with pytest.raises(PreprocessingError) as exc:
        preprocess_image(stream.getvalue())
    assert exc.value.code == "animated"


@pytest.mark.parametrize("data,kwargs,code", [
    (encoded(Image.new("RGB", (2001, 2000))), {}, "too_many_pixels"),
    (encoded(Image.new("RGB", (3, 32))), {}, "invalid_width"),
    (encoded(Image.new("RGB", (4, 100))), {}, "invalid_width"),
    (encoded(Image.new("RGB", (1025, 32))), {}, "invalid_width"),
    (b"not an image", {}, "invalid_image"),
    (b"random bytes", {}, "invalid_image"),
    (encoded(Image.new("RGB", (16, 32)), "BMP"), {}, "unsupported_media"),
    (encoded(Image.new("RGB", (16, 32)))[:-20], {}, "invalid_image"),
    (b"", {}, "invalid_image"),
    (encoded(Image.new("RGB", (16, 32))), {"max_width": 3}, "invalid_width"),
])
def test_rejects_invalid_inputs(data, kwargs, code):
    with pytest.raises(PreprocessingError) as exc:
        preprocess_image(data, **kwargs)
    assert exc.value.code == code


def test_does_not_apply_exif_orientation():
    img = Image.new("RGB", (80, 16), "white")
    exif = img.getexif()
    exif[274] = 6
    raw = encoded(img)
    stream = BytesIO()
    img.save(stream, format="JPEG", exif=exif)
    assert preprocess_image(stream.getvalue()).array.shape == (1, 3, 32, 160)


def assert_error(data, code):
    with pytest.raises(PreprocessingError) as exc:
        preprocess_image(data)
    assert exc.value.code == code
