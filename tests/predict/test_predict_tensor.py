# SPDX-License-Identifier: Apache-2.0
"""Gate on single-image inference from an image already held as a tensor.

Every ``predict_*`` entry point takes either a file path or a decoded ``(3, H, W)`` RGB tensor. The contract worth
pinning is that the two are the same input: a picture passed as ``uint8`` codes, or as floats already in ``[0, 1]``,
gets exactly the answer the same picture gets when saved and read back, because both reach the model through one scale
and one letterbox (:func:`~lucid_yolo.eval.annotations.letterbox_image_tensor`). An untrained module is used on
purpose: its answer depends on every pixel, so equal outputs mean equal inputs reached the network, which a stub
ignoring its input could not show.

The refusals cover the three tensors a caller most plausibly holds by mistake: an ``(H, W, 3)`` array from PIL or numpy,
a float image still in ``0..255``, and signed integer codes; and float images holding a NaN or infinite value, which a
min/max range check alone passes once a NaN makes the extrema NaN.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
from planted import ORIGINAL_SIZE
from torchvision.io import write_png

from lucid_yolo.data._image import decode_rgb
from lucid_yolo.predict import predict_image, predict_segmentation
from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from pathlib import Path

#: Letterbox side: a multiple of every stride, and small enough to run in a blink.
_IMG_SIZE = 64
#: Keep every row the decoder emits, so the comparison covers a full output.
_KEEP_ALL = 0.0


def _module(task: str) -> DetectionLitModule:
    """Build an untrained n-scale module for ``task``.

    Examples:
        >>> _module("segment").task
        'segment'
    """
    return DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=3, task=task).eval()


@pytest.fixture
def noise_file(tmp_path: Path) -> Path:
    """Write a random RGB image, so every pixel influences an untrained module's answer."""
    path = tmp_path / "noise.png"
    write_png(torch.randint(0, 256, (3, *ORIGINAL_SIZE), dtype=torch.uint8), str(path))
    return path


@pytest.mark.parametrize("form", ["uint8", "unit-float"])
def test_tensor_image_answers_as_file(noise_file: Path, form: str) -> None:
    """A decoded image, as codes or as unit floats, yields the file's detections exactly.

    ``unit-float`` is the codes divided by 255 by the caller, the most common way an image arrives already scaled.
    """
    module = _module("detect")
    codes = decode_rgb(noise_file)
    image = codes if form == "uint8" else codes.float() / 255

    from_tensor = predict_image(module, image, img_size=_IMG_SIZE, conf_threshold=_KEEP_ALL)
    from_file = predict_image(module, noise_file, img_size=_IMG_SIZE, conf_threshold=_KEEP_ALL)

    assert from_tensor.shape[0] > 0
    torch.testing.assert_close(from_tensor, from_file, rtol=0, atol=0)


def test_tensor_image_masks_match_file(noise_file: Path) -> None:
    """The segmentation entry point reads a tensor exactly as a file, masks included."""
    module = _module("segment")

    from_tensor = predict_segmentation(module, decode_rgb(noise_file), img_size=_IMG_SIZE, conf_threshold=_KEEP_ALL)
    from_file = predict_segmentation(module, noise_file, img_size=_IMG_SIZE, conf_threshold=_KEEP_ALL)

    torch.testing.assert_close(from_tensor.detections, from_file.detections, rtol=0, atol=0)
    assert torch.equal(from_tensor.masks, from_file.masks)


@pytest.mark.parametrize(
    ("image", "error", "message"),
    [
        pytest.param(torch.zeros(*ORIGINAL_SIZE, 3, dtype=torch.uint8), ValueError, "permute", id="hwc-array"),
        pytest.param(torch.full((3, *ORIGINAL_SIZE), 255.0), ValueError, r"\[0, 1\]", id="float-0-255"),
        pytest.param(torch.full((3, *ORIGINAL_SIZE), torch.nan), ValueError, r"got \d+ NaN or infinite", id="all-nan"),
        pytest.param(
            torch.full((3 * ORIGINAL_SIZE[0] * ORIGINAL_SIZE[1],), 0.5)
            .index_fill_(0, torch.tensor([0]), torch.nan)
            .view(3, *ORIGINAL_SIZE),
            ValueError,
            "got 1 NaN or infinite",
            id="one-nan-in-unit-image",
        ),
        pytest.param(torch.full((3, *ORIGINAL_SIZE), torch.inf), ValueError, r"got \d+ NaN or infinite", id="inf"),
        pytest.param(torch.zeros(3, *ORIGINAL_SIZE, dtype=torch.int32), TypeError, "unsigned", id="signed-codes"),
    ],
)
def test_implausible_tensor_refused(image: torch.Tensor, error: type[Exception], message: str) -> None:
    """A tensor the model would answer plausibly but wrongly is refused, naming the fix."""
    with pytest.raises(error, match=message):
        predict_image(_module("detect"), image, img_size=_IMG_SIZE)
