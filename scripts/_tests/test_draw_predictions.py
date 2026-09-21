# SPDX-License-Identifier: Apache-2.0
"""The prediction-drawing command turns a checkpoint and an image into a written figure (WP-067, WP-189).

The drawing itself — what each tuple becomes on the axes, the confidence cut, the rotated
ring, the mask alignment — is :mod:`lucid_yolo._viz.overlay`'s and is asserted in
``tests/viz/test_overlay.py``, where the code moved. What stays here is the command: the
argument parse, the checkpoint load, the write, and the two refusals the command answers
with a code or a name rather than a traceback. The end-to-end case drives ``main`` through
a real ``.ckpt`` written by ``tests/predict/planted.py``: nothing is asserted about
*which* objects an untrained network finds, only that the checkpoint-to-file path runs
and writes a decodable image.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import pytest
import torch
from planted import IMG_SIZE, write_checkpoint
from torchvision.io import write_png

matplotlib.use("Agg")

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "draw_predictions.py"

#: Image the command runs on, ``(height, width)``: not square, and at the ``planted``
#: letterbox side of 64 the ratio is exactly 1.0 with a vertical pad, so the end-to-end
#: case letterboxes without enlarging the picture it drew.
IMAGE_HEIGHT, IMAGE_WIDTH = 48, 64


def _load_script() -> ModuleType:
    """Load ``scripts/draw_predictions.py`` as an importable module.

    Examples:
        >>> module = _load_script()
        >>> hasattr(module, "render_prediction")
        True
    """
    spec = importlib.util.spec_from_file_location("draw_predictions", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered before exec
    spec.loader.exec_module(module)
    return module


draw = _load_script()


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    """Write a mid-grey picture to disk and return its path.

    Content is irrelevant — nothing asserts on pixels — but the file has to decode, and its
    shape is what fixes the letterbox geometry the end-to-end case runs through.
    """
    path = tmp_path / "scene.png"
    write_png(torch.full((3, IMAGE_HEIGHT, IMAGE_WIDTH), 128, dtype=torch.uint8), str(path))
    return path


def test_the_drawing_functions_stay_importable_from_the_script() -> None:
    """The names the script published before the move are still attributes of the module.

    A caller who imported ``draw_detections`` or ``RenderOptions`` from the script keeps
    working: the script re-exports them from ``lucid_yolo._viz.overlay``, and its
    ``__all__`` is the list this asserts against.
    """
    assert all(hasattr(draw, name) for name in draw.__all__)
    assert {"draw_detections", "draw_segmentation", "draw_oriented", "draw_keypoints", "RenderOptions"} <= set(
        draw.__all__
    )


def test_the_command_renders_a_checkpoint_it_loaded_to_a_file(
    tmp_path: Path,
    image_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``main`` turns a real ``.ckpt`` and an image into a written PNG, creating its directory.

    The one case that proves the command path rather than assuming it: a checkpoint is
    what the script reads, and neither the planted tuples nor a directly called drawing
    function exercises the load, the task lookup, the letterbox side or the write.
    Nothing is asserted about *which* boxes an untrained network draws — only that the
    path from a file on disk to a figure on disk runs and produces a decodable image.
    """
    checkpoint = write_checkpoint("detect", tmp_path / "det.ckpt")
    output = tmp_path / "figures" / "scene_det.png"

    code = draw.main(
        [
            str(checkpoint),
            str(image_file),
            "--output",
            str(output),
            "--img-size",
            str(IMG_SIZE),
            "--no-ema",
        ]
    )

    assert code == 0
    assert output.is_file()
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "the suffix chose the format"
    assert f"figure -> {output}" in capsys.readouterr().out


def test_a_task_the_checkpoint_contradicts_is_refused_by_name(tmp_path: Path, image_file: Path) -> None:
    """``--task obb`` on a detection checkpoint raises, naming the task the checkpoint carries.

    What makes the flag safe to default from the checkpoint and to accept from a caller:
    naming a task selects an entry point, and every entry point in
    :mod:`lucid_yolo.predict` refuses a checkpoint that is not its own. Without that, the
    flag would be the mistake ``lucid-predict`` deliberately has no way to make — an
    oriented drawing of a detector's boxes, at a heading nothing predicted.
    """
    checkpoint = write_checkpoint("detect", tmp_path / "det.ckpt")

    with pytest.raises(ValueError, match="detect"):
        draw.main(
            [
                str(checkpoint),
                str(image_file),
                "--output",
                str(tmp_path / "never.png"),
                "--task",
                "obb",
                "--img-size",
                str(IMG_SIZE),
                "--no-ema",
            ]
        )


def test_a_missing_input_file_reports_it_and_returns_one(
    tmp_path: Path,
    image_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A checkpoint path that does not exist is named on stderr, with exit code 1.

    The mistake an operator makes most often — a stale run directory in the path — and the
    only failure this script answers with a code rather than a traceback, because a
    ``FileNotFoundError`` from inside a Lightning loader names a temporary file rather than
    the argument that was wrong.
    """
    code = draw.main([str(tmp_path / "absent.ckpt"), str(image_file), "--output", str(tmp_path / "never.png")])

    assert code == 1
    assert "no such checkpoint file" in capsys.readouterr().err
