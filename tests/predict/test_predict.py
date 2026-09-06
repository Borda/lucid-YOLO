# SPDX-License-Identifier: Apache-2.0
"""Single-image detection inference and the ``lucid-predict`` command (WP-089).

The substance of this suite is a **geometry round trip**, not detection quality. No
trained weights exist and none may be created (D14), so a detection is *planted*: a
module stub emits one high-confidence anchor whose ltrb distances decode to a box chosen
by hand on the letterboxed canvas, and the assertion is that
:func:`~lucid_yolo.predict.predict_image` lands it on the original pixel grid where the
letterbox geometry says it must. Everything between the plant and the assertion — the
read, the resize, the pad, the decode, the inverse — is the shipping code path.

The image size, the planted box and the original coordinates it must land at live in
``planted.py``, which explains the arithmetic; the segmentation suite asserts against the
same numbers, and its mask-inside-box claim is only meaningful if it is the same box.

The planted logits are written to **both** branches, so one parametrized case covers the
suppression-free path over the one-to-one branch and the suppression path over the dense
one, and neither can pass by reading the other's output.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch
from planted import (
    ABSENT_LOGIT,
    CANVAS_BOX,
    EXPECTED_ORIGINAL_BOX,
    IMG_SIZE,
    NUM_CLASSES,
    PLANTED_LABEL,
    PRESENT_LOGIT,
    SUPERSEDED_SEAM,
    plant_detection,
    write_checkpoint,
)

from lucid_yolo.assign.grid import make_anchor_points
from lucid_yolo.cli import predict as predict_cli
from lucid_yolo.decode.nms_path import NMSDecoder
from lucid_yolo.decode.topk_e2e import TopKDecoder
from lucid_yolo.models.heads.detect import BranchName, BranchOutput, DualHeadOutput
from lucid_yolo.predict import (
    _TASK_ENTRY_POINTS,
    DECODE_PATHS,
    _wrong_task_message,
    predict_image,
)
from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor


class _PlantedDetectionModule(DetectionLitModule):
    """A ``detect`` module whose forward reports exactly one object, at a chosen box.

    Subclasses the real module rather than faking one, so ``task`` is genuinely the
    module's own property and :func:`~lucid_yolo.predict.predict_image` reads it the way
    it reads a checkpoint's. The replaced method is
    :meth:`~lucid_yolo.ptl.module.DetectionLitModule.forward_branch`, which is the seam
    that entry point calls; the untrained backbone would answer with noise, and what is
    under test is the geometry between the file on disk and the returned coordinates, not
    what a network saw.

    The superseded dual-branch :meth:`forward` is made to raise rather than left planting
    outputs nobody reads. A stub that kept answering on the old seam would simply stop
    being consulted, and the assertions below would pass against head noise while proving
    nothing -- see :data:`~planted.SUPERSEDED_SEAM`.

    Args:
        canvas_box: The ``xyxy`` box, in letterboxed-canvas pixels, the single planted
            detection decodes to.
        label: Class index the planted detection carries.
        num_classes: Class count the stub head reports over. Defaults to ``2``.

    Examples:
        >>> module = _PlantedDetectionModule((8.0, 24.0, 40.0, 48.0), label=1)
        >>> module.task
        'detect'
    """

    def __init__(self, canvas_box: tuple[float, ...], label: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__(depth=0.34, width=0.25, max_channels=64, num_classes=num_classes, task="detect")
        self._canvas_box = tuple(float(value) for value in canvas_box)
        self._label = int(label)
        self._num_classes = int(num_classes)

    def forward_branch(self, images: Tensor, branch: BranchName) -> BranchOutput:
        """Emit one confident anchor decoding to the planted box, on whichever branch is asked."""
        # Both decode paths are meant to select the same detection here, so the plant does
        # not vary by branch; which branch was requested is proved by the suites that do
        # vary it (segmentation's coefficients, keypoints' point sets).
        del branch
        grid = plant_detection(images, self._canvas_box, self._label, self._num_classes)
        return BranchOutput(cls=grid.cls, box=grid.box)

    def forward(self, images: Tensor) -> DualHeadOutput:
        """Refuse the superseded dual-branch seam, loudly."""
        del images
        raise AssertionError(SUPERSEDED_SEAM)


@pytest.mark.parametrize("decoder", [pytest.param(path, id=path) for path in DECODE_PATHS])
def test_known_object_lands_in_original_coordinates(image_file: Path, decoder: str) -> None:
    """A detection planted on the canvas comes back at the original coordinates it implies.

    The image is 64x128 letterboxed into 64x64: ratio 0.5, 16 px of padding on top, none
    on the left. A canvas box of ``[8, 24, 40, 48]`` therefore has to land at
    ``[16, 16, 80, 64]`` in the original image, and does so only if the pad is subtracted
    before the ratio is undone and only on the axis that carries it. Both decode paths
    are exercised, from the same planted logits.
    """
    module = _PlantedDetectionModule(CANVAS_BOX, label=PLANTED_LABEL).eval()

    detections = predict_image(module, image_file, img_size=IMG_SIZE, decoder=decoder)

    assert detections.shape == (1, 6)
    assert detections[0, :4].tolist() == pytest.approx(EXPECTED_ORIGINAL_BOX)
    assert int(detections[0, 5]) == PLANTED_LABEL
    assert float(detections[0, 4]) > 0.9


@pytest.mark.parametrize("task", [pytest.param("segment", id="segment"), pytest.param("obb", id="obb")])
def test_a_non_detect_checkpoint_fails_naming_its_task(image_file: Path, task: str) -> None:
    """Segmentation and oriented checkpoints are refused by name, never run as detectors.

    Both build a head whose plain forward returns the same detection output a detector's
    does, so falling through would produce boxes with the mask or angle branch silently
    unread — a plausible answer to a question nobody asked. Each has a destination of its
    own now (:func:`~lucid_yolo.predict.predict_segmentation`, WP-090;
    :func:`~lucid_yolo.predict.predict_oriented`, WP-091), and the refusal names it rather
    than merely declining, so the message is a redirection instead of a dead end.
    """
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=NUM_CLASSES, task=task).eval()

    with pytest.raises(ValueError, match=task):
        predict_image(module, image_file, img_size=IMG_SIZE)


def test_the_report_is_written_into_a_directory_that_does_not_exist_yet(
    tmp_path: Path,
    image_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--output`` creates its parent, and stdout summarises the same run.

    The forward pass is the expensive part of this command, so a missing parent
    directory discovered afterwards discards work that has already been done — the
    WP-105 defect, guarded here so it cannot come back through a second entry point.
    """
    checkpoint = write_checkpoint("detect", tmp_path / "det.ckpt")
    report = tmp_path / "absent" / "nested" / "dets.json"

    code = predict_cli.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--image",
            str(image_file),
            "--ema",
            "false",
            "--img_size",
            str(IMG_SIZE),
            "--output",
            str(report),
        ]
    )

    assert code == 0
    payload = json.loads(report.read_text())
    assert payload["decoder"] == "e2e"
    assert payload["info"]["task"] == "detect"
    assert payload["masks"] is None  # a detector says so, rather than omitting the key
    assert isinstance(payload["detections"], list)
    assert str(image_file) in capsys.readouterr().out


class TestTaskRefusalMessages:
    """Every entry point's refusal names every other entry point (audit L-14)."""

    @pytest.mark.parametrize(
        "handled",
        [pytest.param(task, id=task) for task in ("detect", "segment", "obb", "keypoints")],
    )
    def test_a_refusal_names_all_three_siblings(self, handled: str) -> None:
        """Each task's refusal message names the other three entry points, not a subset.

        Three of the four messages were written by hand when only three tasks existed and
        each named the two siblings of its own era; WP-152 added a fourth entry point
        without revisiting them, so a caller holding a pose checkpoint and calling
        ``predict_image`` was told about segmentation and orientation and not about the
        function they actually wanted. Composing every message from one table is what
        makes a message naming a subset unrepresentable rather than merely unlikely.
        """
        message = _wrong_task_message(handled, "whatever-the-checkpoint-says")
        siblings = [task for task in _TASK_ENTRY_POINTS if task != handled]

        assert _TASK_ENTRY_POINTS[handled][0] in message, "the message names the function called"
        for sibling in siblings:
            assert _TASK_ENTRY_POINTS[sibling][0] in message, f"{sibling}'s entry point is unnamed"

    def test_a_refusal_names_the_task_the_checkpoint_actually_carries(self) -> None:
        """The message reports the checkpoint's own task, which is what the caller has to act on.

        Naming the siblings is only half of a redirection: the caller also needs to know
        which of them is theirs, and the only place that is written is the task the
        checkpoint reported.
        """
        message = _wrong_task_message("detect", "keypoints")

        assert "'keypoints'" in message
        assert "predict_keypoints" in message


class TestDecoderIndexDtype:
    """Both decode paths hand back the same index dtype (audit L-19)."""

    @pytest.mark.parametrize(
        "decoder",
        [pytest.param("e2e", id="e2e"), pytest.param("nms", id="nms")],
    )
    def test_anchor_indices_are_long_on_both_paths(self, decoder: str) -> None:
        """``decode_with_indices`` returns a long tensor whichever path produced it.

        Both decoders document a *long* tensor, and both deliver one -- but only the
        keypoint consumer used to cast the result to ``torch.int64`` before gathering,
        while its mask-path sibling gathered by the same indices with no cast at all. Two
        spellings of one contract read as two contracts. This pins the contract from the
        consumer's side, so dropping the redundant cast is safe and stays safe; it
        characterizes existing behaviour rather than failing on the code before the fix.
        """
        points, strides = make_anchor_points([(2, 2)], [8])
        cls_logits = torch.full((1, 4, NUM_CLASSES), ABSENT_LOGIT)
        cls_logits[0, 1, PLANTED_LABEL] = PRESENT_LOGIT
        raw_ltrb = torch.full((1, 4, 4), 0.25)
        chosen = TopKDecoder(k=3) if decoder == "e2e" else NMSDecoder(conf_threshold=0.5, max_det=3)

        _, anchor_index = chosen.decode_with_indices(cls_logits, raw_ltrb, points, strides)

        assert anchor_index.dtype == torch.int64, "index_select requires a long index"
