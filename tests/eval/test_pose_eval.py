# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the pose acceptance instrument (WP-134, WP-174).

``lucid-eval`` on a keypoints checkpoint is the acceptance scorer of the keypoint tier,
and until this file existed its whole scoring body was unexecuted: the module's only
executable Example is ``>>> callable(run)``, which a broken implementation also satisfies.
:mod:`tests.eval.test_keypoint_eval` pins the pieces underneath — the prediction adapter,
OKS matching, which branch each path reads — on hand-built tensors; what was missing is
the seam that joins them to a directory on disk, which is where the annotation file's
name, the batching, the sigma table and the report's own keys are decided.

There is no COCO checkout on the gate machine, so the split is built here: a
``person_keypoints_val2017.json`` carrying COCO's own 17-point schema beside a ``val2017/``
of tiny PNGs, the exact layout
:func:`~lucid_yolo.eval.pose_eval.run` resolves from ``data_root``. An untrained module
scores near zero and that is fine — what is under test is that the ground truth, the
predictions and the accumulator meet at all, and that the report says which figures it is
reporting. "It ran and produced a float" is precisely the claim ``callable(run)`` already
made.

The refusal is covered from the other side too. ``K != 17`` is a documented contract, and
:mod:`tests.eval.test_keypoint_eval` already exercises it at ``K=7``; the case added here
is ``K=16``, the value adjacent to the boundary, where an off-by-one in the comparison
would let a foreign schema through onto COCO's sigma table.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import torch
from torchvision.io import write_png

from lucid_yolo.eval import pose_eval
from lucid_yolo.eval.coco_eval import COCO_KEYPOINT_OKS_SIGMAS
from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from pathlib import Path

#: COCO's person schema point count, read from the sigma table rather than spelled, so
#: this file and the protocol cannot disagree about what "the 17-point schema" is.
_COCO_POINTS = len(COCO_KEYPOINT_OKS_SIGMAS)

#: Letterbox side: divisible by the level-32 stride and small enough that two forward
#: passes over an untrained backbone stay inside a unit test's budget.
_IMG_SIZE = 64

#: Source image size ``(height, width)`` of every fixture image — not square, so the
#: letterbox carries a real pad rather than a degenerate one.
_IMAGE_SIZE = (40, 56)

#: Images in the built split, and the batch size that leaves the last batch short.
_NUM_IMAGES = 3
_BATCH_SIZE = 2

#: The single category a person-keypoints file names. COCO's own person id is 1, and the
#: protocol's whole argument for this annotation file is that the average is over one
#: category rather than eighty.
_PERSON_CATEGORY = 1


def _person_annotation(image_id: int, origin: float) -> dict[str, object]:
    """Build one 17-point person annotation whose joints march away from ``origin``.

    The point list is COCO's flat ``(x, y, v)`` triple-per-joint encoding, so its length is
    ``3 * 17``; every joint is labeled visible, which is what makes the instance count
    toward OKS at all. ``area`` and ``num_keypoints`` are stated rather than recomputed,
    because :func:`~lucid_yolo.eval.annotations.load_eval_annotations` hands the OKS scorer
    the file's own metadata (WP-134) and a fixture that omitted them would exercise a
    reconstruction this protocol does not perform.

    Args:
        image_id: Image the instance belongs to.
        origin: Top-left ``(x, y)`` the box and the joint ladder start from.

    Returns:
        One COCO annotation record.

    Examples:
        >>> record = _person_annotation(1, 4.0)
        >>> len(record["keypoints"]) == 3 * 17
        True
        >>> record["keypoints"][:3]
        [4.0, 4.0, 2]
        >>> record["category_id"], record["num_keypoints"]
        (1, 17)
    """
    joints: list[float | int] = []
    for index in range(_COCO_POINTS):
        joints.extend([origin + index, origin + index, 2])
    return {
        "image_id": image_id,
        "category_id": _PERSON_CATEGORY,
        "bbox": [origin, origin, 16.0, 16.0],
        "area": 256.0,
        "iscrowd": 0,
        "num_keypoints": _COCO_POINTS,
        "keypoints": joints,
    }


def _keypoint_module(num_keypoints: int) -> DetectionLitModule:
    """Return an eval-mode single-class keypoint module predicting ``num_keypoints`` points.

    Untrained on purpose: the numbers it produces are not the subject, the path they travel
    is. ``K`` is a constructor argument (A64), which is exactly what lets the boundary case
    below hand the protocol a schema it must refuse.

    Args:
        num_keypoints: Point count ``K`` both head branches predict.

    Returns:
        The module, in eval mode.

    Examples:
        >>> _keypoint_module(17).hparams["num_keypoints"]
        17
        >>> _keypoint_module(17).task
        'keypoints'
    """
    module = DetectionLitModule(
        depth=0.34,
        width=0.25,
        max_channels=64,
        num_classes=1,
        task="keypoints",
        num_keypoints=num_keypoints,
    )
    return module.eval()


@pytest.fixture
def pose_root(tmp_path: Path) -> Path:
    """Build the ``val2017/`` plus ``annotations/person_keypoints_val2017.json`` layout.

    Three images, the middle one carrying two instances and the last one none — the
    majority shape of a real person-keypoints split, where most images hold no person and
    the evaluator must still see them.
    """
    root = tmp_path / "coco2017"
    images_dir = root / "val2017"
    annotations_dir = root / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    height, width = _IMAGE_SIZE
    images = []
    annotations = []
    for image_id in range(1, _NUM_IMAGES + 1):
        name = f"{image_id:012d}.png"
        write_png(torch.full((3, height, width), 128, dtype=torch.uint8), str(images_dir / name))
        images.append({"id": image_id, "file_name": name, "height": height, "width": width})
    annotations.append(_person_annotation(1, 4.0))
    annotations.extend([_person_annotation(2, 2.0), _person_annotation(2, 12.0)])

    payload = {"categories": [{"id": _PERSON_CATEGORY}], "images": images, "annotations": annotations}
    (annotations_dir / "person_keypoints_val2017.json").write_text(json.dumps(payload))
    return root


class TestPoseReport:
    """The scoring body: annotation load, batching, OKS accumulation and the written report."""

    def test_the_split_is_scored_and_the_report_names_both_paths(
        self, pose_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A pose evaluation writes both paths' box and OKS figures, into a directory it creates.

        This is the whole span an ``>>> callable(run)`` Example left unexecuted: the
        ``person_keypoints_val2017.json`` name (not ``instances_``, the substitution that
        divides a person detector's AP by eighty), the letterboxed batching, the
        :class:`~lucid_yolo.eval.coco_eval.DualPathEvaluator` built with COCO's sigma
        table, and the report's own keys. The ten keypoint statistics carry the ``oks_``
        prefix and the twelve box ones keep their bare names, so a reader can tell which
        column is which; both paths appear, because the ``e2e`` row is the one a training
        run's ``val/mAP`` is comparable to and quoting a single unlabeled number is the
        confusion the report exists to end.

        ``--output`` points into a directory that does not exist yet, the WP-105 failure:
        an evaluation that completes, prints its numbers, and then dies writing them turns
        minutes of work into a traceback.
        """
        output = tmp_path / "absent" / "nested" / "pose.json"

        code = pose_eval.run(
            _keypoint_module(_COCO_POINTS),
            {"ema": False},
            data_root=pose_root,
            img_size=_IMG_SIZE,
            batch_size=_BATCH_SIZE,
            device_name="cpu",
            limit=0,
            output=output,
        )

        assert code == 0
        payload = json.loads(output.read_text())
        assert payload["info"]["keypoints"] == _COCO_POINTS
        assert payload["info"]["eval_backend"] == "faster_coco_eval"
        assert payload["images"] == _NUM_IMAGES
        assert set(payload["report"]) == {"e2e", "nms"}
        for path in ("e2e", "nms"):
            stats = payload["report"][path]
            assert {"map", "map_50", "map_75"} <= set(stats)
            assert {"oks_AP_all", "oks_AP_50", "oks_AP_75"} <= set(stats)
            assert 0.0 <= stats["oks_AP_all"] <= 1.0
        printed = capsys.readouterr().out
        assert f"K={_COCO_POINTS}" in printed
        assert "the e2e row is the one comparable to a training run's own val/mAP" in printed

    def test_the_limit_scores_a_prefix_rather_than_the_whole_split(self, pose_root: Path, tmp_path: Path) -> None:
        """``--limit`` stops after N images, and the report counts what it actually scored.

        The count in the report is what a reader compares two runs by, so a limit that
        truncated the loop while the report still claimed the full split would make the
        two runs look like the same measurement.
        """
        output = tmp_path / "pose.json"

        code = pose_eval.run(
            _keypoint_module(_COCO_POINTS),
            {"ema": False},
            data_root=pose_root,
            img_size=_IMG_SIZE,
            batch_size=_BATCH_SIZE,
            device_name="cpu",
            limit=1,
            output=output,
        )

        assert code == 0
        assert json.loads(output.read_text())["images"] == 1

    def test_no_output_path_scores_without_writing_anything(self, pose_root: Path, tmp_path: Path) -> None:
        """``output=None`` still returns success and leaves no file behind.

        The report is optional — a caller reading the printed numbers off a terminal is the
        common case — so the write must be genuinely conditional rather than defaulting to
        a path nobody asked for.
        """
        code = pose_eval.run(
            _keypoint_module(_COCO_POINTS),
            {"ema": False},
            data_root=pose_root,
            img_size=_IMG_SIZE,
            batch_size=_BATCH_SIZE,
            device_name="cpu",
            limit=0,
            output=None,
        )

        assert code == 0
        assert list(tmp_path.glob("*.json")) == []


class TestPoseProtocolBoundary:
    """``K != 17`` is refused before any annotation file is opened, at the boundary too."""

    def test_a_sixteen_point_checkpoint_is_refused_at_the_boundary(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """``K=16`` exits non-zero naming both counts, scoring nothing against COCO's sigmas.

        :mod:`tests.eval.test_keypoint_eval` pins the same refusal at ``K=7``, a schema no
        off-by-one produces; sixteen is the value adjacent to the boundary, where a ``<=``
        or a table padded by one joint would let a foreign schema through and report a
        real-looking OKS for a correspondence that does not exist — point *i* of a
        sixteen-point prediction is not point *i* of the annotation. The refusal fires
        before the annotation file is resolved, which is why the data root here is a path
        that was never created.
        """
        code = pose_eval.run(
            _keypoint_module(_COCO_POINTS - 1),
            {},
            data_root=tmp_path / "absent",
            img_size=_IMG_SIZE,
            batch_size=_BATCH_SIZE,
            device_name="cpu",
            limit=0,
            output=None,
        )

        assert code == 1
        message = capsys.readouterr().out
        assert f"{_COCO_POINTS}-point person schema" in message
        assert f"K={_COCO_POINTS - 1}" in message
