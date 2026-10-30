# SPDX-License-Identifier: Apache-2.0
"""Gate on :func:`~lucid_yolo.eval.coco_score.score_coco`, the Python door to the COCO protocol.

The command runners print their numbers and return an exit code; this function returns the numbers. What matters most
is that the two cannot disagree, so the first test scores one module both ways and asserts the report the runner writes
equals the report the function returns. The rest pin the refusals a Python caller reaches without a command line in
front of it: a task the protocol does not score, a keypoint schema it cannot score, and paths it cannot resolve.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from lucid_yolo.eval import detect_eval
from lucid_yolo.eval.coco_score import CocoScore, score_coco
from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from pathlib import Path

#: Category ids 1..4 in the detseg fixture.
_NUM_CLASSES = 4
#: Canvas side: a multiple of every stride, upscaling the 128 px fixture images.
_CANVAS = 160
#: Images scored per test; enough for a batch boundary, few enough to stay quick.
_LIMIT = 3
#: Images per forward pass, so ``_LIMIT`` spans two batches.
_BATCH_SIZE = 2


def _module(task: str, **kwargs: object) -> DetectionLitModule:
    """Build an untrained n-scale eval-mode module for ``task``.

    Examples:
        >>> _module("keypoints", num_keypoints=17).task
        'keypoints'
    """
    return DetectionLitModule(
        depth=0.34, width=0.25, max_channels=1024, num_classes=_NUM_CLASSES, task=task, **kwargs
    ).eval()


@pytest.fixture
def coco_root(detseg_fixture_dir: Path, tmp_path: Path) -> Path:
    """Lay the detseg fixture's train split out under the ``val2017`` layout."""
    root = tmp_path / "coco"
    (root / "annotations").mkdir(parents=True)
    split_dir = detseg_fixture_dir / "train"
    (root / "annotations" / "instances_val2017.json").write_text((split_dir / "_annotations.coco.json").read_text())
    (root / "val2017").symlink_to(split_dir, target_is_directory=True)
    return root


def test_score_equals_report_runner_writes(coco_root: Path, tmp_path: Path) -> None:
    """One module scored by the function and by ``lucid-eval``'s runner yields one report.

    The runner's JSON file is its only durable output, so it is the comparison target: equal reports mean a notebook
    caller and an operator read the same numbers.
    """
    module = _module("detect")
    output = tmp_path / "report.json"
    detect_eval.run(
        module,
        {"ema": False},
        data_root=coco_root,
        img_size=_CANVAS,
        batch_size=_BATCH_SIZE,
        masks=True,
        device_name="cpu",
        limit=_LIMIT,
        output=output,
    )

    score = score_coco(module, coco_root, img_size=_CANVAS, batch_size=_BATCH_SIZE, device="cpu", limit=_LIMIT)

    assert isinstance(score, CocoScore)
    assert score.images == _LIMIT
    assert score.report == json.loads(output.read_text())["report"]


@pytest.mark.parametrize(("masks", "has_segm"), [(True, True), (False, False)])
def test_segmentation_checkpoint_scores_masks_unless_declined(coco_root: Path, masks: bool, has_segm: bool) -> None:
    """A ``segment`` module reports ``segm_`` statistics unless ``masks=False`` declines them."""
    score = score_coco(
        _module("segment"), coco_root, img_size=_CANVAS, batch_size=_BATCH_SIZE, device="cpu", limit=1, masks=masks
    )

    assert ("segm_map" in score.report["e2e"]) is has_segm
    assert ("segm_map" in score.report["nms"]) is has_segm


@pytest.mark.parametrize(
    ("task", "module_kwargs", "message"),
    [
        pytest.param("obb", {}, "task must be one of", id="oriented-task"),
        pytest.param("keypoints", {"num_keypoints": 5}, "checkpoint has K=5", id="non-coco-keypoint-schema"),
    ],
)
def test_unscorable_checkpoint_refused(
    coco_root: Path, task: str, module_kwargs: dict[str, object], message: str
) -> None:
    """A module the protocol cannot score is refused by name instead of being scored as something else.

    An ``obb`` module is not scored as an axis-aligned detector, and a keypoint module whose ``K`` is not COCO's 17 is
    refused before any file is read.
    """
    with pytest.raises(ValueError, match=message):
        score_coco(_module(task, **module_kwargs), coco_root, img_size=_CANVAS, device="cpu")


def test_overrides_without_data_root_need_both(coco_root: Path) -> None:
    """Without ``data_root`` both paths are required; with both, no root is consulted."""
    with pytest.raises(ValueError, match="data_root is required"):
        score_coco(_module("detect"), ann_file=coco_root / "annotations" / "instances_val2017.json")

    score = score_coco(
        _module("detect"),
        ann_file=coco_root / "annotations" / "instances_val2017.json",
        images_dir=coco_root / "val2017",
        img_size=_CANVAS,
        device="cpu",
        limit=1,
    )

    assert score.images == 1


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("limit", -1, "limit must be a number >= 0"),
        ("batch_size", 0, "batch_size must be a number >= 1"),
        ("eval_backend", "pycocotools", "eval_backend must be one of"),
    ],
)
def test_out_of_domain_option_refused(coco_root: Path, option: str, value: object, message: str) -> None:
    """An option outside its domain is refused by name before the ground truth is read."""
    with pytest.raises(ValueError, match=message):
        score_coco(_module("detect"), coco_root, **{option: value})
