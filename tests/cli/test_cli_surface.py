# SPDX-License-Identifier: Apache-2.0
"""Unit gate on the installed command surface (WP-096).

Four console scripts are declared in ``pyproject.toml`` and none of them is exercised by
importing the library: a rename or a move breaks the *installed* command while every
other test keeps passing, which is how ``scripts/`` tooling drifted out of reach of a
wheel in the first place. These tests resolve each declared target the way a console
script does and drive the two new commands end to end.

``lucid-eval``'s dispatch is asserted on both branches with the protocol functions
stubbed: what is under test is that the checkpoint's own task chooses the protocol and
supplies the per-task defaults, not the evaluation itself, which its own suites cover.

The refusal group (WP-171) is the other half of that surface: what a command does with a
value its annotation admits and its arithmetic cannot use. Each case asserts the message
names the flag the caller typed, because the flag is the only part of the refusal they
can act on -- a ``ValueError`` reading "invalid value" leaves a caller re-reading their
own command line. The values are the ones that used to answer *plausibly* rather than
fail: ``--limit -5`` scored every image but the last five, ``--batch_size 0`` reached a
``ZeroDivisionError`` three frames down, ``--img_size 641`` crashed inside the neck's
concatenation, a mis-spelled ``--decoder`` returned the other branch's boxes, a negative
``--conf_threshold`` admitted the decoders' score-zero padding rows as detections, and an
unknown task was scored as detection.
"""

from __future__ import annotations

import importlib
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import torch

from lucid_yolo.assign.grid import HEAD_STRIDES
from lucid_yolo.cli import data as data_cli
from lucid_yolo.cli import eval as eval_cli
from lucid_yolo.cli import predict as predict_cli
from lucid_yolo.data import download
from lucid_yolo.predict import KeypointPrediction, SegmentedPrediction
from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The commands a wheel installs, and what each must resolve to.
EXPECTED_SCRIPTS = {
    "lucid-yolo": "lucid_yolo.cli.train:main",
    "lucid-data": "lucid_yolo.cli.data:main",
    "lucid-eval": "lucid_yolo.cli.eval:main",
    "lucid-predict": "lucid_yolo.cli.predict:main",
}


def _declared_scripts() -> dict[str, str]:
    """Read ``[project.scripts]`` from ``pyproject.toml``.

    Examples:
        >>> _declared_scripts() == EXPECTED_SCRIPTS
        True
    """
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts: dict[str, str] = payload["project"]["scripts"]
    return scripts


#: Point count given to a ``keypoints`` checkpoint. ``num_keypoints`` has no default for
#: that task -- ``K`` belongs to whichever schema supplies the points -- so the stub must
#: name one, and COCO's person schema is the one ``pose_eval`` scores against.
_COCO_PERSON_POINTS = 17


def _write_checkpoint(task: str, path: Path) -> Path:
    """Save a tiny checkpoint of ``task`` that :func:`load_eval_module` can read.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     out = _write_checkpoint("detect", Path(tmp) / "ckpt.pt")
        ...     out.is_file()
        True
    """
    points = _COCO_PERSON_POINTS if task == "keypoints" else None
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=2, task=task, num_keypoints=points)
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "epoch": 0,
            "global_step": 0,
            "pytorch-lightning_version": "2.4.0",
            "loops": {},
            "callbacks": {},
            "optimizer_states": [],
            "lr_schedulers": [],
        },
        path,
    )
    return path


def test_every_declared_console_script_resolves() -> None:
    """Each ``[project.scripts]`` target imports and is callable, as the wrapper does."""
    declared = _declared_scripts()

    assert declared == EXPECTED_SCRIPTS
    for target in declared.values():
        module_name, _, attribute = target.partition(":")
        assert callable(getattr(importlib.import_module(module_name), attribute))


def test_the_data_command_offers_the_three_dataset_operations() -> None:
    """``lucid-data`` covers download, check and build-tiles, and nothing task-specific."""
    assert list(data_cli.SUBCOMMANDS) == ["download", "check", "build-tiles"]


def test_the_check_subcommand_reports_a_missing_root(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``lucid-data check`` exits non-zero and names the root it could not validate."""
    code = data_cli.main(["check", "--data_root", str(tmp_path / "absent")])

    assert code == 1
    assert "absent" in capsys.readouterr().out


def test_build_tiles_rejects_an_unknown_source(tmp_path: Path) -> None:
    """A reader that does not exist fails loudly rather than tiling nothing."""
    with pytest.raises(ValueError, match="unknown source"):
        data_cli.main(["build-tiles", "--root", str(tmp_path), "--out", str(tmp_path / "out"), "--source", "kitti"])


@pytest.mark.parametrize(
    ("task", "expected_module", "expected_img_size", "expected_batch_size"),
    [
        pytest.param("detect", "detect_eval", 640, 32, id="detect"),
        pytest.param("obb", "rotated_eval", 1024, 8, id="obb"),
        pytest.param("keypoints", "pose_eval", 640, 32, id="keypoints"),
    ],
)
def test_eval_dispatches_on_the_checkpoints_own_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    expected_module: str,
    expected_img_size: int,
    expected_batch_size: int,
) -> None:
    """The protocol and its defaults come from the checkpoint, with no task flag anywhere.

    Both protocol entry points are stubbed, so a wrong dispatch shows up as the other
    stub recording the call rather than as a numerically odd report.
    """
    calls: dict[str, dict[str, Any]] = {}

    def _record(name: str) -> Callable[..., int]:
        def stub(_module: object, _info: dict[str, object], **kwargs: Any) -> int:
            calls[name] = kwargs
            return 0

        return stub

    monkeypatch.setattr(eval_cli.detect_eval, "run", _record("detect_eval"))
    monkeypatch.setattr(eval_cli.rotated_eval, "run", _record("rotated_eval"))
    monkeypatch.setattr(eval_cli.pose_eval, "run", _record("pose_eval"))
    checkpoint = _write_checkpoint(task, tmp_path / f"{task}.ckpt")

    code = eval_cli.main(["--checkpoint", str(checkpoint), "--data_root", str(tmp_path), "--ema", "false"])

    assert code == 0
    assert list(calls) == [expected_module]
    assert calls[expected_module]["img_size"] == expected_img_size
    assert calls[expected_module]["batch_size"] == expected_batch_size


def test_an_explicit_size_beats_the_task_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller who names ``--img_size`` gets it, oriented default or not."""
    seen: dict[str, Any] = {}

    def stub(_module: object, _info: dict[str, object], **kwargs: Any) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(eval_cli.rotated_eval, "run", stub)
    checkpoint = _write_checkpoint("obb", tmp_path / "obb.ckpt")

    eval_cli.main(
        ["--checkpoint", str(checkpoint), "--data_root", str(tmp_path), "--ema", "false", "--img_size", "512"]
    )

    assert seen["img_size"] == 512


@pytest.mark.parametrize(
    ("task", "expected_entry_point"),
    [
        pytest.param("detect", "predict_image", id="detect"),
        pytest.param("segment", "predict_segmentation", id="segment"),
        pytest.param("obb", "predict_oriented", id="obb"),
        pytest.param("keypoints", "predict_keypoints", id="keypoints"),
    ],
)
def test_predict_dispatches_on_the_checkpoints_own_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task: str,
    expected_entry_point: str,
) -> None:
    """Each task reaches its own inference entry point, with no task flag anywhere.

    All four entry points are stubbed, so a wrong dispatch shows up as another stub
    recording the call. The keypoint case is why this test exists: until WP-152 the
    command had three branches and a ``keypoints`` checkpoint silently fell through to
    the detection one, answering with boxes and never consulting the point stem -- a
    plausible answer, no error, and the pose the caller asked for absent.
    """
    called: list[str] = []

    def _record(name: str, result: object) -> Callable[..., object]:
        def stub(*_args: Any, **_kwargs: Any) -> object:
            called.append(name)
            return result

        return stub

    empty_detections = torch.zeros(0, 6)
    monkeypatch.setattr(predict_cli, "predict_image", _record("predict_image", empty_detections))
    monkeypatch.setattr(
        predict_cli,
        "predict_segmentation",
        _record("predict_segmentation", SegmentedPrediction(empty_detections, torch.zeros(0, 4, 4, dtype=torch.bool))),
    )
    monkeypatch.setattr(predict_cli, "predict_oriented", _record("predict_oriented", torch.zeros(0, 7)))
    monkeypatch.setattr(
        predict_cli,
        "predict_keypoints",
        _record("predict_keypoints", KeypointPrediction(empty_detections, torch.zeros(0, 17, 2))),
    )
    checkpoint = _write_checkpoint(task, tmp_path / f"{task}.ckpt")

    code = predict_cli.predict(checkpoint=checkpoint, image=tmp_path / "image.jpg", ema=False)

    assert code == 0
    assert called == [expected_entry_point]


def test_a_keypoint_report_names_its_point_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A keypoint report carries the point layout and ``K``; the other reports are untouched.

    ``K`` is a property of the checkpoint's schema (A64), so a report that omitted it
    would leave a reader unable to tell a 17-point human pose from a 15-point letter one
    without re-loading the checkpoint that wrote it.
    """
    points = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    detections = torch.tensor([[0.0, 0.0, 8.0, 8.0, 0.9, 1.0]])

    def stub(*_args: Any, **_kwargs: Any) -> KeypointPrediction:
        return KeypointPrediction(detections, points)

    monkeypatch.setattr(predict_cli, "predict_keypoints", stub)
    checkpoint = _write_checkpoint("keypoints", tmp_path / "keypoints.ckpt")
    report = tmp_path / "report.json"

    predict_cli.predict(checkpoint=checkpoint, image=tmp_path / "image.jpg", output=report, ema=False)

    payload = json.loads(report.read_text())
    assert payload["keypoints"] == "xy-pairs"
    assert payload["info"]["num_keypoints"] == 3
    assert payload["detections"][0]["keypoints"] == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def test_the_deprecated_download_alias_is_gone() -> None:
    """``lucid-download`` is no longer declared, and the entry point it named is gone too.

    0.3.0 deprecated the alias in favour of ``lucid-data download`` and put the removal in
    writing for 0.4.0 (WP-096), which WP-110 carried out. This asserts the absence rather
    than merely dropping the old test: a removal nothing asserts is one a later copy-paste
    can silently undo, and the entry point is the half that would come back unnoticed —
    a stale ``[project.scripts]`` line at least fails to resolve.
    """
    assert "lucid-download" not in _declared_scripts()
    assert not hasattr(download, "main")
    assert "main" not in download.__all__


@pytest.mark.parametrize(
    ("argument", "flag"),
    [
        pytest.param("--limit=-5", "limit", id="negative-limit"),
        pytest.param("--batch_size=0", "batch_size", id="zero-batch-size"),
        pytest.param("--img_size=641", "img_size", id="canvas-off-the-stride"),
    ],
)
def test_eval_refuses_an_argument_outside_its_domain(tmp_path: Path, argument: str, flag: str) -> None:
    """``lucid-eval`` refuses each unusable flag by name, before it opens the checkpoint.

    The checkpoint path does not exist and that is the assertion's other half: the
    refusal has to come from the flag rather than from the first thing downstream to
    trip over it, or the caller reads a ``FileNotFoundError`` and never learns which of
    their flags was the problem. Each value here used to be accepted: ``-5`` is truthy,
    so ``images[:-5]`` scored every image *but* the last five while the banner printed
    the truncated count as the request; ``0`` reached ``math.ceil(len(images) / 0)``; and
    ``641`` reached the neck, which concatenates an upsampled P5 of width 42 with a P4 of
    width 41.
    """
    with pytest.raises(ValueError, match=flag):
        eval_cli.main(["--checkpoint", str(tmp_path / "absent.ckpt"), "--data_root", str(tmp_path), argument])


def test_eval_refuses_a_task_it_has_no_protocol_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkpoint task this command implements nothing for is named, not scored as detection.

    The dispatch branches for ``obb`` and ``keypoints`` and ends in the detection
    protocol, so every other task value fell through and was *scored* — a report carrying
    detection's numbers under the checkpoint's own task name, which nothing in the file
    contradicts. The module is stubbed rather than saved, because
    :class:`~lucid_yolo.ptl.module.DetectionLitModule` refuses an unknown task at
    construction: the gap was only ever reachable through a checkpoint this repository
    cannot write, which is exactly why nothing caught it.
    """
    monkeypatch.setattr(eval_cli, "load_eval_module", lambda *_args, **_kwargs: (SimpleNamespace(task="panoptic"), {}))

    with pytest.raises(ValueError, match="task") as refusal:
        eval_cli.main(["--checkpoint", str(tmp_path / "absent.ckpt"), "--data_root", str(tmp_path)])

    assert "panoptic" in str(refusal.value)


@pytest.mark.parametrize(
    ("overrides", "flag"),
    [
        pytest.param({"decoder": "E2E"}, "decoder", id="decoder-spelling"),
        pytest.param({"decoder": "nms "}, "decoder", id="decoder-trailing-space"),
        pytest.param({"conf_threshold": -0.1}, "conf_threshold", id="negative-threshold"),
        pytest.param({"conf_threshold": 1.5}, "conf_threshold", id="threshold-above-one"),
        pytest.param({"img_size": 641}, "img_size", id="canvas-off-the-stride"),
    ],
)
def test_predict_refuses_an_argument_outside_its_domain(tmp_path: Path, overrides: dict[str, Any], flag: str) -> None:
    """``lucid-predict`` refuses each unusable flag by name, before it reads the image.

    The refusal belongs to the four entry points of :mod:`lucid_yolo.predict` rather than
    to this command — the same ownership the task refusal already has, so a direct
    library call and a command line cannot disagree about which values are usable — and
    this asserts it arrives through the command with the flag's own spelling in it.
    ``"E2E"`` and ``"nms "`` are the cases worth naming: both used to select the
    one-to-many branch by falling through ``decoder == "e2e"``, answering with another
    path's boxes and, for an oriented checkpoint, another path's headings.
    """
    checkpoint = _write_checkpoint("detect", tmp_path / "detect.ckpt")

    with pytest.raises(ValueError, match=flag):
        predict_cli.predict(checkpoint=checkpoint, image=tmp_path / "image.jpg", ema=False, **overrides)


def test_the_per_task_defaults_are_themselves_usable_values() -> None:
    """Both default tables cover the same tasks, and every value in them would be accepted.

    The task guard reads its vocabulary off :data:`~lucid_yolo.cli.eval.DEFAULT_IMG_SIZE`,
    and the resolved defaults skip the flag checks by construction — a caller who names
    nothing is trusting these two tables. That makes the tables the one place an unusable
    value could still reach a protocol, so they are pinned here rather than checked at
    every call: a task added to one table alone, or a side the strides do not divide,
    fails in this suite instead of in a queued run.
    """
    assert eval_cli.DEFAULT_IMG_SIZE.keys() == eval_cli.DEFAULT_BATCH_SIZE.keys()
    assert {side % max(HEAD_STRIDES) for side in eval_cli.DEFAULT_IMG_SIZE.values()} == {0}
    assert min(eval_cli.DEFAULT_BATCH_SIZE.values()) >= 1
