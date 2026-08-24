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
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch

from lucid_yolo.cli import data as data_cli
from lucid_yolo.cli import eval as eval_cli
from lucid_yolo.data import download
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
