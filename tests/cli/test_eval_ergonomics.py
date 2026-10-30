# SPDX-License-Identifier: Apache-2.0
"""Task-specific evaluation options, dataset aliases and readable CLI help."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from cli_checkpoint_helpers import kwargs_recorder, save_lightning_checkpoint

from lucid_yolo.cli import data as data_cli
from lucid_yolo.cli import eval as eval_cli
from lucid_yolo.data.tiles import build_tiles
from lucid_yolo.ptl.module import DetectionLitModule

#: The deprecation torch raises from ``torch.jit.script``, which fvcore triggers at import.
_JIT_DEPRECATION = "`torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`."


def _succeed(*_args: Any, **_kwargs: Any) -> int:
    """Stand in for an evaluation protocol that accepts any call and exits ``0``.

    Examples:
        >>> _succeed(None, {}, img_size=640)
        0
    """
    return 0


def _run_interpreter(code: str, *flags: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter with ``flags`` and capture its text output.

    Examples:
        >>> _run_interpreter("print('fresh')").stdout.strip()
        'fresh'
    """
    return subprocess.run([sys.executable, *flags, "-c", code], capture_output=True, text=True, check=False)


@pytest.fixture
def checkpoint(task: str, tmp_path: Path) -> Path:
    """Save a small real checkpoint so option checks follow checkpoint task discovery."""
    module = DetectionLitModule(
        depth=0.34,
        width=0.25,
        max_channels=64,
        num_classes=1,
        task=task,
        num_keypoints=17 if task == "keypoints" else None,
    )
    return save_lightning_checkpoint(module, tmp_path / "model.ckpt")


@pytest.mark.parametrize(
    ("task", "flag", "value"),
    [
        pytest.param(task, flag, value, id=f"{task}-{flag}")
        for task, options in {
            "detect": [("split", "val"), ("variant", "n"), ("masks", "true")],
            "segment": [("split", "val"), ("variant", "n")],
            "keypoints": [("split", "val"), ("variant", "n"), ("masks", "true")],
            "obb": [("masks", "true"), ("eval_backend", "auto"), ("ann_file", "custom.json"), ("images_dir", "custom")],
        }.items()
        for flag, value in options
    ],
)
@pytest.mark.parametrize("from_config", [False, True])
@patch.object(eval_cli.detect_eval, "run", _succeed)
@patch.object(eval_cli.pose_eval, "run", _succeed)
@patch.object(eval_cli.rotated_eval, "run", _succeed)
def test_eval_refuses_unused_explicit_option(
    task: str,
    flag: str,
    value: str,
    from_config: bool,
    checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Explicit defaults also fail, including options supplied through configuration files."""
    args = ["--checkpoint", str(checkpoint), "--data_root", str(tmp_path), "--ema", "false"]
    if from_config:
        config = tmp_path / "options.json"
        config.write_text(json.dumps({flag: True if value == "true" else value}))
        args += ["--config", str(config)]
    else:
        args += [f"--{flag}", value]
    with pytest.raises(ValueError, match=rf"{flag}.*{task}"):
        eval_cli.main(args)


@pytest.mark.parametrize("task", ["detect", "segment", "keypoints", "obb"])
@patch.object(eval_cli, "_resolve_eval_backend", lambda requested: ("faster_coco_eval", None))
def test_eval_omitted_options_use_protocol_defaults(
    task: str,
    checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unspecified options never reject a task and retain its effective defaults."""
    seen: dict[str, Any] = {}
    for protocol in (eval_cli.detect_eval, eval_cli.pose_eval, eval_cli.rotated_eval):
        monkeypatch.setattr(protocol, "run", kwargs_recorder(seen))
    assert eval_cli.main(["--checkpoint", str(checkpoint), "--data_root", str(tmp_path), "--ema", "false"]) == 0
    if task == "obb":
        assert seen["split"] == "val"
        assert seen["variant"] == "n"
    else:
        assert seen["ann_file"] is None
        assert seen["images_dir"] is None
        assert seen["eval_backend"] == "faster_coco_eval"
        if task in {"detect", "segment"}:
            assert seen["masks"] is True


@pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
def test_eval_forwards_coco_path_overrides(
    task: str,
    checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """COCO file and image overrides reach every COCO protocol unchanged."""
    seen: dict[str, Any] = {}
    monkeypatch.setattr(eval_cli.detect_eval, "run", kwargs_recorder(seen))
    monkeypatch.setattr(eval_cli.pose_eval, "run", kwargs_recorder(seen))
    annotation = tmp_path / "subset.json"
    images = tmp_path / "subset"
    assert (
        eval_cli.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--data_root",
                str(tmp_path),
                "--ema",
                "false",
                "--ann_file",
                str(annotation),
                "--images_dir",
                str(images),
                "--eval_backend",
                "faster_coco_eval",
            ]
        )
        == 0
    )
    assert seen["ann_file"] == annotation
    assert seen["images_dir"] == images


@pytest.mark.parametrize("flag", ["--root", "--data_root"])
def test_build_tiles_accepts_dataset_root_alias(flag: str, tmp_path: Path) -> None:
    """Both root spellings bind to the operation's existing root argument."""
    parsed = data_cli.build_parser().parse_args(["build-tiles", flag, str(tmp_path), "--out", str(tmp_path / "out")])
    assert parsed["build-tiles"].root == tmp_path


def test_build_tiles_help_contains_plain_text(capsys: pytest.CaptureFixture[str]) -> None:
    """User help contains no raw documentation roles."""
    with pytest.raises(SystemExit) as exc:
        data_cli.main(["build-tiles", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "Tile an oriented dataset" in help_text
    assert ":data:" not in help_text
    assert ":func:" not in help_text


def test_build_tiles_suffix_defaults_to_library_default(tmp_path: Path) -> None:
    """Omitting ``--suffix`` hands ``build_tiles`` the default its own signature declares.

    The flag is declared by hand to keep library-only markup out of the help, which makes its default the one value that
    could drift from the function's; this pins them equal.
    """
    parsed = data_cli.build_parser().parse_args(
        ["build-tiles", "--root", str(tmp_path), "--out", str(tmp_path / "out")]
    )
    assert parsed["build-tiles"].suffix == inspect.signature(build_tiles).parameters["suffix"].default


@pytest.mark.parametrize("module", ["eval", "data", "train", "predict"])
def test_cli_import_avoids_jit_deprecation(module: str) -> None:
    """CLI imports tolerate the known third-party JIT warning under warning-as-error."""
    result = _run_interpreter(f"import lucid_yolo.cli.{module}", "-W", "error::FutureWarning")
    assert result.returncode == 0, result.stderr
    assert "torch.jit.script` is deprecated" not in result.stderr


def test_cli_import_leaves_jit_deprecation_visible_to_user_code() -> None:
    """After ``import lucid_yolo.cli``, a ``torch.jit.script`` deprecation still reaches its caller.

    The CLI silences that warning only while its own imports run fvcore's import-time scripting. A filter left installed
    would also swallow the warning for every function a library user scripts later, so the user-side emission -- same
    message, category and module the filter matches -- must still raise under warning-as-error.
    """
    script = (
        "import warnings\n"
        "import lucid_yolo.cli\n"
        f"warnings.warn_explicit({_JIT_DEPRECATION!r}, FutureWarning, 'user.py', 1, module='torch.jit._script')\n"
    )
    result = _run_interpreter(script, "-W", "error::FutureWarning")
    assert result.returncode != 0
    assert f"FutureWarning: {_JIT_DEPRECATION}" in result.stderr


def test_eval_help_prints_no_future_warning() -> None:
    """``lucid-eval --help`` exits 0 with no FutureWarning on stderr, in a fresh interpreter."""
    result = _run_interpreter("from lucid_yolo.cli import eval_main; raise SystemExit(eval_main(['--help']))")
    assert result.returncode == 0, result.stderr
    assert "FutureWarning" not in result.stderr
