# SPDX-License-Identifier: Apache-2.0
"""``lucid-export``: a checkpoint in, one static ONNX file out, per task.

The command composes nothing of its own — :func:`~lucid_yolo.export.graph_from_module`
builds the graph and :func:`~lucid_yolo.export.export_graph` writes it, and
``tests/models/test_onnx_export.py`` gates both on op content and numeric parity. What is
tested here is the command around them: that it parses, that it reads the task from the
checkpoint for every task, that the file it writes is the static, pinned-opset graph the
summary describes, and that onnxruntime answers with the scores the checkpoint answers.

The checkpoints are small and untrained, at multipliers no registry row has, so they
also show the command exports a module the training CLI did not configure. Their
BatchNorm statistics are calibrated by a few forward passes before saving: a freshly
built model emits near-identical scores at every anchor, and a score comparison between
two backends would then pass on an export that never loaded the checkpoint's weights.
Scores are compared sorted, because tied rows may be ordered differently by the two
backends without either being wrong.
"""

from __future__ import annotations

import importlib.util
import os
import re
from typing import TYPE_CHECKING

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from lucid_yolo.cli import export as export_cli
from lucid_yolo.eval.checkpoint import load_eval_module
from lucid_yolo.export import ONNX_OPSET, graph_from_module
from lucid_yolo.ptl.module import DetectionLitModule

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

#: Export side. 128 px yields 336 anchors, above the 300-row cap the segmentation graph
#: requires, and keeps the export fast on CPU.
_SIDE = 128

#: Point count given to a ``keypoints`` checkpoint; deliberately not COCO's 17.
_NUM_KEYPOINTS = 3

_NUM_CLASSES = 2

#: Score agreement between eager PyTorch and onnxruntime, measured well below this on
#: the parity gate's own fixture.
_SCORE_ATOL = 1e-5

#: Column of the score in each task's detection rows.
_SCORE_COLUMN = {"detect": 4, "segment": 4, "obb": 5, "keypoints": 4}


def _write_checkpoint(task: str, path: Path) -> Path:
    """Save a small calibrated checkpoint of ``task`` that :func:`load_eval_module` reads.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     _write_checkpoint("detect", Path(tmp) / "detect.ckpt").is_file()
        True
    """
    points = _NUM_KEYPOINTS if task == "keypoints" else None
    module = DetectionLitModule(
        depth=0.34, width=0.25, max_channels=64, num_classes=_NUM_CLASSES, task=task, num_keypoints=points
    )
    module.train()
    with torch.no_grad():
        for _ in range(3):
            module(torch.rand(2, 3, _SIDE, _SIDE))
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


def _static_dims(value: onnx.ValueInfoProto) -> tuple[int, ...]:
    """Return a graph value's dimensions, ``-1`` standing for a symbolic one.

    Examples:
        >>> from onnx import TensorProto, helper
        >>> _static_dims(helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "batch"]))
        (1, -1)
    """
    return tuple(dim.dim_value if dim.HasField("dim_value") else -1 for dim in value.type.tensor_type.shape.dim)


def test_help_lists_every_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """``--help`` exits zero and lists the six flags the command documents.

    The help text is generated from the function signature, so a renamed parameter would
    silently rename its flag; this pins the spellings the docs give.
    """
    with pytest.raises(SystemExit) as stopped:
        export_cli.main(["--help"])

    assert stopped.value.code == 0
    usage = capsys.readouterr().out
    for flag in ("--checkpoint", "--output", "--canvas", "--img_size", "--ema", "--opset"):
        assert flag in usage


class TestEndToEnd:
    """One export per task, from a checkpoint on disk through the command's own ``main``."""

    @pytest.mark.parametrize("task", ["detect", "segment", "obb", "keypoints"])
    def test_writes_static_graph_at_pinned_opset(
        self, task: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The written file takes one static ``(1, 3, H, W)`` image and declares the pinned opset.

        The output is written into a directory that does not exist yet, which the command
        creates; the summary names the task it read from the checkpoint.
        """
        checkpoint = _write_checkpoint(task, tmp_path / f"{task}.ckpt")
        output = tmp_path / "out" / f"{task}.onnx"

        code = export_cli.main(
            ["--checkpoint", str(checkpoint), "--output", str(output), "--img_size", str(_SIDE), "--ema", "false"]
        )

        assert code == 0
        model = onnx.load(str(output))
        onnx.checker.check_model(model)
        assert [_static_dims(value) for value in model.graph.input] == [(1, 3, _SIDE, _SIDE)]
        assert {imported.version for imported in model.opset_import if imported.domain in {"", "ai.onnx"}} == {
            ONNX_OPSET
        }
        assert f"task={task}" in capsys.readouterr().out

    @pytest.mark.parametrize("task", ["detect", "segment", "obb", "keypoints"])
    def test_onnxruntime_scores_match_checkpoint(self, task: str, tmp_path: Path) -> None:
        """Onnxruntime answers the exported file with the scores the checkpoint itself gives.

        The reference is the checkpoint loaded afresh and decoded eagerly through the same
        bridge, so a file exported from fresh weights, or from the wrong task's graph,
        disagrees here.
        """
        checkpoint = _write_checkpoint(task, tmp_path / f"{task}.ckpt")
        output = tmp_path / f"{task}.onnx"
        export_cli.export(checkpoint, output, img_size=_SIDE, ema=False)
        module, _ = load_eval_module(checkpoint, use_ema=False)
        image = torch.rand(1, 3, _SIDE, _SIDE, generator=torch.Generator().manual_seed(7))

        with torch.no_grad():
            eager = graph_from_module(module, (_SIDE, _SIDE))(image)
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        produced = session.run(None, {session.get_inputs()[0].name: image.numpy()})

        reference = (eager[0] if isinstance(eager, tuple) else eager).numpy()[0, :, _SCORE_COLUMN[task]]
        assert np.ptp(reference) > 0, "every score is tied; the comparison would pass on any weights"
        np.testing.assert_allclose(
            np.sort(produced[0][0, :, _SCORE_COLUMN[task]]), np.sort(reference), atol=_SCORE_ATOL
        )

    def test_canvas_flag_sets_non_square_input(self, tmp_path: Path) -> None:
        """``--canvas "[H, W]"`` exports a non-square graph, height first."""
        checkpoint = _write_checkpoint("detect", tmp_path / "detect.ckpt")
        output = tmp_path / "detect.onnx"

        export_cli.main(
            ["--checkpoint", str(checkpoint), "--output", str(output), "--canvas", "[64, 128]", "--ema", "false"]
        )

        assert _static_dims(onnx.load(str(output)).graph.input[0]) == (1, 3, 64, 128)


class TestRefusals:
    """What the command refuses, and that each refusal comes before the checkpoint is read."""

    def test_canvas_and_img_size_together_refused(self, tmp_path: Path) -> None:
        """The two size flags are exclusive; the refusal names both values.

        The checkpoint path does not exist, so a refusal raised after reading it would be
        a ``FileNotFoundError`` instead.
        """
        with pytest.raises(ValueError, match=r"not both; got canvas=\[64, 64\] and img_size=128"):
            export_cli.export(tmp_path / "absent.ckpt", tmp_path / "out.onnx", canvas=(64, 64), img_size=128)

    def test_missing_onnx_refused_naming_extra(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the ``onnx`` package the command stops at once, naming the extra that installs it.

        torch's exporter only needs the package to finalise the file, so left to itself it
        would fail after the trace rather than before the checkpoint is read.
        """
        real_find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util, "find_spec", lambda name, *rest: None if name == "onnx" else real_find_spec(name, *rest)
        )

        with pytest.raises(ModuleNotFoundError, match=r'pip install "lucid-yolo\[export\]"'):
            export_cli.export(tmp_path / "absent.ckpt", tmp_path / "out.onnx")

    def test_unknown_task_refused_by_name(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A checkpoint whose task has no graph is refused, naming the task and the four that have one."""
        stand_in = DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=_NUM_CLASSES)
        monkeypatch.setattr(stand_in, "_task", "panoptic")
        monkeypatch.setattr(export_cli, "load_eval_module", lambda *_args, **_kwargs: (stand_in, {}))

        with pytest.raises(
            ValueError, match=r"task must be one of \('obb', 'detect', 'segment', 'keypoints'\); got 'panoptic'"
        ):
            export_cli.export(tmp_path / "unknown.ckpt", tmp_path / "out.onnx")

    @pytest.mark.parametrize("spelling", ["detect.ckpt", "runs/../detect.ckpt"])
    def test_output_naming_checkpoint_refused(self, spelling: str, tmp_path: Path) -> None:
        """An ``--output`` that is the checkpoint's own path is refused, the checkpoint left byte for byte.

        The exporter truncates its destination, so without the refusal a real checkpoint
        exported onto itself comes back as an ONNX file and the weights are gone. The
        second spelling reaches the same file through ``..``, which only a resolved-path
        comparison sees.
        """
        checkpoint = _write_checkpoint("detect", tmp_path / "detect.ckpt")
        original = checkpoint.read_bytes()
        output = tmp_path / spelling

        with pytest.raises(ValueError, match=re.escape(f"--output {output} is the checkpoint {checkpoint} itself")):
            export_cli.export(checkpoint, output)

        assert checkpoint.read_bytes() == original

    @pytest.mark.parametrize("link", [pytest.param(os.symlink, id="symlink"), pytest.param(os.link, id="hardlink")])
    def test_output_linked_to_checkpoint_refused(self, link: Callable[[Path, Path], None], tmp_path: Path) -> None:
        """An ``--output`` linked to the checkpoint is refused, the checkpoint left byte for byte.

        Writing through a symbolic link truncates its target, and a hard link is the
        checkpoint's own file under a second name, so either alias would replace the
        weights with ONNX bytes. A hard link resolves to a path of its own, which is why
        the refusal also asks the filesystem whether the two are one file.
        """
        checkpoint = _write_checkpoint("detect", tmp_path / "detect.ckpt")
        original = checkpoint.read_bytes()
        alias = tmp_path / "alias.onnx"
        link(checkpoint, alias)

        with pytest.raises(ValueError, match=re.escape(f"--output {alias} is the checkpoint {checkpoint} itself")):
            export_cli.export(checkpoint, alias)

        assert checkpoint.read_bytes() == original
