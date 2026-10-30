# SPDX-License-Identifier: Apache-2.0
"""The task registry: its four members, their defaults, and the string boundary around them.

:mod:`lucid_yolo.tasks` replaced a set of hand-written copies of the task names and two per-task default tables. The
tests here pin three things: that the registry holds exactly the values those copies held, so the replacement moved
nothing; that the tables the commands still publish are now read off it with their old contents and key order; and that
the enum never crosses the checkpoint boundary, because a saved checkpoint is read under
``torch.load(weights_only=True)`` and that unpickler refuses a class it was not told about.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest
import torch

import lucid_yolo
from lucid_yolo._viz.overlay import DRAWABLE_TASKS
from lucid_yolo.cli.eval import DEFAULT_BATCH_SIZE, DEFAULT_IMG_SIZE
from lucid_yolo.decode.common import COCO_IMG_SIZE
from lucid_yolo.predict import DEFAULT_ORIENTED_IMG_SIZE
from lucid_yolo.ptl import module as module_lib
from lucid_yolo.ptl.module import DetectionLitModule
from lucid_yolo.tasks import TASK_NAMES, Task

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

#: Class count of the throwaway module each checkpoint test saves.
_NUM_CLASSES = 3

#: Point count a ``keypoints`` module needs; any positive value builds the stem.
_NUM_KEYPOINTS = 3


def _write_checkpoint(module: DetectionLitModule, path: Path) -> Path:
    """Save ``module`` as a Lightning-shaped checkpoint at ``path`` and return the path.

    The dict is the one ``tests/predict/planted.py`` writes — what :func:`~lucid_yolo.eval.checkpoint.load_eval_module`
    reads — built here rather than imported across test packages. ``hyper_parameters`` is ``dict(module.hparams)``,
    which is exactly what Lightning's own checkpoint writer stores, so whatever type the constructor recorded is the
    type the file carries.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> module = DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=_NUM_CLASSES)
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = _write_checkpoint(module, Path(tmp) / "detect.ckpt")
        ...     torch.load(path, map_location="cpu", weights_only=True)["hyper_parameters"]["task"]
        'detect'
    """
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


@pytest.fixture
def save_checkpoint(tmp_path: Path) -> Callable[[str], Path]:
    """Return a factory saving a tiny untrained module of a given task as a Lightning-shaped checkpoint."""

    def save(task: str) -> Path:
        num_keypoints = _NUM_KEYPOINTS if task == Task.KEYPOINTS else None
        module = DetectionLitModule(
            depth=0.34, width=0.25, max_channels=64, num_classes=_NUM_CLASSES, task=task, num_keypoints=num_keypoints
        )
        return _write_checkpoint(module, tmp_path / f"{task}.ckpt")

    return save


class TestTask:
    """The closed set itself, and how a member meets the plain spelling."""

    def test_members_are_four_tasks_in_registry_order(self) -> None:
        """The registry holds exactly the four names the module accepted, in its order.

        A fifth member fails here on purpose. Adding a task is not one row: ``lucid-eval`` accepts every registry member
        and its dispatch falls through to ``lucid_yolo.eval.detect_eval.run``, which has no task guard of its own, so a
        fifth task must arrive with its own protocol there, with its own branch in ``lucid-predict``, a renderer in
        ``lucid_yolo._viz.overlay`` and an entry point in ``lucid_yolo.predict`` — this pin is what makes that a
        deliberate change rather than a silent one.
        """
        assert TASK_NAMES == ("detect", "segment", "obb", "keypoints")

    def test_member_equals_its_plain_spelling(self) -> None:
        """A member compares equal to, hashes like and formats as the string a config carries.

        The three properties are what let the string stay at every boundary: a YAML value compares against a member,
        looks up a member-keyed table, and an f-string message prints the spelling the caller typed.
        """
        table = {Task.OBB: "rotated"}

        assert Task("obb") is Task.OBB
        assert Task.OBB == "obb"
        assert table["obb"] == "rotated"
        assert f"{Task.OBB}" == "obb"

    def test_unknown_spelling_refused(self) -> None:
        """A name outside the registry raises rather than parsing to some default task."""
        with pytest.raises(ValueError, match="'panoptic' is not a valid Task"):
            Task("panoptic")


class TestTaskSpec:
    """The per-task defaults, against the literals the tables carried before the registry."""

    @pytest.mark.parametrize(
        ("task", "img_size", "eval_batch_size"),
        [
            pytest.param(Task.DETECT, 640, 32, id="detect"),
            pytest.param(Task.SEGMENT, 640, 32, id="segment"),
            pytest.param(Task.OBB, 1024, 8, id="obb"),
            pytest.param(Task.KEYPOINTS, 640, 32, id="keypoints"),
        ],
    )
    def test_spec_carries_previous_defaults(self, task: Task, img_size: int, eval_batch_size: int) -> None:
        """Each task's spec holds the side and batch size ``lucid-eval``'s tables hard-coded.

        The numbers are written out rather than read from anywhere, because the point is that the move into the registry
        changed none of them.
        """
        assert (task.spec.img_size, task.spec.eval_batch_size) == (img_size, eval_batch_size)

    @pytest.mark.parametrize("task", [Task.DETECT, Task.SEGMENT, Task.KEYPOINTS])
    def test_coco_tier_side_equals_decode_default(self, task: Task) -> None:
        """A COCO tier's side is the decode layer's ``COCO_IMG_SIZE``, which the leaf cannot import.

        ``lucid_yolo.decode`` imports the models and torch, so the registry states 640 itself; this pin is what keeps
        the scoring defaults and the per-task defaults one number.
        """
        assert task.spec.img_size == COCO_IMG_SIZE

    def test_spec_frozen(self) -> None:
        """A spec cannot be re-tuned in place by one consumer for every other one."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            Task.DETECT.spec.img_size = 320  # type: ignore[misc]

    def test_leaf_imports_no_package_module_and_no_torch(self) -> None:
        """Importing the registry loads no other ``lucid_yolo`` module, and neither torch nor numpy.

        Run in a fresh interpreter, because this session has long since imported both. A leaf is what lets
        ``lucid_yolo.models`` read it — that package sits below ``lucid_yolo.decode`` and cannot import upward without a
        cycle.
        """
        probe = (
            "import json, sys, lucid_yolo.tasks; "
            "print(json.dumps(sorted(n for n in sys.modules if n.split('.')[0] in {'lucid_yolo', 'torch', 'numpy'})))"
        )

        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

        assert json.loads(result.stdout) == ["lucid_yolo", "lucid_yolo.tasks"]


class TestDerivedTables:
    """The tables consumers still publish, now read off the registry, against their old literals."""

    @pytest.mark.parametrize(
        ("table", "expected_rows"),
        [
            pytest.param(
                DEFAULT_IMG_SIZE,
                [("obb", 1024), ("detect", 640), ("segment", 640), ("keypoints", 640)],
                id="img-size",
            ),
            pytest.param(
                DEFAULT_BATCH_SIZE,
                [("obb", 8), ("detect", 32), ("segment", 32), ("keypoints", 32)],
                id="batch-size",
            ),
        ],
    )
    def test_eval_tables_keep_contents_and_order(
        self, table: dict[str, int], expected_rows: list[tuple[str, int]]
    ) -> None:
        """``DEFAULT_IMG_SIZE`` and ``DEFAULT_BATCH_SIZE`` keep their old rows in their old key order, oriented first.

        Order is part of the contract: three commands' unknown-task refusal lists these keys, and
        ``tests/cli/test_export.py`` pins that text.
        """
        assert list(table.items()) == expected_rows

    def test_module_accepts_registry_names(self) -> None:
        """The module's accepted names are the registry's, in the order its refusal lists them."""
        assert module_lib._TASKS == ("detect", "segment", "obb", "keypoints")

    def test_oriented_predict_default_reads_registry(self) -> None:
        """``predict_oriented``'s public default side is the oriented spec's."""
        assert DEFAULT_ORIENTED_IMG_SIZE == Task.OBB.spec.img_size == 1024

    def test_drawable_tasks_are_plain_sorted_strings(self) -> None:
        """The script's ``--task`` choices stay plain strings, sorted, as argparse prints them.

        A member in an argparse ``choices`` would still validate, and the refusal that lists the tuple would then print
        ``<Task.OBB: 'obb'>`` instead of ``'obb'``.
        """
        assert [(name, type(name)) for name in DRAWABLE_TASKS] == [
            ("detect", str),
            ("keypoints", str),
            ("obb", str),
            ("segment", str),
        ]


class TestCheckpointTask:
    """``hparams["task"]`` stays a plain ``str``, so a saved checkpoint loads under ``weights_only``."""

    @pytest.mark.parametrize(
        "task",
        [
            pytest.param("detect", id="plain-string"),
            pytest.param(Task.DETECT, id="enum-member"),
            pytest.param(Task.KEYPOINTS, id="enum-member-keypoints"),
        ],
    )
    def test_saved_task_is_plain_str(self, save_checkpoint: Callable[[str], Path], task: str) -> None:
        """A checkpoint's ``task`` is exactly ``str``, even from a module built with a ``Task`` member.

        The member case is the one that matters: the registry makes passing one natural, Lightning's
        ``save_hyperparameters`` records whatever object the constructor's local holds, and a ``Task`` there would be
        pickled by class — which the restricted unpickler then refuses, making the file unloadable by every command.
        """
        payload = torch.load(save_checkpoint(task), map_location="cpu", weights_only=True)

        assert type(payload["hyper_parameters"]["task"]) is str

    def test_member_built_checkpoint_loads_through_load_eval_module(
        self, save_checkpoint: Callable[[str], Path]
    ) -> None:
        """A checkpoint saved from a ``Task.SEGMENT``-built module loads and reports the plain task.

        The end-to-end form of the pin above: :func:`lucid_yolo.load_eval_module` is the loader every command reads
        checkpoints through, and it reads them restricted.
        """
        loaded, _info = lucid_yolo.load_eval_module(save_checkpoint(Task.SEGMENT), use_ema=False)

        assert (loaded.task, type(loaded.task)) == ("segment", str)


class _NamedTaskModule(DetectionLitModule):
    """A subclass with ``task`` in its own signature, forwarding whatever it is given."""

    def __init__(self, task: str) -> None:
        super().__init__(depth=0.34, width=0.25, max_channels=64, num_classes=_NUM_CLASSES, task=task)


class _KwargsTaskModule(DetectionLitModule):
    """A subclass forwarding ``task`` through ``**kwargs``, as a thin configuring wrapper does."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(depth=0.34, width=0.25, max_channels=64, num_classes=_NUM_CLASSES, **kwargs)


class _ResavedTaskModule(_NamedTaskModule):
    """Record constructor arguments again after the base constructor returns."""

    def __init__(self, task: str) -> None:
        super().__init__(task=task)
        self.save_hyperparameters()


class _ResavedKwargsTaskModule(_KwargsTaskModule):
    """Re-record a member forwarded through keyword arguments."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()


@pytest.mark.parametrize(
    "module_cls",
    [
        pytest.param(_NamedTaskModule, id="named-parameter"),
        pytest.param(_KwargsTaskModule, id="kwargs"),
        pytest.param(_ResavedTaskModule, id="resaved-named-parameter"),
        pytest.param(_ResavedKwargsTaskModule, id="resaved-kwargs"),
    ],
)
class TestSubclassCheckpointTask:
    """A subclass forwarding a ``Task`` member still records a plain ``str`` ``task``.

    Lightning's ``save_hyperparameters`` collects constructor arguments from every ``__init__`` frame up the inheritance
    chain and merges the subclass frames over the base's, ``**kwargs`` contents included, so the subclass's
    ``Task.DETECT`` replaces the spelling the base constructor normalized its own local to.
    """

    def test_saved_task_is_plain_str(self, module_cls: Callable[..., DetectionLitModule], tmp_path: Path) -> None:
        """The checkpoint's ``task`` is exactly ``str``, so the restricted unpickler reads the file."""
        module = module_cls(task=Task.DETECT)
        assert type(module.hparams["task"]) is str
        path = _write_checkpoint(module, tmp_path / "subclass.ckpt")

        payload = torch.load(path, map_location="cpu", weights_only=True)

        assert type(payload["hyper_parameters"]["task"]) is str

    def test_initial_snapshot_task_is_plain_str(self, module_cls: Callable[..., DetectionLitModule]) -> None:
        """``hparams_initial``, the copy a logger writes to its hyper-parameter file, holds ``str`` too.

        Lightning takes that snapshot inside ``save_hyperparameters``, so correcting ``hparams`` alone afterwards would
        leave the logged record carrying the member.
        """
        module = module_cls(task=Task.DETECT)

        assert type(module.hparams_initial["task"]) is str

    def test_checkpoint_loads_through_load_eval_module(
        self, module_cls: Callable[..., DetectionLitModule], tmp_path: Path
    ) -> None:
        """The subclass's checkpoint loads through the command loader and reports the plain task."""
        path = _write_checkpoint(module_cls(task=Task.DETECT), tmp_path / "subclass.ckpt")

        loaded, _info = lucid_yolo.load_eval_module(path, use_ema=False)

        assert (loaded.task, type(loaded.task)) == ("detect", str)
