# SPDX-License-Identifier: Apache-2.0
"""Tests for the opt-in compiled detection objective (WP-184).

``compile_step=True`` compiles :meth:`~lucid_yolo.ptl.module.DetectionLitModule._detect_objective`
— forward, both decodes, assignment and the dual loss — as one region on a CUDA
accelerator under ``task="detect"``. No GPU runs here, so what is proved on CPU is
the property the speed-up rests on and the one a later edit breaks silently: the
region traces as **one** Dynamo graph with **zero** breaks, on a batch with ground
truths and on a batch with none (the ``tal.py`` empty-batch branch), with
``capture_scalar_outputs`` as :meth:`setup` sets it and — since the only scalar
read in the region was :func:`~lucid_yolo.assign.tal._check_labels`, now skipped
under tracing — with it off as well. The ``setup`` decision is exercised through
the same mocked CUDA accelerator ``test_memory_format.py`` uses: the wrapper is
installed once and a second ``setup`` keeps it, and a task other than ``detect`` is
refused by name; the off-CUDA refusal is ``test_module.py``'s, beside the loss it leaves
unchanged.

Every test runs under a fixture that resets Dynamo and restores the process-global
config flag, so nothing here leaks into the rest of the suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import torch
import torch._dynamo
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import CUDAAccelerator
from torch import Tensor

from lucid_yolo.data.targets import Targets
from lucid_yolo.ptl import DetectionLitModule, pad_targets

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Class count of the tiny test head.
_NUM_CLASSES = 4
#: Square input side; divisible by every stride for an integer anchor grid.
_IMG_SIZE = 160
#: Images per synthetic batch.
_BATCH_SIZE = 2


@pytest.fixture(autouse=True)
def isolate_dynamo() -> Iterator[None]:
    """Seed torch, then reset Dynamo and restore ``capture_scalar_outputs`` after each test."""
    torch.manual_seed(0)
    flag = torch._dynamo.config.capture_scalar_outputs
    yield
    torch._dynamo.config.capture_scalar_outputs = flag
    torch._dynamo.reset()


def _module(compile_step: bool = False) -> DetectionLitModule:
    """Build an n-scale detection module with a low channel cap.

    Examples:
        >>> _module().task
        'detect'
        >>> _module(compile_step=True).hparams.compile_step
        True
    """
    return DetectionLitModule(
        depth=0.34, width=0.25, max_channels=256, num_classes=_NUM_CLASSES, compile_step=compile_step
    )


def _targets(num_boxes: int) -> Targets:
    """Build ``num_boxes`` valid ``xyxy`` targets inside the test image.

    Examples:
        >>> _targets(2).boxes.shape
        torch.Size([2, 4])
        >>> _targets(0).labels.shape
        torch.Size([0])
    """
    top_left = torch.rand(num_boxes, 2) * 80.0
    size = torch.rand(num_boxes, 2) * 40.0 + 10.0
    boxes = torch.cat([top_left, top_left + size], dim=1)
    return Targets(boxes=boxes, labels=torch.randint(0, _NUM_CLASSES, (num_boxes,)))


def _region_inputs(module: DetectionLitModule, counts: tuple[int, int]) -> tuple[Tensor, ...]:
    """Build the six positional inputs of ``_detect_objective`` for a batch with ``counts`` instances.

    Examples:
        >>> images, points, strides, boxes, labels, mask = _region_inputs(_module(), (2, 1))
        >>> images.shape, boxes.shape
        (torch.Size([2, 3, 160, 160]), torch.Size([2, 2, 4]))
        >>> _region_inputs(_module(), (0, 0))[3].shape
        torch.Size([2, 0, 4])
    """
    images = torch.randn(_BATCH_SIZE, 3, _IMG_SIZE, _IMG_SIZE)
    anchor_points, strides = module._anchor_grid(_IMG_SIZE, _IMG_SIZE, images.device)
    gt_boxes, gt_labels, gt_mask = pad_targets([_targets(count) for count in counts])
    return images, anchor_points, strides, gt_boxes, gt_labels, gt_mask


def _cuda_trainer() -> MagicMock:
    """Mock a trainer reporting a CUDA accelerator, so ``setup`` takes the positive branch on CPU.

    Examples:
        >>> isinstance(_cuda_trainer().accelerator, CUDAAccelerator)
        True
    """
    return MagicMock(spec=Trainer, world_size=1, accelerator=CUDAAccelerator())


class TestDetectObjectiveTracesAsOneGraph:
    """The region compiles to a single graph, so the measured speed-up has its precondition."""

    @pytest.mark.parametrize("capture_scalar_outputs", [True, False])
    @pytest.mark.parametrize("counts", [pytest.param((2, 1), id="with_gts"), pytest.param((0, 0), id="zero_gts")])
    def test_zero_graph_breaks(self, counts: tuple[int, int], capture_scalar_outputs: bool) -> None:
        """One graph, no breaks — with the flag as ``setup`` sets it and without it."""
        torch._dynamo.config.capture_scalar_outputs = capture_scalar_outputs
        module = _module()
        inputs = _region_inputs(module, counts)

        explanation = torch._dynamo.explain(module._detect_objective)(*inputs)

        assert explanation.graph_break_count == 0, explanation.break_reasons
        assert explanation.graph_count == 1

    def test_region_total_matches_the_shared_step(self) -> None:
        """The extracted region scores exactly what ``_shared_step`` scores for ``detect``."""
        module = _module()
        module.log = MagicMock()  # type: ignore[method-assign]
        images = torch.randn(_BATCH_SIZE, 3, _IMG_SIZE, _IMG_SIZE)
        targets = [_targets(2), _targets(1)]
        anchor_points, strides = module._anchor_grid(_IMG_SIZE, _IMG_SIZE, images.device)
        gt_boxes, gt_labels, gt_mask = pad_targets(targets)

        _head_out, out = module._detect_objective(images, anchor_points, strides, gt_boxes, gt_labels, gt_mask)
        total, _head, _seg = module._shared_step((images, targets), "train")

        assert torch.equal(out.total, total)


class TestSetupDecidesOnce:
    """``setup`` installs the wrapper on CUDA + detect, warns and stays eager otherwise, and does so once."""

    def test_cuda_detect_installs_the_compiled_wrapper_once(self) -> None:
        """On a CUDA accelerator the objective is the compiled wrapper, kept across a second ``setup``."""
        torch._dynamo.config.capture_scalar_outputs = False
        module = _module(compile_step=True)
        module._trainer = _cuda_trainer()

        module.setup("fit")
        objective = module._objective
        module.setup("validate")

        assert module._objective_compiled
        assert objective != module._detect_objective
        assert module._objective is objective
        assert torch._dynamo.config.capture_scalar_outputs is True

    def test_state_dict_keys_are_unchanged_with_the_wrapper_installed(self) -> None:
        """Function-level compile leaves every parameter name as built — no ``_orig_mod.`` prefix."""
        expected = list(_module().state_dict())
        module = _module(compile_step=True)
        module._trainer = _cuda_trainer()

        module.setup("fit")

        assert module._objective_compiled
        assert list(module.state_dict()) == expected
        assert not any(name.startswith("_orig_mod.") for name, _ in module.named_parameters())

    def test_other_task_warns_and_stays_eager_even_on_cuda(self) -> None:
        """A non-detect task is refused by name, CUDA or not."""
        module = DetectionLitModule(
            depth=0.34, width=0.25, max_channels=256, num_classes=_NUM_CLASSES, task="segment", compile_step=True
        )
        module._trainer = _cuda_trainer()

        with pytest.warns(UserWarning, match="task='segment'"):
            module.setup("fit")

        assert not module._objective_compiled
        assert module._objective == module._detect_objective

    def test_default_off_never_touches_the_objective(self) -> None:
        """Without the flag a CUDA ``setup`` leaves the bare method and the config flag alone."""
        torch._dynamo.config.capture_scalar_outputs = False
        module = _module()
        module._trainer = _cuda_trainer()

        module.setup("fit")

        assert not module._objective_compiled
        assert module._objective == module._detect_objective
        assert torch._dynamo.config.capture_scalar_outputs is False

    def test_dynamic_marks_apply_only_to_a_compiled_objective(self) -> None:
        """The marks are a no-op while eager, and mark ``B`` and ``N`` once the wrapper is installed."""
        module = _module(compile_step=True)
        images, _points, _strides, gt_boxes, gt_labels, gt_mask = _region_inputs(module, (0, 0))

        module._mark_dynamic(images, gt_boxes, gt_labels, gt_mask)
        assert not hasattr(images, "_dynamo_dynamic_indices")

        module._trainer = _cuda_trainer()
        module.setup("fit")
        module._mark_dynamic(images, gt_boxes, gt_labels, gt_mask)

        assert getattr(images, "_dynamo_dynamic_indices", None) == {0}
        assert all(
            getattr(tensor, "_dynamo_dynamic_indices", None) == {0, 1} for tensor in (gt_boxes, gt_labels, gt_mask)
        )


class TestCompiledStepOverChangingShapes:
    """The enforcing marks hold: no batch shape the loader produces specialises a marked dim."""

    @pytest.mark.parametrize(
        "counts",
        [
            pytest.param((2, 1), id="ragged"),
            pytest.param((0, 0), id="zero_gts"),
            pytest.param((1, 1), id="one_gt_each"),
            pytest.param((3,), id="single_image"),
            pytest.param((4, 2, 0), id="three_images"),
        ],
    )
    def test_matches_eager_through_the_shared_step(self, counts: tuple[int, ...]) -> None:
        """A compiled objective steps every shape to the eager total, marks and all.

        ``mark_dynamic`` is the enforcing form: a marked dim the traced code then pins
        to a constant raises ``ConstraintViolationError`` at the call. The ``eager``
        backend traces the region exactly as Inductor would without lowering it, so
        the ``N`` pin ``F.one_hot`` used to make and the ``B`` pin an unmarked ground
        truth made are both caught here on CPU.
        """
        module = _module(compile_step=True)
        module.log = MagicMock()  # type: ignore[method-assign]
        module._objective = torch.compile(module._detect_objective, backend="eager")
        module._objective_compiled = True
        images = torch.randn(len(counts), 3, _IMG_SIZE, _IMG_SIZE)
        targets = [_targets(count) for count in counts]

        total, _head, _seg = module._shared_step((images, targets), "train")
        _head, expected = module._detect_objective(
            images, *module._anchor_grid(_IMG_SIZE, _IMG_SIZE, images.device), *pad_targets(targets)
        )

        assert torch.equal(total, expected.total)
