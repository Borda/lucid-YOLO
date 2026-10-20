# SPDX-License-Identifier: Apache-2.0
"""Replay the existing detection assignment and loss with masked target capacity.

Only dispatch changes. Each added ground-truth row has a false mask, a zero box
and a zero label. It selects no anchor, contributes zero to integer counts and
maxima, and leaves every real row's anchor ranking unchanged. One fit owns one
recording; a larger target count replaces it rather than retaining another pool.
Label validation stays outside capture and reads the current, unpadded inputs.
"""

from __future__ import annotations

import gc
import math
import warnings
from typing import TYPE_CHECKING, cast

import torch
from packaging.version import Version
from pytorch_lightning import Callback
from torch import Tensor, nn

from lucid_yolo.assign.one_to_one import UniqueAssigner
from lucid_yolo.assign.stal import SmallTargetAssigner
from lucid_yolo.assign.tal import AssignResult, _check_labels
from lucid_yolo.losses.detection_loss import DetectionBranchLoss, DetectionLossOutput
from lucid_yolo.losses.dual_loss import DualBranchLoss, DualLossOutput
from lucid_yolo.ptl.network_capture import _precision_signature

if TYPE_CHECKING:
    from pytorch_lightning import LightningModule, Trainer

_BF16_SUPPORTED = Version(torch.__version__) >= Version("2.9")


def _functions(source: DualBranchLoss) -> tuple[object, ...]:
    """Read the callable identities whose unchanged expressions a recording executes."""
    objects = (source, source.o2m_assigner, source.o2o_assigner, source._o2m_loss, source._o2o_loss)
    names = (
        ("__call__", "_score"),
        (
            "__call__",
            "_assign",
            "_candidate_mask",
            "_alignment_metric",
            "_select_topk",
            "_resolve_conflicts",
            "_finalize_mask",
            "_build_result",
            "_empty_result",
        ),
        (
            "__call__",
            "_assign",
            "_candidate_mask",
            "_alignment_metric",
            "_select_topk",
            "_resolve_conflicts",
            "_finalize_mask",
            "_build_result",
            "_empty_result",
        ),
        ("__call__", "_classification_loss", "_box_losses"),
        ("__call__", "_classification_loss", "_box_losses"),
    )
    return tuple(
        getattr(function, "__func__", function)
        for obj, fields in zip(objects, names, strict=True)
        for name in fields
        for function in (getattr(obj, name),)
    )


_EXPECTED = _functions(DualBranchLoss())


def _source_signature(source: DualBranchLoss) -> tuple[object, ...] | None:
    """Certify the built-in objects and scalar settings, excluding the live alpha input."""
    if type(source) is not DualBranchLoss:
        return None
    objects = (source, source.o2m_assigner, source.o2o_assigner, source._o2m_loss, source._o2o_loss)
    if (
        tuple(type(obj) for obj in objects)
        != (DualBranchLoss, SmallTargetAssigner, UniqueAssigner, DetectionBranchLoss, DetectionBranchLoss)
        or _functions(source) != _EXPECTED
    ):
        return None
    fields = ("topk", "alpha", "beta", "eps", "s_min", "s_ref")
    settings = tuple(getattr(assigner, name) for assigner in objects[1:3] for name in fields)
    settings += tuple(getattr(branch, name) for branch in objects[3:] for name in ("box_gain", "cls_gain", "l1_gain"))
    if any(type(value) not in (int, float) for value in settings):
        return None
    # Padding must keep zero-overlap IoUs finite in float32. A constructor-valid
    # positive Python epsilon can underflow (or flush) to zero on the device.
    if any(
        not math.isfinite(assigner.eps)
        or assigner.eps < torch.finfo(torch.float32).tiny
        or not math.isfinite(assigner.alpha)
        or assigner.alpha < 0
        or not math.isfinite(assigner.beta)
        or assigner.beta < 0
        for assigner in (source.o2m_assigner, source.o2o_assigner)
    ):
        return None
    return tuple(id(obj) for obj in objects), tuple((type(value), value) for value in settings)


class _LossObjective(nn.Module):
    """Expose the unchanged assignment stages and scalar loss fields to the recorder."""

    def __init__(self, source: DualBranchLoss) -> None:
        super().__init__()
        self.source = source
        self.assignments: tuple[AssignResult, AssignResult] | None = None

    def forward(
        self,
        ml: Tensor,
        mb: Tensor,
        ol: Tensor,
        ob: Tensor,
        points: Tensor,
        boxes: Tensor,
        labels: Tensor,
        mask: Tensor,
        strides: Tensor,
        alpha: Tensor,
    ) -> tuple[Tensor, ...]:
        """Run the existing expressions after the holder has checked the current labels."""
        source = self.source
        ma = source.o2m_assigner._assign(ml.sigmoid(), mb, points, boxes, labels, mask, check_labels=False)
        oa = source.o2o_assigner._assign(ol.sigmoid(), ob, points, boxes, labels, mask, check_labels=False)
        out = source._score(ml, mb, ol, ob, ma, oa, strides, alpha)
        self.assignments = ma, oa
        return (
            out.total,
            out.o2m.total,
            out.o2m.box,
            out.o2m.cls,
            out.o2m.l1,
            out.o2o.total,
            out.o2o.box,
            out.o2o.cls,
            out.o2o.l1,
        )


def _copy_assignment(value: AssignResult) -> AssignResult:
    """Return independently owned fields before another replay rewrites its workspace."""
    return AssignResult(
        *(
            tensor.clone()
            for tensor in (value.fg_mask, value.gt_index, value.target_labels, value.target_boxes, value.align_weights)
        )
    )


def _eligible(inputs: tuple[Tensor, ...]) -> bool:
    """Keep unsupported tensor, autograd and CUDA contexts on the native route."""
    if (
        not torch.is_grad_enabled()
        or torch.is_inference_mode_enabled()
        or torch.compiler.is_compiling()
        or torch.is_anomaly_enabled()
        or any(
            getattr(nn.modules.module, name, {})
            for name in (
                "_global_forward_hooks",
                "_global_forward_pre_hooks",
                "_global_backward_hooks",
                "_global_backward_pre_hooks",
            )
        )
        or any(type(value) is not Tensor for value in inputs)
    ):
        return False
    logits, _, _, _, _, boxes, labels, mask, _, alpha = inputs
    return (
        all(value.is_cuda and value.device == logits.device and value.is_contiguous() for value in inputs)
        and all(value.requires_grad for value in inputs[:4])
        and not any(value.requires_grad for value in inputs[4:])
        and boxes.ndim == 3
        and boxes.shape[-1] == 4
        and labels.shape == boxes.shape[:2]
        and mask.shape == labels.shape
        and boxes.dtype == torch.float32
        and labels.dtype == torch.int64
        and mask.dtype == torch.bool
        and alpha.ndim == 0
        and alpha.dtype == torch.float64
        and torch.cuda.current_device() == logits.device.index
        and not torch.cuda.is_current_stream_capturing()
    )


class _LossCapture:
    """Own one transient loss recording and its geometrically grown target workspace.

    Shape changes outside the target count stay eager. Changed source settings or
    precision retire the recording. An outstanding backward keeps later forwards
    eager, because their writes must not replace its saved intermediates.

    Examples:
        ```pycon
        >>> import copy
        >>> copy.deepcopy(_LossCapture())._objective is None
        True

        ```
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        """Release graph ownership and restart the next fit's eligibility decision."""
        previous = getattr(self, "_objective", None)
        if previous is not None:
            vars(previous).pop("forward", None)
        self._objective: _LossObjective | None = None
        self._workspace: tuple[Tensor, Tensor, Tensor] | None = None
        self._signature: tuple[object, ...] | None = None
        self._source: tuple[object, ...] | None = None
        self._precision: tuple[object, ...] | None = None
        self._capacity = 0
        self._pending_backward = False
        self._disabled = False

    def __getstate__(self) -> dict[str, object]:
        """Keep CUDA resources out of pickle and deepcopy."""
        return {}

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore an empty owner rather than a recording from another fit."""
        del state
        self.clear()

    def retire(self, reason: str) -> None:
        """Clear a stale recording and report its first cause once per fit."""
        self.clear()
        self._disabled = True
        warnings.warn(f"capture_loss=True stays eager: {reason}", UserWarning, stacklevel=3)

    def run(
        self,
        source: DualBranchLoss,
        ml: Tensor,
        mb: Tensor,
        ol: Tensor,
        ob: Tensor,
        points: Tensor,
        boxes: Tensor,
        labels: Tensor,
        mask: Tensor,
        strides: Tensor,
        alpha: Tensor,
    ) -> DualLossOutput:
        """Replay an eligible detection loss, otherwise call its existing public route."""
        inputs = ml, mb, ol, ob, points, boxes, labels, mask, strides, alpha
        if self._disabled or self._pending_backward or not _eligible(inputs):
            return source(*inputs[:4], points, boxes, labels, mask, strides=strides, alpha=alpha)
        contract = self._contract(source, ml)
        if contract is None:
            return source(*inputs[:4], points, boxes, labels, mask, strides=strides, alpha=alpha)
        state, precision = contract
        signature = (
            tuple(
                (tuple(value.shape), value.stride(), value.dtype, value.device)
                for value in (*inputs[:5], strides, alpha)
            ),
            tuple(
                (value.shape[0], tuple(value.shape[2:]), value.dtype, value.device) for value in (boxes, labels, mask)
            ),
            torch.cuda.current_stream(ml.device).cuda_stream,
        )
        if self._signature is not None and signature != self._signature:
            return source(*inputs[:4], points, boxes, labels, mask, strides=strides, alpha=alpha)
        if __debug__:
            _check_labels(labels, mask, ml.shape[-1])
            if ol.shape[-1] != ml.shape[-1]:
                _check_labels(labels, mask, ol.shape[-1])
        # Masked rows are neutral only for finite arithmetic: NaN * 0 is NaN.
        # Preserve the native empty-target and non-finite propagation behavior.
        # Only predictions and target boxes meet the padded rows in arithmetic:
        # anchor points enter comparisons only, and strides and alpha are not
        # padded, so their non-finite values match on both routes.
        if boxes.shape[1] == 0 or not bool(
            torch.stack(tuple(value.isfinite().all() for value in (ml, mb, ol, ob, boxes))).all()
        ):
            return source(*inputs[:4], points, boxes, labels, mask, strides=strides, alpha=alpha)
        capacity = max(self._capacity, 1 << (max(1, boxes.shape[1]) - 1).bit_length())
        if capacity != self._capacity:
            self.clear()
            self._workspace = (
                boxes.new_zeros((boxes.shape[0], capacity, boxes.shape[-1])),
                labels.new_zeros((labels.shape[0], capacity)),
                mask.new_zeros((mask.shape[0], capacity)),
            )
        assert self._workspace is not None
        for workspace, value in zip(self._workspace, (boxes, labels, mask), strict=True):
            workspace.zero_()
            workspace[:, : value.shape[1]].copy_(value)
        arguments = (*inputs[:5], *self._workspace, strides, alpha)
        with torch.autocast(
            "cuda", enabled=torch.is_autocast_enabled(), dtype=torch.get_autocast_dtype("cuda"), cache_enabled=False
        ):
            result = self._replay(source, arguments, capacity, signature, state, precision)
        assert self._objective is not None and self._objective.assignments is not None
        ma, oa = (_copy_assignment(value) for value in self._objective.assignments)
        node = result[0].grad_fn
        result = tuple(value.clone() for value in result)
        assert node is not None
        self._pending_backward = True
        node.register_hook(self._backward_done)
        return DualLossOutput(
            result[0], DetectionLossOutput(*result[1:5]), DetectionLossOutput(*result[5:]), source.alpha, ma, oa
        )

    def _contract(self, source: DualBranchLoss, logits: Tensor) -> tuple[tuple[object, ...], tuple[object, ...]] | None:
        """Resolve source and precision guards before entering a recording."""
        state = _source_signature(source)
        if state is None:
            self.retire("the loss or an assignment implementation was customized")
            return None
        try:
            precision = _precision_signature()
        except RuntimeError as error:
            self.retire(f"the CUDA precision policy cannot be read ({error})")
            return None
        if not _BF16_SUPPORTED and (
            logits.dtype == torch.bfloat16 or (precision[0] and precision[1] == torch.bfloat16)
        ):
            return None
        if self._objective is not None and (state != self._source or precision != self._precision):
            self.retire("the loss settings or CUDA precision policy changed after recording")
            return None
        return state, precision

    def _replay(
        self,
        source: DualBranchLoss,
        arguments: tuple[Tensor, ...],
        capacity: int,
        signature: tuple[object, ...],
        state: tuple[object, ...],
        precision: tuple[object, ...],
    ) -> tuple[Tensor, ...]:
        """Finish the first replay and its fence before publishing a new recording."""
        if self._objective is not None:
            return cast("tuple[Tensor, ...]", self._objective(*arguments))
        objective = _LossObjective(source)
        samples = tuple(value.detach().clone().requires_grad_(value.requires_grad) for value in arguments)
        # An automatic cycle collection can destroy an old CUDA graph while this
        # recording is open. Drain unreachable cycles before capture and keep
        # collection disabled only until its first replay has completed.
        gc.collect()
        collection_enabled = gc.isenabled()
        gc.disable()
        try:
            objective = cast("_LossObjective", torch.cuda.make_graphed_callables(objective, samples))
            result = cast("tuple[Tensor, ...]", objective(*arguments))
            torch.cuda.synchronize(arguments[0].device)
        except BaseException:
            vars(objective).pop("forward", None)
            self.clear()
            raise
        finally:
            if collection_enabled:
                gc.enable()
        self._objective = objective
        self._capacity, self._signature = capacity, signature
        self._source, self._precision = state, precision
        return result

    def _backward_done(self, grad_inputs: object, grad_outputs: object) -> None:
        """Release the hold after the replayed scalar total has been differentiated."""
        del grad_inputs, grad_outputs
        self._pending_backward = False

    def start_batch(self) -> None:
        """Discard a hold from a skipped backward at the next automatic training batch."""
        self._pending_backward = False


class _LossCaptureLifecycle(Callback):
    """Bound the loss pool to one automatic fit, including skipped batches and failure."""

    @staticmethod
    def _holder(module: LightningModule) -> _LossCapture:
        return cast("_LossCapture", module._loss_capture)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: object, batch_idx: int) -> None:
        """Allow a new batch after a skipped backward."""
        del trainer, batch, batch_idx
        self._holder(pl_module).start_batch()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Start with no recording from an earlier fit."""
        del trainer
        self._holder(pl_module).clear()

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Release the graph pool after training."""
        del trainer
        self._holder(pl_module).clear()

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        """Release the graph pool when training aborts."""
        del trainer, exception
        self._holder(pl_module).clear()
