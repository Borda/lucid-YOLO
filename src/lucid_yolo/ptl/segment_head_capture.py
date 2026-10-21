# SPDX-License-Identifier: Apache-2.0
"""Record segmentation prediction groups at their native gradient boundaries.

Box/class predictions, mask coefficients, prototypes and semantic logits have
separate backward nodes. In particular, an unused coefficient or prototype
output leaves its parameters' gradients absent, as the native empty-mask path
does. No assignment, loss or optimizer expression is recorded or rewritten.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch
from pytorch_lightning import Callback
from pytorch_lightning.accelerators import CUDAAccelerator
from pytorch_lightning.strategies import SingleDeviceStrategy
from torch import Tensor, nn

from lucid_yolo.models.blocks import ConvBNAct, DepthwiseConv
from lucid_yolo.models.heads.detect import DualDetectionHead, _DetectionBranch
from lucid_yolo.models.heads.proto import ProtoFusion, ProtoNet
from lucid_yolo.models.heads.semantic import SemanticAux
from lucid_yolo.optim.musgd import _memory_overlaps
from lucid_yolo.optim.update_replay import _RecordedSpans
from lucid_yolo.ptl.network_capture import (
    _BF16_REPLAY_SUPPORTED,
    _precision_signature,
)

if TYPE_CHECKING:
    from pytorch_lightning import LightningModule, Trainer

    from lucid_yolo.ptl.module import DetectionLitModule

_Tensors = tuple[Tensor, ...]
_Call = Callable[[_Tensors], _Tensors]
_HEAD_METHODS = {name: getattr(DualDetectionHead, name) for name in ("forward", "forward_branch")}
_BRANCH_METHODS = {
    name: getattr(_DetectionBranch, name)
    for name in ("forward", "_base_predictions", "_coefficients", "_keypoints", "_angles")
}

_STAGE_FORWARDS: dict[type[nn.Module], object] = {
    cls: cls.forward
    for cls in (
        DualDetectionHead,
        _DetectionBranch,
        ProtoFusion,
        ProtoNet,
        SemanticAux,
        ConvBNAct,
        DepthwiseConv,
        nn.Sequential,
        nn.ModuleList,
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.SiLU,
        nn.Upsample,
    )
}
_FORWARD_HELPERS: dict[type[nn.Module], dict[str, object]] = {
    nn.Conv2d: {"_conv_forward": nn.Conv2d._conv_forward},
    nn.BatchNorm2d: {"_check_input_dim": nn.BatchNorm2d._check_input_dim},
}


def _tensor_signature(tensors: tuple[Tensor, ...]) -> tuple[object, ...]:
    """Describe the mutable tensor identities, storage and gradient contract without reading values."""
    return tuple(
        (
            id(tensor),
            tensor.data_ptr(),
            tuple(tensor.shape),
            tensor.stride(),
            tensor.dtype,
            tensor.device,
            tensor.requires_grad,
        )
        for tensor in tensors
    )


def _observer_breach(modules: tuple[nn.Module, ...]) -> str | None:
    """Keep module hooks observable on the native path, even after recording."""
    runtime = nn.modules.module
    if any(
        getattr(runtime, name, {})
        for name in (
            "_global_forward_hooks",
            "_global_forward_pre_hooks",
            "_global_backward_hooks",
            "_global_backward_pre_hooks",
        )
    ):
        return "a global module hook is registered"
    if any(
        layer._forward_hooks or layer._forward_pre_hooks or layer._backward_hooks or layer._backward_pre_hooks
        for layer in modules
    ):
        return "a prediction stage carries a forward or backward hook"
    return None


def _layer_breach(layer: nn.Module) -> str | None:
    """Name why one layer's type, forward implementation or forward helpers leave the recorded route."""
    if type(layer) not in _STAGE_FORWARDS:
        return f"a prediction stage of type {type(layer).__name__} is not in the native stage list"
    if (
        type(layer).forward is not _STAGE_FORWARDS[type(layer)]
        or "forward" in vars(layer)
        or getattr(layer, "_compiled_call_impl", None) is not None
    ):
        return "a prediction stage overrides its native forward implementation"
    if type(layer) is nn.BatchNorm2d and layer.training and layer.track_running_stats and layer.momentum is None:
        return "a cumulative-average BatchNorm reads its count on the host"
    for cls, helpers in _FORWARD_HELPERS.items():
        if type(layer) is cls and any(
            name in vars(layer) or getattr(type(layer), name) is not method for name, method in helpers.items()
        ):
            return "a prediction stage overrides a native forward helper"
    return None


def _stage_breach(owner: nn.Module, modules: tuple[nn.Module, ...] | None = None) -> str | None:
    """Keep custom stage implementations and forward helpers outside the recorded route."""
    modules = tuple(owner.modules()) if modules is None else modules
    reason = _observer_breach(modules)
    if reason is not None:
        return reason
    if not all(stage.training for stage in owner.children()):
        return "a prediction stage is outside training mode"
    for layer in modules:
        reason = _layer_breach(layer)
        if reason is not None:
            return reason
    return None


def _members(owner: nn.Module) -> tuple[tuple[nn.Module, ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Enumerate current topology once for the three checks that consume it.

    Direct registries have the same preorder and identity deduplication as
    ``named_parameters``/``named_buffers``. Nothing is cached across calls: a
    replacement, alias or child-order change still reaches the certificate.
    """
    modules = tuple(owner.modules())
    params = {id(p): p for module in modules for p in module._parameters.values() if p is not None}
    buffers = {id(b): b for module in modules for b in module._buffers.values() if b is not None}
    return modules, tuple(params.values()), tuple(buffers.values())


def _parameter_breach(parameters: tuple[Tensor, ...], buffers: tuple[Tensor, ...], device: torch.device) -> str | None:
    """Read live tensor contracts and hooks independently of the attribute certificate.

    Lazy negative/conjugate views can change values with the same pointer and
    strides. Keep them native rather than replaying their earlier dispatch bits.
    """
    if not parameters or any(
        type(x) not in (Tensor, nn.Parameter)
        or x.layout != torch.strided
        or x.device != device
        or x.is_conj()
        or x.is_neg()
        for x in parameters + buffers
    ):
        return "prediction storage is not plain strided storage on the feature device"
    if any(
        not p.requires_grad or p.dtype != torch.float32 or p._backward_hooks or p._post_accumulate_grad_hooks
        for p in parameters
    ):
        return "a prediction parameter is frozen, not float32 or carries a gradient hook"
    if any(b.requires_grad for b in buffers):
        return "a prediction buffer requires gradients"
    return None


def _storage_breach(owner: nn.Module, device: torch.device) -> str | None:
    """Certify independent prediction storage once, before its keyed lifetime begins."""
    values: list[Tensor] = [x for _, x in owner.named_parameters(remove_duplicate=False)]
    values.extend(x for _, x in owner.named_buffers(remove_duplicate=False))
    if any(x.device != device or x.numel() == 0 for x in values):
        return "prediction storage is empty or spans CUDA devices"
    if _memory_overlaps(values):
        return "prediction parameters or buffers share bytes"
    return None


class _PredictionGroup(nn.Module):
    """Expose only the parameters a native prediction call can read."""

    def __init__(self, stages: tuple[nn.Module, ...], eager: _Call) -> None:
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.eager = eager

    def forward(self, *inputs: Tensor) -> _Tensors:
        """Call the existing prediction routine in its original tensor order."""
        return self.eager(inputs)


def _input_signature(inputs: _Tensors) -> tuple[object, ...]:
    """Keep a recording on one layout, gradient contract and submission stream."""
    return (
        tuple((tuple(x.shape), x.stride(), x.dtype, x.device, x.requires_grad) for x in inputs),
        torch.cuda.current_stream(inputs[0].device).cuda_stream,
    )


class _RecordedGroup:
    """Keep one prediction group's static tensors alive until its backward consumes them."""

    def __init__(self, adapter: _PredictionGroup, inputs: _Tensors, spans: _RecordedSpans) -> None:
        self.adapter = adapter
        self.parameters = tuple(adapter.parameters())
        self.spans = spans
        self.signature = _input_signature(inputs)
        self.pending_backward = False
        self.stream = torch.cuda.current_stream(inputs[0].device)
        snapshots = [(buffer, buffer.detach().clone()) for buffer in self.adapter.buffers()]
        with torch.random.fork_rng(devices=[inputs[0].device.index]):
            try:
                samples = tuple(
                    torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
                    .copy_(x.detach())
                    .requires_grad_(x.requires_grad)
                    for x in inputs
                )
                self.adapter = cast("_PredictionGroup", torch.cuda.make_graphed_callables(self.adapter, samples))
            finally:
                torch.cuda.synchronize(inputs[0].device)
                with torch.no_grad():
                    for buffer, snapshot in snapshots:
                        buffer.copy_(snapshot)

    def run(self, inputs: _Tensors) -> _Tensors:
        """Replay once, or preserve the static activations a previous backward still needs."""
        if self.pending_backward or _input_signature(inputs) != self.signature or self.spans.reached_by(list(inputs)):
            return self.adapter.eager(inputs)
        outputs = self.adapter(*inputs)
        node = outputs[0].grad_fn
        if node is not None:
            self.pending_backward = True
            node.register_prehook(self._before_backward)
            node.register_hook(self._after_backward)
        return cast("_Tensors", outputs)

    def _before_backward(self, grad_outputs: object) -> None:
        """Protect live gradients from the private buffers the recorded backward overwrites."""
        del grad_outputs
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad = parameter.grad.clone()

    def _after_backward(self, grad_inputs: object, grad_outputs: object) -> None:
        """Release this group's activations after its recorded backward finishes."""
        del grad_inputs, grad_outputs
        self.pending_backward = False


def _head_breach(head: DualDetectionHead, levels: int = 3) -> str | None:
    """Refuse custom dispatch before passing a scheduling adapter into the native head."""
    if type(head) is not DualDetectionHead or any(
        name in vars(head) or getattr(type(head), name) is not method for name, method in _HEAD_METHODS.items()
    ):
        return "the detection head overrides its native dispatch"
    for branch in (head.o2m, head.o2o):
        if type(branch) is not _DetectionBranch or any(
            name in vars(branch) or getattr(type(branch), name) is not method
            for name, method in _BRANCH_METHODS.items()
        ):
            return "a prediction branch overrides its native dispatch"
        if branch.coeff_stems is None or branch.angle_stems is not None or branch.keypoint_stems is not None:
            return "the head is not the native segmentation topology"
        if any(len(stems) != levels for stems in (branch.box_stems, branch.cls_stems, branch.coeff_stems)):
            return "the prediction stem count differs from the feature count"
    return None


def _feature_breach(features: _Tensors, precision: tuple[object, ...]) -> str | None:
    """Name input and runtime policies that cannot retain a prediction recording."""
    if not all(
        type(x) is Tensor
        and x.is_cuda
        and x.layout == torch.strided
        and (x.is_contiguous() or x.is_contiguous(memory_format=torch.channels_last))
        for x in features
    ):
        return "the features are not ordinary dense CUDA tensors"
    if any((x.requires_grad and x.is_leaf) or x.retains_grad or x._backward_hooks for x in features):
        return "feature gradients are leaf-owned, retained or hooked"
    if torch.is_anomaly_enabled() or torch.is_inference_mode_enabled() or torch.compiler.is_compiling():
        return "anomaly detection, inference mode or compilation is enabled"
    if not (precision[0] and precision[1] == torch.bfloat16 and all(x.dtype == torch.bfloat16 for x in features)):
        return "replay is verified only for BF16 mixed precision"
    if not _BF16_REPLAY_SUPPORTED:
        return "BF16 replay requires torch>=2.9, the first verified exact runtime"
    return None


class _AttributePlan:
    """Retain attribute bindings while independently checking mutable registries and sequences.

    Stored values stay alive, so a fresh object cannot reuse a recorded identity.
    Tensor identities/storage, hooks, precision and class dispatch are checked
    separately on every call; an unchanged dictionary object certifies none of them.
    """

    def __init__(self, modules: tuple[nn.Module, ...]) -> None:
        self.modules = modules
        self.bindings = tuple(dict(vars(module)) for module in modules)
        self.registries = tuple(
            (tuple(module._parameters), tuple(module._buffers), tuple(module._modules)) for module in modules
        )
        self.sequences = tuple(
            {name: tuple(value) for name, value in vars(module).items() if type(value) in (tuple, list)}
            for module in modules
        )

    def matches(self, modules: tuple[nn.Module, ...]) -> bool:
        """Require the same bindings and reread registry names and sequence elements each time."""
        if tuple(map(id, modules)) != tuple(map(id, self.modules)):
            return False
        for module, saved, registries, sequences in zip(
            modules, self.bindings, self.registries, self.sequences, strict=True
        ):
            current = vars(module)
            if current.keys() != saved.keys() or any(value is not saved[name] for name, value in current.items()):
                return False
            if (tuple(module._parameters), tuple(module._buffers), tuple(module._modules)) != registries:
                return False
            for name, old_items in sequences.items():
                items = current[name]
                if len(items) != len(old_items) or any(a is not b for a, b in zip(items, old_items, strict=True)):
                    return False
        return True


class _SegmentHeadCapture:
    """Own at most sixteen recordings, discarded by copying, serialization and the fit lifecycle.

    The model keeps its original modules and state-dict keys. Recording wrappers
    live only in this ordinary Python holder; no live module forward is replaced.

    A replayed group's ``coefficients``, ``prototypes`` and ``semantic`` outputs are the graphed callable's
    static CUDA-graph tensors, not copies; the box and class predictions are concatenation copies. The next
    replay of that group overwrites them in place, so a consumer must finish with them, and with any ``.grad``
    adopted from their backward, inside the training step that produced them and clone whatever it keeps
    across steps.

    Each prediction group records the input signature of its first eligible call once per fit: shape, stride,
    dtype, device, gradient flag and CUDA stream. A call with any other signature, a partial batch included,
    runs the native routine and records nothing more, while the recorded graphs keep their private pools
    reserved until the fit ends.

    Examples:
        ```pycon
        >>> import copy
        >>> holder = _SegmentHeadCapture(True)
        >>> copy.deepcopy(holder).entries
        {}

        ```
    """

    def __init__(self, requested: bool) -> None:
        self.requested = requested
        self.active = False
        self.clear()

    def clear(self) -> None:
        """Reset the holder, then drain recorded streams before their private pools are released.

        The reset comes first, so a failing drain still leaves an empty holder; the recordings stay
        alive in a local reference until every stream they ran on has finished.
        """
        entries = getattr(self, "entries", {})
        self.entries: dict[tuple[int, str], _RecordedGroup] = {}
        self.owner: nn.Module | None = None
        self.attributes: _AttributePlan | None = None
        self.tensors: tuple[Tensor, ...] = ()
        self.state: tuple[object, ...] | None = None
        self.disabled = False
        for stream in {entry.stream for entry in entries.values()}:
            stream.synchronize()

    def __getstate__(self) -> dict[str, bool]:
        """Serialize the request and no CUDA recording or live module reference."""
        return {"requested": self.requested}

    def __setstate__(self, state: dict[str, bool]) -> None:
        """Restore an inactive holder whose next fit must establish eligibility again."""
        _SegmentHeadCapture.__init__(self, state["requested"])

    def setup(self, module: DetectionLitModule, stage: str) -> None:
        """Activate only the single-device CUDA automatic-optimization training route."""
        trainer = module._trainer
        self.active = bool(
            self.requested
            and trainer is not None
            and isinstance(trainer.accelerator, CUDAAccelerator)
            and isinstance(trainer.strategy, SingleDeviceStrategy)
            and module.automatic_optimization
            and trainer.accumulate_grad_batches == 1
        )
        if self.requested and not self.active and stage == "fit":
            warnings.warn(
                "capture_segment_heads=True is ignored outside single-device CUDA automatic optimization "
                "with accumulate_grad_batches=1",
                UserWarning,
                stacklevel=3,
            )

    def retire(self, reason: str) -> None:
        """Keep the first incompatibility visible and use native calls for the rest of this fit."""
        if self.disabled:
            return
        self.clear()
        self.disabled = True
        warnings.warn(f"capture_segment_heads=True stays eager: {reason}", UserWarning, stacklevel=3)

    def runner(self, module: DetectionLitModule, features: _Tensors) -> _SegmentHeadCapture | None:
        """Certify the current head before any of its prediction groups may replay."""
        trainer = module._trainer
        if (
            not self.active
            or self.disabled
            or trainer is None
            or not trainer.training
            or not module.training
            or not module.automatic_optimization
            or not torch.is_grad_enabled()
        ):
            return None
        # A capture in progress, or an empty batch with nothing to record: this call stays native, replay stays on.
        if torch.cuda.is_current_stream_capturing() or any(x.numel() == 0 for x in features):
            return None
        if trainer.accumulate_grad_batches != 1:
            self.retire("gradient accumulation changed after setup")
            return None
        try:
            precision = _precision_signature()
        except RuntimeError as error:
            self.retire(f"the CUDA precision policy cannot be read ({error})")
            return None
        reason = _head_breach(module.head, len(features))
        reason = reason or _feature_breach(features, precision)
        stages = (module.head, module.proto_fusion, module.protonet, module.semantic)
        if self.owner is not None and tuple(self.owner.children()) != stages:
            reason = "a prediction stage was replaced after recording"
        owner = self.owner if self.owner is not None else nn.ModuleList(stages)
        members = _members(owner)
        reason = reason or _parameter_breach(members[1], members[2], features[0].device)
        reason = reason or _stage_breach(owner, members[0])
        if reason is None and self.owner is None:
            reason = _storage_breach(owner, features[0].device)
        state = (_tensor_signature(members[1] + members[2]), precision) if reason is None else None
        if reason is None and self.attributes is not None and not self.attributes.matches(members[0]):
            reason = "prediction attributes or registration names changed after recording"
        if reason is None and self.owner is not None and state != self.state:
            reason = "prediction storage, topology, modes or precision changed after recording"
        if reason is not None:
            self.retire(reason)
            return None
        if self.owner is None:
            self.owner, self.state = owner, state
            self.attributes = _AttributePlan(members[0])
            self.tensors = members[1] + members[2]
        return self

    def _run(self, key: tuple[int, str], stages: tuple[nn.Module, ...], eager: _Call, inputs: _Tensors) -> _Tensors:
        """Record one group once; preserve native arithmetic on unsupported image signatures."""
        with torch.autocast(
            "cuda", enabled=torch.is_autocast_enabled(), dtype=torch.get_autocast_dtype("cuda"), cache_enabled=False
        ):
            if any(
                (x.requires_grad and x.is_leaf)
                or x.retains_grad
                or x._backward_hooks
                or not (x.is_contiguous() or x.is_contiguous(memory_format=torch.channels_last))
                for x in inputs
            ):
                return eager(inputs)
            if key not in self.entries:
                adapter = _PredictionGroup(stages, eager)
                values = list(self.tensors)
                spans = _RecordedSpans(values, list(inputs))
                if spans.reached_by(list(inputs)):
                    return eager(inputs)
                entry = _RecordedGroup(adapter, inputs, spans)
                self.entries[key] = entry
            return self.entries[key].run(inputs)

    def stem(self, stage: nn.Module, feature: Tensor) -> Tensor:
        """Keep each box/class contribution separate at the shared-feature gradient sum."""
        return self._run((id(stage), "stem"), (stage,), lambda inputs: (stage(inputs[0]),), (feature,))[0]

    def coefficients(self, branch: _DetectionBranch, features: tuple[Tensor, Tensor, Tensor]) -> Tensor | None:
        """Keep unused mask predictions disconnected from the optimizer, as in the native branch."""
        assert branch.coeff_stems is not None  # runner() certified the segmentation topology via _head_breach

        def eager(inputs: _Tensors) -> _Tensors:
            result = branch._coefficients(cast("tuple[Tensor, Tensor, Tensor]", inputs))
            assert result is not None  # _coefficients returns None only without coeff_stems, refused by _head_breach
            return (result,)

        return self._run((id(branch), "coeff"), tuple(branch.coeff_stems), eager, features)[0]

    def mask(self, stage: nn.Module, feature: Tensor) -> Tensor:
        """Record prototypes and semantic logits independently, preserving an unused prototype's None gradients."""
        return self._run((id(stage), "mask"), (stage,), lambda inputs: (stage(inputs[0]),), (feature,))[0]


class _SegmentHeadCaptureLifecycle(Callback):
    """Keep transient prediction recordings inside one fit, including skipped and failed steps."""

    @staticmethod
    def _holder(module: LightningModule) -> _SegmentHeadCapture | None:
        holder = getattr(module, "_segment_head_capture", None)
        return holder if isinstance(holder, _SegmentHeadCapture) else None

    @classmethod
    def _clear(cls, module: LightningModule) -> None:
        holder = cls._holder(module)
        if holder is not None:
            holder.clear()

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Begin the fit without a graph from an earlier model or trainer."""
        self._clear(pl_module)

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: object, batch_idx: int) -> None:
        """A new batch releases holds whose previous training step skipped backward."""
        holder = self._holder(pl_module)
        for entry in holder.entries.values() if holder is not None else ():
            entry.pending_backward = False

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Drain and release recordings at normal fit completion; a failing drain is raised."""
        self._clear(pl_module)

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        """Release recordings when the trainer fails, without replacing its exception or skipping teardown.

        A failed CUDA step usually leaves the device in an error state that the drain raises again.
        Raising it here would hide the trainer's own exception and skip the trainer's teardown, so a
        drain failure becomes a warning; the holder is already empty by then.
        """
        try:
            self._clear(pl_module)
        except RuntimeError as error:
            # Warning filters must not replace the trainer's original exception.
            with contextlib.suppress(UserWarning):
                warnings.warn(
                    f"capture_segment_heads=True could not drain its recordings after the fit failed ({error})",
                    UserWarning,
                    stacklevel=2,
                )
