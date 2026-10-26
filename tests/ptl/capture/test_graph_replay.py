# SPDX-License-Identifier: Apache-2.0
"""The shared replay primitives drain before release and keep the fit lifecycle safe, without a GPU.

CUDA graph execution is covered by the ``gpu`` tests of each holder. These cases check the
host-side contracts the network, segmentation-head and loss holders now share: what a holder
drains when it releases a recording, that the drain runs before the recording is dropped,
and that the lifecycle callback neither fails on a module without its holder nor lets a drain
failure replace the trainer's own exception. CUDA initialization and the device synchronize are the
only boundaries replaced.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from capture_access import pose_stems
from torch import nn

from lucid_yolo.ptl import network_capture
from lucid_yolo.ptl._graph_replay import ReplayLifecycle, drain_device, release_graphs
from lucid_yolo.ptl.loss_capture import _LossCapture, _LossCaptureLifecycle
from lucid_yolo.ptl.network_capture import _NetworkCapture, _NetworkCaptureLifecycle
from lucid_yolo.ptl.segment_head_capture import _SegmentHeadCaptureLifecycle

#: The CUDA device every stand-in recording below claims to have been made on.
_DEVICE = torch.device("cuda", 0)

#: Builds a holder that owns one stand-in recording, for the holders that drain their whole device.
_Factory = Callable[[nn.Module], "_NetworkCapture | _LossCapture"]


def _graphed() -> nn.Module:
    """Stand in for a recorded module: the recorder installs its graphed forward as an instance attribute.

    Examples:
        >>> "forward" in vars(_graphed())
        True
    """
    module = nn.Identity()
    module.forward = lambda value: value
    return module


def _network_holder(recording: nn.Module) -> _NetworkCapture:
    """Hold ``recording`` as a network recording made on the first CUDA device.

    Examples:
        >>> _network_holder(_graphed())._device
        device(type='cuda', index=0)
    """
    holder = _NetworkCapture()
    holder._network, holder._device = recording, _DEVICE
    return holder


def _loss_holder(recording: nn.Module) -> _LossCapture:
    """Hold ``recording`` as a loss recording made on the first CUDA device.

    Examples:
        >>> _loss_holder(_graphed())._device
        device(type='cuda', index=0)
    """
    holder = _LossCapture()
    holder._objective, holder._device = recording, _DEVICE
    return holder


@pytest.fixture
def cuda_sync(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Report CUDA as initialized and stand in for the device synchronize, so a CPU run reaches the drain.

    Examples:
        >>> callable(cuda_sync)
        True
    """
    synchronize = MagicMock()
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    return synchronize


class TestReleaseGraphs:
    """Each distinct stream is drained once, and only an initialized CUDA recording drains its device."""

    def test_drains_each_distinct_stream_once_in_order(self) -> None:
        """Groups sharing a stream fence it once, in the order the streams were first seen.

        The segmentation holder passes one stream per recorded group, so a shared stream must not
        be synchronized once per group.
        """
        order: list[str] = []
        first = MagicMock(synchronize=MagicMock(side_effect=lambda: order.append("first")))
        second = MagicMock(synchronize=MagicMock(side_effect=lambda: order.append("second")))
        release_graphs([second, first, second, first])
        assert order == ["second", "first"]

    @pytest.mark.parametrize(
        ("device", "initialized"),
        [
            pytest.param(None, True, id="no-recording"),
            pytest.param(torch.device("cpu"), True, id="cpu-device"),
            pytest.param(_DEVICE, False, id="cuda-uninitialized"),
        ],
    )
    def test_never_touches_cuda_without_initialized_cuda_recording(
        self, device: torch.device | None, initialized: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No recording, a CPU device or an uninitialized runtime never reaches the CUDA synchronize.

        This is what keeps ``clear`` free on a CPU run and in ``__setstate__``: the synchronize is
        replaced by a failure, so reaching it at all fails the test.
        """
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized)
        monkeypatch.setattr(torch.cuda, "synchronize", MagicMock(side_effect=AssertionError("CUDA touched")))
        drain_device(device)

    def test_synchronizes_whole_recorded_device(self, cuda_sync: MagicMock) -> None:
        """An initialized runtime synchronizes the device the recording was made on, every stream included.

        A replay launched from a side stream must have finished too before its graph is dropped, so
        the drain is device-wide rather than the current stream's.
        """
        drain_device(_DEVICE)
        cuda_sync.assert_called_once_with(_DEVICE)


@pytest.mark.parametrize("make", [pytest.param(_network_holder, id="network"), pytest.param(_loss_holder, id="loss")])
class TestReleaseOnClear:
    """The network and loss holders drain their recorded device before dropping the recording."""

    def test_drains_before_recording_is_dropped(self, make: _Factory, cuda_sync: MagicMock) -> None:
        """The graphed forward is still installed while the device drains and gone once ``clear`` returns.

        Dropping a graph's private pool while a replay on it may still be queued is the hazard the
        drain rules out (D6), so the order is the contract, not merely that both happen.
        """
        recording = _graphed()
        installed: list[bool] = []
        cuda_sync.side_effect = lambda device: installed.append("forward" in vars(recording))
        holder = make(recording)
        holder.clear()
        assert installed == [True]
        assert "forward" not in vars(recording)
        assert holder._device is None

    def test_clear_without_recording_never_drains(self, make: _Factory, cuda_sync: MagicMock) -> None:
        """A holder that recorded nothing releases nothing, so the fit boundaries cost no synchronize."""
        holder = make(_graphed())
        holder.clear()
        cuda_sync.reset_mock()
        holder.clear()
        cuda_sync.assert_not_called()


@pytest.mark.parametrize(
    ("stem_error", "drain_error", "raised"),
    [
        pytest.param(None, None, None, id="both-ok"),
        pytest.param(None, RuntimeError("drain"), "drain", id="drain-fails"),
        pytest.param(RuntimeError("stem"), None, "stem", id="stem-fails"),
        pytest.param(RuntimeError("stem"), RuntimeError("drain"), "stem", id="both-fail"),
    ],
)
def test_network_clear_releases_everything_and_raises_first_failure(
    stem_error: RuntimeError | None,
    drain_error: RuntimeError | None,
    raised: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every stem holder is cleared, the device drains and the forward is dropped whichever of them fails.

    A pose network holds one stem holder per keypoint stem, and after a device error each of their
    drains, and the holder's own, may fail. None of them may leave a graphed forward installed, and
    the first failure is the one raised, so a later drain error never masks the stem's.
    """
    first, second = MagicMock(spec=_NetworkCapture), MagicMock(spec=_NetworkCapture)
    first.clear.side_effect = stem_error
    drain = MagicMock(side_effect=drain_error)
    monkeypatch.setattr(network_capture, "drain_device", drain)
    recording = _graphed()
    holder = _network_holder(recording)
    holder._stems = {1: first, 2: second}
    with pytest.raises(RuntimeError, match=rf"^{raised}$") if raised else nullcontext():
        holder.clear()
    second.clear.assert_called_once_with()
    drain.assert_called_once_with(_DEVICE)
    assert pose_stems(holder) == {} and "forward" not in vars(recording)


@pytest.mark.parametrize(
    ("lifecycle", "make"),
    [
        pytest.param(_NetworkCaptureLifecycle, _network_holder, id="network"),
        pytest.param(_LossCaptureLifecycle, _loss_holder, id="loss"),
    ],
)
class TestFailingDrain:
    """A drain failure is raised after a completed fit and becomes a warning after a failed one."""

    @pytest.fixture
    def failing(self, cuda_sync: MagicMock) -> MagicMock:
        """Make the stand-in device synchronize fail, as after a CUDA device error.

        Examples:
            >>> callable(TestFailingDrain.failing)
            True
        """
        cuda_sync.side_effect = RuntimeError("device-side assert triggered")
        return cuda_sync

    def test_raises_at_fit_end_after_reset(
        self, lifecycle: type[ReplayLifecycle], make: _Factory, failing: MagicMock
    ) -> None:
        """Nothing else is in flight at normal fit completion, so a failing drain is the error to report.

        The reset comes before the drain, so the holder is already empty and its recording released.
        """
        recording = _graphed()
        holder = make(recording)
        with pytest.raises(RuntimeError, match="device-side assert"):
            lifecycle().on_fit_end(None, SimpleNamespace(**{lifecycle.attribute: holder}))
        failing.assert_called_once_with(_DEVICE)
        assert holder._device is None and "forward" not in vars(recording)

    def test_warns_on_failed_fit_without_replacing_exception(
        self, lifecycle: type[ReplayLifecycle], make: _Factory, failing: MagicMock
    ) -> None:
        """After a failed fit the drain failure is a warning naming the holder, so the trainer's exception propagates.

        A failed CUDA step usually leaves the device erroring on every later synchronize; raising that
        from ``on_exception`` would replace the original exception and skip the trainer's teardown.
        """
        recording = _graphed()
        holder = make(recording)
        message = rf"^{lifecycle.flag}=True could not drain its recordings after the fit failed \(device-side assert"
        with pytest.warns(UserWarning, match=message):
            lifecycle().on_exception(
                None,
                SimpleNamespace(**{lifecycle.attribute: holder}),
                RuntimeError("step"),
            )
        failing.assert_called_once_with(_DEVICE)
        assert holder._device is None and "forward" not in vars(recording)


@pytest.mark.parametrize(
    "lifecycle",
    [
        pytest.param(_NetworkCaptureLifecycle, id="network"),
        pytest.param(_SegmentHeadCaptureLifecycle, id="segment"),
        pytest.param(_LossCaptureLifecycle, id="loss"),
    ],
)
@pytest.mark.parametrize(
    "module",
    [
        pytest.param(SimpleNamespace(), id="missing"),
        pytest.param(
            SimpleNamespace(_network_capture=object(), _segment_head_capture=object(), _loss_capture=object()),
            id="foreign",
        ),
    ],
)
def test_module_without_holder_makes_every_hook_no_op(
    lifecycle: type[ReplayLifecycle], module: SimpleNamespace
) -> None:
    """A trainer reused on another module class does not fail in any replay lifecycle, the loss one included.

    The loss lifecycle used to read its holder without a check, so a module lacking it raised from
    every hook. A foreign object would raise on ``clear`` or ``start_batch`` were it ever called.
    """
    callback = lifecycle()
    callback.on_fit_start(None, module)
    callback.on_train_batch_start(None, module, None, 0)
    callback.on_fit_end(None, module)
    callback.on_exception(None, module, RuntimeError("abort"))
