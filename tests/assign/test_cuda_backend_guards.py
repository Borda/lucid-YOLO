# SPDX-License-Identifier: Apache-2.0
"""The optional CUDA assignment backend loads lazily and certifies the device passes it loads.

Every test here runs offline: import behaviour is observed in fresh interpreters, and the loaded-passes certificate
uses a stand-in module, so neither CUDA nor the optional compiler is needed.
"""

from __future__ import annotations

import sys
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
import torch
from isolated_interpreter_helpers import run_isolated

from lucid_yolo.assign import _cuda
from lucid_yolo.losses.dual_loss import DualBranchLoss
from lucid_yolo.ptl.loss_capture import _source_signature

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Names the device-pass module binds; the stand-in mirrors them.
_PASS_NAMES = (
    "_divide",
    "_maximum",
    "_minimum",
    "_iou_pass",
    "_metric_pass",
    "_inside_pass",
    "_first_maximum",
    "_conflicts_pass",
)


@pytest.fixture(autouse=True)
def fresh_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Start and end every test with empty launch decisions and an unloaded pass cache."""
    loader = _cuda._kernels
    monkeypatch.setattr(_cuda, "_STATE", _cuda._BackendState())
    loader.cache_clear()
    yield
    loader.cache_clear()


@pytest.fixture
def loaded_passes(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Serve a stand-in device-pass module from the import system, as a loaded compiler would."""
    module = ModuleType(_cuda._KERNELS_MODULE)
    for name in _PASS_NAMES:
        setattr(module, name, Mock(name=name))
    module._IS_TRITON_3_AVAILABLE = True  # type: ignore[attr-defined]
    module._PASSES = MappingProxyType({name: getattr(module, name) for name in _PASS_NAMES})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _cuda._KERNELS_MODULE, module)
    return module


class TestLazyDevicePasses:
    """The device passes and their compiler load on the first admitted launch, never on import."""

    def test_import_and_native_calls_leave_passes_unloaded(self) -> None:
        """Importing assignment and the loss recorder, then assigning and certifying on CPU, loads no pass.

        A CPU call with ``fused_assignment=True`` declines at the device gate before the loader runs, and a certificate
        for a loss requesting no fusion has no pass to check.
        """
        code = """
import sys
import torch
import lucid_yolo.assign
from lucid_yolo.assign.tal import TaskAlignedAssigner
from lucid_yolo.losses.dual_loss import DualBranchLoss
from lucid_yolo.ptl.loss_capture import _source_signature
boxes = torch.tensor([[[0., 0., 2., 2.]]])
result = TaskAlignedAssigner(1, fused_assignment=True)(
    torch.tensor([[[.8]]]), boxes.clone(), torch.tensor([[1., 1.]]),
    boxes, torch.tensor([[0]]), torch.tensor([[True]]),
)
assert result.fg_mask.tolist() == [[True]]
assert _source_signature(DualBranchLoss()) is not None
assert "lucid_yolo.assign._cuda_kernels" not in sys.modules, "device passes loaded eagerly"
print("PASSES_UNLOADED")
"""
        assert "PASSES_UNLOADED" in run_isolated(code)

    def test_broken_compiler_warns_once_and_stays_unavailable(self) -> None:
        """A compiler that imports but fails while the passes load warns once, then stays native.

        The stand-in compiler reports an unparsable version, which raises a non-``ImportError`` from the pass module; a
        second lookup reuses the decision. It is installed after the package import, which itself reads the real import
        system.
        """
        code = """
import sys
import types
import warnings
from lucid_yolo.assign import _cuda
compiler = types.ModuleType("triton")
compiler.__version__ = "not-a-version"
compiler.language = types.ModuleType("triton.language")
sys.modules["triton"] = compiler
sys.modules["triton.language"] = compiler.language
with warnings.catch_warnings(record=True) as record:
    warnings.simplefilter("always")
    loaded = _cuda._kernels(), _cuda._kernels()
assert loaded == (None, None), loaded
messages = [str(item.message) for item in record if "cannot load its device passes" in str(item.message)]
assert len(messages) == 1, [str(item.message) for item in record]
assert "InvalidVersion" in messages[0], messages
print("BROKEN_COMPILER_ONE_WARNING")
"""
        assert "BROKEN_COMPILER_ONE_WARNING" in run_isolated(code)

    @pytest.mark.parametrize("missing", ["triton", "lucid_yolo.assign._cuda_kernels"])
    def test_missing_module_stays_unavailable_without_warning(self, missing: str) -> None:
        """An absent compiler or pass module leaves the backend unavailable and raises no warning.

        This is an install without the ``cuda-assignment`` extra: nothing is broken, so nothing is reported.
        """
        code = f"""
import sys
import warnings
sys.modules[{missing!r}] = None
from lucid_yolo.assign import _cuda
with warnings.catch_warnings():
    warnings.simplefilter("error")
    assert _cuda._kernels() is None
print("MISSING_MODULE_SILENT")
"""
        assert "MISSING_MODULE_SILENT" in run_isolated(code)


class TestLoadedPassCertificate:
    """A fused recording certifies the device passes that loaded, against their module's own binding."""

    def test_loaded_passes_certify_fused_loss(self, loaded_passes: ModuleType) -> None:
        """Unchanged passes leave a fused built-in loss certifiable after a lazy load."""
        assert _source_signature(DualBranchLoss(fused_assignment=True)) is not None

    @pytest.mark.parametrize("name", ["_iou_pass", "_divide"])
    def test_rebound_pass_invalidates_fused_loss(
        self, monkeypatch: pytest.MonkeyPatch, loaded_passes: ModuleType, name: str
    ) -> None:
        """Rebinding a launched pass, or one another pass inlines by global name, refuses the recording."""
        monkeypatch.setattr(loaded_passes, name, Mock())

        assert _source_signature(DualBranchLoss(fused_assignment=True)) is None

    def test_rebound_pass_leaves_unfused_loss_certifiable(
        self, monkeypatch: pytest.MonkeyPatch, loaded_passes: ModuleType
    ) -> None:
        """A loss requesting no fusion never launches a pass, so a rebinding does not concern it."""
        monkeypatch.setattr(loaded_passes, "_iou_pass", Mock())

        assert _source_signature(DualBranchLoss()) is not None

    @patch.object(_cuda, "_kernels", new=Mock(wraps=_cuda._kernels))
    def test_replaced_loader_invalidates_fused_loss(self) -> None:
        """The loader decides which passes run, so replacing it refuses the recording."""
        assert _source_signature(DualBranchLoss(fused_assignment=True)) is None


class TestFirstLaunchLatch:
    """Only a failing first launch keeps its family native; workload and device errors propagate."""

    @pytest.fixture(autouse=True)
    def eager_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Report an uncaptured stream, which a CPU-only build cannot answer itself."""
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    @patch.object(torch.cuda, "synchronize")
    def test_first_launch_publishes_after_fence(self, fence: Mock) -> None:
        """A successful first launch is fenced on its device, then marked ready and verified."""
        device = torch.device("cuda", 0)
        result = torch.tensor([3.0])

        assert _cuda._run("metric", (1,), device, lambda: result) is result
        fence.assert_called_once_with(device)
        assert _cuda._STATE.ready == {("metric", device, 1)}
        assert _cuda._STATE.verified == {("metric", device)}

    @patch.object(torch.cuda, "synchronize")
    def test_out_of_memory_propagates_unlatched(self, fence: Mock) -> None:
        """An allocation failure in the first launch reaches the caller and disables nothing.

        Out-of-memory belongs to the workload: latching it would keep the family native for the rest of the process
        after one oversized batch.
        """
        produce = Mock(side_effect=torch.cuda.OutOfMemoryError("CUDA out of memory"))

        with pytest.raises(torch.cuda.OutOfMemoryError):
            _cuda._run("metric", (1,), torch.device("cuda", 0), produce)

        fence.assert_not_called()
        assert not _cuda._STATE.unavailable and not _cuda._STATE.ready and not _cuda._STATE.verified

    @patch.object(torch.cuda, "synchronize", new=Mock(side_effect=RuntimeError("device-side assert")))
    def test_fence_error_propagates_unlatched(self) -> None:
        """A device error surfacing at the completion fence reaches the caller and publishes nothing.

        Such an error is sticky on the device, so a native rerun could not recover it.
        """
        with pytest.raises(RuntimeError, match="device-side assert"):
            _cuda._run("metric", (1,), torch.device("cuda", 0), lambda: torch.tensor([3.0]))

        assert not _cuda._STATE.unavailable and not _cuda._STATE.ready and not _cuda._STATE.verified
