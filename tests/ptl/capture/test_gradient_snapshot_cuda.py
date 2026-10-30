# SPDX-License-Identifier: Apache-2.0
"""Native CUDA closures discard snapshots without changing training state."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any
from unittest.mock import patch

import pytest
import torch
from markers import requires_cuda, skip_unless_native_zeroing
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.plugins.precision import MixedPrecision
from torch.utils.data import DataLoader

from lucid_yolo.data.targets import Targets
from lucid_yolo.ptl import _graph_replay, network_capture
from lucid_yolo.ptl._graph_replay import BF16_REPLAY_SUPPORTED
from lucid_yolo.ptl.module import DetectionLitModule


@requires_cuda
@skip_unless_native_zeroing
@pytest.mark.usefixtures("deterministic_kernels_pinning_cudnn_benchmark")
@pytest.mark.parametrize(
    "precision",
    [
        "32-true",
        "16-mixed",
        pytest.param(
            "bf16-mixed",
            marks=pytest.mark.skipif(not BF16_REPLAY_SUPPORTED, reason="BF16 recording requires torch>=2.9"),
        ),
    ],
)
def test_cuda_fit_omits_snapshots_with_identical_state(monkeypatch: pytest.MonkeyPatch, precision: str) -> None:
    """Actual replay removes snapshots and preserves weights, buffers and momentum bytes."""
    states = []
    counts = []
    raw_gradients = []
    original = _graph_replay.copy_out_live_gradients
    for enabled in (False, True):
        torch.manual_seed(41)
        module = DetectionLitModule(
            0.5,
            0.25,
            64,
            2,
            capture_network=True,
            capture_updates=True,
            capture_loss=True,
            lr=0.001,
            warmup_epochs=0,
        )
        images = torch.rand(2, 3, 64, 64)
        targets = [
            Targets(boxes=torch.zeros(0, 4), labels=torch.zeros(0, dtype=torch.long)),
            Targets(boxes=torch.tensor([[8.0, 6.0, 48.0, 52.0]]), labels=torch.tensor([1])),
        ]
        loader = DataLoader([(images, targets)] * 4, batch_size=None)
        observed = {"skipped": 0, "copied": 0}

        def copy(parameters, observed=observed):
            parameters = tuple(parameters)
            gradients = tuple(parameter.grad for parameter in parameters)
            original(parameters)
            for parameter, gradient in zip(parameters, gradients, strict=True):
                if gradient is not None:
                    observed["skipped" if parameter.grad is gradient else "copied"] += 1

        monkeypatch.setattr(network_capture, "copy_out_live_gradients", copy)
        snapshots = []

        class _RawGradients(Callback):
            def __init__(self, snapshots):
                self.snapshots = snapshots

            def on_before_optimizer_step(self, trainer, module, optimizer):
                self.snapshots.append(
                    {
                        name: None
                        if parameter.grad is None
                        else (
                            tuple(parameter.grad.shape),
                            parameter.grad.dtype,
                            parameter.grad.stride(),
                            parameter.grad.detach().cpu().contiguous().reshape(-1).view(torch.uint8).clone(),
                        )
                        for name, parameter in module.named_parameters()
                    }
                )

        precision_options: dict[str, Any] = {"precision": precision}
        if precision == "16-mixed":
            # Unit scale keeps the short fixture from skipping every update at the default scale.
            scaler = torch.amp.GradScaler("cuda", init_scale=1.0)
            precision_options = {"plugins": [MixedPrecision("16-mixed", "cuda", scaler=scaler)]}
        trainer = Trainer(
            accelerator="cuda",
            devices=1,
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            gradient_clip_val=10.0,
            callbacks=[_RawGradients(snapshots)],
            deterministic=True,
            **precision_options,
        )
        scope = nullcontext() if enabled else patch.object(module, "_parameters_cleared_after_forward", return_value=())
        with scope:
            trainer.fit(module, train_dataloaders=loader)
        tensors = list(module.state_dict().values())
        for state in trainer.optimizers[0].state.values():
            tensors.extend(value for value in state.values() if isinstance(value, torch.Tensor))
        states.append([tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).clone() for tensor in tensors])
        counts.append(observed)
        raw_gradients.append(snapshots)
    assert states[0] and len(states[0]) == len(states[1])
    assert all(torch.equal(left, right) for left, right in zip(*states, strict=True))
    assert counts[0]["copied"] > 0 and counts[0]["skipped"] == 0
    assert counts[1]["skipped"] > 0 and counts[1]["copied"] == 0

    assert len(raw_gradients[0]) == len(raw_gradients[1]) == 4
    for before, after in zip(*raw_gradients, strict=True):
        assert before.keys() == after.keys()
        assert any(value is not None for value in before.values())
        for name, left in before.items():
            right = after[name]
            assert (left is None) == (right is None)
            if left is not None:
                assert left[:3] == right[:3]
                assert torch.equal(left[3], right[3])
