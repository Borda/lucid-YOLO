# SPDX-License-Identifier: Apache-2.0
"""Checkpoint writing and protocol stubs shared by the ``lucid-*`` command tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from lucid_yolo.ptl.module import DetectionLitModule


def save_lightning_checkpoint(module: DetectionLitModule, path: Path) -> Path:
    """Save ``module`` in the Lightning checkpoint layout that ``load_eval_module`` reads.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from lucid_yolo.ptl.module import DetectionLitModule
        >>> tiny = DetectionLitModule(depth=0.34, width=0.25, max_channels=64, num_classes=2)
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     save_lightning_checkpoint(tiny, Path(tmp) / "tiny.ckpt").is_file()
        True
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


def kwargs_recorder(sink: dict[str, Any]) -> Callable[..., int]:
    """Build an evaluation-protocol stub that stores its keyword arguments in ``sink`` and returns ``0``.

    Examples:
        >>> seen: dict[str, Any] = {}
        >>> kwargs_recorder(seen)(None, {}, img_size=512)
        0
        >>> seen
        {'img_size': 512}
    """

    def stub(_module: object, _info: object, **kwargs: Any) -> int:
        sink.update(kwargs)
        return 0

    return stub
