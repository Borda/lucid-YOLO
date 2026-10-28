# SPDX-License-Identifier: Apache-2.0
"""lucid-yolo: independent PyTorch Lightning reproduction of the YOLO26 methods.

Implements the detection, instance-segmentation, and oriented-detection methods
published in arXiv:2606.03748, plus a keypoint task whose loss and evaluation
protocol come from arXiv:2107.11291 instead. Every method input is a paper, its
cited primary literature, or a recorded assumption; a small set of permissively
licensed implementations is admitted for diagnostic reading only, never copied
(D13 / ADR-004). See docs/PROVENANCE.md and docs/ASSUMPTIONS.md for the
clean-room record.

The names a Python caller needs after training are importable from here: load a
checkpoint (:func:`load_eval_module`, :func:`pick_device`), predict on one image file or
tensor (:func:`predict_image`, :func:`predict_segmentation`, :func:`predict_oriented`,
:func:`predict_keypoints`), and score it on COCO val2017 (:func:`score_coco`). They are
resolved on first access rather than at import: each pulls in torch and Lightning, and
``import lucid_yolo`` -- which reading ``__version__`` needs -- stays free of both.

Examples:
    ```pycon
    >>> import lucid_yolo
    >>> lucid_yolo.score_coco.__module__
    'lucid_yolo.eval.coco_score'
    >>> "predict_image" in dir(lucid_yolo)
    True

    ```
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lucid_yolo.eval.checkpoint import load_eval_module, pick_device
    from lucid_yolo.eval.coco_score import CocoScore, score_coco
    from lucid_yolo.predict import (
        KeypointPrediction,
        SegmentedPrediction,
        predict_image,
        predict_keypoints,
        predict_oriented,
        predict_segmentation,
    )

#: Single source of truth for the package version (pyproject reads this attribute
#: statically). Pre-release wheels for a tier run carry a PEP 440 ``.devN`` suffix,
#: which sorts *below* the release it leads to, so ``pip install lucid-yolo`` never
#: resolves to one by accident and a run's artifacts record exactly which build trained
#: them. ``N`` counts from zero. Bump it for every wheel that leaves this machine, even a rebuild of the same
#: tree: a 15-hour run once trained on a stale wheel because the intended version was
#: never published and pip silently resolved the newest that existed.
__version__ = "0.8.0"

__all__ = [
    "CocoScore",
    "KeypointPrediction",
    "SegmentedPrediction",
    "__version__",
    "load_eval_module",
    "pick_device",
    "predict_image",
    "predict_keypoints",
    "predict_oriented",
    "predict_segmentation",
    "score_coco",
]

#: Each re-exported name and the module defining it, imported on first access (PEP 562).
_LAZY_EXPORTS: dict[str, str] = {
    "CocoScore": "lucid_yolo.eval.coco_score",
    "KeypointPrediction": "lucid_yolo.predict",
    "SegmentedPrediction": "lucid_yolo.predict",
    "load_eval_module": "lucid_yolo.eval.checkpoint",
    "pick_device": "lucid_yolo.eval.checkpoint",
    "predict_image": "lucid_yolo.predict",
    "predict_keypoints": "lucid_yolo.predict",
    "predict_oriented": "lucid_yolo.predict",
    "predict_segmentation": "lucid_yolo.predict",
    "score_coco": "lucid_yolo.eval.coco_score",
}


def __getattr__(name: str) -> Any:
    """Import a re-exported name from its defining module on first access, then cache it.

    Args:
        name: The attribute requested from the package.

    Returns:
        The defining module's object of that name.

    Raises:
        AttributeError: If ``name`` is not one of the package's re-exports.

    Examples:
        ```pycon
        >>> import lucid_yolo
        >>> lucid_yolo.no_such_name
        Traceback (most recent call last):
            ...
        AttributeError: module 'lucid_yolo' has no attribute 'no_such_name'

        ```
    """
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the package's attributes, the not-yet-imported re-exports included.

    Examples:
        ```pycon
        >>> import lucid_yolo
        >>> {"load_eval_module", "score_coco"} <= set(dir(lucid_yolo))
        True

        ```
    """
    return sorted({*globals(), *_LAZY_EXPORTS})
