# SPDX-License-Identifier: Apache-2.0
"""The closed set of supervision tasks, and the per-task defaults the library reads off it.

Four tasks — ``detect``, ``segment``, ``obb`` and ``keypoints`` — and until this module
each consumer spelled them out for itself: the module's accepted names, the evaluation
command's two default tables, the inference entry points' refusals, the drawing tables.
Each copy was right when written, and one went wrong without being touched: when the
fourth task arrived, three inference refusals went on naming only the siblings of their
own era. :class:`Task` is the one declaration, and :class:`TaskSpec` holds the per-task
numbers those tables restated, so a consumer derives its table rather than writing it.

**The enum is for declaration, membership and comparison; the string stays at every
boundary.** A constructor argument, a CLI flag, a config YAML, a JSON report and a saved
checkpoint all carry the plain spelling, and that last one is not a preference: a
checkpoint is read under ``torch.load(weights_only=True)``, whose restricted unpickler
refuses any class it was not told about, so a :class:`Task` member pickled into
``hparams["task"]`` would make the file unloadable. :class:`Task` is a
:class:`~enum.StrEnum` so that the two meet without a conversion at each comparison —
``Task.OBB == "obb"`` holds, a plain string looks up a ``Task``-keyed table, and
``str()`` and an f-string give the plain spelling. ``repr()`` does not, so a message that
lists tasks formats ``.value`` (or :data:`TASK_NAMES`) rather than the members.

A **leaf**: nothing here imports from :mod:`lucid_yolo`, and nothing beyond the standard
library, so any module — :mod:`lucid_yolo.models` included, which sits below
:mod:`lucid_yolo.decode` and cannot import it — can read the registry without an import
cycle or a torch import.

Provenance: R1 sec. 4.4, R18 sec. 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["TASK_NAMES", "Task", "TaskSpec"]

#: Letterbox side of the COCO tiers — detection, segmentation and keypoints (R1 sec. 4.4).
#: :data:`~lucid_yolo.decode.common.COCO_IMG_SIZE` states the same number for the decode
#: and scoring defaults, which this leaf cannot import; a test pins the two equal.
_COCO_IMG_SIZE = 640

#: Per-image detection cap shared by model ranking and decoding (R3 sec. 4, A9).
_MAX_DETECTIONS = 300

#: Letterbox side of the oriented tier, which trains on 1024 px DOTA tiles (R18 sec. 4),
#: so an oriented checkpoint has never seen 640.
_OBB_IMG_SIZE = 1024

#: Images per evaluation forward pass on a COCO tier.
_COCO_EVAL_BATCH_SIZE = 32

#: Tiles per evaluation forward pass on the oriented tier. A 1024 px tile is 2.5x the
#: pixels of a 640 px image, so the default is smaller for the same memory.
_OBB_EVAL_BATCH_SIZE = 8


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """The per-task defaults a consumer reads instead of restating.

    Only facts more than one table already carried live here; what a task *does* — which
    loss terms, which decoder, which evaluation protocol — stays with the code that does
    it. Frozen, because a spec is shared by every reader in the process.

    Attributes:
        img_size: Letterbox side the task's tier trains at, and so the default canvas of
            its evaluation, prediction, drawing and export: 640 for the COCO tiers, 1024
            for the oriented one.
        eval_batch_size: Images (or, for ``obb``, tiles) per forward pass when an
            evaluation is not told otherwise.

    Examples:
        ```pycon
        >>> Task.OBB.spec
        TaskSpec(img_size=1024, eval_batch_size=8)

        ```
    """

    img_size: int
    eval_batch_size: int


class Task(StrEnum):
    """One supervision task the module trains and the commands serve.

    Member order is the registry order: :data:`TASK_NAMES` and every table derived by
    iterating :class:`Task` list the tasks in it.

    Attributes:
        DETECT: Axis-aligned boxes, the dual detection objective alone.
        SEGMENT: Boxes and their instance masks.
        OBB: Rotated boxes, a centre, two extents and an angle.
        KEYPOINTS: Boxes and a point set per box.

    Examples:
        ```pycon
        >>> Task("obb") is Task.OBB, Task.OBB == "obb", f"{Task.OBB}"
        (True, True, 'obb')
        >>> Task.DETECT.spec.img_size, Task.OBB.spec.img_size
        (640, 1024)

        ```
    """

    DETECT = "detect"
    SEGMENT = "segment"
    OBB = "obb"
    KEYPOINTS = "keypoints"

    @property
    def spec(self) -> TaskSpec:
        """Return this task's :class:`TaskSpec`."""
        return _TASK_SPECS[self]


#: Each task's defaults. Read-only, so no consumer can re-tune another's default in place.
_TASK_SPECS: Mapping[Task, TaskSpec] = MappingProxyType(
    {
        Task.DETECT: TaskSpec(img_size=_COCO_IMG_SIZE, eval_batch_size=_COCO_EVAL_BATCH_SIZE),
        Task.SEGMENT: TaskSpec(img_size=_COCO_IMG_SIZE, eval_batch_size=_COCO_EVAL_BATCH_SIZE),
        Task.OBB: TaskSpec(img_size=_OBB_IMG_SIZE, eval_batch_size=_OBB_EVAL_BATCH_SIZE),
        Task.KEYPOINTS: TaskSpec(img_size=_COCO_IMG_SIZE, eval_batch_size=_COCO_EVAL_BATCH_SIZE),
    }
)

#: The four spellings as plain strings, in registry order — what a boundary validates
#: against or a message lists, where an enum member's ``repr`` would read as noise.
TASK_NAMES: tuple[str, ...] = tuple(task.value for task in Task)
