# SPDX-License-Identifier: Apache-2.0
"""``lucid-eval`` — score a checkpoint on the protocol its own task defines (WP-096).

One command, no task flag::

    lucid-eval --checkpoint runs/det.ckpt --data_root ~/data/coco2017
    lucid-eval --checkpoint runs/obb.ckpt --data_root /data/dota_tiles --split val

Which protocol runs is read from the checkpoint's ``task``: ``detect`` and ``segment``
take :mod:`lucid_yolo.eval.detect_eval` (dual-path COCO, with the twelve ``segm_``
statistics when the checkpoint has a mask branch), ``obb`` takes
:mod:`lucid_yolo.eval.rotated_eval` (per-tile rotated mAP). That is the rule the
detection evaluator already applied to masks — "driven by the checkpoint's own task, not
by a flag the caller has to remember" — one level up: a caller who names the wrong task
gets a wrong report, and a caller who names none cannot.

``img_size`` and ``batch_size`` follow the same source. Both stay unset until the task is
known, so an oriented run gets the tier's 1024 px crop side and a COCO run gets 640, and
neither caller has to remember which; an explicit value always wins.

Arguments that belong to one protocol are accepted and unused by the other (``masks`` is
detection-only; ``split`` and ``variant`` are oriented-only). Subcommands per task would
make the caller state what the checkpoint already knows, which is the property this
command exists to avoid.

Provenance: R1 Table 7, R1 sec. 4.4; R18 sec. 4. Assumptions: A9, A10, A37, A46, A47, A48.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jsonargparse import auto_cli

from lucid_yolo.eval import detect_eval, rotated_eval
from lucid_yolo.eval.checkpoint import load_eval_module

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["DEFAULT_BATCH_SIZE", "DEFAULT_IMG_SIZE", "evaluate", "main"]

#: Letterbox side used when ``img_size`` is not given, per checkpoint task. The oriented
#: tier trains on 1024 px crops (R18 sec. 4); COCO runs at 640 (R1 sec. 4.4).
DEFAULT_IMG_SIZE = {"obb": 1024, "detect": 640, "segment": 640}
#: Batch size used when ``batch_size`` is not given. A 1024 px tile is 2.5x the pixels of
#: a 640 px image, so the oriented default is smaller for the same memory.
DEFAULT_BATCH_SIZE = {"obb": 8, "detect": 32, "segment": 32}


def evaluate(
    checkpoint: Path,
    data_root: Path,
    ema: bool = True,
    masks: bool = True,
    split: str = "val",
    variant: str = "n",
    batch_size: int | None = None,
    img_size: int | None = None,
    device: str = "auto",
    limit: int = 0,
    output: Path | None = None,
) -> int:
    """Score a checkpoint on the acceptance protocol its task defines.

    Args:
        checkpoint: Lightning ``.ckpt`` to evaluate.
        data_root: COCO root, or the tiled layout root for an oriented checkpoint.
        ema: Evaluate the EMA shadow stored in the checkpoint rather than raw weights.
        masks: detect/segment only; set false to score boxes from a segmentation
            checkpoint.
        split: obb only; split of the tiled layout to score.
        variant: obb only; scale letter of the trained model.
        batch_size: Images or tiles per forward pass. Defaults per task.
        img_size: Letterbox side. Defaults per task.
        device: ``auto``, ``cpu``, ``mps`` or ``cuda``.
        limit: Score only the first N images or tiles; ``0`` scores all.
        output: Write the JSON report here.

    Returns:
        The protocol's exit code.

    Examples:
        >>> evaluate(Path("/nonexistent.ckpt"), Path("/data"))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        FileNotFoundError: ...
    """
    module, info = load_eval_module(checkpoint, use_ema=ema)
    task = str(module.task)
    info["task"] = task
    resolved_img_size = DEFAULT_IMG_SIZE.get(task, 640) if img_size is None else img_size
    resolved_batch_size = DEFAULT_BATCH_SIZE.get(task, 32) if batch_size is None else batch_size
    if task == "obb":
        return rotated_eval.run(
            module,
            info,
            data_root=data_root,
            split=split,
            variant=variant,
            img_size=resolved_img_size,
            batch_size=resolved_batch_size,
            device_name=device,
            limit=limit,
            output=output,
        )
    return detect_eval.run(
        module,
        info,
        data_root=data_root,
        img_size=resolved_img_size,
        batch_size=resolved_batch_size,
        masks=masks,
        device_name=device,
        limit=limit,
        output=output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``lucid-eval`` from the command line.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        The protocol's exit code.

    Examples:
        >>> main(["--help"])  # doctest: +SKIP
        0
    """
    return int(auto_cli(evaluate, args=None if argv is None else list(argv), as_positional=False))


if __name__ == "__main__":  # pragma: no cover - `python -m lucid_yolo.cli.eval`
    raise SystemExit(main())
