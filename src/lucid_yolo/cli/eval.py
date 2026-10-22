# SPDX-License-Identifier: Apache-2.0
"""``lucid-eval`` — score a checkpoint on the protocol its own task defines (WP-096).

One command, no task flag::

    lucid-eval --checkpoint runs/det.ckpt --data_root ~/data/coco2017
    lucid-eval --checkpoint runs/obb.ckpt --data_root /data/dota_tiles --split val

Which protocol runs is read from the checkpoint's ``task``: ``detect`` and ``segment``
take :mod:`lucid_yolo.eval.detect_eval` (dual-path COCO, with the twelve ``segm_``
statistics when the checkpoint has a mask branch), ``obb`` takes
:mod:`lucid_yolo.eval.rotated_eval` (rotated mAP, reported per tile **and** per whole
source image once the tiles are merged, WP-107), and ``keypoints`` takes
:mod:`lucid_yolo.eval.pose_eval` (box AP against the person-keypoints ground truth plus
the ten ``oks_`` statistics, WP-134). That is the rule the
detection evaluator already applied to masks — "driven by the checkpoint's own task, not
by a flag the caller has to remember" — one level up: a caller who names the wrong task
gets a wrong report, and a caller who names none cannot.

``img_size`` and ``batch_size`` follow the same source. Both stay unset until the task is
known, so an oriented run gets the tier's 1024 px crop side and a COCO run gets 640, and
neither caller has to remember which; an explicit value always wins.

Explicit protocol-specific arguments are refused when the checkpoint's task does not
use them. Unset arguments retain the protocol's defaults. COCO annotation and image
paths can be overridden independently without changing the default val2017 layout.

Provenance: R1 Table 7, R1 sec. 4.4; R18 sec. 4. Assumptions: A9, A10, A37, A46, A47, A48.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jsonargparse import auto_cli

from lucid_yolo.assign.grid import require_grid_side
from lucid_yolo.eval import detect_eval, pose_eval, rotated_eval
from lucid_yolo.eval.checkpoint import load_eval_module
from lucid_yolo.eval.coco_eval import hotcoco_available
from lucid_yolo.tasks import Task
from lucid_yolo.validate import require_at_least, require_one_of

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["DEFAULT_BATCH_SIZE", "DEFAULT_IMG_SIZE", "evaluate", "main"]

#: Row order of the two tables below: the oriented task first, then every other task in
#: registry order. Kept rather than tidied because the order is visible: ``lucid-eval``,
#: ``lucid-predict`` and ``lucid-export`` list these keys, in this order, in the refusal
#: they raise for an unknown task.
_TABLE_ORDER: tuple[Task, ...] = (Task.OBB, *(task for task in Task if task is not Task.OBB))

#: Letterbox side used when ``img_size`` is not given, per checkpoint task, read off
#: :attr:`~lucid_yolo.tasks.TaskSpec.img_size`. The oriented tier trains on 1024 px crops
#: (R18 sec. 4); COCO runs at 640 (R1 sec. 4.4).
DEFAULT_IMG_SIZE: dict[str, int] = {task.value: task.spec.img_size for task in _TABLE_ORDER}
#: Batch size used when ``batch_size`` is not given, read off
#: :attr:`~lucid_yolo.tasks.TaskSpec.eval_batch_size`. A 1024 px tile is 2.5x the pixels
#: of a 640 px image, so the oriented default is smaller for the same memory.
DEFAULT_BATCH_SIZE: dict[str, int] = {task.value: task.spec.eval_batch_size for task in _TABLE_ORDER}


def _resolve_eval_backend(requested: str) -> tuple[str, str | None]:
    """Resolve the box/segm scoring engine, per WP-138's own "safe to default" call.

    hotcoco ships prebuilt wheels for this project's real target platforms
    (macOS/Linux/Windows, cp39-abi3), which is why ``"auto"`` — the default —
    picks it without hedging. The fallback exists for what R36 documented rather
    than what is expected here: no musllinux wheel, so an
    Alpine-style deploy falls back to a source build that needs a Rust
    toolchain this project has no way to guarantee. An explicit
    ``eval_backend="hotcoco"`` is a stated requirement, not a preference, and
    raises rather than silently substituting an engine the caller did not ask
    for — the same "explicit override wins, and is trusted" shape
    ``--data.layout`` already has (A63).

    Args:
        requested: ``"auto"``, ``"hotcoco"`` or ``"faster_coco_eval"``.

    Returns:
        ``(engine, fallback_reason)`` — ``fallback_reason`` is ``None`` unless
        ``"auto"`` fell back, in which case it is hotcoco's own probe failure,
        recorded in the report rather than only printed.

    Examples:
        ```pycon
        >>> _resolve_eval_backend("faster_coco_eval")
        ('faster_coco_eval', None)

        ```
    """
    if requested == "faster_coco_eval":
        return "faster_coco_eval", None
    available, reason = hotcoco_available()
    if requested == "hotcoco":
        if not available:
            raise RuntimeError(f"eval_backend='hotcoco' was requested explicitly but is not usable here: {reason}")
        return "hotcoco", None
    if requested != "auto":
        raise ValueError(f"eval_backend must be one of ('auto', 'hotcoco', 'faster_coco_eval'), got {requested!r}")
    return ("hotcoco", None) if available else ("faster_coco_eval", reason)


def _check_arguments(batch_size: int | None, img_size: int | None, limit: int) -> None:
    """Refuse the argument values the three protocols below would otherwise run on.

    One check at the single entry point all three pass through, rather than one per
    protocol: :mod:`~lucid_yolo.eval.detect_eval`, :mod:`~lucid_yolo.eval.pose_eval` and
    :mod:`~lucid_yolo.eval.rotated_eval` each read ``limit`` and ``batch_size`` in their
    own idiom, and three copies of one rule are three chances for one to be edited alone.
    What those idioms do with a value outside its domain is why this is not merely tidy.
    ``if limit:`` is true for ``-5``, so ``images[:-5]`` scores every image *but* the last
    five while the banner prints the truncated count as though it were the request; on the
    oriented path ``len(predictions) >= -5`` holds after the first batch, so the run breaks
    out immediately and reports a number from a handful of tiles; and
    ``math.ceil(len(images) / 0)`` raises from inside the progress-bar construction rather
    than at the flag that caused it.

    ``None`` is not a value here but "unset": both defaults are resolved from the
    checkpoint's own task once it is known, out of tables this module owns.

    Args:
        batch_size: The caller's ``batch_size``, or ``None`` to take the task default.
        img_size: The caller's ``img_size``, or ``None`` to take the task default.
        limit: The caller's ``limit``; ``0`` scores everything.

    Raises:
        ValueError: If ``limit`` is negative, if ``batch_size`` is below ``1``, or if
            ``img_size`` is not a positive multiple of every head stride.

    Examples:
        ```pycon
        >>> _check_arguments(32, 640, 0)  # a usable trio: returns nothing
        >>> _check_arguments(0, None, 0)
        Traceback (most recent call last):
            ...
        ValueError: batch_size must be a number >= 1; got 0

        ```
    """
    require_at_least("limit", limit, 0)
    if batch_size is not None:
        require_at_least("batch_size", batch_size, 1)
    if img_size is not None:
        require_grid_side("img_size", img_size)


def evaluate(  # noqa: PLR0913 - Explicit CLI options supply jsonargparse's flags and help.
    checkpoint: Path,
    data_root: Path,
    ema: bool = True,
    masks: bool | None = None,
    split: str | None = None,
    variant: str | None = None,
    batch_size: int | None = None,
    img_size: int | None = None,
    device: str = "auto",
    limit: int = 0,
    output: Path | None = None,
    eval_backend: str | None = None,
    *,
    ann_file: Path | None = None,
    images_dir: Path | None = None,
) -> int:
    """Score a checkpoint on the acceptance protocol its task defines.

    Args:
        checkpoint: Lightning ``.ckpt`` to evaluate.
        data_root: COCO root, or the tiled layout root for an oriented checkpoint.
        ema: Evaluate the EMA shadow stored in the checkpoint rather than raw weights.
        masks: Segment only; score masks by default. Set false to score boxes alone.
        split: OBB only; split of the tiled layout to score. Defaults to val.
        variant: OBB only; scale letter of the trained model. Defaults to n.
        batch_size: Images or tiles per forward pass. Defaults per task.
        img_size: Letterbox side. Defaults per task.
        device: ``auto``, ``cpu``, ``mps`` or ``cuda``.
        limit: Score only the first N images or tiles; ``0`` scores all.
        output: Write the JSON report here.
        eval_backend: COCO box/mask scoring engine; defaults to auto, which prefers
            hotcoco and falls back to faster_coco_eval when unavailable. An explicit
            engine must be usable. OBB does not accept this option. Keypoint OKS
            statistics always use faster_coco_eval.
        ann_file: COCO only; annotation JSON path. Defaults to annotations/instances_val2017.json
            under data_root, or annotations/person_keypoints_val2017.json for keypoints.
        images_dir: COCO only; image directory. Defaults to val2017 under data_root.

    Returns:
        The protocol's exit code.

    Raises:
        ValueError: If ``limit``, ``batch_size`` or ``img_size`` is outside its domain
            (:func:`_check_arguments`), the checkpoint's task is unsupported, or an
            explicit option is unused by that task.

    Examples:
        ```pycon
        >>> evaluate(Path("/nonexistent.ckpt"), Path("/data"))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        FileNotFoundError: ...

        ```

        The arguments are checked before the checkpoint is opened, so a refused flag
        costs nothing and names itself rather than the file it never got to:

        ```pycon
        >>> evaluate(Path("/nonexistent.ckpt"), Path("/data"), limit=-5)
        Traceback (most recent call last):
            ...
        ValueError: limit must be a number >= 0; got -5

        ```
    """
    _check_arguments(batch_size, img_size, limit)
    module, info = load_eval_module(checkpoint, use_ema=ema)
    task = str(module.task)
    # Refused rather than defaulted: the dispatch below ends in the detection protocol, so
    # a task this command implements nothing for would be *scored* as detection — a report
    # carrying one protocol's numbers under another's name, which is unfalsifiable from the
    # file. The four are read off the defaults table, and the table off the task registry,
    # so a fifth task arrives with its own defaults or does not arrive at all. It is then
    # accepted here, and the dispatch below must learn its protocol in the same change or
    # score it as detection; `tests/test_tasks.py` pins the four members so that a fifth is
    # a deliberate edit rather than a silent one.
    require_one_of("task", task, tuple(DEFAULT_IMG_SIZE))
    # None distinguishes omission from explicitly passing even a protocol's default,
    # including values supplied by --config. Validate before backend setup or scoring.
    for name, value, applicable in (
        ("split", split, task == "obb"),
        ("variant", variant, task == "obb"),
        ("masks", masks, task == "segment"),
        ("eval_backend", eval_backend, task != "obb"),
        ("ann_file", ann_file, task != "obb"),
        ("images_dir", images_dir, task != "obb"),
    ):
        if value is not None and not applicable:
            raise ValueError(f"{name} is not used by checkpoint task {task!r}")
    info["task"] = task
    resolved_img_size = DEFAULT_IMG_SIZE[task] if img_size is None else img_size
    resolved_batch_size = DEFAULT_BATCH_SIZE[task] if batch_size is None else batch_size
    if task == "obb":
        return rotated_eval.run(
            module,
            info,
            data_root=data_root,
            split="val" if split is None else split,
            variant="n" if variant is None else variant,
            img_size=resolved_img_size,
            batch_size=resolved_batch_size,
            device_name=device,
            limit=limit,
            output=output,
        )
    resolved_backend, fallback_reason = _resolve_eval_backend("auto" if eval_backend is None else eval_backend)
    if fallback_reason is not None:
        info["eval_backend_fallback_reason"] = fallback_reason
        print(
            f"eval_backend='hotcoco' requested by default but unusable here, using faster_coco_eval: {fallback_reason}"
        )
    if task == "keypoints":
        return pose_eval.run(
            module,
            info,
            data_root=data_root,
            img_size=resolved_img_size,
            batch_size=resolved_batch_size,
            device_name=device,
            limit=limit,
            output=output,
            eval_backend=resolved_backend,
            ann_file=ann_file,
            images_dir=images_dir,
        )
    return detect_eval.run(
        module,
        info,
        data_root=data_root,
        img_size=resolved_img_size,
        batch_size=resolved_batch_size,
        masks=True if masks is None else masks,
        device_name=device,
        limit=limit,
        output=output,
        eval_backend=resolved_backend,
        ann_file=ann_file,
        images_dir=images_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``lucid-eval`` from the command line.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        The protocol's exit code.

    Examples:
        ```pycon
        >>> main(["--help"])  # argparse exits the process on --help  # doctest: +SKIP
        0

        ```
    """
    return int(auto_cli(evaluate, args=None if argv is None else list(argv), as_positional=False))


if __name__ == "__main__":  # pragma: no cover - `python -m lucid_yolo.cli.eval`
    raise SystemExit(main())
