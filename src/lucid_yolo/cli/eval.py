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

from lucid_yolo.assign.grid import require_grid_side
from lucid_yolo.eval import detect_eval, pose_eval, rotated_eval
from lucid_yolo.eval.checkpoint import load_eval_module
from lucid_yolo.eval.coco_eval import hotcoco_available
from lucid_yolo.validate import require_at_least, require_one_of

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["DEFAULT_BATCH_SIZE", "DEFAULT_IMG_SIZE", "evaluate", "main"]

#: Letterbox side used when ``img_size`` is not given, per checkpoint task. The oriented
#: tier trains on 1024 px crops (R18 sec. 4); COCO runs at 640 (R1 sec. 4.4).
DEFAULT_IMG_SIZE = {"obb": 1024, "detect": 640, "segment": 640, "keypoints": 640}
#: Batch size used when ``batch_size`` is not given. A 1024 px tile is 2.5x the pixels of
#: a 640 px image, so the oriented default is smaller for the same memory.
DEFAULT_BATCH_SIZE = {"obb": 8, "detect": 32, "segment": 32, "keypoints": 32}


def _resolve_eval_backend(requested: str) -> tuple[str, str | None]:
    """Resolve the box/segm scoring engine, per WP-138's own "safe to default" call.

    hotcoco ships prebuilt wheels for this project's real target platforms
    (macOS/Linux/Windows, cp39-abi3), which is why ``"auto"`` — the default —
    picks it without hedging. The fallback exists for what rf-detr PR 1402
    documented rather than what is expected here: no musllinux wheel, so an
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
        >>> _resolve_eval_backend("faster_coco_eval")
        ('faster_coco_eval', None)
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
        >>> _check_arguments(32, 640, 0)  # a usable trio: returns nothing
        >>> _check_arguments(0, None, 0)
        Traceback (most recent call last):
            ...
        ValueError: batch_size must be a number >= 1; got 0
    """
    require_at_least("limit", limit, 0)
    if batch_size is not None:
        require_at_least("batch_size", batch_size, 1)
    if img_size is not None:
        require_grid_side("img_size", img_size)


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
    eval_backend: str = "auto",
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
        eval_backend: bbox/segm scoring engine (WP-138) — ``"auto"`` (default)
            prefers ``hotcoco``, falling back to ``faster_coco_eval`` only if
            hotcoco is not usable *here* (see :func:`_resolve_eval_backend`);
            ``"hotcoco"`` or ``"faster_coco_eval"`` states one explicitly and
            raises rather than substituting if it is not available. ``obb``
            ignores this — WP-063's rotated mAP is not a COCOeval protocol at
            all — and the ten OKS keypoint statistics stay on hand-driven
            ``faster_coco_eval`` regardless (A73); this only ever selects the
            box (and, for a segmentation checkpoint, mask) engine.

    Returns:
        The protocol's exit code.

    Raises:
        ValueError: If ``limit``, ``batch_size`` or ``img_size`` is outside its domain
            (:func:`_check_arguments`), or if the checkpoint's task is not one this
            command implements a protocol for.

    Examples:
        >>> evaluate(Path("/nonexistent.ckpt"), Path("/data"))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        FileNotFoundError: ...

        The arguments are checked before the checkpoint is opened, so a refused flag
        costs nothing and names itself rather than the file it never got to:

        >>> evaluate(Path("/nonexistent.ckpt"), Path("/data"), limit=-5)
        Traceback (most recent call last):
            ...
        ValueError: limit must be a number >= 0; got -5
    """
    _check_arguments(batch_size, img_size, limit)
    module, info = load_eval_module(checkpoint, use_ema=ema)
    task = str(module.task)
    # Refused rather than defaulted: the dispatch below ends in the detection protocol, so
    # a task this command implements nothing for would be *scored* as detection — a report
    # carrying one protocol's numbers under another's name, which is unfalsifiable from the
    # file. The four are read off the defaults table rather than restated here, so a fifth
    # task arrives with its own defaults or does not arrive at all.
    require_one_of("task", task, tuple(DEFAULT_IMG_SIZE))
    info["task"] = task
    resolved_img_size = DEFAULT_IMG_SIZE[task] if img_size is None else img_size
    resolved_batch_size = DEFAULT_BATCH_SIZE[task] if batch_size is None else batch_size
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
    resolved_backend, fallback_reason = _resolve_eval_backend(eval_backend)
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
        eval_backend=resolved_backend,
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
