# SPDX-License-Identifier: Apache-2.0
"""``lucid-export`` — one checkpoint to one ONNX file.

One command, no task flag::

    lucid-export --checkpoint runs/det.ckpt --output det.onnx
    lucid-export --checkpoint runs/seg.ckpt --output seg.onnx --img_size 320
    lucid-export --checkpoint runs/obb.ckpt --output obb.onnx --canvas "[768, 1024]"
    lucid-export --checkpoint runs/det.ckpt --output det_raw.onnx --ema false

The task is read from the checkpoint, exactly as ``lucid-eval`` and ``lucid-predict``
read it, and the graph is the suppression-free one-to-one path that task deploys on:
:func:`~lucid_yolo.export.graph_from_module` rebuilds the task's model from the
checkpoint under a strict key check and :func:`~lucid_yolo.export.export_graph` writes
it at the pinned opset. Nothing here composes a graph of its own, so the file this
command writes is the file the ONNX export gate certifies.

``--ema`` and ``--img_size`` keep the other commands' spellings and semantics: the EMA
shadow by default, and a square side defaulting per task through the same
:data:`~lucid_yolo.cli.eval.DEFAULT_IMG_SIZE` table. ``--canvas`` takes a non-square
``[height, width]`` instead, written as one JSON list because that is how jsonargparse
reads a pair; the two flags are exclusive.

Limits a caller should know before deploying the file (the
:mod:`lucid_yolo.export` docstring has the reasons):

- The input shape is static, batch included: ``(1, 3, H, W)`` for the canvas named. A
  second canvas or batch size is a second export.
- The output is a fixed ``300`` rows per image, best first; rows no anchor produced are
  zero-score padding. A segmentation export refuses canvases yielding fewer than 300
  anchors (96 px a side and below).
- Opset 17 by default, through torch's legacy TorchScript exporter, which needs the
  ``onnx`` package: ``pip install "lucid-yolo[export]"``. This command checks for it
  before reading the checkpoint.

Provenance: R1 sec. 3.2.1, R3 sec. 4. Assumptions: A9, A45.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from jsonargparse import auto_cli

from lucid_yolo.cli.eval import DEFAULT_IMG_SIZE
from lucid_yolo.eval.checkpoint import load_eval_module
from lucid_yolo.export import ONNX_OPSET, export_graph, graph_from_module
from lucid_yolo.validate import require_one_of

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["export", "main"]

#: What each output of a task's graph holds, in output order, for the summary line.
_OUTPUT_LAYOUT: dict[str, tuple[str, ...]] = {
    "detect": ("detections [x1, y1, x2, y2, score, class]",),
    "segment": ("detections [x1, y1, x2, y2, score, class]", "masks (bool, letterboxed canvas)"),
    "obb": ("detections [cx, cy, w, h, theta, score, class]",),
    "keypoints": ("detections [x1, y1, x2, y2, score, class]", "keypoints [x, y] (letterboxed canvas)"),
}

#: Install hint naming the extra that carries the exporter's one extra dependency.
_ONNX_HINT = 'pip install "lucid-yolo[export]"'


def _require_onnx() -> None:
    """Refuse up front when the ``onnx`` package torch's exporter needs is absent.

    The exporter imports it only to finalise the file, after the trace, so without this
    check a missing package surfaces at the end of the run rather than before the
    checkpoint is read.

    Raises:
        ModuleNotFoundError: If ``onnx`` cannot be imported; the message names the extra.

    Examples:
        ```pycon
        >>> _require_onnx()  # the dev environment has it: returns nothing

        ```
    """
    if importlib.util.find_spec("onnx") is None:
        raise ModuleNotFoundError(f"lucid-export needs the onnx package to write the file: {_ONNX_HINT}", name="onnx")


def _refuse_output_onto_checkpoint(checkpoint: Path, output: Path) -> None:
    """Refuse an ``--output`` that names the checkpoint file itself, under any spelling.

    The exporter opens the destination for writing, which truncates it, so an output that
    is the checkpoint replaces the weights with ONNX bytes and leaves no checkpoint to
    export again. Two tests catch the aliases: resolved-path equality covers the same path
    spelled differently and a symbolic link, and :func:`os.path.samefile` covers a hard
    link, which resolves to a path of its own while sharing the checkpoint's file.

    Args:
        checkpoint: ``--checkpoint``, the file about to be read.
        output: ``--output``, the file about to be written.

    Raises:
        ValueError: If ``output`` refers to the same file as ``checkpoint``; nothing has
            been read or written by then.

    Examples:
        ```pycon
        >>> _refuse_output_onto_checkpoint(Path("runs/det.ckpt"), Path("det.onnx"))  # returns nothing
        >>> _refuse_output_onto_checkpoint(Path("runs/det.ckpt"), Path("runs/../runs/det.ckpt"))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        ValueError: --output runs/../runs/det.ckpt is the checkpoint runs/det.ckpt itself; writing there would ...

        ```
    """
    same = checkpoint.resolve() == output.resolve()
    if not same and output.exists() and checkpoint.exists():
        same = os.path.samefile(checkpoint, output)
    if same:
        raise ValueError(
            f"--output {output} is the checkpoint {checkpoint} itself; writing there would replace the checkpoint "
            f"with the ONNX file, so nothing was written: pick another --output"
        )


def _resolve_canvas(task: str, canvas: tuple[int, int] | None, img_size: int | None) -> tuple[int, int]:
    """Pick the export canvas from the two exclusive flags, or the task's default side.

    Args:
        task: The checkpoint's task, already known to have a default side.
        canvas: ``--canvas``, a ``(height, width)`` pair; wins when given, the caller
            having already refused it beside ``img_size``.
        img_size: ``--img_size``, a square side.

    Returns:
        The ``(height, width)`` canvas to build the graph for.

    Examples:
        ```pycon
        >>> _resolve_canvas("obb", None, None), _resolve_canvas("detect", None, 320)
        ((1024, 1024), (320, 320))
        >>> _resolve_canvas("detect", (480, 640), None)
        (480, 640)

        ```
    """
    if canvas is not None:
        return (int(canvas[0]), int(canvas[1]))
    side = DEFAULT_IMG_SIZE[task] if img_size is None else img_size
    return (side, side)


def export(
    checkpoint: Path,
    output: Path,
    canvas: tuple[int, int] | None = None,
    img_size: int | None = None,
    ema: bool = True,
    opset: int = ONNX_OPSET,
) -> int:
    """Export one checkpoint's suppression-free graph to an ONNX file.

    Which graph is built is the checkpoint's ``task``, not a flag, for the reason
    ``lucid-predict`` gives: a flag is how a caller would export a segmentation model
    with its mask branch unread.

    Args:
        checkpoint: Lightning ``.ckpt`` to export; its task must be ``detect``,
            ``segment``, ``obb`` or ``keypoints``.
        output: Destination ``.onnx`` file; the parent directory is created if absent.
        canvas: Input ``[height, width]`` in pixels, each a positive multiple of every
            head stride. Exclusive with ``img_size``.
        img_size: Square input side in pixels. Defaults per the checkpoint's task when
            neither this nor ``canvas`` is given.
        ema: Export the EMA shadow stored in the checkpoint rather than raw weights.
        opset: ONNX opset to emit. Defaults to the pinned
            :data:`~lucid_yolo.export.ONNX_OPSET`, the only one the export gate
            certifies.

    Returns:
        ``0``; a failure here raises rather than returning a code.

    Raises:
        ModuleNotFoundError: If the ``onnx`` package is not installed.
        ValueError: If both ``canvas`` and ``img_size`` are given, if ``output`` is the
            checkpoint file itself (the same path, a symbolic link or a hard link to it;
            refused before the checkpoint is read), if the checkpoint's task is unknown,
            if its weights do not match the task's model
            (:func:`~lucid_yolo.export.graph_from_module` names the keys), or if the
            canvas is refused.

    Examples:
        ```pycon
        >>> export(Path("/nonexistent.ckpt"), Path("/none.onnx"))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        FileNotFoundError: ...

        ```
    """
    _require_onnx()
    if canvas is not None and img_size is not None:
        raise ValueError(f"give --canvas or --img_size, not both; got canvas={list(canvas)} and img_size={img_size}")
    _refuse_output_onto_checkpoint(checkpoint, output)
    module, info = load_eval_module(checkpoint, use_ema=ema)
    task = str(module.task)
    require_one_of("task", task, tuple(DEFAULT_IMG_SIZE))
    height, width = _resolve_canvas(task, canvas, img_size)
    graph = graph_from_module(module, (height, width))
    with torch.no_grad():
        produced = graph(torch.zeros(1, 3, height, width))
    outputs = produced if isinstance(produced, tuple) else (produced,)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = export_graph(graph, output, opset_version=opset)
    print(f"export: {checkpoint} -> {written} (task={task}, ema={ema}, epoch={info.get('epoch')}, opset={opset})")
    print(f"input: image (1, 3, {height}, {width}) float32, static shape")
    for tensor, layout in zip(outputs, _OUTPUT_LAYOUT[task], strict=True):
        print(f"output: {tuple(tensor.shape)} {layout}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``lucid-export`` from the command line.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        The command's exit code.

    Examples:
        ```pycon
        >>> main(["--help"])  # argparse exits the process on --help  # doctest: +SKIP
        0

        ```
    """
    return int(auto_cli(export, args=None if argv is None else list(argv), as_positional=False))


if __name__ == "__main__":  # pragma: no cover - `python -m lucid_yolo.cli.export`
    raise SystemExit(main())
