#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Draw one checkpoint's predictions over the image they were made on (WP-067, WP-152).

The command-line face of :mod:`lucid_yolo._viz.overlay` (WP-189): the four drawing
functions, the renderer table and the checkpoint-to-axes path live there, where the
notebooks share them and the gate tests them, and this script owns the argument parse
and the file write. The release's worked example: a checkpoint, an image, and a figure
showing what the model actually answered — boxes for a ``detect`` checkpoint, boxes and
instance masks for a ``segment`` one, rotated quadrilaterals for an ``obb`` one, boxes
with their point sets for a ``keypoints`` one. Why each is drawn the way it is — and why
a rotated box is never an upright rectangle — is documented on the module that draws it.

Assumptions:
    ``--task`` defaults to **the checkpoint's own task**, as ``lucid-predict`` reads it,
    rather than being required. A named task selects an entry point, and the library then
    refuses a checkpoint that does not match it by name (a ``segment`` checkpoint drawn as
    ``detect`` would answer plausibly with its mask branch unread), so the flag can narrow
    the choice and cannot silently mislead.

Examples:
    ```console
    $ python scripts/draw_predictions.py runs/det.ckpt street.jpg --output street_det.png
    $ python scripts/draw_predictions.py runs/seg.ckpt street.jpg --output street_seg.png \
        --conf-threshold 0.4 --title "seg-smoke v9"
    $ python scripts/draw_predictions.py runs/obb.ckpt aerial.png --output aerial_obb.svg \
        --decoder nms --img-size 1024
    ```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

# Chosen here, by the one process that is headless, and never by the package: pyplot
# binds whatever backend is current when it is first imported, and `lucid_yolo._viz`
# performs that import. Same order, same reason, as `scripts/plot_training.py`.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from lucid_yolo._viz.overlay import (
    DRAWABLE_TASKS,
    RenderOptions,
    class_color,
    draw_detections,
    draw_keypoints,
    draw_oriented,
    draw_segmentation,
    predict_and_draw,
)
from lucid_yolo.eval.checkpoint import load_eval_module
from lucid_yolo.predict import DECODE_PATHS, DEFAULT_CONF_THRESHOLD

if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "RenderOptions",
    "class_color",
    "draw_detections",
    "draw_keypoints",
    "draw_oriented",
    "draw_segmentation",
    "main",
    "render_prediction",
]

#: Raster resolution of the written figure. A vector suffix (``.svg``) ignores it.
_FIGURE_DPI = 150


def render_prediction(checkpoint: Path, image: Path, output: Path, options: RenderOptions) -> Path:
    """Predict one image with one checkpoint and write the annotated figure.

    :func:`~lucid_yolo._viz.overlay.predict_and_draw` does the prediction and the
    drawing; this function loads the checkpoint, writes the figure and closes it.

    Args:
        checkpoint: Lightning ``.ckpt`` to predict with. Releases ship none (D14): this is
            a checkpoint the operator trained or was given.
        image: Image file to run on and draw over.
        output: Destination figure path; the suffix chooses the format and parent
            directories are created.
        options: The rest of the request — see :class:`~lucid_yolo._viz.overlay.RenderOptions`.

    Returns:
        The written ``output`` path.

    Raises:
        ValueError: If the checkpoint's task (or the one ``options`` names) is not one
            this script draws.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> written = render_prediction(
        ...     Path("det.ckpt"), Path("a.jpg"), Path("a.png"), RenderOptions()
        ... )  # needs a checkpoint  # doctest: +SKIP

        ```
    """
    module, _info = load_eval_module(checkpoint, use_ema=options.ema)
    axes = predict_and_draw(module, image, options)
    figure = axes.get_figure(root=True)
    if figure is None:  # an axes drawn into always has one; only the annotation admits None
        raise RuntimeError("predict_and_draw returned axes that belong to no figure")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=_FIGURE_DPI)
    plt.close(figure)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` when the checkpoint or the image does not exist.

    Examples:
        ```pycon
        >>> main(["--help"])  # prints usage and exits  # doctest: +SKIP

        ```
    """
    parser = argparse.ArgumentParser(description="Draw a checkpoint's predictions over the image it read.")
    parser.add_argument("checkpoint", type=Path, help="Lightning .ckpt to predict with")
    parser.add_argument("image", type=Path, help="image file to run on and draw over")
    parser.add_argument("--output", type=Path, required=True, help="destination figure path (.png, .svg, ...)")
    parser.add_argument(
        "--task",
        choices=DRAWABLE_TASKS,
        default=None,
        help="entry point to draw through (default: the checkpoint's own task)",
    )
    parser.add_argument("--decoder", choices=DECODE_PATHS, default="e2e", help="decode path (default: e2e)")
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=DEFAULT_CONF_THRESHOLD,
        help=f"drop detections at or below this score (default: {DEFAULT_CONF_THRESHOLD})",
    )
    parser.add_argument("--img-size", type=int, default=None, help="letterbox side (default: the task's own)")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps or cuda (default: auto)")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True, help="predict with EMA weights")
    parser.add_argument("--title", default=None, help="figure title (default: the image's name and the task)")
    args = parser.parse_args(argv)
    for label, path in (("checkpoint", args.checkpoint), ("image", args.image)):
        if not path.is_file():
            print(f"no such {label} file: {path}", file=sys.stderr)
            return 1
    written = render_prediction(
        args.checkpoint,
        args.image,
        args.output,
        RenderOptions(
            task=args.task,
            decoder=args.decoder,
            conf_threshold=args.conf_threshold,
            img_size=args.img_size,
            device=args.device,
            ema=args.ema,
            title=args.title,
        ),
    )
    print(f"figure -> {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
