#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Render a training run's curves as a vector figure for the reproduction report (WP-080).

The command-line face of :mod:`lucid_yolo._viz.curves` (WP-189): the panels, the series
tables and the CSV reader live there, where the notebooks share them and the gate tests
them, and this script owns what only a committed figure needs — the deterministic SVG
save. Reads the CSV a run's :class:`~pytorch_lightning.loggers.CSVLogger` wrote
(``lightning_logs/version_N/metrics.csv``) and emits one three- or four-panel SVG; which
panels, and why, is documented on :func:`~lucid_yolo._viz.curves.build_run_figure`.

The output is deterministic: SVG ids are salted with a fixed value and the
``Date`` metadata is suppressed, so regenerating an unchanged run reproduces the
file byte-for-byte instead of churning the diff.

That determinism only helps if the figure is regenerated rather than patched.
Matplotlib draws the title as glyph outlines and emits the source text beside it
as an XML comment, so a search-and-replace over a committed SVG rewrites the
comment and leaves the drawn title untouched — the file then reports one name and
displays another. Retitling means rerunning this script.

The committed figures caption themselves with typographic punctuation — em dashes,
and a multiplication sign between batch and epochs — which the linter rejects as
ambiguous in source, so the examples below spell both in ASCII. Copying one
verbatim reproduces the right figure under a slightly plainer caption; the strings
the committed files actually carry are in each SVG's own title comment.

Examples:
    ```console
    $ python scripts/plot_training.py lightning_logs/version_8/metrics.csv \
        --output docs/figures/det_smoke_training.svg --close-mosaic 40 \
        --title "Det-smoke - run v8 (dev12, batch 128 x 50 epochs, lr 0.02) - ACCEPTED"
    $ python scripts/plot_training.py lightning_logs/version_9/metrics.csv \
        --output docs/figures/seg_smoke_training.svg --close-mosaic 40 \
        --title "Seg-smoke - run v9 (batch 128 x 50 epochs, lr 0.02)"
    ```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

# The backend and the hashsalt are process-wide settings, so they are chosen here, by
# the one process that is headless and writes a committed file, and never by the
# package: `lucid_yolo._viz` selecting `Agg` would switch a notebook away from its
# inline display. Both must precede the first pyplot import, which `_viz` performs.
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "lucid-yolo"

import matplotlib.pyplot as plt  # noqa: E402  — backend must be selected before pyplot

from lucid_yolo._viz.curves import build_run_figure, epoch_means, load_series  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = ["epoch_means", "load_series", "main", "plot_run"]


def plot_run(metrics: Path, output: Path, title: str, close_mosaic: int | None = None) -> Path:
    """Render one run's figure and write it as SVG.

    :func:`~lucid_yolo._viz.curves.build_run_figure` builds the panels; this function
    adds the write the reproduction report depends on — SVG, with the ``Date`` metadata
    suppressed so an unchanged run regenerates byte-for-byte — and closes the figure.

    Args:
        metrics: Path to the run's ``metrics.csv``.
        output: Destination ``.svg`` path; parent directories are created.
        title: Figure suptitle, e.g. ``"Det-smoke - run v8"``. This is the figure's
            caption as readers see it, drawn as glyphs — changing it later means
            rerunning this function, not editing the SVG.
        close_mosaic: Epoch at which mosaic augmentation was disabled, marked on
            the mAP panel. ``None`` omits the marker.

    Returns:
        The written ``output`` path.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     metrics = Path(tmp) / "metrics.csv"
        ...     _ = metrics.write_text("epoch,step,val/mAP,train/loss,val/o2o_cls\\n0,10,0.1,2.0,0.5\\n")
        ...     plot_run(metrics, Path(tmp) / "fig.svg", "demo").read_bytes()[:5]
        b'<?xml'

        ```
    """
    figure = build_run_figure(metrics, title, close_mosaic)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, format="svg", metadata={"Date": None})
    plt.close(figure)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument vector; ``None`` reads ``sys.argv[1:]``.

    Returns:
        ``0`` on success, ``1`` when the metrics file does not exist.

    Examples:
        ```pycon
        >>> main(["--help"])  # prints usage and exits  # doctest: +SKIP

        ```
    """
    parser = argparse.ArgumentParser(description="Plot a training run's curves as a vector figure.")
    parser.add_argument("metrics", type=Path, help="path to lightning_logs/version_N/metrics.csv")
    parser.add_argument("--output", type=Path, required=True, help="destination .svg path")
    parser.add_argument("--title", default="training run", help="figure suptitle")
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=None,
        help="epoch at which mosaic was disabled (marked on the mAP panel)",
    )
    args = parser.parse_args(argv)
    if not args.metrics.is_file():
        print(f"no such metrics file: {args.metrics}", file=sys.stderr)
        return 1
    written = plot_run(args.metrics, args.output, args.title, args.close_mosaic)
    print(f"figure -> {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
