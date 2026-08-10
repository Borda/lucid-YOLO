#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Render a training run's curves as a vector figure for the reproduction report (WP-080).

Reads the CSV a run's :class:`~pytorch_lightning.loggers.CSVLogger` wrote
(``lightning_logs/version_N/metrics.csv``) and emits one three- or four-panel SVG:

1. **val/mAP** — the epoch-end E2E proxy logged by WP-077, the run's headline curve;
2. **loss** — train and validation totals on one axis;
3. **o2o components** — the one-to-one branch's classification, CIoU and L1 terms,
   which is where a mis-scaled term shows up as a flat line (the WP-078 signature);
4. **segmentation terms** — the pre-gain instance-mask and auxiliary semantic terms
   (WP-087, A38), which only a ``task: segment`` run logs.

Panels are skipped when the run predates the column they need, so the script also
works on the older runs whose CSVs carry no ``val/mAP``. The fourth panel follows
the same rule from the other side: a detection run's CSV has no ``mask`` or
``semantic`` column, so it renders the three panels it always did, at the width it
always did.

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
import csv
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "lucid-yolo"

import matplotlib.pyplot as plt  # noqa: E402  — backend must be selected before pyplot

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes


__all__ = ["load_series", "main", "plot_run"]

#: Panel 3 series: CSV column suffix -> legend label, in plot order.
_COMPONENTS = (("o2o_cls", "classification"), ("o2o_box", "CIoU"), ("o2o_l1", "L1 (stride units)"))

#: Panel 4 series: CSV column -> legend label and colour, in plot order. ``val/semantic``
#: is missing on purpose, not by oversight — see :func:`_panel_segmentation`. The colours
#: repeat the loss panel's train-grey / validation-blue pairing so the mask term reads the
#: same way there as here.
_SEG_SERIES = (
    ("train/mask", "mask (train)", "#7f7f7f"),
    ("val/mask", "mask (validation)", "#1f77b4"),
    ("train/semantic", "semantic (train only)", "#2ca02c"),
)

#: Figure geometry in inches: each panel gets this width, and the figure this height.
_PANEL_WIDTH, _FIGURE_HEIGHT = 4.5, 3.9


def _figure_size(panels: int) -> tuple[float, float]:
    """Return the figure size in inches for a given panel count.

    The three-panel value is load-bearing rather than cosmetic: the committed
    detection figure was rendered at it, and widening the figure by a fraction of
    an inch moves every coordinate in the SVG, so the byte-identity the module
    docstring promises would break for a run whose numbers never changed.

    Examples:
        ```pycon
        >>> _figure_size(3)
        (13.5, 3.9)
        >>> _figure_size(4)
        (18.0, 3.9)

        ```
    """
    return (_PANEL_WIDTH * panels, _FIGURE_HEIGHT)


def load_series(path: Path, column: str) -> tuple[list[int], list[float]]:
    """Read one metric column from a Lightning CSV log, dropping rows that lack it.

    Lightning writes a sparse table — a step row carries only the training
    metrics, an epoch-end row only the validation ones — so every column needs
    its own row filter rather than a shared index.

    Args:
        path: The ``metrics.csv`` written by the run's ``CSVLogger``.
        column: Column name to extract, e.g. ``"val/mAP"``.

    Returns:
        The ``(epochs, values)`` pair for the rows where ``column`` is non-empty.
        Both lists are empty when the run never logged that column.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> epochs, values = load_series(Path("nonexistent.csv"), "val/mAP")  # doctest: +SKIP

        ```
    """
    epochs: list[int] = []
    values: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = row.get(column)
            if raw:
                epochs.append(int(row["epoch"]))
                values.append(float(raw))
    return epochs, values


def _panel_map(axis: Axes, path: Path, close_mosaic: int | None) -> bool:
    """Draw the val/mAP panel; return whether the run logged the metric at all."""
    epochs, values = load_series(path, "val/mAP")
    if not epochs:
        return False
    axis.plot(epochs, values, color="#1f77b4", linewidth=1.8)
    axis.set_title("validation mAP50-95 (E2E proxy)")
    axis.set_ylabel("mAP50-95")
    axis.set_ylim(bottom=0.0)
    if close_mosaic is not None:
        axis.axvline(close_mosaic, color="#d62728", linestyle="--", linewidth=1.0)
        axis.annotate(
            "close-mosaic",
            xy=(close_mosaic, axis.get_ylim()[1] * 0.12),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color="#d62728",
        )
    axis.annotate(
        f"{values[-1]:.3f}",
        xy=(epochs[-1], values[-1]),
        xytext=(-34, -12),
        textcoords="offset points",
        fontsize=9,
        color="#1f77b4",
    )
    return True


def epoch_means(epochs: Sequence[int], values: Sequence[float]) -> tuple[list[int], list[float]]:
    """Collapse a per-step series to one mean per epoch.

    Training metrics are logged every N steps, so a raw plot is a sawtooth whose
    first point (the untrained opening loss) sets the axis range for the whole
    run. Averaging within the epoch gives a curve comparable to the once-per-epoch
    validation series.

    Args:
        epochs: Per-sample epoch indices, non-decreasing.
        values: Values aligned with ``epochs``.

    Returns:
        The ``(epochs, means)`` pair with one entry per distinct epoch, in order.

    Examples:
        ```pycon
        >>> epoch_means([0, 0, 1], [1.0, 3.0, 5.0])
        ([0, 1], [2.0, 5.0])

        ```
    """
    totals: dict[int, list[float]] = {}
    for epoch, value in zip(epochs, values, strict=True):
        totals.setdefault(epoch, []).append(value)
    ordered = sorted(totals)
    return ordered, [sum(totals[epoch]) / len(totals[epoch]) for epoch in ordered]


def _panel_loss(axis: Axes, path: Path) -> None:
    """Draw epoch-mean train and validation totals on one log axis."""
    for column, label, color in (("train/loss", "train", "#7f7f7f"), ("val/loss", "validation", "#1f77b4")):
        epochs, values = load_series(path, column)
        if epochs:
            axis.plot(*epoch_means(epochs, values), label=label, color=color, linewidth=1.4)
    axis.set_title("loss (epoch mean)")
    axis.set_ylabel("total loss")
    axis.set_yscale("log")
    axis.legend(frameon=False, fontsize=9)


def _panel_components(axis: Axes, path: Path) -> None:
    """Draw the one-to-one branch's pre-gain loss components."""
    for column, label in _COMPONENTS:
        epochs, values = load_series(path, f"val/{column}")
        if epochs:
            axis.plot(epochs, values, label=label, linewidth=1.4)
    axis.set_title("validation o2o components (pre-gain)")
    axis.set_ylabel("term value")
    axis.set_yscale("log")
    axis.legend(frameon=False, fontsize=9)


def _has_segmentation(path: Path) -> bool:
    """Report whether the run logged any of the segmentation terms."""
    return any(load_series(path, column)[0] for column, _, _ in _SEG_SERIES)


def _panel_segmentation(axis: Axes, path: Path) -> None:
    """Draw the epoch-mean pre-gain segmentation terms (A38): instance mask, and semantic.

    ``val/semantic`` is left out deliberately. The auxiliary semantic branch is
    training-only (A17), so the run does log the validation column but every entry
    in it is exactly 0.0 — which on this axis is a flat line pinned to the bottom,
    the same shape a dead loss term makes. Drawing it would make a working run
    look like the WP-078 failure it is not, so the panel shows the training curve,
    which descends, and the legend says which of the two is train-only.

    The axis is linear rather than the log used by the loss and component panels:
    these are per-pixel means confined to roughly one decade, and a log axis would
    stretch the semantic term's approach to zero into the panel's dominant feature.
    """
    for column, label, color in _SEG_SERIES:
        epochs, values = load_series(path, column)
        if epochs:
            axis.plot(*epoch_means(epochs, values), label=label, color=color, linewidth=1.4)
    axis.set_title("segmentation terms (pre-gain, epoch mean)")
    axis.set_ylabel("term value")
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False, fontsize=9)


def plot_run(metrics: Path, output: Path, title: str, close_mosaic: int | None = None) -> Path:
    """Render one run's figure and write it as SVG.

    A segmentation run gets a fourth panel for its mask terms; a detection run,
    whose CSV has no such column, gets the three panels and the exact geometry it
    got before that panel existed.

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
        >>> from pathlib import Path
        >>> plot_run(Path("metrics.csv"), Path("fig.svg"), "demo")  # doctest: +SKIP

        ```
    """
    segmented = _has_segmentation(metrics)
    panels = 4 if segmented else 3
    figure, axes = plt.subplots(1, panels, figsize=_figure_size(panels))
    has_map = _panel_map(axes[0], metrics, close_mosaic)
    if not has_map:
        axes[0].set_title("validation mAP50-95 — not logged by this run")
        axes[0].set_axis_off()
    _panel_loss(axes[1], metrics)
    _panel_components(axes[2], metrics)
    if segmented:
        _panel_segmentation(axes[3], metrics)
    for axis in axes:
        if axis.axison:
            axis.set_xlabel("epoch")
            axis.grid(visible=True, alpha=0.25, linewidth=0.6)
            axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(title, fontsize=12)
    figure.tight_layout()
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
        >>> main(["--help"])  # doctest: +SKIP

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
