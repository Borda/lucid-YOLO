# SPDX-License-Identifier: Apache-2.0
"""Training-curve figures from a run's Lightning CSV log (WP-080, WP-189).

Reads the CSV a run's :class:`~pytorch_lightning.loggers.CSVLogger` wrote
(``lightning_logs/version_N/metrics.csv``) and builds one three- or four-panel figure:

1. **val/mAP** — the epoch-end E2E proxy logged by WP-077, the run's headline curve;
2. **loss** — train and validation totals on one axis;
3. **o2o components** — the one-to-one branch's classification, CIoU and L1 terms,
   which is where a mis-scaled term shows up as a flat line (the WP-078 signature);
4. **segmentation, oriented or keypoint terms** — a ``task: segment`` run's pre-gain
   instance-mask and auxiliary semantic terms (WP-087, A38); an ``obb`` run's rotated
   ProbIoU, retargeted L1 and angle terms (WP-088); a ``keypoints`` run's RLE term or
   its Laplace-NLL control (WP-123, WP-135) — whichever the CSV's own columns name.

Panels are skipped when the run predates the column they need, so the figure also
works on the older runs whose CSVs carry no ``val/mAP``. The fourth panel follows
the same rule from the other side: a detection run's CSV has none of the mask,
oriented or keypoint columns, so it renders the three panels it always did, at the
width it always did.

Two callers, one figure. :func:`build_run_figure` returns the figure and nothing
else; ``scripts/plot_training.py`` wraps it in the deterministic SVG save the
reproduction report needs (fixed hashsalt, suppressed date metadata), and
:func:`plot_curves` is the notebook entry, where the figure is displayed rather than
written and the title defaults to the run directory's name. The panels are the same
object either way, so a figure a notebook shows is the figure the report commits.

Nothing here selects a backend. ``matplotlib.use("Agg")`` is a process-wide setting
that switches a notebook *away* from its inline display, so the headless choice
belongs to the script and the test session that need it, not to a module a notebook
imports.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


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

#: Panel 4 series for an oriented run: the pre-gain terms that replace and extend the box
#: pair (A49, A50, WP-088). Validation only, matching panel 3 — both splits would put six
#: curves on one axis to say what three already say.
_OBB_SERIES = (("rbox", "rotated ProbIoU"), ("rl1", "L1 (stride units)"), ("angle", "angle"))

#: Panel 4 series for a keypoints run: the single term ``keypoint_loss`` selects
#: between (WP-135) — RLE's flow-plus-Laplace NLL by default, or the Laplace-NLL
#: control with the flow term removed. Both write the same ``keypoint`` column, so
#: the panel cannot and does not distinguish which loss a given run trained under;
#: that distinction lives in the run's own ``config.yaml``, not its metrics.
_KEYPOINT_SERIES = (
    ("train/keypoint", "keypoint (train)", "#7f7f7f"),
    ("val/keypoint", "keypoint (validation)", "#1f77b4"),
)

#: Panel 1 series when a run logs no ``val/mAP``: the oriented metric that replaces it
#: (WP-102). mAP50 leads because the oriented DoD is stated at IoU 0.50.
_ROTATED_SERIES = (
    ("val/rotated_mAP50", "mAP50", "#1f77b4"),
    ("val/rotated_mAP", "mAP50-95", "#7f7f7f"),
)

#: Figure geometry in inches: each panel gets this width, and the figure this height.
_PANEL_WIDTH, _FIGURE_HEIGHT = 4.5, 3.9


def _figure_size(panels: int) -> tuple[float, float]:
    """Return the figure size in inches for a given panel count.

    The three-panel value is load-bearing rather than cosmetic: the committed
    detection figure was rendered at it, and widening the figure by a fraction of
    an inch moves every coordinate in the SVG, so the byte-identity the script
    promises would break for a run whose numbers never changed.

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
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "metrics.csv"
        ...     _ = path.write_text("epoch,step,val/mAP,train/loss\\n0,9,,2.0\\n0,10,0.1,\\n1,20,0.3,\\n")
        ...     load_series(path, "val/mAP")
        ([0, 1], [0.1, 0.3])

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


def _mark_close_mosaic(axis: Axes, close_mosaic: int | None) -> None:
    """Mark the epoch mosaic augmentation was disabled, on whichever metric panel was drawn."""
    if close_mosaic is None:
        return
    axis.axvline(close_mosaic, color="#d62728", linestyle="--", linewidth=1.0)
    axis.annotate(
        "close-mosaic",
        xy=(close_mosaic, axis.get_ylim()[1] * 0.12),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=8,
        color="#d62728",
    )


def _panel_rotated_map(axis: Axes, path: Path, close_mosaic: int | None) -> bool:
    """Draw the oriented metric panel: rotated mAP50 and mAP50-95 (WP-063).

    Both are drawn because they answer different questions about the same run. mAP50
    leads and is annotated — the oriented DoD is stated at IoU 0.50, and localization
    at 0.50 is what an orientation either gets right or does not. The stricter mean
    over 0.50-0.95 sits beneath it and is where a systematically slightly-wrong
    heading shows up as a gap rather than as a failure.

    Neither is a whole-image number: both score tiles as supplied, and the merge
    policy for overlapping tiles belongs to WP-064 (`eval/dota_eval.py`).
    """
    drawn = False
    for column, label, color in _ROTATED_SERIES:
        epochs, values = load_series(path, column)
        if not epochs:
            continue
        axis.plot(epochs, values, label=label, color=color, linewidth=1.8 if not drawn else 1.4)
        if not drawn:
            axis.annotate(
                f"{values[-1]:.3f}",
                xy=(epochs[-1], values[-1]),
                xytext=(-34, -12),
                textcoords="offset points",
                fontsize=9,
                color=color,
            )
        drawn = True
    if not drawn:
        return False
    axis.set_title("validation rotated mAP (per tile)")
    axis.set_ylabel("rotated mAP")
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False, fontsize=9)
    _mark_close_mosaic(axis, close_mosaic)
    return True


def _panel_map(axis: Axes, path: Path, close_mosaic: int | None) -> bool:
    """Draw the headline metric panel; return whether the run logged one at all.

    An oriented run logs no ``val/mAP`` — that figure reads the A44 composition's
    pre-rotation rectangle and was dropped rather than carried beside a metric that
    answers the question (WP-102) — so the panel falls back to the rotated pair. The
    annotated value is the leading series either way, which for an oriented run is
    mAP50, where its acceptance criterion is stated.
    """
    epochs, values = load_series(path, "val/mAP")
    if not epochs:
        return _panel_rotated_map(axis, path, close_mosaic)
    axis.plot(epochs, values, color="#1f77b4", linewidth=1.8)
    axis.set_title("validation mAP50-95 (E2E proxy)")
    axis.set_ylabel("mAP50-95")
    axis.set_ylim(bottom=0.0)
    _mark_close_mosaic(axis, close_mosaic)
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


def _has_oriented(path: Path) -> bool:
    """Report whether the run logged the oriented loss terms."""
    return any(load_series(path, f"val/{column}")[0] for column, _ in _OBB_SERIES)


def _panel_oriented(axis: Axes, path: Path) -> None:
    """Draw the pre-gain oriented terms an ``obb`` run adds to the shared objective.

    Two of the three replace terms panel 3 also carries: ``rbox`` is the rotated
    ProbIoU that stands in for CIoU (A49) and ``rl1`` the L1 retargeted onto the
    rotated box's own extents (A50), so panel 3's ``o2o_box`` and ``o2o_l1`` are the
    zeros the dual loss is constructed with rather than live terms. ``angle`` is the
    one term with no axis-aligned counterpart at all.

    The axis is log, as panel 3 is: the three terms sit decades apart, and the angle
    term in particular is small enough that a linear axis would draw it flat against
    the bottom whether it was learning or dead.
    """
    for column, label in _OBB_SERIES:
        epochs, values = load_series(path, f"val/{column}")
        if epochs:
            axis.plot(epochs, values, label=label, linewidth=1.4)
    axis.set_title("validation oriented terms (pre-gain)")
    axis.set_ylabel("term value")
    axis.set_yscale("log")
    axis.legend(frameon=False, fontsize=9)


def _has_keypoint(path: Path) -> bool:
    """Report whether the run logged the keypoint loss term."""
    return any(load_series(path, column)[0] for column, _, _ in _KEYPOINT_SERIES)


def _panel_keypoint(axis: Axes, path: Path) -> None:
    """Draw the epoch-mean keypoint loss term, whichever of WP-135's two arms trained it.

    A single series each split, unlike the segmentation and oriented panels' three:
    the module logs one aggregate ``keypoint`` value regardless of which loss produced
    it (RLE's flow-plus-Laplace NLL, or the Laplace-NLL control alone). The axis is
    linear rather than log, since RLE's negative log-likelihood is signed — a run's
    ``keypoint`` value crosses zero as the flow's density concentrates, which a log
    axis cannot represent at all.
    """
    for column, label, color in _KEYPOINT_SERIES:
        epochs, values = load_series(path, column)
        if epochs:
            axis.plot(*epoch_means(epochs, values), label=label, color=color, linewidth=1.4)
    axis.set_title("keypoint loss (epoch mean)")
    axis.set_ylabel("term value")
    axis.legend(frameon=False, fontsize=9)


def build_run_figure(metrics: Path, title: str, close_mosaic: int | None = None) -> Figure:
    """Build one run's figure and return it, unsaved and open.

    A segmentation, oriented or keypoints run gets a fourth panel for the terms
    unique to its task; a detection run, whose CSV has none of those columns, gets
    the three panels and the exact geometry it got before that panel existed.

    The figure is returned open on purpose: the script saves it and closes it, a
    notebook displays it, and closing it here would serve the first caller by
    breaking the second.

    Args:
        metrics: Path to the run's ``metrics.csv``.
        title: Figure suptitle, e.g. ``"Det-smoke - run v8"``. Drawn as glyphs —
            changing it later means rebuilding the figure, not editing an SVG.
        close_mosaic: Epoch at which mosaic augmentation was disabled, marked on
            the mAP panel. ``None`` omits the marker.

    Returns:
        The figure, with one axes per panel in ``figure.axes``.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "metrics.csv"
        ...     _ = path.write_text("epoch,step,val/mAP,train/loss,val/o2o_cls\\n0,10,0.1,2.0,0.5\\n")
        ...     figure = build_run_figure(path, "demo")
        >>> len(figure.axes), figure.get_suptitle()
        (3, 'demo')
        >>> plt.close(figure)

        ```
    """
    segmented = _has_segmentation(metrics)
    oriented = _has_oriented(metrics)
    keypoint = _has_keypoint(metrics)
    panels = 4 if segmented or oriented or keypoint else 3
    figure, axes = plt.subplots(1, panels, figsize=_figure_size(panels))
    has_map = _panel_map(axes[0], metrics, close_mosaic)
    if not has_map:
        axes[0].set_title("validation mAP50-95 — not logged by this run")
        axes[0].set_axis_off()
    _panel_loss(axes[1], metrics)
    _panel_components(axes[2], metrics)
    if segmented:
        _panel_segmentation(axes[3], metrics)
    elif oriented:
        _panel_oriented(axes[3], metrics)
    elif keypoint:
        _panel_keypoint(axes[3], metrics)
    for axis in axes:
        if axis.axison:
            axis.set_xlabel("epoch")
            axis.grid(visible=True, alpha=0.25, linewidth=0.6)
            axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    return figure


def plot_curves(metrics: Path, title: str | None = None, close_mosaic: int | None = None) -> Figure:
    """Plot a run's curves for a notebook: the report's panels, titled by the run directory.

    The same figure :func:`build_run_figure` builds, with the one default a notebook
    wants and a report does not: no title given, the suptitle is the run directory's
    own name (``version_8`` for ``lightning_logs/version_8/metrics.csv``), which is
    how a reader scrolling past several runs tells them apart.

    Args:
        metrics: Path to the run's ``metrics.csv``.
        title: Figure suptitle; ``None`` uses ``metrics.parent.name``.
        close_mosaic: Epoch at which mosaic augmentation was disabled, marked on
            the mAP panel. ``None`` omits the marker.

    Returns:
        The open figure, for a notebook cell to display.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     run = Path(tmp) / "version_8"
        ...     run.mkdir()
        ...     _ = (run / "metrics.csv").write_text("epoch,step,val/mAP,train/loss,val/o2o_cls\\n0,10,0.1,2.0,0.5\\n")
        ...     figure = plot_curves(run / "metrics.csv")
        >>> figure.get_suptitle()
        'version_8'
        >>> plt.close(figure)

        ```
    """
    return build_run_figure(metrics, metrics.parent.name if title is None else title, close_mosaic)
