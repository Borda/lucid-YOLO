# SPDX-License-Identifier: Apache-2.0
"""The curve figure carries the panels a run's CSV can fill, and no others (WP-189).

Every assertion reads the figure's **axes** — how many, what each is titled, whether a
panel is switched off — and never a rendered pixel, for the reason ``tests/viz/test_overlay.py``
gives: a raster moves with a font metric and a backend version, and the axes are what the
code decided. The CSVs are written by the tests, column by column, because the panel count
is a function of which columns a run logged and that is the claim under test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from lucid_yolo._viz.curves import build_run_figure, epoch_means, load_series, plot_curves

if TYPE_CHECKING:
    from pathlib import Path

#: The columns every run logs, and one row of each. Panel 1 (``val/mAP``), panel 2 (the
#: two totals) and panel 3 (one o2o term) are all fed, so a three-panel figure draws
#: every panel it has rather than an empty legend.
_DETECTION_HEADER = "epoch,step,val/mAP,train/loss,val/loss,val/o2o_cls"
_DETECTION_ROWS = ("0,10,0.10,2.0,1.5,0.5", "1,20,0.20,1.5,1.2,0.4")


def _write_metrics(directory: Path, header: str, rows: tuple[str, ...]) -> Path:
    """Write a Lightning-style ``metrics.csv`` under ``directory`` and return its path.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = _write_metrics(Path(tmp), "epoch,step,val/mAP", ("0,1,0.5",))
        ...     path.read_text().splitlines()
        ['epoch,step,val/mAP', '0,1,0.5']
    """
    path = directory / "metrics.csv"
    path.write_text("\n".join((header, *rows)) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def detection_metrics(tmp_path: Path) -> Path:
    """A two-epoch detection CSV inside a ``version_8`` run directory."""
    run = tmp_path / "version_8"
    run.mkdir()
    return _write_metrics(run, _DETECTION_HEADER, _DETECTION_ROWS)


class TestLoadSeries:
    """``load_series`` reads one column of a sparse Lightning table."""

    def test_keeps_only_the_rows_that_carry_the_column(self, tmp_path: Path) -> None:
        """A step row without the validation column is skipped rather than read as zero.

        Lightning writes a sparse table — a step row carries only the training metrics —
        so a reader that took every row would plot a zero at every training step.
        """
        path = _write_metrics(tmp_path, "epoch,step,val/mAP,train/loss", ("0,9,,2.0", "0,10,0.1,", "1,20,0.3,"))

        epochs, values = load_series(path, "val/mAP")

        assert (epochs, values) == ([0, 1], [0.1, 0.3])

    def test_a_column_the_run_never_logged_is_empty(self, tmp_path: Path) -> None:
        """Asking for an absent column yields two empty lists, not a ``KeyError``.

        The panel logic keys off emptiness — a run predating ``val/mAP`` still plots —
        so absence has to be a value the caller can test, not an exception.
        """
        path = _write_metrics(tmp_path, "epoch,step,train/loss", ("0,1,2.0",))

        assert load_series(path, "val/mAP") == ([], [])


def test_epoch_means_collapse_a_per_step_series_to_one_point_per_epoch() -> None:
    """Three samples over two epochs become two means, in epoch order.

    The training sawtooth is what this smooths: without it the untrained opening loss
    sets the axis range for the whole run.
    """
    assert epoch_means([0, 0, 1], [1.0, 3.0, 5.0]) == ([0, 1], [2.0, 5.0])


class TestBuildRunFigure:
    """``build_run_figure`` chooses its panels from the CSV's own columns."""

    def test_a_detection_run_gets_three_panels(self, detection_metrics: Path) -> None:
        """A CSV with none of the task-specific columns draws exactly the three shared panels.

        The three-panel geometry is the committed detection figure's, so a fourth panel
        appearing on a detection run would move every coordinate in a regenerated SVG.
        """
        figure = build_run_figure(detection_metrics, "det")

        assert len(figure.axes) == 3
        assert [axis.get_title() for axis in figure.axes] == [
            "validation mAP50-95 (E2E proxy)",
            "loss (epoch mean)",
            "validation o2o components (pre-gain)",
        ]

    @pytest.mark.parametrize(
        ("column", "title"),
        [
            pytest.param("train/mask", "segmentation terms (pre-gain, epoch mean)", id="segment"),
            pytest.param("val/rbox", "validation oriented terms (pre-gain)", id="obb"),
            pytest.param("train/keypoint", "keypoint loss (epoch mean)", id="keypoints"),
        ],
    )
    def test_a_task_column_adds_its_own_fourth_panel(self, tmp_path: Path, column: str, title: str) -> None:
        """One task-specific column is enough to add the fourth panel, titled for that task.

        Which panel appears is decided by the columns and nothing else — there is no
        ``--task`` flag — so a run's CSV is the whole input.
        """
        path = _write_metrics(tmp_path, f"{_DETECTION_HEADER},{column}", tuple(f"{row},0.3" for row in _DETECTION_ROWS))

        figure = build_run_figure(path, "task")

        assert len(figure.axes) == 4
        assert figure.axes[3].get_title() == title

    def test_a_run_without_any_metric_switches_the_first_panel_off(self, tmp_path: Path) -> None:
        """No ``val/mAP`` and no rotated pair leaves panel 1 present but axis-less.

        The older runs predate the headline column; they still plot, with the first
        panel saying so rather than drawing an empty frame.
        """
        path = _write_metrics(tmp_path, "epoch,step,train/loss,val/loss,val/o2o_cls", ("0,10,2.0,1.5,0.5",))

        figure = build_run_figure(path, "old")

        assert len(figure.axes) == 3
        assert not figure.axes[0].axison
        assert figure.axes[1].axison

    def test_the_close_mosaic_epoch_is_marked_on_the_metric_panel(self, detection_metrics: Path) -> None:
        """``close_mosaic`` adds one vertical line at that epoch to the first panel only.

        The marker is what lets a reader attribute a late-run kink to the augmentation
        switch rather than to the model.
        """
        figure = build_run_figure(detection_metrics, "det", close_mosaic=1)

        marker_lines = [line for line in figure.axes[0].lines if line.get_linestyle() == "--"]
        assert len(marker_lines) == 1
        assert marker_lines[0].get_xdata()[0] == 1
        assert all(line.get_linestyle() != "--" for line in figure.axes[1].lines)


class TestPlotCurves:
    """``plot_curves`` is the notebook entry: same figure, titled by the run directory."""

    def test_returns_the_report_figure_with_every_panel(self, detection_metrics: Path) -> None:
        """The notebook figure has the same panels the script's figure has.

        One implementation behind both callers is the point of the move; a panel that
        differed between them would be a figure nothing tests.
        """
        figure = plot_curves(detection_metrics)

        assert [axis.get_title() for axis in figure.axes] == [
            axis.get_title() for axis in build_run_figure(detection_metrics, "x").axes
        ]

    def test_the_title_defaults_to_the_run_directory_name(self, detection_metrics: Path) -> None:
        """With no title, the suptitle is ``version_8`` — the directory the CSV sits in.

        A reader scrolling past several runs in a notebook tells them apart by that
        name, which is also how Lightning names the run on disk.
        """
        assert plot_curves(detection_metrics).get_suptitle() == "version_8"

    def test_an_explicit_title_is_used_verbatim(self, detection_metrics: Path) -> None:
        """A given title replaces the directory default rather than prefixing it."""
        assert plot_curves(detection_metrics, title="Det-smoke v8").get_suptitle() == "Det-smoke v8"
