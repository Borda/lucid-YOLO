# SPDX-License-Identifier: Apache-2.0
"""Figure helpers for the notebooks and the figure scripts (WP-189).

Private on purpose: the notebooks are this package's only callers, so its signatures
move with them and nothing outside ``lucid_yolo._viz`` imports it — ``import lucid_yolo``
and the shipped surface beneath it never load this package, which
``tests/viz/test_import_surface.py`` pins in a subprocess and by reading the source.
``matplotlib`` is a declared runtime dependency (WP-189; ``pytorch-lightning[extra]``
already required it), and this package is the one place under ``lucid_yolo`` allowed
to import it. The package exists so that the four demo notebooks and the two figure
scripts draw through one implementation the gate tests, rather than each carrying a
copy nothing runs.

One function per figure, for a notebook cell:

- :func:`plot_curves` — a run's ``metrics.csv`` as the report's three- or four-panel figure;
- :func:`show_predictions` — one checkpoint over several images, predictions solid and,
  given an annotation file, ground truth dashed under them.

The building blocks — the per-task ``draw_*`` functions, :func:`draw_ground_truth`,
:func:`predict_and_draw`, :func:`build_run_figure` — live in :mod:`lucid_yolo._viz.curves`
and :mod:`lucid_yolo._viz.overlay` and are imported from there.

No backend is selected here. A headless process chooses ``Agg`` before importing this
package, as ``scripts/plot_training.py`` does; a notebook keeps its inline display.
"""

from __future__ import annotations

from lucid_yolo._viz.curves import plot_curves as plot_curves
from lucid_yolo._viz.overlay import RenderOptions as RenderOptions
from lucid_yolo._viz.overlay import draw_ground_truth as draw_ground_truth
from lucid_yolo._viz.overlay import show_predictions as show_predictions
