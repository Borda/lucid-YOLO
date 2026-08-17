# SPDX-License-Identifier: Apache-2.0
"""The prediction overlay draws what each of the three tuples says (WP-067).

Every assertion here reads a matplotlib **artist** — a patch's corners, an overlay's alpha
channel, a label's string — and never a rendered pixel. No golden image is written, and
none could be trusted: a figure's bytes move with a font metric, a backend version and a
default rcParam, so a pixel comparison fails for reasons that have nothing to do with the
geometry, and passes on a drawing that is subtly wrong at sub-pixel scale. The artists are
what the drawing code decided; the raster is what matplotlib did with that afterwards.

Four claims are the reason this suite exists, one per shape the overlay can get wrong:

- an **empty** prediction still draws the picture, with nothing over it — the case a
  freshly-trained checkpoint hits on its first image and the one where a naive
  ``argmax``-style drawing raises instead;
- the **confidence cut** actually removes rows, with the same strict ``>`` the library
  filters its own output with, so a figure and a report of one prediction agree;
- a **rotated** box comes out as a rotated quadrilateral whose four vertices are the
  rectangle's own corners, and is provably not its axis-aligned envelope — the failure
  that looks correct in every review because it is still a rectangle around the object;
- a **mask** overlay is opaque over its own instance and transparent everywhere else,
  including after a threshold has dropped its neighbour, which is the row-alignment defect
  :class:`~lucid_yolo.predict.SegmentedPrediction` exists to make unrepresentable.

The expected rotated corners are derived here from a rotation matrix applied to the
rectangle's local corners — a different formulation from
:func:`~lucid_yolo.data.rotated_geom.rboxes_to_polygons`' basis-vector construction, which
is the point: a test that called the function under test to compute its own expectation
would agree with a broken implementation.

The last case drives the command end to end, through a real ``.ckpt`` written by
``tests/predict/planted.py``: nothing is asserted about *which* objects an untrained
network finds, only that the checkpoint-to-figure path runs and writes a decodable file,
which is the one claim the drawing tests above cannot make for it.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
import pytest
import torch
from planted import IMG_SIZE, write_checkpoint
from torchvision.io import write_png

from lucid_yolo.predict import SegmentedPrediction

matplotlib.use("Agg")

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "draw_predictions.py"

#: Image the drawing cases work on, ``(height, width)``: not square, so a figure geometry
#: that swapped the axes cannot pass by symmetry. It matches ``conftest.IMAGE_SIZE``, and
#: at the ``planted`` letterbox side of 64 the ratio is exactly 1.0 with a vertical pad,
#: so the end-to-end case letterboxes without enlarging the picture it drew.
IMAGE_HEIGHT, IMAGE_WIDTH = 48, 64

#: The rotated box the oriented cases plant, in original-image pixels: a centre inside the
#: picture, a long edge first (canonical, A23) and an angle that is not a multiple of a
#: quarter turn — the only kind of angle that can tell a rotated polygon apart from the
#: axis-aligned envelope of the same box.
PLANTED_RBOX = (32.0, 24.0, 20.0, 8.0, 0.5)

#: Distance, in pixels, within which a drawn vertex counts as a planted corner. The
#: corners travel through float32 tensors, so the agreement is exact to about 1e-6; 1e-4
#: is far below anything a wrong construction could hide in and far above the noise.
CORNER_TOLERANCE = 1e-4

#: Length, in pixels, above which an edge component counts as non-zero. An axis-aligned
#: rectangle has one zero component on every edge; a rotated one has none.
AXIS_ALIGNED_TOLERANCE = 1e-3


def _load_script() -> ModuleType:
    """Load ``scripts/draw_predictions.py`` as an importable module.

    Examples:
        >>> module = _load_script()
        >>> hasattr(module, "draw_detections")
        True
    """
    spec = importlib.util.spec_from_file_location("draw_predictions", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module registered before exec
    spec.loader.exec_module(module)
    return module


draw = _load_script()


@pytest.fixture
def picture() -> NDArray[np.uint8]:
    """A mid-grey ``(height, width, 3)`` image array, the shape the drawing functions take."""
    return np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 128, dtype=np.uint8)


@pytest.fixture
def image_file(tmp_path: Path) -> Path:
    """Write the same picture to disk and return its path, for the cases that run the command.

    Content is irrelevant — nothing asserts on pixels — but the file has to decode, and its
    shape is what fixes the letterbox geometry the end-to-end case runs through.
    """
    path = tmp_path / "scene.png"
    write_png(torch.full((3, IMAGE_HEIGHT, IMAGE_WIDTH), 128, dtype=torch.uint8), str(path))
    return path


def _two_instances() -> SegmentedPrediction:
    """Two row-aligned detections with disjoint masks, scoring 0.9 and 0.4.

    The masks share no pixel, so "which overlay is this" is answerable from the artist
    alone; the scores straddle 0.5, so one threshold separates them.

    Examples:
        >>> prediction = _two_instances()
        >>> prediction.detections.shape
        torch.Size([2, 6])
        >>> prediction.masks.shape
        torch.Size([2, 48, 64])
    """
    masks = torch.zeros(2, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.bool)
    masks[0, 4:16, 4:20] = True
    masks[1, 28:44, 40:60] = True
    detections = torch.tensor(
        [
            [4.0, 4.0, 20.0, 16.0, 0.9, 1.0],
            [40.0, 28.0, 60.0, 44.0, 0.4, 0.0],
        ]
    )
    return SegmentedPrediction(detections=detections, masks=masks)


def _rotated_corners(rbox: tuple[float, float, float, float, float]) -> NDArray[np.float64]:
    """Return a rotated rectangle's four corners, by rotating its local corners.

    An independent statement of the same geometry the drawing reads out of
    :func:`~lucid_yolo.data.rotated_geom.rboxes_to_polygons`: the local corner ring is
    rotated by ``theta`` about the origin and translated to the centre, measuring from
    ``+x`` towards ``+y`` on the y-down image grid, which is this project's convention.

    Examples:
        >>> _rotated_corners((0.0, 0.0, 4.0, 4.0, 0.0)).tolist()
        [[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]]
    """
    centre_x, centre_y, width, height, theta = rbox
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    local = np.array(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
        ]
    )
    return np.asarray(local @ rotation.T + np.array([centre_x, centre_y]))


def test_each_detection_becomes_one_labelled_rectangle(picture: NDArray[np.uint8]) -> None:
    """Two A9 rows draw two rectangles at their own corners, each labelled class and score.

    The baseline claim the other detection cases vary: the box a caller reads off the
    figure is the box the tuple carried, corner for corner, and the label says which class
    at what confidence rather than merely marking that something is there.
    """
    detections = torch.tensor([[4.0, 4.0, 20.0, 16.0, 0.9, 1.0], [24.0, 8.0, 44.0, 28.0, 0.6, 0.0]])

    axes = draw.draw_detections(picture, detections)

    assert len(axes.images) == 1, "the picture, and no overlay a detection figure has no use for"
    assert len(axes.patches) == 2
    assert [patch.get_xy() for patch in axes.patches] == [(4.0, 4.0), (24.0, 8.0)]
    assert [patch.get_width() for patch in axes.patches] == [16.0, 20.0]
    assert [patch.get_height() for patch in axes.patches] == [12.0, 20.0]
    assert [text.get_text() for text in axes.texts] == ["1 0.90", "0 0.60"]
    assert axes.texts[0].get_position() == (4.0, 4.0)


@pytest.mark.parametrize(
    "draw_empty",
    [
        pytest.param(lambda image: draw.draw_detections(image, torch.zeros(0, 6)), id="detect"),
        pytest.param(
            lambda image: draw.draw_segmentation(
                image,
                SegmentedPrediction(torch.zeros(0, 6), torch.zeros(0, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.bool)),
            ),
            id="segment",
        ),
        pytest.param(lambda image: draw.draw_oriented(image, torch.zeros(0, 7)), id="obb"),
    ],
)
def test_an_empty_prediction_draws_the_picture_and_nothing_over_it(
    picture: NDArray[np.uint8],
    draw_empty: Callable[[NDArray[np.uint8]], Axes],
) -> None:
    """An image with nothing above the threshold still renders, with no annotation artifact.

    The live case for a checkpoint early in training and for any picture the model finds
    empty, on all three tasks: the answer is a picture of the scene, not an exception and
    not a blank canvas. The segmentation path is the one at risk — an empty instance axis
    is what ``F.interpolate`` rejects outright and what an overlay loop over zero masks
    must simply skip.
    """
    axes = draw_empty(picture)

    assert len(axes.images) == 1
    assert len(axes.patches) == 0
    assert len(axes.texts) == 0


def test_the_confidence_threshold_drops_every_row_at_or_below_it(picture: NDArray[np.uint8]) -> None:
    """Only rows scoring strictly above the threshold are drawn, boundary row included.

    Three rows straddling the cut, one of them exactly on it. The strict comparison is the
    library's — :func:`~lucid_yolo.predict.predict_image` filters its own output with
    ``> conf_threshold`` — and the two surfaces have to agree, or a figure drawn at the
    threshold a report was written at shows a different set of objects than the report
    lists.
    """
    detections = torch.tensor(
        [
            [4.0, 4.0, 20.0, 16.0, 0.9, 1.0],
            [8.0, 8.0, 24.0, 20.0, 0.5, 1.0],
            [24.0, 8.0, 44.0, 28.0, 0.1, 0.0],
        ]
    )

    axes = draw.draw_detections(picture, detections, conf_threshold=0.5)

    assert len(axes.patches) == 1
    assert axes.patches[0].get_xy() == (4.0, 4.0)
    assert [text.get_text() for text in axes.texts] == ["1 0.90"]


def test_a_class_keeps_its_colour_whatever_else_is_in_the_picture(picture: NDArray[np.uint8]) -> None:
    """Class 2 is drawn in one colour in both figures, and never class 5's.

    A palette walked in encounter order would pass a single-figure test and fail here: it
    gives class 2 the first colour when it is alone and the second when class 5 precedes
    it, so two figures of the same scene disagree about which object is which. Colour is a
    function of the class index, which is what makes two predictions comparable by eye.
    """
    alone = draw.draw_detections(picture, torch.tensor([[4.0, 4.0, 20.0, 16.0, 0.9, 2.0]]))

    accompanied = draw.draw_detections(
        picture,
        torch.tensor([[24.0, 8.0, 44.0, 28.0, 0.8, 5.0], [4.0, 4.0, 20.0, 16.0, 0.9, 2.0]]),
    )

    assert accompanied.patches[1].get_edgecolor() == alone.patches[0].get_edgecolor()
    assert accompanied.patches[0].get_edgecolor() != accompanied.patches[1].get_edgecolor()


def test_an_oriented_box_is_drawn_as_the_rotated_rectangle_it_describes(picture: NDArray[np.uint8]) -> None:
    """The polygon's four vertices are the rotated box's corners, and are not axis-aligned.

    Both halves matter and only together. Matching the corners proves the drawing reads
    ``[cx, cy, w, h, theta]`` as a rotated rectangle; the non-axis-aligned check proves it
    is not the envelope of that rectangle, which at this angle contains all four corners
    too and would satisfy a looser assertion while showing the reader a 21x14 upright box
    where the model reported a 20x8 rotated one.
    """
    detections = torch.tensor([[*PLANTED_RBOX, 0.8, 1.0]])

    axes = draw.draw_oriented(picture, detections)

    ring = axes.patches[0].get_xy()
    assert len(ring) == 5, "four corners with the ring closed back onto the first"
    drawn = np.asarray(ring[:4], dtype=np.float64)
    distances = [
        min(float(np.hypot(*(corner - vertex))) for vertex in drawn) for corner in _rotated_corners(PLANTED_RBOX)
    ]
    assert distances == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=CORNER_TOLERANCE)
    edge = drawn[1] - drawn[0]
    assert abs(edge[0]) > AXIS_ALIGNED_TOLERANCE
    assert abs(edge[1]) > AXIS_ALIGNED_TOLERANCE
    assert len({round(float(x), 6) for x in drawn[:, 0]}) == 4, "an upright rectangle would repeat two x values"


@pytest.mark.parametrize(("instance", "label"), [pytest.param(0, 1, id="first"), pytest.param(1, 0, id="second")])
def test_a_mask_overlay_is_opaque_over_its_own_instance_only(
    picture: NDArray[np.uint8],
    instance: int,
    label: int,
) -> None:
    """Each instance gets its own overlay, opaque exactly where its mask is and nowhere else.

    One overlay per instance rather than one composited layer: the question a segmentation
    figure is read to answer is *which* instance covers a pixel, and a single layer can
    only answer that some instance does. The alpha channel is compared against the mask
    itself, so an overlay that leaked outside its instance — a crop applied in the wrong
    frame, an interpolation smearing the boundary — fails on the pixels it gained.
    """
    prediction = _two_instances()

    axes = draw.draw_segmentation(picture, prediction)

    assert len(axes.images) == 3, "the picture, then one overlay per instance"
    overlay = np.asarray(axes.images[1 + instance].get_array())
    opaque = overlay[..., 3] > 0
    assert np.array_equal(opaque, prediction.masks[instance].numpy())
    assert overlay[..., :3][opaque][0].tolist() == pytest.approx(list(draw.class_color(label)))


def test_the_threshold_drops_a_mask_together_with_its_own_box(picture: NDArray[np.uint8]) -> None:
    """A dropped detection takes its mask with it, and the survivor keeps the mask it came with.

    The row-alignment defect this path is most exposed to: two filters, or one filter
    applied twice, leave the boxes right and shift every mask by a row — a plausible
    figure that no shape check catches, because the counts still agree. The surviving
    overlay is compared against instance 0's own mask, so a shift to instance 1's fails
    here rather than in a reader's judgement of the picture.
    """
    prediction = _two_instances()

    axes = draw.draw_segmentation(picture, prediction, conf_threshold=0.5)

    assert len(axes.images) == 2
    assert len(axes.patches) == 1
    overlay = np.asarray(axes.images[1].get_array())
    assert np.array_equal(overlay[..., 3] > 0, prediction.masks[0].numpy())


@pytest.mark.parametrize(
    "draw_mismatched",
    [
        pytest.param(
            lambda image: draw.draw_detections(image, torch.zeros(1, 7)),
            id="oriented-tuple-to-draw_detections",
        ),
        pytest.param(
            lambda image: draw.draw_oriented(image, torch.zeros(1, 6)),
            id="axis-aligned-tuple-to-draw_oriented",
        ),
        pytest.param(
            lambda image: draw.draw_segmentation(
                image,
                SegmentedPrediction(torch.zeros(2, 6), torch.zeros(1, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.bool)),
            ),
            id="one-mask-for-two-detections",
        ),
        pytest.param(
            lambda image: draw.draw_segmentation(
                image,
                SegmentedPrediction(torch.zeros(1, 6), torch.zeros(1, 8, 8, dtype=torch.bool)),
            ),
            id="mask-on-another-grid",
        ),
    ],
)
def test_a_prediction_that_cannot_belong_to_this_picture_is_refused(
    picture: NDArray[np.uint8],
    draw_mismatched: Callable[[NDArray[np.uint8]], Axes],
) -> None:
    """Mismatched tuples and mask stacks raise instead of drawing something plausible.

    Each of these four has a silent drawing available to it, which is why the guard is
    worth its lines: an A45 row's first four columns drawn as corners give a rectangle in
    the wrong place at the wrong size; an A9 row read as a rotated box gives a heading of
    ``score`` radians; a short mask stack shifts every later pairing; and a mask on another
    grid is stretched by ``imshow`` onto the image's extent, landing over a neighbour.
    """
    with pytest.raises(ValueError):
        draw_mismatched(picture)


def test_the_command_renders_a_checkpoint_it_loaded_to_a_file(
    tmp_path: Path,
    image_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``main`` turns a real ``.ckpt`` and an image into a written PNG, creating its directory.

    The one case that proves the command path rather than assuming it: a checkpoint is
    what the script reads, and neither the planted tuples above nor a directly called
    drawing function exercises the load, the task lookup, the letterbox side or the write.
    Nothing is asserted about *which* boxes an untrained network draws — only that the
    path from a file on disk to a figure on disk runs and produces a decodable image.
    """
    checkpoint = write_checkpoint("detect", tmp_path / "det.ckpt")
    output = tmp_path / "figures" / "scene_det.png"

    code = draw.main(
        [
            str(checkpoint),
            str(image_file),
            "--output",
            str(output),
            "--img-size",
            str(IMG_SIZE),
            "--no-ema",
        ]
    )

    assert code == 0
    assert output.is_file()
    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "the suffix chose the format"
    assert f"figure -> {output}" in capsys.readouterr().out


def test_a_task_the_checkpoint_contradicts_is_refused_by_name(tmp_path: Path, image_file: Path) -> None:
    """``--task obb`` on a detection checkpoint raises, naming the task the checkpoint carries.

    What makes the flag safe to default from the checkpoint and to accept from a caller:
    naming a task selects an entry point, and every entry point in
    :mod:`lucid_yolo.predict` refuses a checkpoint that is not its own. Without that, the
    flag would be the mistake ``lucid-predict`` deliberately has no way to make — an
    oriented drawing of a detector's boxes, at a heading nothing predicted.
    """
    checkpoint = write_checkpoint("detect", tmp_path / "det.ckpt")

    with pytest.raises(ValueError, match="detect"):
        draw.main(
            [
                str(checkpoint),
                str(image_file),
                "--output",
                str(tmp_path / "never.png"),
                "--task",
                "obb",
                "--img-size",
                str(IMG_SIZE),
                "--no-ema",
            ]
        )


def test_a_missing_input_file_reports_it_and_returns_one(
    tmp_path: Path,
    image_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A checkpoint path that does not exist is named on stderr, with exit code 1.

    The mistake an operator makes most often — a stale run directory in the path — and the
    only failure this script answers with a code rather than a traceback, because a
    ``FileNotFoundError`` from inside a Lightning loader names a temporary file rather than
    the argument that was wrong.
    """
    code = draw.main([str(tmp_path / "absent.ckpt"), str(image_file), "--output", str(tmp_path / "never.png")])

    assert code == 1
    assert "no such checkpoint file" in capsys.readouterr().err
