# SPDX-License-Identifier: Apache-2.0
"""The prediction overlay draws what each tuple says, and the label layer beside it (WP-067, WP-189).

Every assertion here reads a matplotlib **artist** — a patch's corners, an overlay's alpha
channel, a label's string, a line style — and never a rendered pixel. No golden image is
written, and none could be trusted: a figure's bytes move with a font metric, a backend
version and a default rcParam, so a pixel comparison fails for reasons that have nothing
to do with the geometry, and passes on a drawing that is subtly wrong at sub-pixel scale.
The artists are what the drawing code decided; the raster is what matplotlib did with
that afterwards.

Four claims are the reason the prediction half exists, one per shape the overlay can get
wrong:

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

The ground-truth half asserts one property per task: **every** instance becomes exactly
one dashed, unfilled artist (plus a hollow ring per visible point for keypoints), so a
prediction drawn solid on the same axes stays distinguishable from its target. The
checkpoint-driven cases run through a real ``.ckpt`` written by ``tests/predict/planted.py``
over the synthetic COCO slices: nothing is asserted about *which* objects an untrained
network finds, only that the checkpoint-to-grid path runs, titles every panel by its image,
and lays the labels under the predictions when an annotation file is given.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from planted import IMG_SIZE, write_checkpoint

from lucid_yolo._viz.overlay import (
    RenderOptions,
    class_color,
    draw_detections,
    draw_ground_truth,
    draw_keypoints,
    draw_oriented,
    draw_segmentation,
    load_ground_truth,
    predict_and_draw,
    show_predictions,
)
from lucid_yolo.data.targets import Targets
from lucid_yolo.eval.checkpoint import load_eval_module
from lucid_yolo.predict import KeypointPrediction, SegmentedPrediction

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

#: Image the drawing cases work on, ``(height, width)``: not square, so a figure geometry
#: that swapped the axes cannot pass by symmetry.
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

#: Two labelled boxes with distinct classes: the instance set every ground-truth case
#: starts from, so "one dashed artist per instance" is asserted against two, not one.
_TWO_BOXES = torch.tensor([[4.0, 4.0, 20.0, 16.0], [40.0, 28.0, 60.0, 44.0]])
_TWO_LABELS = torch.tensor([1, 0])


@pytest.fixture
def picture() -> NDArray[np.uint8]:
    """A mid-grey ``(height, width, 3)`` image array, the shape the drawing functions take."""
    return np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 128, dtype=np.uint8)


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


def _dashed_unfilled(axes: Axes) -> list[bool]:
    """Report, per patch, whether it is dashed and unfilled — the ground-truth signature.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> from matplotlib.patches import Rectangle
        >>> _, axes = plt.subplots()
        >>> _ = axes.add_patch(Rectangle((0, 0), 1, 1, fill=False, linestyle="--"))
        >>> _ = axes.add_patch(Rectangle((0, 0), 1, 1, fill=True))
        >>> _dashed_unfilled(axes)
        [True, False]
        >>> plt.close(axes.figure)
    """
    return [patch.get_linestyle() == "--" and not patch.get_fill() for patch in axes.patches]


def test_each_detection_becomes_one_labelled_rectangle(picture: NDArray[np.uint8]) -> None:
    """Two A9 rows draw two rectangles at their own corners, each labelled class and score.

    The baseline claim the other detection cases vary: the box a caller reads off the
    figure is the box the tuple carried, corner for corner, and the label says which class
    at what confidence rather than merely marking that something is there.
    """
    detections = torch.tensor([[4.0, 4.0, 20.0, 16.0, 0.9, 1.0], [24.0, 8.0, 44.0, 28.0, 0.6, 0.0]])

    axes = draw_detections(picture, detections)

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
        pytest.param(lambda image: draw_detections(image, torch.zeros(0, 6)), id="detect"),
        pytest.param(
            lambda image: draw_segmentation(
                image,
                SegmentedPrediction(torch.zeros(0, 6), torch.zeros(0, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.bool)),
            ),
            id="segment",
        ),
        pytest.param(lambda image: draw_oriented(image, torch.zeros(0, 7)), id="obb"),
        pytest.param(
            lambda image: draw_keypoints(image, KeypointPrediction(torch.zeros(0, 6), torch.zeros(0, 3, 2))),
            id="keypoints",
        ),
    ],
)
def test_an_empty_prediction_draws_the_picture_and_nothing_over_it(
    picture: NDArray[np.uint8],
    draw_empty: Callable[[NDArray[np.uint8]], Axes],
) -> None:
    """An image with nothing above the threshold still renders, with no annotation artifact.

    The live case for a checkpoint early in training and for any picture the model finds
    empty, on all four tasks: the answer is a picture of the scene, not an exception and
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

    axes = draw_detections(picture, detections, conf_threshold=0.5)

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
    alone = draw_detections(picture, torch.tensor([[4.0, 4.0, 20.0, 16.0, 0.9, 2.0]]))

    accompanied = draw_detections(
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

    axes = draw_oriented(picture, detections)

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

    axes = draw_segmentation(picture, prediction)

    assert len(axes.images) == 3, "the picture, then one overlay per instance"
    overlay = np.asarray(axes.images[1 + instance].get_array())
    opaque = overlay[..., 3] > 0
    assert np.array_equal(opaque, prediction.masks[instance].numpy())
    assert overlay[..., :3][opaque][0].tolist() == pytest.approx(list(class_color(label)))


def test_the_threshold_drops_a_mask_together_with_its_own_box(picture: NDArray[np.uint8]) -> None:
    """A dropped detection takes its mask with it, and the survivor keeps the mask it came with.

    The row-alignment defect this path is most exposed to: two filters, or one filter
    applied twice, leave the boxes right and shift every mask by a row — a plausible
    figure that no shape check catches, because the counts still agree. The surviving
    overlay is compared against instance 0's own mask, so a shift to instance 1's fails
    here rather than in a reader's judgement of the picture.
    """
    prediction = _two_instances()

    axes = draw_segmentation(picture, prediction, conf_threshold=0.5)

    assert len(axes.images) == 2
    assert len(axes.patches) == 1
    overlay = np.asarray(axes.images[1].get_array())
    assert np.array_equal(overlay[..., 3] > 0, prediction.masks[0].numpy())


@pytest.mark.parametrize(
    "draw_mismatched",
    [
        pytest.param(
            lambda image: draw_detections(image, torch.zeros(1, 7)),
            id="oriented-tuple-to-draw_detections",
        ),
        pytest.param(
            lambda image: draw_oriented(image, torch.zeros(1, 6)),
            id="axis-aligned-tuple-to-draw_oriented",
        ),
        pytest.param(
            lambda image: draw_segmentation(
                image,
                SegmentedPrediction(torch.zeros(2, 6), torch.zeros(1, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.bool)),
            ),
            id="one-mask-for-two-detections",
        ),
        pytest.param(
            lambda image: draw_segmentation(
                image,
                SegmentedPrediction(torch.zeros(1, 6), torch.zeros(1, 8, 8, dtype=torch.bool)),
            ),
            id="mask-on-another-grid",
        ),
        pytest.param(
            lambda image: draw_keypoints(
                image, KeypointPrediction(torch.zeros(1, 6), torch.zeros(1, 2, 2)), skeleton=[(0, 2)]
            ),
            id="skeleton-edge-past-the-point-count",
        ),
    ],
)
def test_a_prediction_that_cannot_belong_to_this_picture_is_refused(
    picture: NDArray[np.uint8],
    draw_mismatched: Callable[[NDArray[np.uint8]], Axes],
) -> None:
    """Mismatched tuples, mask stacks and skeletons raise instead of drawing something plausible.

    Each of these has a silent drawing available to it, which is why the guard is worth
    its lines: an A45 row's first four columns drawn as corners give a rectangle in the
    wrong place at the wrong size; an A9 row read as a rotated box gives a heading of
    ``score`` radians; a short mask stack shifts every later pairing; a mask on another
    grid is stretched by ``imshow`` onto the image's extent, landing over a neighbour;
    and a skeleton edge past ``K`` would index into the next point set.
    """
    with pytest.raises(ValueError):
        draw_mismatched(picture)


class TestDrawGroundTruth:
    """``draw_ground_truth`` draws one dashed, unfilled artist per instance, per task."""

    def test_detect_outlines_every_box_dashed(self, picture: NDArray[np.uint8]) -> None:
        """Two labelled boxes become two dashed rectangles at their own corners, with no text.

        The signature every other task case inherits: dashed and unfilled is what tells a
        label from a prediction on shared axes, and a label has no score to write.
        """
        axes = draw_detections(picture, torch.zeros(0, 6))

        draw_ground_truth(axes, Targets(boxes=_TWO_BOXES, labels=_TWO_LABELS), "detect")

        assert _dashed_unfilled(axes) == [True, True]
        assert [patch.get_xy() for patch in axes.patches] == [(4.0, 4.0), (40.0, 28.0)]
        assert [patch.get_edgecolor()[:3] for patch in axes.patches] == [class_color(1), class_color(0)]
        assert len(axes.texts) == 0, "a label has no score to write"

    def test_segment_outlines_every_ring_dashed(self, picture: NDArray[np.uint8]) -> None:
        """Two instance rings become two dashed polygons through the rings' own vertices.

        A ring rather than its envelope: the segmentation label is the outline the mask
        was rasterized from, and a box would hide exactly the boundary a mask is judged on.
        """
        rings = [
            torch.tensor([[4.0, 4.0], [20.0, 4.0], [12.0, 16.0]]),
            torch.tensor([[40.0, 28.0], [60.0, 28.0], [50.0, 44.0]]),
        ]
        axes = draw_detections(picture, torch.zeros(0, 6))

        draw_ground_truth(axes, Targets(boxes=_TWO_BOXES, labels=_TWO_LABELS, polygons=rings), "segment")

        assert _dashed_unfilled(axes) == [True, True]
        assert [len(patch.get_xy()) for patch in axes.patches] == [4, 4], "three vertices, closed"
        assert np.asarray(axes.patches[0].get_xy())[:3].tolist() == rings[0].tolist()

    def test_segment_without_rings_falls_back_to_dashed_boxes(self, picture: NDArray[np.uint8]) -> None:
        """An image the reader gave no rings still draws its boxes, dashed.

        The COCO reader collapses ``polygons`` to ``[]`` when any instance lacks a ring,
        and every box is then ``bbox`` — the label the segmentation path trained against
        on that image, so it is what is drawn.
        """
        axes = draw_detections(picture, torch.zeros(0, 6))

        draw_ground_truth(axes, Targets(boxes=_TWO_BOXES, labels=_TWO_LABELS), "segment")

        assert _dashed_unfilled(axes) == [True, True]
        assert [patch.get_width() for patch in axes.patches] == [16.0, 20.0]

    def test_obb_outlines_every_rotated_box_as_a_dashed_quadrilateral(self, picture: NDArray[np.uint8]) -> None:
        """A rotated label becomes a dashed four-cornered ring at the box's own corners.

        Through the same corner function the predicted rings use, so a predicted ring and
        its label ring are the same four points when the model is right — and the label
        is never the envelope a ``[:, :4]`` misread would draw.
        """
        rboxes = torch.tensor([[*PLANTED_RBOX], [48.0, 36.0, 12.0, 6.0, 1.0]])
        axes = draw_oriented(picture, torch.zeros(0, 7))

        draw_ground_truth(axes, Targets(boxes=_TWO_BOXES, labels=_TWO_LABELS, rboxes=rboxes), "obb")

        assert _dashed_unfilled(axes) == [True, True]
        drawn = np.asarray(axes.patches[0].get_xy()[:4], dtype=np.float64)
        distances = [
            min(float(np.hypot(*(corner - vertex))) for vertex in drawn) for corner in _rotated_corners(PLANTED_RBOX)
        ]
        assert distances == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=CORNER_TOLERANCE)

    def test_keypoints_draws_dashed_boxes_and_a_hollow_ring_per_visible_point(self, picture: NDArray[np.uint8]) -> None:
        """Each labelled instance gets its dashed box and one unfilled ring per point with visibility > 0.

        Visibility ``0`` is COCO's "not labelled", written as ``(0, 0, 0)``: drawing it
        would put a ring at the origin for every point the annotator never placed. The
        prediction has no visibility to consult and draws all ``K``; the label does, and
        uses it.
        """
        points = torch.tensor([[[8.0, 8.0], [16.0, 12.0], [0.0, 0.0]], [[44.0, 30.0], [0.0, 0.0], [0.0, 0.0]]])
        visibility = torch.tensor([[2, 1, 0], [2, 0, 0]])
        axes = draw_detections(picture, torch.zeros(0, 6))

        draw_ground_truth(
            axes,
            Targets(boxes=_TWO_BOXES, labels=_TWO_LABELS, keypoints=points, keypoint_vis=visibility),
            "keypoints",
        )

        assert _dashed_unfilled(axes)[:2] == [True, True], "the two boxes lead"
        rings = axes.patches[2:]
        assert len(rings) == 3, "three visible points across the two instances"
        assert all(not ring.get_fill() for ring in rings)
        assert [ring.center for ring in rings] == [(8.0, 8.0), (16.0, 12.0), (44.0, 30.0)]

    def test_a_task_with_no_drawer_is_refused_by_name(self, picture: NDArray[np.uint8]) -> None:
        """An unknown task raises, naming the four that are drawable, rather than drawing boxes.

        Falling back to boxes would be a plausible figure for a task whose geometry is
        something else — the same silent-misread class the oriented guard exists for.
        """
        axes = draw_detections(picture, torch.zeros(0, 6))

        with pytest.raises(ValueError, match="obb"):
            draw_ground_truth(axes, Targets(boxes=_TWO_BOXES, labels=_TWO_LABELS), "classify")


class TestLoadGroundTruth:
    """``load_ground_truth`` keys the COCO reader's targets by image file name."""

    def test_names_every_image_in_the_file(self, detseg_fixture_dir: Path) -> None:
        """Each image of the synthetic detection slice maps to targets with one label per box.

        The reader is the training path's, so the boxes drawn as labels are the boxes
        the model trained against — the file is not parsed a second time here.
        """
        train = detseg_fixture_dir / "train"

        labels = load_ground_truth(train / "_annotations.coco.json", "detect")

        assert set(labels) == {path.name for path in train.glob("*.jpg")}
        assert all(len(targets.boxes) == len(targets.labels) for targets in labels.values())

    def test_an_oriented_read_carries_rotated_boxes(self, obb_fixture_dir: Path) -> None:
        """Under ``obb`` every instance's ring is fitted to a rotated box, one per label.

        ``rboxes`` is what the oriented drawer outlines; a plain read would leave it
        empty and draw nothing for an oriented label.
        """
        train = obb_fixture_dir / "train"

        labels = load_ground_truth(train / "_annotations.coco.json", "obb")

        assert all(targets.rboxes.shape[0] == targets.labels.shape[0] for targets in labels.values())
        assert any(targets.rboxes.shape[0] > 0 for targets in labels.values())


class TestShowPredictions:
    """``show_predictions`` loads once, draws one titled panel per image, labels dashed on request."""

    @pytest.fixture
    def checkpoint(self, tmp_path: Path) -> Path:
        """A tiny untrained detection checkpoint ``load_eval_module`` reads."""
        return write_checkpoint("detect", tmp_path / "det.ckpt")

    @pytest.fixture
    def options(self) -> RenderOptions:
        """Options matching the planted checkpoint: its letterbox side, raw weights, CPU."""
        return RenderOptions(img_size=IMG_SIZE, ema=False, device="cpu")

    def test_one_titled_panel_per_image_and_blank_cells_after(
        self, checkpoint: Path, detseg_fixture_dir: Path, options: RenderOptions
    ) -> None:
        """Three images in two columns give a 2x2 grid: three panels titled by name, one switched off.

        The panel count and the padding cell are what a reader sees first; a grid that
        dropped the odd image or left a stray empty frame with ticks would be read as a
        missing prediction.
        """
        images = sorted((detseg_fixture_dir / "train").glob("*.jpg"))[:3]

        figure = show_predictions(checkpoint, images, options=options, columns=2)

        assert len(figure.axes) == 4
        assert [axes.get_title() for axes in figure.axes[:3]] == [image.name for image in images]
        assert all(len(axes.images) == 1 for axes in figure.axes[:3]), "every panel holds its picture"
        assert not figure.axes[3].axison

    def test_annotations_lay_dashed_labels_under_every_panel(
        self, checkpoint: Path, detseg_fixture_dir: Path, options: RenderOptions
    ) -> None:
        """With an annotation file, each panel carries one dashed patch per labelled instance.

        The labels are matched by file name and drawn after the prediction, so the dashed
        count per panel equals that image's instance count whatever the untrained network
        drew solid.
        """
        train = detseg_fixture_dir / "train"
        images = sorted(train.glob("*.jpg"))[:2]
        labels = load_ground_truth(train / "_annotations.coco.json", "detect")

        figure = show_predictions(checkpoint, images, train / "_annotations.coco.json", options=options)

        for image, axes in zip(images, figure.axes[:2], strict=True):
            assert sum(_dashed_unfilled(axes)) == len(labels[image.name].boxes)

    def test_an_image_the_annotations_do_not_name_is_refused(
        self, checkpoint: Path, detseg_fixture_dir: Path, options: RenderOptions, tmp_path: Path
    ) -> None:
        """An image absent from the file raises, naming it, rather than drawing a panel without labels.

        A panel quietly missing its labels reads as "the model found something the
        dataset did not", which is the wrong conclusion about the wrong layer.
        """
        train = detseg_fixture_dir / "train"
        stranger = tmp_path / "stranger.jpg"
        stranger.write_bytes(next(train.glob("*.jpg")).read_bytes())

        with pytest.raises(ValueError, match=r"stranger\.jpg"):
            show_predictions(checkpoint, [stranger], train / "_annotations.coco.json", options=options)

    @pytest.mark.parametrize(
        ("columns", "images"),
        [pytest.param(0, 1, id="no-columns"), pytest.param(2, 0, id="no-images")],
    )
    def test_a_grid_with_no_cells_is_refused(
        self, checkpoint: Path, detseg_fixture_dir: Path, options: RenderOptions, columns: int, images: int
    ) -> None:
        """Zero columns or zero images raise before any checkpoint is read.

        ``plt.subplots`` would raise its own, less useful error on a zero-sized grid, and
        an empty figure is not the answer a caller who passed nothing expects.
        """
        paths = sorted((detseg_fixture_dir / "train").glob("*.jpg"))[:images]

        with pytest.raises(ValueError):
            show_predictions(checkpoint, paths, options=options, columns=columns)


def test_predict_and_draw_titles_the_axes_with_the_image_task_and_decoder(
    tmp_path: Path, detseg_fixture_dir: Path
) -> None:
    """A loaded module and an image give axes holding the picture, titled ``name (task, decoder)``.

    The half of the script's render that is not a write: one prediction, drawn into fresh
    axes, with the title the script's figures have always carried.
    """
    module, _ = load_eval_module(write_checkpoint("detect", tmp_path / "det.ckpt"), use_ema=False)
    image = next(iter(sorted((detseg_fixture_dir / "train").glob("*.jpg"))))

    axes = predict_and_draw(module, image, RenderOptions(img_size=IMG_SIZE, device="cpu"))

    assert len(axes.images) == 1
    assert axes.get_title() == f"{image.name} (detect, e2e)"


def test_predict_and_draw_passes_the_skeleton_through_to_the_keypoints_renderer(
    tmp_path: Path, detseg_fixture_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``RenderOptions.skeleton`` reaches ``draw_keypoints``: one edge in, one line drawn.

    The notebooks draw a pose grid through ``predict_and_draw`` like every other task, so
    the skeleton has to travel inside the options rather than through a second call
    path. The prediction is planted, because the checkpoint is a detector: only the
    dispatch and the argument threading are under test.
    """
    module, _ = load_eval_module(write_checkpoint("detect", tmp_path / "det.ckpt"), use_ema=False)
    image = next(iter(sorted((detseg_fixture_dir / "train").glob("*.jpg"))))
    planted = KeypointPrediction(
        torch.tensor([[2.0, 2.0, 20.0, 20.0, 0.9, 0.0]]), torch.tensor([[[4.0, 4.0], [16.0, 16.0]]])
    )
    monkeypatch.setattr("lucid_yolo._viz.overlay.predict_keypoints", lambda *args, **kwargs: planted)

    with_edge = predict_and_draw(module, image, RenderOptions(task="keypoints", skeleton=[(0, 1)], device="cpu"))
    without = predict_and_draw(module, image, RenderOptions(task="keypoints", device="cpu"))

    assert len(with_edge.lines) == 1
    assert list(with_edge.lines[0].get_xdata()) == [4.0, 16.0]
    assert len(without.lines) == 0
