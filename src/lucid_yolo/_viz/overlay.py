# SPDX-License-Identifier: Apache-2.0
"""Draw a checkpoint's predictions, and the ground truth beside them (WP-067, WP-152, WP-189).

A checkpoint, an image, and a figure showing what the model actually answered — boxes
for a ``detect`` checkpoint, boxes and instance masks for a ``segment`` one, rotated
quadrilaterals for an ``obb`` one, boxes with their point sets for a ``keypoints`` one.
The prediction itself is not recomputed here in any form:
:func:`~lucid_yolo.predict.predict_image`,
:func:`~lucid_yolo.predict.predict_segmentation`,
:func:`~lucid_yolo.predict.predict_oriented` and
:func:`~lucid_yolo.predict.predict_keypoints` answer, and this module draws their answer.

Four drawing functions rather than one that switches, because the four answers are four
different shapes and the split is the one the library already makes: a ``(N, 6)``
A9 tuple, that tuple beside a row-aligned mask stack, a ``(N, 7)`` A45 tuple whose
box columns are a centre and two extents rather than corners, and that A9 tuple beside a
``(N, K, 2)`` point stack. Each takes an
already-computed prediction and an image array, so a drawing can be asserted without an
inference — which is what makes the geometry below testable at all.

**A rotated box is drawn as a rotated polygon.** The A45 tuple's first four columns look
like an ``xyxy`` box and are not one, so the tempting ``[:, :4]`` slice draws a plausible
upright rectangle that is not the object the model reported — the exact misreading
:func:`~lucid_yolo.predict.predict_oriented` warns about, and one that survives review
because the picture looks right. The corners come from
:func:`~lucid_yolo.data.rotated_geom.rboxes_to_polygons`, the same function the oriented
report and the DOTA tile writer state an object with, so a drawn ring and a written ring
are the same four points.

**Colour is a function of the class index alone.** A palette walked in encounter order
would give the same class two colours in two pictures — and two classes the same colour
in one — which is exactly what a reader comparing two predictions is not expecting. Class
``k`` is always :func:`class_color`'s ``k``.

**Ground truth is dashed and never filled** (:func:`draw_ground_truth`). A prediction and
its label share a picture in the notebooks, and the two have to be tellable apart at a
glance without a legend: predictions are solid outlines and translucent masks, labels
are dashed outlines and hollow markers, in the same class colours. The label geometry is
what :class:`~lucid_yolo.data.coco.CocoDetectionDataset` yields — the reader the
training path uses — so a drawn label is the target the model trained against, not a
second parse of the JSON.

This is not ``scripts/dump_debug_grid.py`` re-spelled. That one rasterizes *ground truth*
through the augmentation pipeline into a contact sheet, in one colour, to answer whether
the augmentation is sane; this one draws *predictions* on one picture, coloured by class
and labelled by score, to answer what a checkpoint sees. Neither's drawing code is usable
from the other: theirs is tensor-in, tensor-out raster compositing, and this one emits
matplotlib artists, which is what makes it assertable without golden images.

Nothing here selects a backend. ``matplotlib.use("Agg")`` is a process-wide setting
that switches a notebook *away* from its inline display, so the headless choice belongs
to ``scripts/draw_predictions.py`` and the test session that need it, not to a module a
notebook imports.

Assumptions:
    ``RenderOptions.task`` defaults to **the checkpoint's own task**, as ``lucid-predict``
    reads it, rather than being required. A named task selects an entry point, and the
    library then refuses a checkpoint that does not match it by name (a ``segment``
    checkpoint drawn as ``detect`` would answer plausibly with its mask branch unread),
    so the option can narrow the choice and cannot silently mislead.

    Labels are contiguous class **indices**, not names: a checkpoint carries
    ``num_classes`` and no names, and a single image carries no annotation file to map
    them through. A caller who knows the dataset owns that mapping, exactly as they do for
    ``lucid-predict``'s report.

    The confidence cut is applied with a strict ``>``, the comparison
    :func:`~lucid_yolo.predict.predict_image` filters its own rows with. The drawing
    functions repeat it because they also accept a tuple that was never filtered — one
    read back from a report written at ``--conf_threshold 0`` — and two surfaces that
    disagree about whether a score *equal* to the threshold is in the picture would be
    worse than either answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Polygon, Rectangle
from torchvision.io import ImageReadMode, read_image

from lucid_yolo.cli.eval import DEFAULT_IMG_SIZE
from lucid_yolo.data.coco import CocoDetectionDataset
from lucid_yolo.data.rotated_geom import rboxes_to_polygons
from lucid_yolo.decode.common import BOX_CORNERS, DET_WIDTH, LABEL_COLUMN, RBOX_COLUMNS, SCORE_COLUMN
from lucid_yolo.eval.checkpoint import load_eval_module, pick_device
from lucid_yolo.predict import (
    DEFAULT_CONF_THRESHOLD,
    KeypointPrediction,
    SegmentedPrediction,
    predict_image,
    predict_keypoints,
    predict_oriented,
    predict_segmentation,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray
    from torch import Tensor

    from lucid_yolo.data.targets import Targets
    from lucid_yolo.predict import DecodePath
    from lucid_yolo.ptl.module import DetectionLitModule

    #: An image as this module draws it: ``(height, width, 3)`` 8-bit RGB, the layout
    #: :meth:`~matplotlib.axes.Axes.imshow` takes without a conversion of its own.
    ImageArray = NDArray[np.uint8]


#: Class palette: ten qualitative hues indexed by class id, so the colour of class ``k``
#: is a property of ``k`` and of nothing else in the picture. Classes past the tenth
#: repeat a hue rather than fading into a continuous ramp — a reader can tell two adjacent
#: hues apart, which is the property a sequential colormap loses at the tenth class.
_PALETTE = matplotlib.colormaps["tab10"]

#: Opacity of an instance mask overlay. High enough to read as a filled region, low enough
#: that the object under it stays visible — the mask is being checked *against* the
#: picture, so hiding the picture defeats the figure.
_MASK_ALPHA = 0.45

#: Outline width in points for a box, a rotated polygon, and the label's own frame.
_OUTLINE_WIDTH = 1.6

#: Label font size in points, and the padding of the filled frame drawn behind it.
_LABEL_FONT_SIZE = 7.0
_LABEL_PAD = 0.2

#: Relative luminance above which a class colour needs black text rather than white
#: (ITU-R BT.601 weights). Half of ``tab10`` is light enough to swallow white text.
_LUMINANCE_WEIGHTS = (0.299, 0.587, 0.114)
_LIGHT_COLOUR_LUMINANCE = 0.6

#: Figure width in inches; the height follows the image's own aspect ratio, so a drawn
#: box has the shape it has in the picture rather than the shape a fixed canvas gives it.
_FIGURE_WIDTH = 7.5

#: Panel geometry of a :func:`show_predictions` grid, in inches per column and per row.
#: Each panel keeps its image's aspect inside that cell (``imshow`` never stretches), so
#: the cell is a budget rather than a shape.
_GRID_COLUMN_WIDTH, _GRID_ROW_HEIGHT = 5.0, 4.0

#: Column count of the A45 oriented tuple ``[cx, cy, w, h, theta, score, class]``: the
#: five box columns, then the two the A9 tuple also ends with.
_RBOX_WIDTH = RBOX_COLUMNS + 2

#: Column index of an oriented row's score and class, which sit two columns right of the
#: A9 tuple's because the box is a centre, two extents and an angle rather than corners.
_RBOX_SCORE_COLUMN = RBOX_COLUMNS
_RBOX_LABEL_COLUMN = RBOX_COLUMNS + 1

#: Confidence cut the drawing functions default to: draw every row handed over. The rows
#: :mod:`lucid_yolo.predict` returns are already filtered, and a strict ``>`` against zero
#: still drops the score-zero padding rows a report written at ``--conf_threshold 0``
#: carries.
_DRAW_EVERYTHING = 0.0

#: The four tasks, named as :attr:`~lucid_yolo.ptl.module.DetectionLitModule.task` and
#: :data:`~lucid_yolo.cli.eval.DEFAULT_IMG_SIZE` already spell them.
_DETECT_TASK = "detect"
_SEGMENT_TASK = "segment"
_OBB_TASK = "obb"
_KEYPOINTS_TASK = "keypoints"

#: Radius in pixels of a drawn keypoint marker. Fixed rather than scaled to the box: a
#: pose figure is read by whether a point sits on its feature, and a marker that grew with
#: the object would hide exactly the small-object error worth seeing.
_KEYPOINT_RADIUS = 2.0

#: Line width of a skeleton edge, thinner than a box edge so the two read apart.
_SKELETON_WIDTH = 1.0

#: How ground truth is drawn apart from a prediction: dashed where a prediction is solid,
#: and never filled. The dash pattern is matplotlib's named ``"--"``; a marker is a ring
#: rather than a disc. Both are the property the notebooks read — same colour, other
#: line style — so a reader can tell the two layers apart without a legend.
_GROUND_TRUTH_LINESTYLE = "--"
_GROUND_TRUTH_LINEWIDTH = 1.2


def class_color(label: int) -> tuple[float, float, float]:
    """Return the RGB colour class ``label`` is always drawn in.

    A pure function of the class index, which is the whole point: two figures drawn from
    two checkpoints of the same dataset use the same colour for the same class, and a
    class missing from one image does not shift the colours of the classes around it.

    Args:
        label: Contiguous class index. Indices past the palette's length wrap around.

    Returns:
        The ``(red, green, blue)`` triple in matplotlib's ``[0, 1]`` range.

    Examples:
        ```pycon
        >>> class_color(3) == class_color(3)  # stable across calls
        True
        >>> class_color(0) == class_color(1)  # and distinct between neighbours
        False

        ```
    """
    rgba = _PALETTE(int(label) % _PALETTE.N)
    return (float(rgba[0]), float(rgba[1]), float(rgba[2]))


def _figure_size(image: ImageArray) -> tuple[float, float]:
    """Return the figure size in inches for an image of a given shape.

    The width is fixed and the height follows the image's aspect, so the drawn geometry is
    never stretched: a square box on a 2:1 picture has to come out square, or every
    judgement a reader makes about the shape of a prediction is made about the canvas.

    Examples:
        ```pycon
        >>> import numpy as np
        >>> _figure_size(np.zeros((64, 128, 3), dtype=np.uint8))
        (7.5, 3.75)

        ```
    """
    height, width = int(image.shape[0]), int(image.shape[1])
    return (_FIGURE_WIDTH, _FIGURE_WIDTH * height / width)


def _show_image(image: ImageArray, axes: Axes | None) -> Axes:
    """Draw ``image`` as the bottom layer and return the axes holding it.

    Every public drawing function starts here, so ``axes.images[0]`` is the picture and
    anything after it is an overlay this module added. A caller supplying its own axes
    (a grid of predictions, say) gets the image drawn into it for the same reason: an
    annotation without its picture is not a figure.
    """
    target = plt.subplots(figsize=_figure_size(image))[1] if axes is None else axes
    target.imshow(image)
    target.set_axis_off()
    return target


def _require_width(detections: Tensor, width: int, drawer: str) -> None:
    """Refuse a detection tensor whose row is not the tuple ``drawer`` draws.

    The guard exists because the two tuples are assignment-compatible and visually
    plausible in each other's place: an A45 row's first four columns are a centre and two
    extents, which :func:`draw_detections` would draw as corners — a rectangle in the
    wrong place, at the wrong size, with no error and nothing in the figure to show it.
    """
    if detections.ndim != 2 or int(detections.shape[1]) != width:
        raise ValueError(
            f"{drawer} draws {width}-column rows; got shape {tuple(detections.shape)}. The A9 tuple "
            f"[x1, y1, x2, y2, score, class] goes to draw_detections or draw_segmentation, and the A45 "
            f"tuple [cx, cy, w, h, theta, score, class] to draw_oriented, whose box columns are a centre "
            f"and two extents rather than corners."
        )


def _above(detections: Tensor, conf_threshold: float, score_column: int) -> Tensor:
    """Return the row mask of detections scoring **strictly above** ``conf_threshold``."""
    return detections[:, score_column] > conf_threshold


def _label_text_color(colour: tuple[float, float, float]) -> str:
    """Return the text colour readable on a filled ``colour`` frame."""
    luminance = sum(weight * channel for weight, channel in zip(_LUMINANCE_WEIGHTS, colour, strict=True))
    return "black" if luminance > _LIGHT_COLOUR_LUMINANCE else "white"


def _draw_label(axes: Axes, position: tuple[float, float], label: int, score: float) -> None:
    """Write ``class score`` at ``position``, framed in that class's own colour."""
    colour = class_color(label)
    axes.text(
        position[0],
        position[1],
        f"{label} {score:.2f}",
        color=_label_text_color(colour),
        fontsize=_LABEL_FONT_SIZE,
        va="top",
        ha="left",
        bbox={"boxstyle": f"square,pad={_LABEL_PAD}", "facecolor": colour, "linewidth": 0.0},
    )


def _draw_boxes(axes: Axes, detections: Tensor) -> None:
    """Outline every A9 row as a rectangle in its class colour, labelled with its score."""
    for row in detections:
        label = int(row[LABEL_COLUMN])
        x1, y1, x2, y2 = (float(value) for value in row[:BOX_CORNERS])
        axes.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=class_color(label),
                linewidth=_OUTLINE_WIDTH,
            )
        )
        _draw_label(axes, (x1, y1), label, float(row[SCORE_COLUMN]))


def draw_detections(
    image: ImageArray,
    detections: Tensor,
    conf_threshold: float = _DRAW_EVERYTHING,
    axes: Axes | None = None,
) -> Axes:
    """Draw an A9 detection tuple over the image it was predicted on.

    Args:
        image: The **original** image as ``(height, width, 3)`` 8-bit RGB — the frame
            :func:`~lucid_yolo.predict.predict_image` returns its boxes in (A10).
        detections: ``(N, 6)`` rows ``[x1, y1, x2, y2, score, class]`` in original-image
            pixels, as :func:`~lucid_yolo.predict.predict_image` returns them.
        conf_threshold: Rows scoring at or below this are not drawn. Defaults to drawing
            every row handed over, since a prediction has already been filtered once.
        axes: Draw into these axes instead of a new figure's; the image is drawn into them
            either way.

    Returns:
        The axes drawn into: the image as ``images[0]``, then one rectangle patch and one
        label text per surviving detection, in the order the rows arrived.

    Raises:
        ValueError: If ``detections`` is not a 2-D ``(N, 6)`` tensor.

    Examples:
        ```pycon
        >>> import numpy as np, torch
        >>> row = torch.tensor([[4.0, 4.0, 20.0, 16.0, 0.9, 1.0]])
        >>> axes = draw_detections(np.zeros((32, 32, 3), dtype=np.uint8), row)
        >>> len(axes.patches), axes.patches[0].get_width(), axes.texts[0].get_text()
        (1, 16.0, '1 0.90')
        >>> plt.close(axes.figure)

        ```
    """
    _require_width(detections, DET_WIDTH, "draw_detections")
    target = _show_image(image, axes)
    _draw_boxes(target, detections[_above(detections, conf_threshold, SCORE_COLUMN)])
    return target


def _overlay_mask(axes: Axes, mask: Tensor, label: int) -> None:
    """Lay one instance mask over the image as a translucent patch of its class colour.

    One overlay per instance rather than one composited layer for all of them: masks
    routinely touch, and a single layer would answer "some instance covers this pixel"
    where the question a segmentation figure is read to answer is "*which* instance".
    """
    red, green, blue = class_color(label)
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., 0], overlay[..., 1], overlay[..., 2] = red, green, blue
    overlay[..., 3] = mask.numpy(force=True).astype(np.float32) * _MASK_ALPHA
    axes.imshow(overlay, interpolation="nearest")


def draw_segmentation(
    image: ImageArray,
    prediction: SegmentedPrediction,
    conf_threshold: float = _DRAW_EVERYTHING,
    axes: Axes | None = None,
) -> Axes:
    """Draw an instance-segmentation prediction: its masks, then its boxes.

    Boxes and masks are filtered by **one** row mask, computed once, because they are only
    meaningful row-aligned — the pairing
    :class:`~lucid_yolo.predict.SegmentedPrediction` exists to protect. Two filters, or
    one filter applied twice, is how a mask ends up drawn on its neighbour: a plausible
    figure that no shape check catches.

    Args:
        image: The **original** image as ``(height, width, 3)`` 8-bit RGB.
        prediction: The row-aligned detections and masks
            :func:`~lucid_yolo.predict.predict_segmentation` returns, in original-image
            coordinates.
        conf_threshold: Rows scoring at or below this are not drawn, with their masks.
        axes: Draw into these axes instead of a new figure's.

    Returns:
        The axes drawn into: the image as ``images[0]``, then one mask overlay per
        surviving detection in ``images[1:]``, then that detection's rectangle and label.

    Raises:
        ValueError: If the detections are not ``(N, 6)``, if the masks are not one
            per detection, or if a mask's grid is not the image's own.

    Examples:
        ```pycon
        >>> import numpy as np, torch
        >>> from lucid_yolo.predict import SegmentedPrediction
        >>> mask = torch.zeros(1, 32, 32, dtype=torch.bool)
        >>> mask[0, 4:16, 4:20] = True
        >>> prediction = SegmentedPrediction(torch.tensor([[4.0, 4.0, 20.0, 16.0, 0.9, 1.0]]), mask)
        >>> axes = draw_segmentation(np.zeros((32, 32, 3), dtype=np.uint8), prediction)
        >>> len(axes.images), len(axes.patches)  # the picture, one overlay, one box
        (2, 1)
        >>> plt.close(axes.figure)

        ```
    """
    _require_width(prediction.detections, DET_WIDTH, "draw_segmentation")
    _require_masks(prediction, image)
    keep = _above(prediction.detections, conf_threshold, SCORE_COLUMN)
    detections, masks = prediction.detections[keep], prediction.masks[keep]
    target = _show_image(image, axes)
    for row, mask in zip(detections, masks, strict=True):
        _overlay_mask(target, mask, int(row[LABEL_COLUMN]))
    _draw_boxes(target, detections)
    return target


def _require_masks(prediction: SegmentedPrediction, image: ImageArray) -> None:
    """Refuse a mask stack that is not one mask per detection on the image's own grid.

    Both halves are drawing-fatal in the same silent way. A count mismatch pairs mask
    ``n`` with box ``n`` for as long as the shorter stack lasts and shifts every pairing
    after it; a grid mismatch is stretched by ``imshow`` onto the image's extent, which
    lands a mask over a neighbouring object at a plausible size.
    """
    rows = int(prediction.detections.shape[0])
    if int(prediction.masks.shape[0]) != rows:
        raise ValueError(
            f"masks and detections must be row-aligned: got {int(prediction.masks.shape[0])} masks for "
            f"{rows} detections. Filter the two together, as predict_segmentation returns them."
        )
    grid = tuple(int(value) for value in prediction.masks.shape[1:])
    expected = (int(image.shape[0]), int(image.shape[1]))
    if rows and grid != expected:
        raise ValueError(
            f"masks live on a {grid} grid but the image is {expected}. Both must be the original "
            f"image's, which is the frame predict_segmentation maps its masks back onto (A10)."
        )


def _draw_polygons(axes: Axes, detections: Tensor) -> None:
    """Outline every A45 row as the rotated quadrilateral it describes.

    The corners are :func:`~lucid_yolo.data.rotated_geom.rboxes_to_polygons`', computed for
    the whole stack in one call, so ring ``n`` is row ``n``'s own rectangle and no corner
    arithmetic is stated a second time in this repository.
    """
    rings = rboxes_to_polygons(detections[:, :RBOX_COLUMNS])
    for row, ring in zip(detections, rings, strict=True):
        label = int(row[_RBOX_LABEL_COLUMN])
        corners = ring.numpy(force=True)
        axes.add_patch(
            Polygon(corners, closed=True, fill=False, edgecolor=class_color(label), linewidth=_OUTLINE_WIDTH)
        )
        # The topmost corner rather than a fixed one: which corner leads the ring is a
        # function of the heading, so anchoring on ring[0] would put the label under the
        # box for half the angles a canonical theta can take.
        highest = int(corners[:, 1].argmin())
        _draw_label(
            axes, (float(corners[highest, 0]), float(corners[highest, 1])), label, float(row[_RBOX_SCORE_COLUMN])
        )


def draw_oriented(
    image: ImageArray,
    detections: Tensor,
    conf_threshold: float = _DRAW_EVERYTHING,
    axes: Axes | None = None,
) -> Axes:
    """Draw an A45 oriented tuple as rotated quadrilaterals over its image.

    Never as upright rectangles: the four box columns are ``[cx, cy, w, h, theta]``, and
    drawing their axis-aligned envelope would show a rectangle the model never predicted,
    at a size the heading does not imply. The whole point of an oriented figure is the
    heading, so the polygon is the only honest artist for it.

    Args:
        image: The **original** image as ``(height, width, 3)`` 8-bit RGB.
        detections: ``(N, 7)`` rows ``[cx, cy, w, h, theta, score, class]`` in
            original-image pixels, as :func:`~lucid_yolo.predict.predict_oriented` returns
            them (canonical per A23, ``theta`` in radians).
        conf_threshold: Rows scoring at or below this are not drawn.
        axes: Draw into these axes instead of a new figure's.

    Returns:
        The axes drawn into: the image as ``images[0]``, then one four-cornered polygon
        patch and one label text per surviving detection.

    Raises:
        ValueError: If ``detections`` is not a 2-D ``(N, 7)`` tensor.

    Examples:
        ```pycon
        >>> import numpy as np, torch
        >>> row = torch.tensor([[16.0, 16.0, 20.0, 8.0, 0.5, 0.9, 0.0]])
        >>> axes = draw_oriented(np.zeros((32, 32, 3), dtype=np.uint8), row)
        >>> len(axes.patches), len(axes.patches[0].get_xy())  # four corners, ring closed
        (1, 5)
        >>> plt.close(axes.figure)

        ```
    """
    _require_width(detections, _RBOX_WIDTH, "draw_oriented")
    target = _show_image(image, axes)
    _draw_polygons(target, detections[_above(detections, conf_threshold, _RBOX_SCORE_COLUMN)])
    return target


def draw_keypoints(
    image: ImageArray,
    prediction: KeypointPrediction,
    conf_threshold: float = _DRAW_EVERYTHING,
    skeleton: Sequence[tuple[int, int]] | None = None,
    axes: Axes | None = None,
) -> Axes:
    """Draw a keypoint prediction: each detection's box, its points, and any skeleton edges.

    Boxes and point sets are filtered by **one** row mask, computed once, for the reason
    :func:`draw_segmentation` gives: the two are only meaningful row-aligned, and a pose
    drawn on the neighbouring object is a plausible figure no shape check catches.

    ``skeleton`` is a caller argument with no default edges, and deliberately so. Which
    points connect to which is dataset metadata, exactly as the left/right flip pairs are
    (A64): the head predicts ``K`` points and knows nothing about what they mean, so
    baking COCO's human topology in here would make a figure of a 15-point letter
    skeleton or a 12-point animal one silently wrong. With no skeleton the points are
    drawn on their own, which is the honest figure for a schema nobody has named.

    Args:
        image: The **original** image as ``(height, width, 3)`` 8-bit RGB.
        prediction: The row-aligned detections and point sets
            :func:`~lucid_yolo.predict.predict_keypoints` returns, in original-image
            coordinates.
        conf_threshold: Rows scoring at or below this are not drawn, with their points.
        skeleton: Optional ``(start, end)`` index pairs on the point axis, drawn as edges
            in the detection's class colour. Defaults to no edges.
        axes: Draw into these axes instead of a new figure's.

    Returns:
        The axes drawn into: the image as ``images[0]``, then per surviving detection its
        rectangle and label, its skeleton edges, and one marker patch per point.

    Raises:
        ValueError: If the detections are not ``(N, 6)``, if the point sets are not one
            per detection, or if a skeleton edge names a point the prediction lacks.

    Examples:
        ```pycon
        >>> import numpy as np, torch
        >>> from lucid_yolo.predict import KeypointPrediction
        >>> points = torch.tensor([[[8.0, 8.0], [16.0, 12.0]]])
        >>> posed = KeypointPrediction(torch.tensor([[4.0, 4.0, 20.0, 16.0, 0.9, 1.0]]), points)
        >>> axes = draw_keypoints(np.zeros((32, 32, 3), dtype=np.uint8), posed, skeleton=[(0, 1)])
        >>> len(axes.patches), len(axes.lines)  # one box plus two markers, one edge
        (3, 1)
        >>> plt.close(axes.figure)

        ```
    """
    detections, keypoints = prediction.detections, prediction.keypoints
    _require_width(detections, DET_WIDTH, "draw_keypoints")
    if keypoints.shape[0] != detections.shape[0]:
        raise ValueError(
            f"draw_keypoints needs one point set per detection; got {keypoints.shape[0]} sets "
            f"for {detections.shape[0]} detections. The pairing is what KeypointPrediction exists "
            f"to protect, and drawing them unpaired would put every pose on the wrong object."
        )
    for start, end in skeleton or ():
        if not (0 <= start < prediction.num_keypoints and 0 <= end < prediction.num_keypoints):
            raise ValueError(f"skeleton edge {(start, end)} is out of range for K={prediction.num_keypoints}")

    keep = _above(detections, conf_threshold, SCORE_COLUMN)
    target = _show_image(image, axes)
    _draw_boxes(target, detections[keep])
    for row, points in zip(detections[keep], keypoints[keep], strict=True):
        colour = class_color(int(row[LABEL_COLUMN]))
        for start, end in skeleton or ():
            target.plot(
                [float(points[start, 0]), float(points[end, 0])],
                [float(points[start, 1]), float(points[end, 1])],
                color=colour,
                linewidth=_SKELETON_WIDTH,
            )
        for point in points:
            target.add_patch(
                Circle((float(point[0]), float(point[1])), radius=_KEYPOINT_RADIUS, facecolor=colour, edgecolor="none")
            )
    return target


def _dashed_ring(axes: Axes, corners: NDArray[np.float32], label: int) -> None:
    """Outline one closed ring as a dashed, unfilled polygon in its class colour."""
    axes.add_patch(
        Polygon(
            corners,
            closed=True,
            fill=False,
            edgecolor=class_color(label),
            linewidth=_GROUND_TRUTH_LINEWIDTH,
            linestyle=_GROUND_TRUTH_LINESTYLE,
        )
    )


def _dashed_boxes(axes: Axes, targets: Targets) -> None:
    """Outline every labelled box as a dashed rectangle, with no score to write."""
    for box, label in zip(targets.boxes, targets.labels, strict=True):
        x1, y1, x2, y2 = (float(value) for value in box)
        axes.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=class_color(int(label)),
                linewidth=_GROUND_TRUTH_LINEWIDTH,
                linestyle=_GROUND_TRUTH_LINESTYLE,
            )
        )


def _dashed_polygons(axes: Axes, targets: Targets) -> None:
    """Outline each instance ring dashed; an image with no rings falls back to its boxes.

    The reader collapses ``polygons`` to ``[]`` the moment one instance of the image has
    no usable ring, and then every box is COCO's ``bbox`` — so the fallback draws what
    the segmentation path trained against on that image, which is the box.
    """
    if not targets.polygons:
        _dashed_boxes(axes, targets)
        return
    for ring, label in zip(targets.polygons, targets.labels, strict=True):
        _dashed_ring(axes, ring.numpy(force=True), int(label))


def _dashed_rotated(axes: Axes, targets: Targets) -> None:
    """Outline every rotated box as a dashed quadrilateral, through the prediction's own corners."""
    rings = rboxes_to_polygons(targets.rboxes)
    for ring, label in zip(rings, targets.labels, strict=True):
        _dashed_ring(axes, ring.numpy(force=True), int(label))


def _hollow_points(axes: Axes, targets: Targets) -> None:
    """Draw each labelled point set as hollow rings over its dashed box, visible points only.

    A point with visibility ``0`` is unlabelled — COCO writes ``(0, 0, 0)`` for it — so
    drawing it would put a ring at the image origin for every point the annotator never
    placed. The prediction draws every one of its ``K`` points because it has no
    visibility to consult; the label has one and uses it.
    """
    _dashed_boxes(axes, targets)
    for points, visibility, label in zip(targets.keypoints, targets.keypoint_vis, targets.labels, strict=True):
        colour = class_color(int(label))
        for point in points[visibility > 0]:
            axes.add_patch(
                Circle(
                    (float(point[0]), float(point[1])),
                    radius=_KEYPOINT_RADIUS,
                    fill=False,
                    edgecolor=colour,
                    linewidth=_GROUND_TRUTH_LINEWIDTH,
                )
            )


#: Which ground-truth drawer serves which task. The keys are the four task names, so a
#: task with no drawer is refused by name rather than drawn as boxes.
_GROUND_TRUTH_DRAWERS: dict[str, Callable[[Axes, Targets], None]] = {
    _DETECT_TASK: _dashed_boxes,
    _SEGMENT_TASK: _dashed_polygons,
    _OBB_TASK: _dashed_rotated,
    _KEYPOINTS_TASK: _hollow_points,
}


def draw_ground_truth(axes: Axes, targets: Targets, task: str) -> Axes:
    """Draw one image's labels into ``axes``, dashed and unfilled, in the task's own geometry.

    What is drawn is what the task trains against: ``detect`` outlines the boxes,
    ``segment`` the instance rings (or the boxes, on an image the reader gave no rings),
    ``obb`` the rotated quadrilaterals through the same corner function the predicted
    rings use, and ``keypoints`` the boxes with a hollow ring at every labelled point.
    Nothing is filled and no score is written — a label has none — so a prediction drawn
    solid over the same axes stays distinguishable from its target.

    Args:
        axes: Axes already holding the picture, typically the ones a ``draw_*`` function
            returned; nothing is drawn under the labels here.
        targets: The image's :class:`~lucid_yolo.data.targets.Targets` in original-image
            pixels, as :class:`~lucid_yolo.data.coco.CocoDetectionDataset` yields them
            with no transform.
        task: One of ``detect``, ``segment``, ``obb``, ``keypoints``.

    Returns:
        ``axes``, with one dashed patch per instance appended (plus one hollow marker per
        visible point for ``keypoints``).

    Raises:
        ValueError: If ``task`` names no drawer.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> _, axes = plt.subplots()
        >>> labels = Targets(boxes=torch.tensor([[4.0, 4.0, 20.0, 16.0]]), labels=torch.tensor([1]))
        >>> len(draw_ground_truth(axes, labels, "detect").patches), axes.patches[0].get_linestyle()
        (1, '--')
        >>> plt.close(axes.figure)

        ```
    """
    drawer = _GROUND_TRUTH_DRAWERS.get(task)
    if drawer is None:
        raise ValueError(
            f"cannot draw ground truth for task {task!r}; this module draws {sorted(_GROUND_TRUTH_DRAWERS)}"
        )
    drawer(axes, targets)
    return axes


def load_ground_truth(annotations: Path, task: str) -> dict[str, Targets]:
    """Read a COCO annotation file into per-image targets keyed by image file name.

    The reader is :class:`~lucid_yolo.data.coco.CocoDetectionDataset`, the same one the
    training path uses, in the reading mode the task needs — ``oriented`` for ``obb`` so
    each four-point ring becomes a rotated box, ``keypoints`` for ``keypoints`` so each
    annotation's points come through. Nothing is parsed a second time here: the file
    name index and the precomputed targets are the dataset's own.

    Args:
        annotations: A COCO ``instances`` or ``person_keypoints`` JSON file.
        task: One of ``detect``, ``segment``, ``obb``, ``keypoints``; selects the reading
            mode and nothing else.

    Returns:
        Each image's file name, as the JSON spells it, to its
        :class:`~lucid_yolo.data.targets.Targets` in original-image pixels.

    Examples:
        ```pycon
        >>> import json, tempfile
        >>> from pathlib import Path
        >>> payload = {
        ...     "categories": [{"id": 1}],
        ...     "images": [{"id": 3, "file_name": "a.png", "height": 32, "width": 32}],
        ...     "annotations": [{"id": 1, "image_id": 3, "category_id": 1, "bbox": [4, 4, 16, 12], "iscrowd": 0}],
        ... }
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "instances.json"
        ...     _ = path.write_text(json.dumps(payload), encoding="utf-8")
        ...     load_ground_truth(path, "detect")["a.png"].boxes.tolist()
        [[4.0, 4.0, 20.0, 16.0]]

        ```
    """
    dataset = CocoDetectionDataset(
        annotations.parent, annotations, oriented=task == _OBB_TASK, keypoints=task == _KEYPOINTS_TASK
    )
    return {name: dataset.targets_at(index) for index, name in enumerate(dataset.file_names)}


@dataclass(frozen=True)
class RenderOptions:
    """What the caller asked for, before any checkpoint has been read.

    Frozen and named rather than a bag of parameters threaded through four renderers:
    the values arrive together from one parse, travel together, and none of them is
    meaningful to change halfway through a render.

    Attributes:
        task: Which entry point to draw through, or ``None`` to read the checkpoint's own
            task — the default, and what ``lucid-predict`` does. A named task that the
            checkpoint contradicts is refused by the library, by name.
        decoder: ``"e2e"`` for the suppression-free top-k path over the one-to-one branch,
            ``"nms"`` for the suppression path over the dense one.
        conf_threshold: Detections at or below this score are neither predicted nor drawn.
        img_size: Letterbox side the model sees, or ``None`` for the task's own default
            from :data:`~lucid_yolo.cli.eval.DEFAULT_IMG_SIZE`.
        device: ``auto``, ``cpu``, ``mps`` or ``cuda``, as ``lucid-eval`` spells it.
        ema: Predict with the EMA shadow stored in the checkpoint rather than raw weights,
            as ``lucid-predict`` defaults to.
        title: Figure title, or ``None`` for the image's name and the task drawn.
        skeleton: ``(start, end)`` point-index pairs drawn as edges by the keypoints
            renderer, or ``None`` for points alone; the other renderers ignore it. Dataset
            metadata, as :func:`draw_keypoints` explains — COCO's ``categories[].skeleton``
            is 1-based, so subtract one from each index before passing it.

    Examples:
        ```pycon
        >>> RenderOptions().task is None, RenderOptions().decoder, RenderOptions().ema
        (True, 'e2e', True)

        ```
    """

    task: str | None = None
    decoder: DecodePath = "e2e"
    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    img_size: int | None = None
    device: str = "auto"
    ema: bool = True
    title: str | None = None
    skeleton: Sequence[tuple[int, int]] | None = None


@dataclass(frozen=True)
class _Run:
    """One prediction's arguments, after the checkpoint has answered what it is.

    :class:`RenderOptions` carries what a caller asked for, including the two questions
    only a loaded checkpoint can settle — which task, and therefore which letterbox side.
    Resolving them once into this pair keeps the four renderers free of a defaulting rule
    each, which is how the oriented path would end up drawn at 640 px.
    """

    image: Path
    img_size: int
    decoder: DecodePath
    conf_threshold: float
    device: torch.device
    skeleton: Sequence[tuple[int, int]] | None


def _render_detect(module: DetectionLitModule, run: _Run, image: ImageArray, axes: Axes | None) -> Axes:
    """Predict axis-aligned boxes and draw them."""
    detections = predict_image(
        module,
        run.image,
        img_size=run.img_size,
        decoder=run.decoder,
        conf_threshold=run.conf_threshold,
        device=run.device,
    )
    # The drawing threshold is left at its default: these rows have already been filtered
    # by `predict_image` at exactly this cut, and restating it here would put one number
    # in two places for no second effect.
    return draw_detections(image, detections, axes=axes)


def _render_segment(module: DetectionLitModule, run: _Run, image: ImageArray, axes: Axes | None) -> Axes:
    """Predict boxes with their instance masks and draw both."""
    prediction = predict_segmentation(
        module,
        run.image,
        img_size=run.img_size,
        decoder=run.decoder,
        conf_threshold=run.conf_threshold,
        device=run.device,
    )
    return draw_segmentation(image, prediction, axes=axes)


def _render_oriented(module: DetectionLitModule, run: _Run, image: ImageArray, axes: Axes | None) -> Axes:
    """Predict rotated boxes and draw them as rotated polygons."""
    detections = predict_oriented(
        module,
        run.image,
        img_size=run.img_size,
        decoder=run.decoder,
        conf_threshold=run.conf_threshold,
        device=run.device,
    )
    return draw_oriented(image, detections, axes=axes)


def _render_keypoints(module: DetectionLitModule, run: _Run, image: ImageArray, axes: Axes | None) -> Axes:
    """Predict boxes with their point sets and draw both."""
    prediction = predict_keypoints(
        module,
        run.image,
        img_size=run.img_size,
        decoder=run.decoder,
        conf_threshold=run.conf_threshold,
        device=run.device,
    )
    return draw_keypoints(image, prediction, skeleton=run.skeleton, axes=axes)


#: Which entry point draws which task. The keys are also the script's ``--task`` choices,
#: so the flag cannot offer a task no renderer serves, and adding one is a single entry.
_RENDERERS: dict[str, Callable[[DetectionLitModule, _Run, ImageArray, Axes | None], Axes]] = {
    _DETECT_TASK: _render_detect,
    _SEGMENT_TASK: _render_segment,
    _OBB_TASK: _render_oriented,
    _KEYPOINTS_TASK: _render_keypoints,
}

#: The tasks a checkpoint can be drawn through, in a stable order — what a CLI offers as
#: its ``--task`` choices, read off the renderer table rather than restated.
DRAWABLE_TASKS: tuple[str, ...] = tuple(sorted(_RENDERERS))


def _read_image_array(path: Path) -> ImageArray:
    """Read an image file as ``(height, width, 3)`` 8-bit RGB.

    The same decode the evaluation path uses — ``read_image(..., ImageReadMode.RGB)``, per
    :func:`~lucid_yolo.eval.annotations.read_letterboxed_image` — so the pixels drawn under
    a prediction are the pixels the prediction was made from. What is deliberately *not*
    reused is the letterboxing that follows it: predictions come back in original-image
    coordinates (A10), so the original image is the frame they are drawn on.
    """
    raw = read_image(str(path), ImageReadMode.RGB)
    array: ImageArray = raw.permute(1, 2, 0).numpy()
    return array


def _resolve_task(module: DetectionLitModule, options: RenderOptions) -> str:
    """Return the task to draw through, refusing one no renderer serves."""
    task = str(module.task) if options.task is None else options.task
    if task not in _RENDERERS:
        raise ValueError(
            f"cannot draw task {task!r}; this module draws {list(DRAWABLE_TASKS)}. A checkpoint of another "
            f"task has no entry point in lucid_yolo.predict either."
        )
    return task


def _resolve_run(image: Path, task: str, options: RenderOptions) -> _Run:
    """Settle the letterbox side and the device once the task is known."""
    return _Run(
        image=image,
        img_size=DEFAULT_IMG_SIZE[task] if options.img_size is None else options.img_size,
        decoder=options.decoder,
        conf_threshold=options.conf_threshold,
        device=pick_device(options.device),
        skeleton=options.skeleton,
    )


def predict_and_draw(
    module: DetectionLitModule,
    image: Path,
    options: RenderOptions | None = None,
    axes: Axes | None = None,
) -> Axes:
    """Predict one image with a loaded module and draw the answer over the picture.

    The half of the script's ``render_prediction`` that is not a file write: the task is
    read off the module (or taken from ``options``), the letterbox side and device are
    settled, the image is decoded once for the drawing, and the task's renderer predicts
    and draws. The axes are titled with the image's name, the task and the decode path
    unless ``options.title`` says otherwise.

    Args:
        module: An eval-mode module, as :func:`~lucid_yolo.eval.checkpoint.load_eval_module`
            returns it — loaded once by the caller so a grid of images pays for it once.
        image: Image file to run on and draw over.
        options: The rest of the request; ``None`` means :class:`RenderOptions`' defaults.
        axes: Draw into these axes; ``None`` opens a new figure sized to the image.

    Returns:
        The axes drawn into, holding the picture and the prediction's artists.

    Raises:
        ValueError: If the checkpoint's task (or the one ``options`` names) is not one
            this module draws.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> from lucid_yolo.eval.checkpoint import load_eval_module
        >>> module, _ = load_eval_module(Path("det.ckpt"), use_ema=False)  # needs a checkpoint  # doctest: +SKIP
        >>> axes = predict_and_draw(module, Path("a.jpg"))  # needs a checkpoint  # doctest: +SKIP

        ```
    """
    resolved = RenderOptions() if options is None else options
    task = _resolve_task(module, resolved)
    run = _resolve_run(image, task, resolved)
    picture = _read_image_array(image)
    target = _RENDERERS[task](module, run, picture, axes)
    target.set_title(f"{image.name} ({task}, {run.decoder})" if resolved.title is None else resolved.title, fontsize=10)
    return target


def show_predictions(
    checkpoint: Path,
    images: Sequence[Path],
    annotations: Path | None = None,
    *,
    options: RenderOptions | None = None,
    columns: int = 2,
) -> Figure:
    """Draw one checkpoint's predictions over several images, one panel each, for a notebook.

    The checkpoint is loaded once and every panel predicts through the same module. With
    ``annotations`` given, each image's labels are drawn dashed under the same axes, so
    the panel reads as prediction against target; an image the file does not name is an
    error rather than a panel quietly missing its labels. Images are matched to their
    annotations by file name, which is how COCO names them.

    Args:
        checkpoint: Lightning ``.ckpt`` to predict with. Releases ship none (D14): this is
            a checkpoint the operator trained or was given.
        images: Image files to run on, one panel each, in this order.
        annotations: Optional COCO JSON naming every image in ``images``; ``None`` draws
            predictions alone.
        options: The rest of the request; ``None`` means :class:`RenderOptions`' defaults.
            ``title`` is ignored — every panel is titled with its image's name.
        columns: Panels per row. The last row is padded with blank, axis-less cells.

    Returns:
        The open figure, one titled axes per image in ``figure.axes[:len(images)]``.

    Raises:
        ValueError: If ``columns`` is not positive, if ``images`` is empty, if the task is
            not one this module draws, or if an image is missing from ``annotations``.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> figure = show_predictions(
        ...     Path("det.ckpt"), [Path("a.jpg"), Path("b.jpg")]
        ... )  # needs a checkpoint  # doctest: +SKIP

        ```
    """
    if columns < 1:
        raise ValueError(f"columns must be positive; got {columns}")
    if not images:
        raise ValueError("show_predictions needs at least one image")
    resolved = RenderOptions() if options is None else options
    module, _info = load_eval_module(checkpoint, use_ema=resolved.ema)
    task = _resolve_task(module, resolved)
    labels = None if annotations is None else _labels_for(images, load_ground_truth(annotations, task))
    rows = math.ceil(len(images) / columns)
    figure, grid = plt.subplots(rows, columns, figsize=(_GRID_COLUMN_WIDTH * columns, _GRID_ROW_HEIGHT * rows))
    panels = list(np.asarray(grid).reshape(-1))
    for image, axes in zip(images, panels, strict=False):
        predict_and_draw(module, image, resolved, axes)
        if labels is not None:
            draw_ground_truth(axes, labels[image.name], task)
        axes.set_title(image.name, fontsize=10)
    for axes in panels[len(images) :]:
        axes.set_axis_off()
    figure.tight_layout()
    return figure


def _labels_for(images: Sequence[Path], ground_truth: dict[str, Targets]) -> dict[str, Targets]:
    """Return the labels of exactly ``images``, refusing an image the annotations never name."""
    missing = sorted({image.name for image in images} - ground_truth.keys())
    if missing:
        raise ValueError(f"annotations name no image called {missing}; matching is by file name, as COCO names images")
    return {image.name: ground_truth[image.name] for image in images}
