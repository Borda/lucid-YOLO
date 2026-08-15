#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Draw one checkpoint's predictions over the image they were made on (WP-067).

The release's worked example: a checkpoint, an image, and a figure showing what the model
actually answered — boxes for a ``detect`` checkpoint, boxes and instance masks for a
``segment`` one, rotated quadrilaterals for an ``obb`` one. The prediction itself is not
recomputed here in any form: :func:`~lucid_yolo.predict.predict_image`,
:func:`~lucid_yolo.predict.predict_segmentation` and
:func:`~lucid_yolo.predict.predict_oriented` answer, and this module draws their answer.

Three drawing functions rather than one that switches, because the three answers are
three different shapes and the split is the one the library already makes: a ``(N, 6)``
A9 tuple, that tuple beside a row-aligned mask stack, and a ``(N, 7)`` A45 tuple whose
box columns are a centre and two extents rather than corners. Each takes an
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

This is not ``scripts/dump_debug_grid.py`` re-spelled. That one rasterizes *ground truth*
through the augmentation pipeline into a contact sheet, in one colour, to answer whether
the augmentation is sane; this one draws *predictions* on one picture, coloured by class
and labelled by score, to answer what a checkpoint sees. Neither's drawing code is usable
from the other: theirs is tensor-in, tensor-out raster compositing, and this one emits
matplotlib artists, which is what makes it assertable without golden images.

``matplotlib`` is a ``dev`` dependency and stays one: nothing under ``src/lucid_yolo/``
imports it, at any level, so a wheel a consumer installs pulls no plotting stack. That is
why this ships as a script rather than as a ``lucid_yolo.draw`` module, and why the
figure-producing code is here beside ``plot_training.py``.

Assumptions:
    ``--task`` defaults to **the checkpoint's own task**, as ``lucid-predict`` reads it,
    rather than being required. A named task selects an entry point, and the library then
    refuses a checkpoint that does not match it by name (a ``segment`` checkpoint drawn as
    ``detect`` would answer plausibly with its mask branch unread), so the flag can narrow
    the choice and cannot silently mislead.

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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import numpy as np
import torch
from torchvision.io import ImageReadMode, read_image

from lucid_yolo.cli.eval import DEFAULT_IMG_SIZE
from lucid_yolo.data.rotated_geom import rboxes_to_polygons
from lucid_yolo.decode.common import BOX_CORNERS, DET_WIDTH, LABEL_COLUMN, RBOX_COLUMNS, SCORE_COLUMN
from lucid_yolo.eval.checkpoint import load_eval_module, pick_device
from lucid_yolo.predict import (
    DECODE_PATHS,
    DEFAULT_CONF_THRESHOLD,
    SegmentedPrediction,
    predict_image,
    predict_oriented,
    predict_segmentation,
)

matplotlib.use("Agg")

# Imported below the backend selection, not above it: pyplot binds whatever backend is
# current when it is first imported, so a figure would go looking for a display that no
# gate has. Same order, same reason, as `scripts/plot_training.py`.
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes
    from numpy.typing import NDArray
    from torch import Tensor

    from lucid_yolo.predict import DecodePath
    from lucid_yolo.ptl.module import DetectionLitModule

    #: An image as this module draws it: ``(height, width, 3)`` 8-bit RGB, the layout
    #: :meth:`~matplotlib.axes.Axes.imshow` takes without a conversion of its own.
    ImageArray = NDArray[np.uint8]


__all__ = [
    "RenderOptions",
    "class_color",
    "draw_detections",
    "draw_oriented",
    "draw_segmentation",
    "main",
    "render_prediction",
]

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

#: Raster resolution of the written figure. A vector suffix (``.svg``) ignores it.
_FIGURE_DPI = 150

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

#: The three tasks, named as :attr:`~lucid_yolo.ptl.module.DetectionLitModule.task` and
#: :data:`~lucid_yolo.cli.eval.DEFAULT_IMG_SIZE` already spell them.
_DETECT_TASK = "detect"
_SEGMENT_TASK = "segment"
_OBB_TASK = "obb"


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


@dataclass(frozen=True)
class RenderOptions:
    """What the command was asked for, before any checkpoint has been read.

    Frozen and named rather than a bag of parameters threaded through three renderers:
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


@dataclass(frozen=True)
class _Run:
    """One prediction's arguments, after the checkpoint has answered what it is.

    :class:`RenderOptions` carries what a caller asked for, including the two questions
    only a loaded checkpoint can settle — which task, and therefore which letterbox side.
    Resolving them once into this pair keeps the three renderers free of a defaulting rule
    each, which is how the oriented path would end up drawn at 640 px.
    """

    image: Path
    img_size: int
    decoder: DecodePath
    conf_threshold: float
    device: torch.device


def _render_detect(module: DetectionLitModule, run: _Run, image: ImageArray, axes: Axes) -> Axes:
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


def _render_segment(module: DetectionLitModule, run: _Run, image: ImageArray, axes: Axes) -> Axes:
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


def _render_oriented(module: DetectionLitModule, run: _Run, image: ImageArray, axes: Axes) -> Axes:
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


#: Which entry point draws which task. The keys are also the ``--task`` choices, so the
#: flag cannot offer a task no renderer serves, and adding one is a single entry.
_RENDERERS: dict[str, Callable[[DetectionLitModule, _Run, ImageArray, Axes], Axes]] = {
    _DETECT_TASK: _render_detect,
    _SEGMENT_TASK: _render_segment,
    _OBB_TASK: _render_oriented,
}


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


def render_prediction(checkpoint: Path, image: Path, output: Path, options: RenderOptions) -> Path:
    """Predict one image with one checkpoint and write the annotated figure.

    Args:
        checkpoint: Lightning ``.ckpt`` to predict with. Releases ship none (D14): this is
            a checkpoint the operator trained or was given.
        image: Image file to run on and draw over.
        output: Destination figure path; the suffix chooses the format and parent
            directories are created.
        options: The rest of the request — see :class:`RenderOptions`.

    Returns:
        The written ``output`` path.

    Raises:
        ValueError: If the checkpoint's task (or the one ``options`` names) is not one
            this script draws.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> render_prediction(Path("det.ckpt"), Path("a.jpg"), Path("a.png"), RenderOptions())  # doctest: +SKIP

        ```
    """
    module, _info = load_eval_module(checkpoint, use_ema=options.ema)
    task = str(module.task) if options.task is None else options.task
    renderer = _RENDERERS.get(task)
    if renderer is None:
        raise ValueError(
            f"cannot draw task {task!r}; this script draws {sorted(_RENDERERS)}. A checkpoint of another "
            f"task has no entry point in lucid_yolo.predict either."
        )
    run = _Run(
        image=image,
        img_size=DEFAULT_IMG_SIZE[task] if options.img_size is None else options.img_size,
        decoder=options.decoder,
        conf_threshold=options.conf_threshold,
        device=pick_device(options.device),
    )
    picture = _read_image_array(image)
    figure, axes = plt.subplots(figsize=_figure_size(picture))
    renderer(module, run, picture, axes)
    axes.set_title(f"{image.name} ({task}, {run.decoder})" if options.title is None else options.title, fontsize=10)
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
        >>> main(["--help"])  # doctest: +SKIP

        ```
    """
    parser = argparse.ArgumentParser(description="Draw a checkpoint's predictions over the image it read.")
    parser.add_argument("checkpoint", type=Path, help="Lightning .ckpt to predict with")
    parser.add_argument("image", type=Path, help="image file to run on and draw over")
    parser.add_argument("--output", type=Path, required=True, help="destination figure path (.png, .svg, ...)")
    parser.add_argument(
        "--task",
        choices=sorted(_RENDERERS),
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
