# SPDX-License-Identifier: Apache-2.0
"""Debug visualizer: dump an annotated grid of augmented training samples (WP-015).

Draws ``--samples`` samples through the Phase-1 augmentation pipeline
(:class:`~open_yolos.ptl.datamodule._TrainPipeline`: mosaic, affine, letterbox,
mixup, copy-paste, HSV jitter, flip) over a fixture-style COCO directory and
writes a single annotated grid PNG: axis-aligned boxes as rectangles (via
:func:`torchvision.utils.draw_bounding_boxes`) and instance polygons as outlines
(via :class:`PIL.ImageDraw`). It is a hand-run inspection aid for the pipeline —
it never downloads a dataset or reaches the network, only reads an already-present
COCO split (typically the seeded WP-007 synthetic fixtures).

Examples:
    Dump an eight-sample grid from a generated fixture set::

        python scripts/dump_debug_grid.py \\
            --data-root tests/fixtures/_generated/detseg \\
            --out /tmp/detseg_grid.png --samples 8 --seed 0
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torchvision.utils import draw_bounding_boxes, make_grid

from open_yolos.data.coco import CocoDetectionDataset, build_scale_policy
from open_yolos.data.targets import Targets
from open_yolos.ptl.datamodule import _TrainPipeline

#: Per-split COCO annotation filename emitted by the fixture generator's CocoWriter.
_COCO_ANNOTATION = "_annotations.coco.json"

#: Default square letterbox side for the rendered samples.
_DEFAULT_IMG_SIZE = 320

#: Default augmentation-strength policy variant (mildest recipe).
_DEFAULT_VARIANT = "n"

#: 8-bit pixel ceiling used when converting a ``[0, 1]`` float image to ``uint8``.
_UINT8_MAX = 255.0

#: Box rectangle colour and instance-polygon outline colour for the annotations.
_BOX_COLOR = "red"
_POLYGON_COLOR = "lime"

#: Minimum vertices a polygon ring needs before it can be outlined.
_MIN_RING_POINTS = 2


def resolve_coco_split(data_root: Path) -> tuple[Path, Path]:
    """Resolve the images directory and annotation file for a fixture-style COCO dir.

    Accepts either a split directory that directly holds ``_annotations.coco.json``
    (and its images), or a parent dataset directory whose ``train/`` subdirectory
    does — the shape the WP-007 fixture generator writes.

    Args:
        data_root: The COCO directory to resolve.

    Returns:
        An ``(images_dir, annotation_file)`` pair.

    Raises:
        FileNotFoundError: If no ``_annotations.coco.json`` is found at either
            ``data_root`` or ``data_root/train``.

    Examples:
        ```pycon
        >>> resolve_coco_split(Path("/no/such/dir"))  # doctest: +SKIP
        >>> # -> (images_dir, annotation_file) when the annotation JSON exists

        ```
    """
    for images_dir in (data_root, data_root / "train"):
        annotation_file = images_dir / _COCO_ANNOTATION
        if annotation_file.is_file():
            return images_dir, annotation_file
    raise FileNotFoundError(f"no {_COCO_ANNOTATION} under {data_root} or {data_root / 'train'}")


def build_pipeline(data_root: Path, seed: int, img_size: int, variant: str) -> _TrainPipeline:
    """Build the train augmentation pipeline over a fixture-style COCO directory.

    Args:
        data_root: The COCO directory (a split dir or its parent) to read.
        seed: Seed for the pipeline's single sampling generator.
        img_size: Square letterbox side for every emitted sample.
        variant: Model-size letter selecting the augmentation-strength policy.

    Returns:
        A ready-to-draw :class:`~open_yolos.ptl.datamodule._TrainPipeline`.

    Examples:
        ```pycon
        >>> build_pipeline(Path("."), 0, 320, "n")  # doctest: +SKIP
        >>> # -> _TrainPipeline over data_root's COCO split

        ```
    """
    images_dir, annotation_file = resolve_coco_split(data_root)
    base = CocoDetectionDataset(images_dir, annotation_file)
    return _TrainPipeline(base, img_size, build_scale_policy(variant), seed)


def _to_uint8(image: Tensor) -> Tensor:
    """Convert a CHW float image in ``[0, 1]`` to a CHW ``uint8`` tensor."""
    return (image.detach().clamp(0.0, 1.0) * _UINT8_MAX).round().to(torch.uint8)


def _draw_boxes(image_u8: Tensor, boxes: Tensor) -> Tensor:
    """Draw the non-degenerate axis-aligned boxes onto a CHW ``uint8`` image."""
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if not bool(valid.any()):
        return image_u8
    drawn = draw_bounding_boxes(image_u8, boxes[valid], colors=_BOX_COLOR, width=2)
    return cast(Tensor, drawn.to(torch.uint8))


def _draw_polygons(image_u8: Tensor, polygons: list[Tensor]) -> Tensor:
    """Outline every instance polygon ring onto a CHW ``uint8`` image via PIL."""
    rings = [ring for ring in polygons if ring.shape[0] >= _MIN_RING_POINTS]
    if not rings:
        return image_u8
    canvas = Image.fromarray(image_u8.permute(1, 2, 0).cpu().numpy())
    draw = ImageDraw.Draw(canvas)
    for ring in rings:
        draw.polygon([(float(x), float(y)) for x, y in ring.tolist()], outline=_POLYGON_COLOR)
    restored = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).contiguous()
    return restored.to(torch.uint8)


def render_sample(image: Tensor, targets: Targets) -> Tensor:
    """Annotate one augmented sample with its boxes and polygon outlines.

    Args:
        image: The sample's CHW float image in ``[0, 1]``.
        targets: The sample's geometry (boxes and instance polygons).

    Returns:
        A CHW ``uint8`` tensor with red box rectangles and lime polygon outlines
        drawn over the image.

    Examples:
        ```pycon
        >>> import torch
        >>> from open_yolos.data.targets import Targets
        >>> img = torch.zeros(3, 8, 8)
        >>> t = Targets(boxes=torch.tensor([[1.0, 1.0, 6.0, 6.0]]), labels=torch.tensor([0]))
        >>> render_sample(img, t).shape
        torch.Size([3, 8, 8])

        ```
    """
    image_u8 = _to_uint8(image)
    if targets.boxes.shape[0] > 0:
        image_u8 = _draw_boxes(image_u8, targets.boxes)
    return _draw_polygons(image_u8, targets.polygons)


def render_grid(pipeline: _TrainPipeline, num_samples: int) -> Tensor:
    """Render the first ``num_samples`` augmented samples into one annotated grid.

    Args:
        pipeline: The augmentation pipeline to draw from.
        num_samples: Requested sample count, clamped to the dataset size.

    Returns:
        A CHW ``uint8`` grid image laid out in a near-square arrangement.

    Raises:
        ValueError: If ``num_samples`` is not positive.

    Examples:
        ```pycon
        >>> render_grid(pipeline, 4)  # doctest: +SKIP
        >>> # -> (3, H, W) uint8 grid of four annotated samples

        ```
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive; got {num_samples}")
    count = min(num_samples, len(pipeline))
    tiles = [render_sample(*pipeline[index]) for index in range(count)]
    grid = make_grid(tiles, nrow=math.ceil(math.sqrt(count)), padding=2, pad_value=255)
    return cast(Tensor, grid.to(torch.uint8))


def save_grid(grid: Tensor, out: Path) -> None:
    """Write a CHW ``uint8`` grid tensor to ``out`` as a PNG (creating parent dirs).

    Args:
        grid: The CHW ``uint8`` grid image.
        out: Destination PNG path.

    Examples:
        ```pycon
        >>> save_grid(grid, Path("/tmp/grid.png"))  # doctest: +SKIP

        ```
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid.permute(1, 2, 0).cpu().numpy()).save(out)


def main(argv: list[str] | None = None) -> int:
    """Render an annotated augmentation grid from a fixture-style COCO directory.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` on success.

    Examples:
        ```pycon
        >>> main(["--data-root", "tests/fixtures/_generated/detseg", "--out", "/tmp/g.png"])  # doctest: +SKIP
        0

        ```
    """
    parser = argparse.ArgumentParser(description="Dump an annotated grid of augmented training samples.")
    parser.add_argument("--data-root", type=Path, required=True, help="fixture-style COCO directory to read")
    parser.add_argument("--out", type=Path, required=True, help="output PNG path for the annotated grid")
    parser.add_argument("--samples", type=int, default=8, help="number of samples to draw (default: 8)")
    parser.add_argument("--seed", type=int, default=0, help="pipeline sampling seed (default: 0)")
    parser.add_argument("--img-size", type=int, default=_DEFAULT_IMG_SIZE, help="square letterbox side (default: 320)")
    parser.add_argument(
        "--variant", type=str, default=_DEFAULT_VARIANT, help="augmentation policy variant (default: n)"
    )
    args = parser.parse_args(argv)

    pipeline = build_pipeline(args.data_root, args.seed, args.img_size, args.variant)
    grid = render_grid(pipeline, args.samples)
    save_grid(grid, args.out)
    print(f"wrote {args.samples}-sample grid to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
