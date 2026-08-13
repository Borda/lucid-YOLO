# SPDX-License-Identifier: Apache-2.0
"""Build the tiled COCO layout the OBB-smoke tier trains on, from a DOTA-v1.0 root (WP-094).

WP-057 built the tiling geometry and WP-056 the label parsing, and both deliberately
write nothing: "the tiles themselves are a **build artifact** (AGENTS.md sec. 3), so
nothing here writes to disk on its own." This script is the caller that decides where
they land. Until it existed, :func:`~lucid_yolo.data.tiling.tile_image_targets` had no
production consumer at all and ``obb_smoke.yaml``'s ``data_root`` was a placeholder
pointing at a directory nothing could produce.

What it emits:
    A COCO-layout root — ``<split>/`` image directories beside
    ``annotations/instances_<split>.json`` — whose ``segmentation`` is the four-corner
    quadrilateral of the oriented annotation, which is exactly what
    :class:`~lucid_yolo.data.coco.CocoDetectionDataset` reads under ``oriented=True``
    (its "COCO file written for an oriented task" path). Nothing about the layout is
    DOTA-specific, so the same recipe trains on any oriented dataset converted into it.

Category ids (R18):
    ``DOTA_CLASSES`` index plus one. The reader assigns dense labels by *sorted category
    id*, so this offset preserves R18's published class order through the label space:
    category ``i + 1`` becomes label ``i``. The offset itself is COCO convention (no
    category 0); nothing downstream depends on it beyond the ordering it protects.

The ``difficult`` flag (A53):
    Tiling **creates** difficult instances — R18 flags any part below 0.7 of its original
    area, so a label file with no difficult objects still produces tiles that have them —
    and COCO has no field for that. Each annotation therefore carries a non-schema
    ``difficult`` key, forwarded by the reader onto the A51 channel of
    :class:`~lucid_yolo.data.targets.Targets` and acted on at the metric under A48.
    ``visible_fraction`` (R18's ``U_i``) rides along on the same annotation: the fitted
    box's area is not the clipped area, so the quantity the rule turns on is otherwise
    unrecoverable, and a sweep over the threshold should not have to re-tile.

Window provenance (A53):
    Every image record carries ``source_image`` and ``window``. A per-tile number is not
    comparable to anything published — in an NMS-free path two tiles detecting one object
    have nothing to suppress the duplicate — so whole-image evaluation needs a merge step
    (WP-064). That step needs to know which source image a tile came from and where it
    sat; recovering it by parsing file names would make the naming a load-bearing
    interface. The name still encodes it, for a human reading a directory listing.

Empty tiles (A52):
    Kept by default. A 1024 px window over a DOTA image frequently contains no annotated
    object, and neither R1 nor R18 says whether such crops are trained on. Dropping them
    would remove exactly the background the one-to-one branch must learn not to fire on;
    keeping them costs disk. ``--drop-empty-tiles`` selects the other reading, and the
    report prints how many tiles the choice covers either way.

This module never downloads anything (AGENTS.md sec. 3). It ships in the wheel as
``lucid-data build-tiles`` (WP-096), because a remote tier run installs a wheel and the
build is a step of that run, not of developing this repository.

Examples:
    Build both annotated splits at R18's own stride (1024 patch, 512 overlap)::

        lucid-data build-tiles --root /data/dota --out /data/dota_tiles \\
            --splits train,val --overlap 512

    Smoke-size the build to the first 20 source images of each split::

        lucid-data build-tiles --root /data/dota --out /data/dota_tiles --limit 20
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torchvision.io import ImageReadMode, read_image, write_png
from tqdm.auto import tqdm

from lucid_yolo.data.dota import DOTA_CLASSES, load_dota_targets
from lucid_yolo.data.rotated_geom import rboxes_to_polygons
from lucid_yolo.data.tiling import CROP_OVERLAP, PATCH_SIZE, TiledTargets, tile_image_targets

#: Image suffixes read from a DOTA ``images/`` directory. DOTA-v1.0 publishes PNG.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
#: Separator between a source stem and its window origin in a tile file name.
TILE_SEPARATOR = "__"
#: Corner coordinates per quadrilateral, flattened into a COCO ``segmentation`` ring.
_RING_VALUES = 8
#: Id written by a worker, which cannot know how many tiles preceded it. The parent
#: overwrites every one in source order, so this value never reaches the JSON — it is
#: out of COCO's 1-based range so that a missed overwrite fails a reader rather than
#: silently colliding with a real record.
_UNNUMBERED = 0
#: One written tile: its ``images`` record and its ``annotations`` records, unnumbered.
_TileResult = tuple[dict[str, object], list[dict[str, object]]]


@dataclass(frozen=True)
class SplitReport:
    """What one split's build produced, for the printed report and the tests.

    Attributes:
        split: The split name, e.g. ``"train"``.
        source_images: Source images read.
        tiles: Tile images written.
        empty_tiles: Tiles written that carry no annotation (A52).
        instances: Annotations written across all tiles.
        difficult: How many of those annotations carry R18's flag.

    Examples:
        >>> SplitReport("val", 1, 4, 1, 6, 2).empty_tiles
        1
    """

    split: str
    source_images: int
    tiles: int
    empty_tiles: int
    instances: int
    difficult: int


def dota_categories() -> list[dict[str, object]]:
    """Build the COCO ``categories`` list for DOTA-v1.0, in R18's published order.

    Returns:
        One ``{"id", "name", "supercategory"}`` entry per class, ids ``1..15``.

    Examples:
        >>> categories = dota_categories()
        >>> len(categories), categories[0]["id"], categories[0]["name"]
        (15, 1, 'plane')
    """
    return [{"id": index + 1, "name": name, "supercategory": "dota"} for index, name in enumerate(DOTA_CLASSES)]


def tile_file_name(stem: str, window: torch.Tensor, suffix: str = ".png") -> str:
    """Name one tile after its source stem and window origin.

    Args:
        stem: The source image's file stem.
        window: The ``(4,)`` int64 ``(x0, y0, x1, y1)`` window.
        suffix: Image suffix to write, including the dot.

    Returns:
        A file name of the form ``<stem>__<x0>_<y0><suffix>``.

    Examples:
        >>> import torch
        >>> tile_file_name("P0007", torch.tensor([824, 0, 1848, 1024]))
        'P0007__824_0.png'
    """
    x0, y0 = int(window[0]), int(window[1])
    return f"{stem}{TILE_SEPARATOR}{x0}_{y0}{suffix}"


def annotation_records(image_id: int, first_id: int, tiled: TiledTargets) -> list[dict[str, object]]:
    """Turn one tile's targets into COCO annotation records.

    ``segmentation`` is the rotated box's own quadrilateral (the oriented reading fits
    ``rboxes`` back out of it, so the ring is the load-bearing field and ``bbox`` is
    advisory); ``area`` is that quadrilateral's area, which for a rotated box is
    ``w * h`` and not the axis-aligned envelope's area.

    Args:
        image_id: The tile's image id.
        first_id: Id of the first annotation emitted here; ids run consecutively.
        tiled: The window's targets, flags and visible fractions.

    Returns:
        One record per instance, in instance order.

    Examples:
        >>> import torch
        >>> from lucid_yolo.data.targets import Targets
        >>> from lucid_yolo.data.tiling import TiledTargets
        >>> targets = Targets(
        ...     boxes=torch.tensor([[1.0, 1.0, 3.0, 3.0]]),
        ...     labels=torch.tensor([0]),
        ...     rboxes=torch.tensor([[2.0, 2.0, 2.0, 2.0, 0.0]]),
        ... )
        >>> tiled = TiledTargets(targets, torch.tensor([True]), torch.tensor([0.5]))
        >>> record = annotation_records(7, 1, tiled)[0]
        >>> record["bbox"], record["area"], record["difficult"], record["category_id"]
        ([1.0, 1.0, 2.0, 2.0], 4.0, 1, 1)
    """
    targets = tiled.targets
    if targets.boxes.shape[0] == 0:
        return []
    rings = rboxes_to_polygons(targets.rboxes).reshape(-1, _RING_VALUES)
    records: list[dict[str, object]] = []
    for index in range(targets.boxes.shape[0]):
        x0, y0, x1, y1 = (round(float(v), 4) for v in targets.boxes[index])
        width, height = targets.rboxes[index, 2], targets.rboxes[index, 3]
        records.append(
            {
                "id": first_id + index,
                "image_id": image_id,
                "category_id": int(targets.labels[index]) + 1,
                "bbox": [x0, y0, round(x1 - x0, 4), round(y1 - y0, 4)],
                "area": round(float(width * height), 4),
                "iscrowd": 0,
                "segmentation": [[round(float(v), 4) for v in rings[index]]],
                "difficult": int(bool(tiled.difficult[index])),
                "visible_fraction": round(float(tiled.visible_fraction[index]), 6),
            }
        )
    return records


def source_images(split_dir: Path) -> list[Path]:
    """List a split's source images in a deterministic order.

    Args:
        split_dir: A DOTA split directory holding ``images/`` and ``labelTxt/``.

    Returns:
        Every readable image path under ``images/``, sorted by name.

    Raises:
        FileNotFoundError: If ``images/`` or ``labelTxt/`` is absent.

    Examples:
        >>> source_images(Path("/nonexistent"))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        FileNotFoundError: ...
    """
    for name in ("images", "labelTxt"):
        if not (split_dir / name).is_dir():
            raise FileNotFoundError(f"{split_dir} is not a DOTA split: {name}/ is missing")
    return sorted(path for path in (split_dir / "images").iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def tile_source_image(
    path: Path, *, labels_dir: Path, images_out: Path, patch: int, overlap: int, keep_empty: bool
) -> tuple[list[_TileResult], int]:
    """Tile one source image, writing its tiles and returning their unnumbered records.

    This is the unit of parallelism, and it is the whole of the per-image work: reading
    the label file, decoding the image, cropping, and writing every tile PNG. What comes
    back is metadata only — a tile tensor returned to the parent would be shipped through
    a pickle for no reason, and the tiles of one DOTA image outweigh the image.

    Ids are **not** assigned here. A worker cannot know how many tiles the images before
    it produced, and numbering by anything a worker does know (its own index, a shared
    counter) would make the file depend on scheduling order. The parent numbers them in
    source order, so the JSON is byte-identical however many workers ran.

    Args:
        path: The source image.
        labels_dir: The split's ``labelTxt`` directory.
        images_out: Directory the tile images are written into.
        patch: Crop side in pixels.
        overlap: Nominal crop overlap in pixels (A21).
        keep_empty: Whether tiles with no annotation are written (A52).

    Returns:
        A ``(tiles, empty)`` pair: one ``(image record, annotation records)`` entry per
        tile written, in window order, and how many windows carried no annotation.

    Examples:
        >>> tile_source_image(  # doctest: +IGNORE_EXCEPTION_DETAIL
        ...     Path("/nonexistent.png"),
        ...     labels_dir=Path("/nonexistent"),
        ...     images_out=Path("/tmp"),
        ...     patch=1024,
        ...     overlap=512,
        ...     keep_empty=True,
        ... )
        Traceback (most recent call last):
        FileNotFoundError: ...
    """
    targets = load_dota_targets(labels_dir / f"{path.stem}.txt", keep_difficult=True)
    image = read_image(str(path), ImageReadMode.RGB)
    tiles: list[_TileResult] = []
    empty = 0
    for window, tile, tiled in tile_image_targets(
        image, targets, difficult=targets.difficult, patch=patch, overlap=overlap
    ):
        records = annotation_records(_UNNUMBERED, _UNNUMBERED, tiled)
        if not records:
            empty += 1
            if not keep_empty:
                continue
        file_name = tile_file_name(path.stem, window)
        write_png(tile.contiguous(), str(images_out / file_name))
        tiles.append((_image_record(_UNNUMBERED, file_name, tile, window, path.name), records))
    return tiles, empty


def _number(tiles: list[_TileResult], images: list[dict[str, object]], annotations: list[dict[str, object]]) -> None:
    """Assign one source image's ids in place, continuing the split's running counts.

    Args:
        tiles: One worker's ``(image record, annotation records)`` entries, in window order.
        images: The split's image records so far; extended here.
        annotations: The split's annotation records so far; extended here.
    """
    for image_record, records in tiles:
        image_id = len(images) + 1
        image_record["id"] = image_id
        for offset, record in enumerate(records):
            record["id"] = len(annotations) + offset + 1
            record["image_id"] = image_id
        images.append(image_record)
        annotations.extend(records)


def _worker_setup() -> None:
    """Pin each pool worker to one torch thread.

    Tiling is one process per image already; letting every worker also fan out over the
    machine's cores turns the pool into oversubscription, which on a many-core host is
    slower than the serial build it replaced.
    """
    torch.set_num_threads(1)


def _results(
    paths: list[Path], job: partial[tuple[list[_TileResult], int]], workers: int, split: str, progress: bool
) -> Iterator[tuple[list[_TileResult], int]]:
    """Yield each source image's tiling result in source order, serially or through a pool.

    Args:
        paths: Source images, in the order their ids are assigned.
        job: The per-image call, with everything but the path already bound.
        workers: Pool processes; ``0`` and ``1`` run in this process instead.
        split: Split name, shown on the progress bar.
        progress: Whether to draw the progress bar.

    Yields:
        ``(tiles, empty)`` pairs, always in ``paths`` order.
    """
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_setup) as pool:
            yield from tqdm(pool.map(job, paths), total=len(paths), desc=split, unit="img", disable=not progress)
        return
    yield from tqdm((job(path) for path in paths), total=len(paths), desc=split, unit="img", disable=not progress)


def convert_split(
    split_dir: Path,
    out_dir: Path,
    split: str,
    *,
    patch: int = PATCH_SIZE,
    overlap: int = CROP_OVERLAP,
    keep_empty: bool = True,
    limit: int | None = None,
    workers: int = 1,
    progress: bool = True,
) -> SplitReport:
    """Tile one DOTA split into the COCO layout and write its instances JSON.

    Args:
        split_dir: The source split directory (``images/`` and ``labelTxt/``).
        out_dir: Root of the layout being written.
        split: Split name, used for the image directory and the JSON file name.
        patch: Crop side in pixels.
        overlap: Nominal crop overlap in pixels (A21).
        keep_empty: Whether tiles with no annotation are written (A52).
        limit: Read at most this many source images; ``None`` reads all.
        workers: Pool processes tiling source images; ``1`` stays in this process.
        progress: Whether to draw a per-image progress bar.

    Returns:
        The split's :class:`SplitReport`.

    Examples:
        >>> convert_split(Path("/nonexistent"), Path("/tmp/out"), "val")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        FileNotFoundError: ...
    """
    paths = source_images(split_dir)[:limit]
    images_out = out_dir / split
    images_out.mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)
    job = partial(
        tile_source_image,
        labels_dir=split_dir / "labelTxt",
        images_out=images_out,
        patch=patch,
        overlap=overlap,
        keep_empty=keep_empty,
    )

    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    empty = 0
    for tiles, empty_here in _results(paths, job, workers, split, progress):
        _number(tiles, images, annotations)
        empty += empty_here

    _write_instances(out_dir / "annotations" / f"instances_{split}.json", images, annotations, split)
    return SplitReport(
        split=split,
        source_images=len(paths),
        tiles=len(images),
        empty_tiles=empty if keep_empty else 0,
        instances=len(annotations),
        difficult=sum(int(record["difficult"]) for record in annotations),  # type: ignore[call-overload]
    )


def _image_record(
    image_id: int, file_name: str, tile: torch.Tensor, window: torch.Tensor, source: str
) -> dict[str, object]:
    """Build one COCO ``images`` record, carrying its window provenance (A53)."""
    return {
        "id": image_id,
        "file_name": file_name,
        "height": int(tile.shape[1]),
        "width": int(tile.shape[2]),
        "source_image": source,
        "window": [int(v) for v in window],
    }


def _write_instances(
    path: Path, images: list[dict[str, object]], annotations: list[dict[str, object]], split: str
) -> None:
    """Write one split's instances JSON, categories included."""
    payload = {
        "info": {"description": f"DOTA-v1.0 {split}, tiled (WP-094)", "version": "1.0"},
        "images": images,
        "annotations": annotations,
        "categories": dota_categories(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


#: Annotation readers ``build_tiles`` can tile from. ``source`` names the reader rather
#: than the command doing so: what this writes is a plain COCO container with
#: quadrilateral rings, so a second oriented dataset is a reader, not a second command
#: with its own flags to keep in step.
SOURCES = ("dota",)


def build_tiles(
    root: Path,
    out: Path,
    source: str = "dota",
    splits: str = "train,val",
    patch: int = PATCH_SIZE,
    overlap: int = CROP_OVERLAP,
    drop_empty_tiles: bool = False,
    limit: int | None = None,
    workers: int | None = None,
    progress: bool = True,
) -> int:
    """Tile an oriented dataset into the trainable COCO layout, split by split.

    Args:
        root: Provisioned dataset root, holding one directory per split.
        out: Where the tiled COCO layout is written.
        source: Annotation reader; only ``dota`` is implemented.
        splits: Comma-separated split names.
        patch: Crop side in pixels.
        overlap: Nominal crop overlap in pixels (A21; R18's own protocol is 512).
        drop_empty_tiles: Skip tiles that carry no annotation (A52).
        limit: Read at most this many source images per split.
        workers: Processes tiling source images in parallel. ``None`` (default)
            takes one per available CPU; ``1`` stays in this process. The output is
            byte-identical either way — ids are assigned by the parent in source
            order, never by a worker.
        progress: Whether to draw a per-image progress bar. A full DOTA build is
            thousands of multi-megapixel decodes, so silence for an hour is not a
            reasonable thing to ask of an operator.

    Returns:
        ``0`` on success; ``1`` when a split directory is not a split of ``source``.

    Raises:
        ValueError: If ``source`` is not a known reader.

    Examples:
        >>> build_tiles(Path("/nonexistent"), Path("/tmp/tiles"), splits="val")  # doctest: +ELLIPSIS
        FAIL: ...
        1
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; known readers are {list(SOURCES)}")
    resolved_workers = max(1, os.cpu_count() or 1) if workers is None else max(1, workers)
    for split in (name.strip() for name in splits.split(",") if name.strip()):
        try:
            report = convert_split(
                root / split,
                out,
                split,
                patch=patch,
                overlap=overlap,
                keep_empty=not drop_empty_tiles,
                limit=limit,
                workers=resolved_workers,
                progress=progress,
            )
        except FileNotFoundError as error:
            print(f"FAIL: {error}")
            return 1
        print(
            f"{report.split}: {report.source_images} images -> {report.tiles} tiles "
            f"({report.empty_tiles} empty), {report.instances} instances, {report.difficult} difficult"
        )
    print(f"wrote {out}")
    return 0
