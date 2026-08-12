# SPDX-License-Identifier: Apache-2.0
"""DOTA-v1.0 label parsing and long-edge conversion for the oriented path (WP-056).

The dataset (R18) annotates each object with an "arbitrary quadrilateral bounding box"
``{(x_i, y_i), i=1,2,3,4}`` whose "vertices are arranged in a clockwise order", one object
per line of a ``labelTxt`` file::

    imagesource:GoogleEarth
    gsd:0.146343590398
    x1 y1 x2 y2 x3 y3 x4 y4 category difficult

Leading ``key:value`` metadata lines (``imagesource``, ``gsd``, and in some releases
``'acquisition dates'``) are skipped; blank lines are ignored; every other line must carry
exactly ten whitespace-separated fields, ``difficult`` being "1 for difficult, 0 for not
difficult". Anything else is rejected with the file, the 1-based line number and the
offending text — a label file this reader cannot account for is a dataset problem, not a
line to drop quietly.

Vertex order carries meaning in R18 (the first point marks the "head" for plane, ship,
harbor, helicopter, large/small vehicle and baseball diamond, and the top-left corner
otherwise), but none of it survives the conversion to a long-edge box, which is symmetric
under a half turn by construction (see :mod:`lucid_yolo.data.rotated_geom`). This reader
therefore neither relies on the ordering nor enforces it: :func:`~lucid_yolo.data.rotated_geom.polygons_to_rboxes`
is invariant to any cyclic rotation or reversal of the ring.

Category names:
    :data:`DOTA_CLASSES` lists the 15 categories in the order R18 publishes them; the
    index into it *is* the class id. The label files on disk write the multi-word names
    with hyphens or underscores (``storage-tank``, ``soccer-ball-field``,
    ``ground_track_field``), so names are **normalized on read** — lower-cased, with ``-``
    and ``_`` treated as spaces and runs of whitespace collapsed. A name that does not
    resolve after normalization raises rather than being dropped: a silently skipped
    category is a silently shrunken dataset.

Instance-axis invariant (relied on by WP-058, WP-061 and WP-088):
    The oriented path keeps the rotated-box axis **1:1 with the instance axis**.
    ``rboxes[i]`` is the oriented box of the same instance as ``boxes[i]`` and
    ``labels[i]``, so ``targets.filter(keep, rkeep=keep)`` keeps every modality aligned
    with a single mask. :class:`~lucid_yolo.data.targets.Targets` permits this — it
    validates the two axes separately and never couples their lengths — and the pairing is
    what gives the existing axis-aligned transforms an envelope to work with.

    ``boxes[i]`` is the axis-aligned envelope of the **annotated quadrilateral**, not of
    the fitted rectangle. The two coincide when the quad is an exact rectangle and differ
    by the hand-drawn quad's own residual otherwise; the envelope of what was annotated is
    the honest axis-aligned reading of the annotation. Boxes are **not** clipped to the
    image: a label file carries no image size, and WP-057's 1024 px tiling is where
    geometry meets the canvas.

    ``polygons`` stays empty. The quad is already carried losslessly by ``rboxes`` up to
    the rectangle fit, and a second copy would be one more modality WP-058 has to keep
    consistent through every warp for no reader.

The ``difficult`` flag:
    Neither R1 nor R18 settles whether difficult instances are trained on, so this module
    does not decide either (A39). The flag is parsed and carried on every
    :class:`DotaObject`, and ``keep_difficult`` is a **required** keyword argument of both
    loaders, so every call site states its choice. The tier-level policy lands with
    WP-063's evaluation protocol and WP-088's training path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lucid_yolo.data.rotated_geom import polygons_to_rboxes
from lucid_yolo.data.targets import Targets
from lucid_yolo.data.transforms import boxes_from_polygons

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = ["DOTA_CLASSES", "DotaObject", "dota_targets", "load_dota_targets", "parse_dota_label_file"]

#: The 15 DOTA-v1.0 categories in the order R18 publishes them; the index is the class id.
DOTA_CLASSES: tuple[str, ...] = (
    "plane",
    "ship",
    "storage tank",
    "baseball diamond",
    "tennis court",
    "basketball court",
    "ground track field",
    "harbor",
    "bridge",
    "large vehicle",
    "small vehicle",
    "helicopter",
    "roundabout",
    "soccer ball field",
    "swimming pool",
)

#: Normalized category name to class id, built once from :data:`DOTA_CLASSES`.
_LABEL_BY_NAME: dict[str, int] = {name: label for label, name in enumerate(DOTA_CLASSES)}

#: Fields on an object line: eight coordinates, the category, the difficult flag.
_FIELDS_PER_OBJECT = 10
#: The leading fields that spell out the quadrilateral.
_COORD_FIELDS = 8
#: Corner count of the quadrilateral form of a rotated box.
_QUAD_CORNERS = 4
#: Column count of a point ``(x, y)``.
_POINT_DIM = 2
#: Accepted spellings of the ``difficult`` flag, mapped to their boolean meaning.
_DIFFICULT_VALUES = {"0": False, "1": True}
#: A metadata header line: an alphabetic key (optionally quoted, possibly with spaces as in
#: ``'acquisition dates'``) followed by a colon. Object lines start with a coordinate, so
#: they never match.
_HEADER_RE = re.compile(r"^'?[A-Za-z][A-Za-z ]*'?\s*:")


@dataclass(frozen=True)
class DotaObject:
    """One object line of a ``labelTxt`` file, parsed but not yet converted.

    Attributes:
        polygon: ``(4, 2)`` float32 quadrilateral corners ``(x, y)`` as annotated.
        label: Class id, the index of the category into :data:`DOTA_CLASSES`.
        difficult: The line's ``difficult`` flag (A39 — carried, never acted on here).

    Examples:
        ```pycon
        >>> import torch
        >>> quad = torch.tensor([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])
        >>> DotaObject(polygon=quad, label=0, difficult=False).label
        0

        ```
    """

    polygon: Tensor
    label: int
    difficult: bool


def parse_dota_label_file(label_file: Path) -> list[DotaObject]:
    """Parse one ``labelTxt`` file into its objects, in file order.

    Metadata header lines and blank lines are skipped; every remaining line must be a
    well-formed object line (see the module docstring). Category names are normalized
    before lookup, so ``storage-tank``, ``storage_tank`` and ``Storage Tank`` all resolve
    to the same class id.

    Args:
        label_file: Path to the image's ``labelTxt`` file.

    Returns:
        The parsed objects, one per object line; an empty list for a file that holds only
        headers or nothing at all.

    Raises:
        ValueError: If any line is not a header and not a well-formed object line — wrong
            field count, non-numeric coordinate, unknown category or a ``difficult`` flag
            that is neither ``0`` nor ``1``. The message names the file and the 1-based
            line number.
        OSError: If the file cannot be read.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> text = "imagesource:GoogleEarth\\ngsd:0.146\\n0 0 4 0 4 2 0 2 large-vehicle 0\\n"
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "P0001.txt"
        ...     _ = path.write_text(text, encoding="utf-8")
        ...     objects = parse_dota_label_file(path)
        >>> len(objects), objects[0].label, objects[0].difficult
        (1, 9, False)

        ```
    """
    lines = label_file.read_text(encoding="utf-8").splitlines()
    return _parse_lines(lines, source=str(label_file))


def dota_targets(objects: Sequence[DotaObject], *, keep_difficult: bool) -> Targets:
    """Convert parsed objects into :class:`~lucid_yolo.data.targets.Targets`.

    Each kept object contributes one entry to all three of ``boxes``, ``labels`` and
    ``rboxes``, on the shared instance axis the module docstring pins: ``rboxes[i]`` is the
    canonical long-edge box fitted to the same quadrilateral whose envelope is ``boxes[i]``.
    ``polygons`` is left empty.

    ``keep_difficult`` has no default on purpose (A39): the papers do not settle whether
    difficult instances are trained on, so the caller says.

    Args:
        objects: Parsed objects, e.g. from :func:`parse_dota_label_file`.
        keep_difficult: Whether objects flagged ``difficult`` are kept. Required.

    Returns:
        The targets for one image; :meth:`~lucid_yolo.data.targets.Targets.empty` when
        nothing survives the filter.

    Examples:
        ```pycon
        >>> import torch
        >>> quad = torch.tensor([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])
        >>> objects = [DotaObject(polygon=quad, label=1, difficult=True)]
        >>> dota_targets(objects, keep_difficult=False).boxes.shape
        torch.Size([0, 4])
        >>> kept = dota_targets(objects, keep_difficult=True)
        >>> [round(v, 4) for v in kept.rboxes[0].tolist()]
        [2.0, 1.0, 4.0, 2.0, 0.0]

        ```
    """
    kept = [obj for obj in objects if keep_difficult or not obj.difficult]
    if not kept:
        return Targets.empty()
    quads = [obj.polygon for obj in kept]
    return Targets(
        boxes=boxes_from_polygons(quads),
        labels=torch.tensor([obj.label for obj in kept], dtype=torch.int64),
        rboxes=polygons_to_rboxes(torch.stack(quads, dim=0)),
        # Carried, never acted on here (A39). Under `keep_difficult=True` this is the only
        # thing that still distinguishes a difficult instance from an ordinary one, and A48
        # makes the evaluation loader depend on that distinction surviving to the metric.
        difficult=torch.tensor([obj.difficult for obj in kept], dtype=torch.bool),
    )


def load_dota_targets(label_file: Path, *, keep_difficult: bool) -> Targets:
    """Load one image's ``labelTxt`` file straight into targets.

    A thin composition of :func:`parse_dota_label_file` and :func:`dota_targets`; the
    intermediate objects are worth reaching for only when the ``difficult`` flags
    themselves matter.

    Args:
        label_file: Path to the image's ``labelTxt`` file.
        keep_difficult: Whether objects flagged ``difficult`` are kept. Required (A39).

    Returns:
        The targets for that image, with ``boxes``, ``labels`` and ``rboxes`` on one
        instance axis.

    Raises:
        ValueError: If any line is malformed (see :func:`parse_dota_label_file`).
        OSError: If the file cannot be read.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = Path(tmp) / "P0002.txt"
        ...     _ = path.write_text("gsd:null\\n1 1 5 1 5 3 1 3 harbor 1\\n", encoding="utf-8")
        ...     targets = load_dota_targets(path, keep_difficult=True)
        >>> targets.boxes.tolist(), targets.labels.tolist()
        ([[1.0, 1.0, 5.0, 3.0]], [7])

        ```
    """
    return dota_targets(parse_dota_label_file(label_file), keep_difficult=keep_difficult)


def _parse_lines(lines: Sequence[str], source: str) -> list[DotaObject]:
    """Parse the object lines of a ``labelTxt`` file, skipping headers and blanks.

    Args:
        lines: The file's lines, without their terminators.
        source: File path used in error messages.

    Returns:
        One :class:`DotaObject` per object line, in order.

    Examples:
        >>> _parse_lines(["gsd:0.1", "", "0 0 2 0 2 1 0 1 plane 1"], "mem")[0].difficult
        True
    """
    objects: list[DotaObject] = []
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or _HEADER_RE.match(stripped):
            continue
        objects.append(_parse_object_line(stripped, source, number))
    return objects


def _parse_object_line(line: str, source: str, number: int) -> DotaObject:
    """Parse one object line into a :class:`DotaObject`.

    Args:
        line: The stripped, non-empty, non-header line.
        source: File path used in error messages.
        number: 1-based line number used in error messages.

    Returns:
        The parsed object.

    Raises:
        ValueError: If the field count is wrong or any field fails to parse.

    Examples:
        >>> _parse_object_line("0 0 2 0 2 1 0 1 swimming-pool 0", "mem", 1).label
        14
    """
    fields = line.split()
    if len(fields) != _FIELDS_PER_OBJECT:
        raise ValueError(
            f"{source}:{number}: expected {_FIELDS_PER_OBJECT} fields "
            f"(8 coordinates, category, difficult); got {len(fields)} in {line!r}"
        )
    return DotaObject(
        polygon=_parse_polygon(fields[:_COORD_FIELDS], source, number),
        label=_resolve_category(fields[_COORD_FIELDS], source, number),
        difficult=_parse_difficult(fields[_COORD_FIELDS + 1], source, number),
    )


def _parse_polygon(fields: Sequence[str], source: str, number: int) -> Tensor:
    """Parse the eight coordinate fields into a ``(4, 2)`` float32 quadrilateral.

    Args:
        fields: The eight coordinate fields ``x1 y1 ... x4 y4``.
        source: File path used in error messages.
        number: 1-based line number used in error messages.

    Returns:
        A ``(4, 2)`` float32 tensor of corners ``(x, y)`` in annotation order.

    Raises:
        ValueError: If any coordinate is not a number.

    Examples:
        >>> _parse_polygon(["0", "0", "2", "0", "2", "1", "0", "1"], "mem", 1).shape
        torch.Size([4, 2])
    """
    try:
        coordinates = [float(value) for value in fields]
    except ValueError:
        raise ValueError(f"{source}:{number}: coordinates must be numeric; got {list(fields)}") from None
    return torch.tensor(coordinates, dtype=torch.float32).reshape(_QUAD_CORNERS, _POINT_DIM)


def _resolve_category(raw: str, source: str, number: int) -> int:
    """Resolve a raw category name to its class id, rejecting unknown names.

    Args:
        raw: The category field exactly as written in the file.
        source: File path used in error messages.
        number: 1-based line number used in error messages.

    Returns:
        The index of the category into :data:`DOTA_CLASSES`.

    Raises:
        ValueError: If the normalized name is not a DOTA-v1.0 category.

    Examples:
        >>> _resolve_category("Ground_Track_Field", "mem", 1)
        6
    """
    normalized = _normalize_category(raw)
    label = _LABEL_BY_NAME.get(normalized)
    if label is None:
        known = ", ".join(DOTA_CLASSES)
        raise ValueError(
            f"{source}:{number}: unknown category {raw!r} (normalized to {normalized!r}); expected one of {known}"
        )
    return label


def _normalize_category(raw: str) -> str:
    """Lower-case a category name and treat ``-``/``_`` and whitespace runs as one space.

    Examples:
        >>> _normalize_category("Soccer-ball_field")
        'soccer ball field'
    """
    return " ".join(raw.lower().replace("-", " ").replace("_", " ").split())


def _parse_difficult(raw: str, source: str, number: int) -> bool:
    """Parse the ``difficult`` field, which R18 defines as exactly ``0`` or ``1``.

    Args:
        raw: The difficult field exactly as written in the file.
        source: File path used in error messages.
        number: 1-based line number used in error messages.

    Returns:
        ``True`` for ``1``, ``False`` for ``0``.

    Raises:
        ValueError: For any other value.

    Examples:
        >>> _parse_difficult("1", "mem", 1)
        True
    """
    difficult = _DIFFICULT_VALUES.get(raw)
    if difficult is None:
        raise ValueError(f"{source}:{number}: difficult flag must be 0 or 1; got {raw!r}")
    return difficult
