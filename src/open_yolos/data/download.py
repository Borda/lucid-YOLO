# SPDX-License-Identifier: Apache-2.0
"""Download COCO 2017 from the official public hosting into the repo's layout.

This module fetches the COCO 2017 archives from the project's canonical public
host (``images.cocodataset.org``) and extracts them into a data root that
:mod:`scripts.check_data` and :class:`~open_yolos.data.coco.CocoDetectionDataset`
already expect::

    <data_root>/
        train2017/000000000009.jpg ...
        val2017/000000000139.jpg ...
        annotations/instances_train2017.json
        annotations/instances_val2017.json

The official archives extract to exactly those top-level directories, so
unpacking them into ``data_root`` produces the validated layout with no
post-processing. Only the standard library is used (``urllib.request`` for the
streaming transfer, ``zipfile`` for extraction), so no runtime dependency is
added.

Download behaviour:
    * ``val`` is the default split (5,000 images, ~1 GB); ``train`` is 18 GB and
      must be opted into explicitly (``--splits train val``).
    * Transfers stream to ``<archive>.part`` and are renamed into place on
      completion (atomic), so a killed run never leaves a truncated ``.zip``.
    * A partial ``.part`` is resumed with an HTTP ``Range`` request when the
      server honours it (``206``); otherwise the transfer cleanly restarts.
    * An already-extracted split/annotations tree is skipped without touching the
      network (idempotent), unless ``force`` is set.

Checksum policy:
    The official COCO archives publish no authoritative SHA-256 digests, so none
    are hard-coded here. The computed digest of every downloaded archive is
    printed, and an expected digest may be supplied per archive
    (``--sha256 val2017.zip=<hex>``) to enforce a mismatch failure.

Examples:
    Fetch the validation split plus annotations::

        python -m open_yolos.data.download --data-root /data/coco

    Or via the console script (both splits)::

        open-yolos-download --data-root /data/coco --splits train val
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["download_coco", "main"]

#: Official public COCO 2017 image host (blueprint sec. 14.3); no mirrors. The
#: S3 path-style form is used because the ``images.cocodataset.org`` CNAME is an
#: S3 bucket whose TLS certificate does not cover that hostname (the official
#: site links plain http); path-style keeps the same bucket behind a valid cert.
COCO_ZIP_BASE = "https://s3.amazonaws.com/images.cocodataset.org/zips"
#: Official public COCO 2017 annotation host (same bucket, path-style TLS).
COCO_ANNOTATION_BASE = "https://s3.amazonaws.com/images.cocodataset.org/annotations"

#: Per-split image archive file names; each extracts to ``<split>2017/``.
SPLIT_ARCHIVES: dict[str, str] = {
    "train": "train2017.zip",
    "val": "val2017.zip",
}
#: Shared train+val annotation archive; extracts to ``annotations/``.
ANNOTATIONS_ARCHIVE = "annotations_trainval2017.zip"

#: Streaming read size (1 MiB) shared by the download and hashing loops.
_CHUNK_SIZE = 1 << 20
#: HTTP status marking a resumed (partial-content) transfer.
_HTTP_PARTIAL = 206
#: Bytes-per-megabyte divisor for human-readable progress.
_BYTES_PER_MB = 1e6


@dataclass(frozen=True)
class _Archive:
    """One downloadable COCO archive and its on-disk extraction marker.

    Attributes:
        name: Archive file name; also the key used for a checksum override.
        url: Fully-qualified download URL on the official host.
        sentinel: Path relative to the data root whose presence means the archive
            is already extracted (the idempotency marker).
    """

    name: str
    url: str
    sentinel: str


def _plan_archives(splits: Sequence[str], annotations: bool) -> list[_Archive]:
    """Resolve the requested ``splits`` and ``annotations`` flag to archives.

    Args:
        splits: Image split names (``"train"`` / ``"val"``), order preserved.
        annotations: Whether to append the shared annotations archive.

    Returns:
        The archives to download, in request order.

    Raises:
        ValueError: If a split name is not one of the known COCO 2017 splits.
    """
    archives: list[_Archive] = []
    for split in splits:
        try:
            file_name = SPLIT_ARCHIVES[split]
        except KeyError:
            known = ", ".join(sorted(SPLIT_ARCHIVES))
            raise ValueError(f"unknown split {split!r}; expected one of {known}") from None
        archives.append(_Archive(file_name, f"{COCO_ZIP_BASE}/{file_name}", f"{split}2017"))
    if annotations:
        url = f"{COCO_ANNOTATION_BASE}/{ANNOTATIONS_ARCHIVE}"
        archives.append(_Archive(ANNOTATIONS_ARCHIVE, url, "annotations"))
    return archives


def _report_progress(downloaded: int, total: int | None) -> None:
    """Write a single-line byte/percent progress update to stderr.

    Args:
        downloaded: Bytes transferred so far.
        total: Expected total bytes, or ``None`` when the server omits a length.
    """
    if total:
        pct = downloaded / total * 100.0
        line = f"\r  {downloaded / _BYTES_PER_MB:,.1f} / {total / _BYTES_PER_MB:,.1f} MB ({pct:5.1f}%)"
    else:
        line = f"\r  {downloaded / _BYTES_PER_MB:,.1f} MB"
    sys.stderr.write(line)
    sys.stderr.flush()


def _content_total(response: Any, already: int) -> int | None:
    """Return the expected total transfer size, or ``None`` if unknown.

    Args:
        response: The open ``urlopen`` response.
        already: Bytes already on disk that the response continues from.

    Returns:
        ``already`` plus the response ``Content-Length``, or ``None`` when the
        header is absent.
    """
    length = response.headers.get("Content-Length")
    if length is None:
        return None
    return already + int(length)


def _stream(response: Any, part: Path, mode: str, already: int, total: int | None, *, progress: bool) -> None:
    """Stream ``response`` into ``part``, reporting progress if requested.

    Args:
        response: The open ``urlopen`` response to drain.
        part: Partial-download file to write.
        mode: File open mode (``"wb"`` to restart, ``"ab"`` to resume).
        already: Bytes already present when resuming (the progress baseline).
        total: Expected total size for the percentage, or ``None``.
        progress: Whether to emit progress lines to stderr.
    """
    downloaded = already
    with part.open(mode) as handle:
        while True:
            chunk = response.read(_CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if progress:
                _report_progress(downloaded, total)


def _download_to(url: str, dest: Path, *, progress: bool) -> None:
    """Download ``url`` to ``dest`` atomically, resuming a partial file if any.

    A partial ``<dest>.part`` is resumed via an HTTP ``Range`` request; if the
    server ignores the range (responds ``200``) the transfer restarts cleanly.
    The completed ``.part`` is renamed onto ``dest`` so an interrupted run never
    leaves a truncated archive at ``dest``.

    Args:
        url: Source URL on the official host.
        dest: Final archive path; the ``.part`` sibling is the scratch file.
        progress: Whether to emit progress lines to stderr.
    """
    part = dest.with_name(dest.name + ".part")
    resume_from = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url)
    if resume_from:
        request.add_header("Range", f"bytes={resume_from}-")
    if progress:
        sys.stderr.write(f"downloading {dest.name} from {url}\n")
    with urllib.request.urlopen(request) as response:
        resumed = resume_from > 0 and response.status == _HTTP_PARTIAL
        mode = "ab" if resumed else "wb"
        already = resume_from if resumed else 0
        total = _content_total(response, already)
        _stream(response, part, mode, already, total, progress=progress)
    if progress:
        sys.stderr.write("\n")
    part.replace(dest)


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of ``path`` (streamed, memory-bounded).

    Args:
        path: File to hash.

    Returns:
        The lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum(path: Path, expected: str | None) -> str:
    """Compute ``path``'s digest and enforce ``expected`` when supplied.

    Args:
        path: Archive to hash.
        expected: Expected hex SHA-256, or ``None`` to only compute and report.

    Returns:
        The computed hex digest.

    Raises:
        ValueError: If ``expected`` is given and does not match (case-insensitive).
    """
    actual = _sha256(path)
    if expected is not None and actual.lower() != expected.strip().lower():
        raise ValueError(f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}")
    return actual


def _safe_extract(archive: Path, dest: Path) -> None:
    """Extract ``archive`` into ``dest``, rejecting zip-slip members.

    Every member is resolved against ``dest`` before extraction; any member that
    would land outside the destination (absolute path or ``..`` traversal) aborts
    the whole extraction.

    Args:
        archive: Zip archive to extract.
        dest: Destination directory (the data root).

    Raises:
        ValueError: If any member resolves outside ``dest``.
    """
    dest_root = dest.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            target = (dest / member).resolve()
            if target != dest_root and dest_root not in target.parents:
                raise ValueError(f"unsafe member path in {archive.name}: {member!r}")
        bundle.extractall(dest)


def _process_archive(
    archive: _Archive, data_root: Path, expected: str | None, *, force: bool, keep: bool, progress: bool
) -> None:
    """Download, verify and extract one ``archive`` into ``data_root``.

    Skips all work (no network) when the extraction sentinel already exists and
    ``force`` is not set.

    Args:
        archive: The archive to process.
        data_root: Destination root directory.
        expected: Expected hex SHA-256 for this archive, or ``None``.
        force: Re-download even when already extracted.
        keep: Keep the downloaded ``.zip`` after a successful extraction.
        progress: Whether to emit progress lines to stderr.
    """
    sentinel = data_root / archive.sentinel
    if sentinel.exists() and not force:
        sys.stderr.write(f"skipping {archive.name}: {sentinel} already present\n")
        return
    zip_path = data_root / archive.name
    _download_to(archive.url, zip_path, progress=progress)
    digest = _verify_checksum(zip_path, expected)
    sys.stderr.write(f"{archive.name} sha256={digest}\n")
    _safe_extract(zip_path, data_root)
    if not keep:
        zip_path.unlink()


def download_coco(
    data_root: Path,
    splits: Sequence[str] = ("val",),
    *,
    annotations: bool = True,
    checksums: Mapping[str, str] | None = None,
    force: bool = False,
    keep_archives: bool = False,
    progress: bool = True,
) -> Path:
    """Download and extract COCO 2017 into ``data_root``.

    Fetches each requested image split plus (by default) the shared annotations
    archive from the official public host and extracts them so the tree matches
    the layout validated by :mod:`scripts.check_data`. Already-extracted archives
    are skipped without network access unless ``force`` is set.

    Args:
        data_root: Target root directory (created if missing).
        splits: Image splits to fetch; defaults to ``("val",)`` (the smallest
            useful split). Duplicates are collapsed and order is preserved.
        annotations: Whether to also fetch the shared annotations archive.
        checksums: Optional per-archive expected SHA-256 digests, keyed by archive
            file name (e.g. ``{"val2017.zip": "<hex>"}``); a mismatch aborts.
        force: Re-download and re-extract even when the sentinel already exists.
        keep_archives: Keep the downloaded ``.zip`` files after extraction.
        progress: Emit byte/percent progress to stderr.

    Returns:
        The ``data_root`` path (created and populated).

    Raises:
        ValueError: On an unknown split name or a checksum mismatch.
        OSError: On a network or filesystem failure during transfer/extraction.

    Examples:
        ```pycon
        >>> from pathlib import Path
        >>> download_coco(Path("/data/coco"), ["val"])  # doctest: +SKIP
        PosixPath('/data/coco')

        ```
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    overrides = dict(checksums or {})
    unique_splits = list(dict.fromkeys(splits))
    for archive in _plan_archives(unique_splits, annotations):
        _process_archive(
            archive,
            data_root,
            overrides.get(archive.name),
            force=force,
            keep=keep_archives,
            progress=progress,
        )
    return data_root


def _parse_checksums(items: Iterable[str]) -> dict[str, str]:
    """Parse ``NAME=HEX`` checksum overrides into a mapping.

    Args:
        items: Raw ``--sha256`` argument strings.

    Returns:
        A mapping of archive file name to expected hex digest.

    Raises:
        ValueError: If any item is not of the form ``NAME=HEX``.
    """
    checksums: dict[str, str] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name or not value:
            raise ValueError(f"invalid --sha256 {item!r}; expected NAME=HEX")
        checksums[name] = value
    return checksums


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``open-yolos-download`` argument parser."""
    parser = argparse.ArgumentParser(description="Download COCO 2017 into the lit-YOLO data layout.")
    parser.add_argument("--data-root", type=Path, required=True, help="target root directory")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=sorted(SPLIT_ARCHIVES),
        default=["val"],
        help="image splits to fetch (default: val)",
    )
    parser.add_argument(
        "--no-annotations", dest="annotations", action="store_false", help="skip the annotations archive"
    )
    parser.add_argument(
        "--sha256",
        action="append",
        default=[],
        metavar="NAME=HEX",
        help="expected SHA-256 for an archive, e.g. val2017.zip=<hex> (repeatable)",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if already extracted")
    parser.add_argument("--keep-archives", action="store_true", help="keep the downloaded .zip files after extraction")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the COCO 2017 downloader command line.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` on success, ``1`` on a download/verification/extraction failure.

    Examples:
        ```pycon
        >>> main(["--data-root", "/data/coco", "--splits", "val"])  # doctest: +SKIP
        0

        ```
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        checksums = _parse_checksums(args.sha256)
        download_coco(
            args.data_root,
            args.splits,
            annotations=args.annotations,
            checksums=checksums,
            force=args.force,
            keep_archives=args.keep_archives,
            progress=not args.quiet,
        )
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
