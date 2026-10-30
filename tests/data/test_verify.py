# SPDX-License-Identifier: Apache-2.0
"""Offline unit gate for the packaged dataset verifier (``lucid_yolo.data.verify``).

Every test builds a tiny fake COCO layout on ``tmp_path`` and exercises the per-file existence checks without any
network or real dataset: a clean root passes, an interrupted-extraction root (annotations list more images than the
directory holds) fails with the right counts and a named missing sample, and a missing annotation file is reported.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from lucid_yolo.data import verify


def _write_split(
    root: Path,
    split_dir: str,
    annotation_name: str,
    annotated: Sequence[str],
    *,
    on_disk: Sequence[str] | None = None,
) -> None:
    """Write a fake split: annotations listing ``annotated``, files ``on_disk``.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     _write_split(root, "val2017", "instances_val2017.json", _names(2))
        ...     sorted(p.name for p in (root / "val2017").iterdir())
        ['000000000000.jpg', '000000000001.jpg']
    """
    (root / split_dir).mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    images = [{"id": i, "file_name": name, "height": 4, "width": 4} for i, name in enumerate(annotated)]
    (root / "annotations" / annotation_name).write_text(json.dumps({"images": images}), encoding="utf-8")
    for name in annotated if on_disk is None else on_disk:
        (root / split_dir / name).write_bytes(b"jpegbytes")


def _names(count: int) -> list[str]:
    """Return ``count`` COCO-style zero-padded jpg file names.

    Examples:
        >>> _names(2)
        ['000000000000.jpg', '000000000001.jpg']
    """
    return [f"{i:012d}.jpg" for i in range(count)]


def _verify_val2017(root: Path, **kwargs: int) -> verify.SplitVerification:
    """Verify the ``val2017`` split of a COCO 2017-spelled ``root``; ``kwargs`` (e.g. ``sample``) go to the verifier.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     _write_split(Path(tmp), "val2017", "instances_val2017.json", _names(2))
        ...     _verify_val2017(Path(tmp)).ok
        True
    """
    images_dir, annotation_file = root / "val2017", root / "annotations" / "instances_val2017.json"
    return verify.verify_split("val2017", images_dir, annotation_file, **kwargs)


class TestVerifySplit:
    """Tests for ``verify.verify_split``."""

    def test_clean_passes(self, tmp_path: Path) -> None:
        """A split whose every annotated image is on disk verifies OK."""
        _write_split(tmp_path, "val2017", "instances_val2017.json", _names(3))

        result = _verify_val2017(tmp_path)

        assert result.ok
        assert (result.expected, result.present, result.missing_count) == (3, 3, 0)
        assert result.missing == []
        assert result.annotation_present is True

    def test_missing_images_reports_counts_and_sample(self, tmp_path: Path) -> None:
        """A partial extraction fails with exact counts and the named missing files."""
        annotated = _names(5)
        _write_split(tmp_path, "val2017", "instances_val2017.json", annotated, on_disk=annotated[:2])

        result = _verify_val2017(tmp_path)

        assert not result.ok
        assert (result.expected, result.present, result.missing_count) == (5, 2, 3)
        assert result.missing == annotated[2:]
        assert any("3 of 5 annotated images missing" in problem for problem in result.problems)

    def test_missing_annotation_file_fails(self, tmp_path: Path) -> None:
        """A split with no annotation JSON fails and flags the annotation absent."""
        (tmp_path / "val2017").mkdir()

        result = _verify_val2017(tmp_path)

        assert not result.ok
        assert result.annotation_present is False
        assert any("annotation file missing" in problem for problem in result.problems)

    def test_unparsable_annotation_fails(self, tmp_path: Path) -> None:
        """A corrupt (non-JSON) annotation file fails and flags the annotation absent."""
        (tmp_path / "val2017").mkdir()
        (tmp_path / "annotations").mkdir()
        (tmp_path / "annotations" / "instances_val2017.json").write_text("{ not json", encoding="utf-8")

        result = _verify_val2017(tmp_path)

        assert not result.ok
        assert result.annotation_present is False
        assert any("failed to parse" in problem for problem in result.problems)

    @pytest.mark.parametrize(
        ("content", "reason"),
        [
            pytest.param(b"\xff\xfe{", "'utf-8' codec can't decode", id="invalid-utf8"),
            pytest.param(b'[{"file_name": "a.jpg"}]', "top level is not a JSON object", id="top-level-array"),
            pytest.param(b'{"images": null}', "images is not a JSON array", id="images-null"),
            pytest.param(b'{"images": {"file_name": "a.jpg"}}', "images is not a JSON array", id="images-object"),
            pytest.param(b'{"images": [{"file_name": "a.jpg"}, 7]}', "images entry 1 is not", id="entry-number"),
            pytest.param(b'{"images": ["file_name"]}', "images entry 0 is not", id="entry-string"),
            pytest.param(b'{"images": [], "n": ' + b"9" * 5000 + b"}", "integer string conversion", id="huge-int"),
            pytest.param(b"[" * 200_000 + b"]" * 200_000, "failed to parse", id="deep-nesting"),
        ],
    )
    def test_malformed_annotation_fails_naming_file(self, tmp_path: Path, content: bytes, reason: str) -> None:
        """A malformed annotation file is a located parse failure naming the file, never an exception.

        Each case used to escape ``verify_split``: a decode error, an ``AttributeError`` on a top level without
        ``.get``, or a ``TypeError`` from iterating ``images`` or indexing an entry.
        """
        (tmp_path / "val2017").mkdir()
        (tmp_path / "annotations").mkdir()
        annotation_file = tmp_path / "annotations" / "instances_val2017.json"
        annotation_file.write_bytes(content)

        result = verify.verify_split("val2017", tmp_path / "val2017", annotation_file)

        assert not result.ok
        assert result.annotation_present is False
        assert result.expected == 0
        assert result.problems[0].startswith(f"annotation file failed to parse: {annotation_file} (")
        assert reason in result.problems[0]

    def test_missing_images_directory_fails(self, tmp_path: Path) -> None:
        """A split whose image directory is absent fails even with valid annotations."""
        (tmp_path / "annotations").mkdir()
        images = [{"id": 0, "file_name": "000000000000.jpg", "height": 4, "width": 4}]
        (tmp_path / "annotations" / "instances_val2017.json").write_text(
            json.dumps({"images": images}), encoding="utf-8"
        )

        result = _verify_val2017(tmp_path)

        assert not result.ok
        assert any("images directory missing" in problem for problem in result.problems)

    def test_caps_missing_sample(self, tmp_path: Path) -> None:
        """The missing sample is capped while ``missing_count`` stays the true total."""
        annotated = _names(30)
        _write_split(tmp_path, "val2017", "instances_val2017.json", annotated, on_disk=[])

        result = _verify_val2017(tmp_path, sample=4)

        assert result.missing_count == 30
        assert result.missing == annotated[:4]

    def test_nested_file_name_present_not_missing(self, tmp_path: Path) -> None:
        """A nested ``file_name`` whose file exists under the images directory verifies OK.

        The reader opens ``images_dir / file_name``, so a flat directory listing alone reported ``nested/a.jpg``
        missing although the reader could load it.
        """
        _write_split(tmp_path, "val2017", "instances_val2017.json", ["flat.jpg", "nested/a.jpg"], on_disk=["flat.jpg"])
        (tmp_path / "val2017" / "nested").mkdir()
        (tmp_path / "val2017" / "nested" / "a.jpg").write_bytes(b"jpegbytes")

        result = _verify_val2017(tmp_path)

        assert result.ok, result.problems
        assert (result.expected, result.present, result.missing_count) == (2, 2, 0)

    def test_nested_file_name_absent_reported_missing(self, tmp_path: Path) -> None:
        """A nested ``file_name`` with no file on disk is reported missing by its annotated name.

        The nested branch must still catch an absent file, not treat every name with a separator as present.
        """
        _write_split(tmp_path, "val2017", "instances_val2017.json", ["flat.jpg", "nested/a.jpg"], on_disk=["flat.jpg"])

        result = _verify_val2017(tmp_path)

        assert not result.ok
        assert (result.expected, result.present, result.missing_count) == (2, 1, 1)
        assert result.missing == ["nested/a.jpg"]

    def test_flat_file_name_absent_beside_nested_reported_missing(self, tmp_path: Path) -> None:
        """A flat ``file_name`` absent from disk is reported missing alongside a present nested one.

        Mixed annotations keep the flat fast path: the directory listing still decides flat names while nested names
        are checked by path.
        """
        _write_split(tmp_path, "val2017", "instances_val2017.json", ["flat.jpg", "nested/a.jpg"], on_disk=[])
        (tmp_path / "val2017" / "nested").mkdir()
        (tmp_path / "val2017" / "nested" / "a.jpg").write_bytes(b"jpegbytes")

        result = _verify_val2017(tmp_path)

        assert (result.expected, result.present, result.missing_count) == (2, 1, 1)
        assert result.missing == ["flat.jpg"]


class TestVerifyCocoRoot:
    """Tests for ``verify.verify_coco_root``."""

    def test_clean_passes(self, tmp_path: Path) -> None:
        """A fully provisioned val root verifies OK end to end."""
        _write_split(tmp_path, "val2017", "instances_val2017.json", _names(4))

        result = verify.verify_coco_root(tmp_path, ["val"])

        assert result.ok
        assert [split.name for split in result.splits] == ["val2017"]

    def test_incomplete_split_fails(self, tmp_path: Path) -> None:
        """A root with a partially extracted split reports not-ok with the shortfall."""
        annotated = _names(6)
        _write_split(tmp_path, "val2017", "instances_val2017.json", annotated, on_disk=annotated[:1])

        result = verify.verify_coco_root(tmp_path, ["val"])

        assert not result.ok
        (split,) = result.splits
        assert (split.expected, split.present, split.missing_count) == (6, 1, 5)

    def test_collapses_duplicate_splits(self, tmp_path: Path) -> None:
        """Duplicate split names are collapsed to a single verification entry."""
        _write_split(tmp_path, "val2017", "instances_val2017.json", _names(2))

        result = verify.verify_coco_root(tmp_path, ["val", "val"])

        assert [split.name for split in result.splits] == ["val2017"]

    def test_plain_spelling_resolved_through_layout(self, tmp_path: Path) -> None:
        """A ``val/`` + ``instances_val.json`` root is verified where the reader would read it.

        The layout probe and the reader both accept this spelling, so a verifier that knew only ``val2017``
        reported a correct tree as missing its images and its annotations.
        """
        _write_split(tmp_path, "val", "instances_val.json", _names(3))

        result = verify.verify_coco_root(tmp_path, ["val"])

        assert result.ok, [split.problems for split in result.splits]
        assert [split.name for split in result.splits] == ["val"]

    def test_unresolved_split_named_by_2017_spelling(self, tmp_path: Path) -> None:
        """A split no spelling resolves is reported under COCO 2017's names, as before.

        The resolver falls back to the published pair when nothing matches, so a missing download still names the
        ``val2017`` paths the downloader would have written.
        """
        result = verify.verify_coco_root(tmp_path, ["val"])

        assert [split.name for split in result.splits] == ["val2017"]
        assert result.splits[0].problems[0] == (
            f"annotation file missing: {tmp_path / 'annotations' / 'instances_val2017.json'}"
        )

    def test_unknown_split_raises(self, tmp_path: Path) -> None:
        """An unknown split name raises a helpful ValueError."""
        with pytest.raises(ValueError, match="unknown split 'test'"):
            verify.verify_coco_root(tmp_path, ["test"])


class TestFormatReport:
    """Tests for ``verify.format_report``."""

    def test_clean_ends_in_pass(self, tmp_path: Path) -> None:
        """The clean-root report ends in the overall PASS verdict."""
        _write_split(tmp_path, "val2017", "instances_val2017.json", _names(2))

        report = verify.format_report(verify.verify_coco_root(tmp_path, ["val"]), tmp_path)

        assert report.splitlines()[-1] == "PASS: every annotated image is present"
        assert "PASS val2017" in report

    def test_missing_lists_sample_and_fail(self, tmp_path: Path) -> None:
        """The failing report names the missing sample and ends in the FAIL verdict."""
        annotated = _names(15)
        _write_split(tmp_path, "val2017", "instances_val2017.json", annotated, on_disk=[])

        report = verify.format_report(verify.verify_coco_root(tmp_path, ["val"]), tmp_path)

        assert report.splitlines()[-1] == "FAIL: dataset is incomplete"
        assert "missing e.g.:" in report
        assert "more)" in report
