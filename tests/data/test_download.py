# SPDX-License-Identifier: Apache-2.0
"""Offline unit gate for the COCO 2017 downloader (``lit_yolo.data.download``).

Every test here runs without network: the transfer path is exercised by
monkeypatching ``urllib.request.urlopen`` with an in-memory zip server, and the
idempotency path monkeypatches it to raise if it is ever called. The tests cover
the URL constants, the split/annotation-to-archive plan (and its check-data
layout compatibility), the zip-slip guard, resume/atomic/skip behaviour, the
SHA-256 verification hook, and the CLI argument parsing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lit_yolo.data import download as dl

_CHECK_DATA_PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_data.py"


def _load_check_data() -> ModuleType:
    """Load ``scripts/check_data.py`` as a module (``scripts`` is not a package)."""
    spec = importlib.util.spec_from_file_location("check_data", _CHECK_DATA_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_data = _load_check_data()


class _FakeResponse:
    """Minimal ``urlopen`` stand-in serving ``payload`` with resume support."""

    def __init__(self, payload: bytes, *, resume_from: int = 0) -> None:
        self._served = payload[resume_from:]
        self._buf = io.BytesIO(self._served)
        self.status = dl._HTTP_PARTIAL if resume_from else 200
        self.headers = {"Content-Length": str(len(self._served))}

    def read(self, size: int = -1) -> bytes:
        return self._buf.read(size)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _make_zip(members: dict[str, bytes]) -> bytes:
    """Build an in-memory zip archive from ``member name -> bytes``."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return buffer.getvalue()


def _val_archive_bytes(num_images: int = 1) -> bytes:
    """A fake ``val2017.zip`` producing a check-data-compatible val split."""
    images = [{"id": i, "file_name": f"{i:012d}.jpg", "height": 4, "width": 4} for i in range(num_images)]
    annotations = {"images": images, "annotations": [], "categories": [{"id": 1, "name": "thing"}]}
    members: dict[str, bytes] = {f"val2017/{i:012d}.jpg": b"jpegbytes" for i in range(num_images)}
    members["annotations/instances_val2017.json"] = json.dumps(annotations).encode("utf-8")
    return _make_zip(members)


def _serve(mapping: dict[str, bytes]) -> Any:
    """Return a fake ``urlopen`` serving ``url -> payload`` with Range resume."""

    def fake_urlopen(request: Any) -> _FakeResponse:
        url = request.full_url
        payload = mapping[url]
        range_header = request.headers.get("Range")
        resume_from = int(range_header.removeprefix("bytes=").rstrip("-")) if range_header else 0
        return _FakeResponse(payload, resume_from=resume_from)

    return fake_urlopen


def _raise_if_called(*_args: object, **_kwargs: object) -> Any:
    """A ``urlopen`` replacement asserting the network is never touched."""
    raise AssertionError("network access attempted during an offline test")


# --------------------------------------------------------------------------- #
# URL constants and archive planning
# --------------------------------------------------------------------------- #


def test_url_constants_use_official_https_host() -> None:
    assert dl.COCO_ZIP_BASE == "https://s3.amazonaws.com/images.cocodataset.org/zips"
    assert dl.COCO_ANNOTATION_BASE == "https://s3.amazonaws.com/images.cocodataset.org/annotations"
    assert dl.SPLIT_ARCHIVES == {"train": "train2017.zip", "val": "val2017.zip"}
    assert dl.ANNOTATIONS_ARCHIVE == "annotations_trainval2017.zip"


def test_plan_archives_val_only_default() -> None:
    archives = dl._plan_archives(["val"], annotations=True)
    names = [a.name for a in archives]
    sentinels = [a.sentinel for a in archives]
    assert names == ["val2017.zip", "annotations_trainval2017.zip"]
    assert sentinels == ["val2017", "annotations"]
    assert archives[0].url == "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip"
    assert archives[1].url == "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"


def test_plan_archives_train_and_val() -> None:
    archives = dl._plan_archives(["train", "val"], annotations=True)
    assert [a.sentinel for a in archives] == ["train2017", "val2017", "annotations"]


def test_plan_archives_no_annotations() -> None:
    archives = dl._plan_archives(["val"], annotations=False)
    assert [a.name for a in archives] == ["val2017.zip"]


def test_plan_archives_sentinels_match_check_data_layout(tmp_path: Path) -> None:
    """Each archive's sentinel is exactly the directory check_data validates."""
    for split, images_dir, ann in (
        ("train", "train2017", "instances_train2017.json"),
        ("val", "val2017", "instances_val2017.json"),
    ):
        (tmp_path / images_dir).mkdir()
        (tmp_path / images_dir / "000000000001.jpg").write_bytes(b"x")
        (tmp_path / "annotations").mkdir(exist_ok=True)
        payload = {"images": [{"id": 1, "file_name": "000000000001.jpg", "height": 4, "width": 4}]}
        (tmp_path / "annotations" / ann).write_text(json.dumps(payload), encoding="utf-8")
        result = check_data.check_split(images_dir, tmp_path / images_dir, tmp_path / "annotations" / ann, 1)
        assert result.ok, result.problems
        # sentinel used by the downloader is the same directory
        [archive] = dl._plan_archives([split], annotations=False)
        assert (tmp_path / archive.sentinel).is_dir()


def test_plan_archives_unknown_split_raises() -> None:
    with pytest.raises(ValueError, match="unknown split 'test'"):
        dl._plan_archives(["test"], annotations=False)


# --------------------------------------------------------------------------- #
# Zip-slip guard
# --------------------------------------------------------------------------- #


def test_safe_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    archive.write_bytes(_make_zip({"../escape.txt": b"pwn"}))
    with pytest.raises(ValueError, match="unsafe member path"):
        dl._safe_extract(archive, tmp_path / "dest")


def test_safe_extract_rejects_absolute_member(tmp_path: Path) -> None:
    # zipfile stores an absolute name with the leading slash stripped, so craft
    # a traversal that still resolves outside the destination root.
    archive = tmp_path / "evil.zip"
    archive.write_bytes(_make_zip({"../../etc/pwned": b"pwn"}))
    with pytest.raises(ValueError, match="unsafe member path"):
        dl._safe_extract(archive, tmp_path / "dest")


def test_safe_extract_accepts_normal_members(tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    dest.mkdir()
    archive = tmp_path / "ok.zip"
    archive.write_bytes(_make_zip({"val2017/a.jpg": b"x", "annotations/i.json": b"{}"}))
    dl._safe_extract(archive, dest)
    assert (dest / "val2017" / "a.jpg").is_file()
    assert (dest / "annotations" / "i.json").is_file()


# --------------------------------------------------------------------------- #
# SHA-256 verification
# --------------------------------------------------------------------------- #


def test_verify_checksum_returns_digest_without_expected(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"hello coco")

    expected = hashlib.sha256(b"hello coco").hexdigest()
    assert dl._verify_checksum(path, None) == expected


def test_verify_checksum_accepts_matching_digest(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"hello coco")

    expected = hashlib.sha256(b"hello coco").hexdigest().upper()  # case-insensitive
    assert dl._verify_checksum(path, expected)


def test_verify_checksum_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"hello coco")
    with pytest.raises(ValueError, match=r"SHA-256 mismatch for blob\.bin"):
        dl._verify_checksum(path, "deadbeef")


# --------------------------------------------------------------------------- #
# End-to-end transfer (monkeypatched network)
# --------------------------------------------------------------------------- #


def test_download_coco_produces_check_data_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    val_bytes = _val_archive_bytes(num_images=2)
    mapping = {"https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip": val_bytes}
    monkeypatch.setattr(urllib.request, "urlopen", _serve(mapping))

    root = dl.download_coco(tmp_path / "coco", ["val"], annotations=False, progress=False)

    assert root == tmp_path / "coco"
    assert (root / "val2017" / "000000000000.jpg").is_file()
    assert (root / "val2017" / "000000000001.jpg").is_file()
    result = check_data.check_split("val2017", root / "val2017", root / "annotations" / "instances_val2017.json", 2)
    # annotations dir shipped inside the same archive; the split validates
    assert (root / "annotations" / "instances_val2017.json").is_file()
    assert result.ok, result.problems


def test_download_coco_removes_archive_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mapping = {"https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip": _val_archive_bytes()}
    monkeypatch.setattr(urllib.request, "urlopen", _serve(mapping))
    root = dl.download_coco(tmp_path / "coco", ["val"], annotations=False, progress=False)
    assert not (root / "val2017.zip").exists()


def test_download_coco_keeps_archive_when_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mapping = {"https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip": _val_archive_bytes()}
    monkeypatch.setattr(urllib.request, "urlopen", _serve(mapping))
    root = dl.download_coco(tmp_path / "coco", ["val"], annotations=False, keep_archives=True, progress=False)
    assert (root / "val2017.zip").is_file()


def test_download_coco_verifies_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    payload = _val_archive_bytes()
    mapping = {"https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip": payload}
    monkeypatch.setattr(urllib.request, "urlopen", _serve(mapping))
    good = hashlib.sha256(payload).hexdigest()
    root = dl.download_coco(
        tmp_path / "coco", ["val"], annotations=False, checksums={"val2017.zip": good}, progress=False
    )
    assert (root / "val2017").is_dir()

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        dl.download_coco(
            tmp_path / "coco2", ["val"], annotations=False, checksums={"val2017.zip": "deadbeef"}, progress=False
        )


def test_download_coco_skips_extracted_split_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "coco"
    (root / "val2017").mkdir(parents=True)
    monkeypatch.setattr(urllib.request, "urlopen", _raise_if_called)
    # Must not raise: sentinel present, no download attempted.
    dl.download_coco(root, ["val"], annotations=False, progress=False)


def test_download_coco_force_redownloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "coco"
    (root / "val2017").mkdir(parents=True)
    mapping = {"https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip": _val_archive_bytes()}
    monkeypatch.setattr(urllib.request, "urlopen", _serve(mapping))
    dl.download_coco(root, ["val"], annotations=False, force=True, progress=False)
    assert (root / "val2017" / "000000000000.jpg").is_file()


def test_download_resumes_partial_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _val_archive_bytes()
    root = tmp_path / "coco"
    root.mkdir()
    # Seed a partial .part with the first 10 bytes already fetched.
    (root / "val2017.zip.part").write_bytes(payload[:10])
    mapping = {"https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip": payload}
    monkeypatch.setattr(urllib.request, "urlopen", _serve(mapping))
    dl.download_coco(root, ["val"], annotations=False, progress=False)
    # Resume path reconstructs the full archive and extracts it.
    assert (root / "val2017" / "000000000000.jpg").is_file()
    assert not (root / "val2017.zip.part").exists()


# --------------------------------------------------------------------------- #
# CLI parsing
# --------------------------------------------------------------------------- #


def test_parse_checksums_valid() -> None:
    assert dl._parse_checksums(["val2017.zip=abc123", "train2017.zip=def456"]) == {
        "val2017.zip": "abc123",
        "train2017.zip": "def456",
    }


@pytest.mark.parametrize("item", ["novalue=", "=nokey", "noequals"])
def test_parse_checksums_invalid_raises(item: str) -> None:
    with pytest.raises(ValueError, match="invalid --sha256"):
        dl._parse_checksums([item])


def test_cli_requires_data_root() -> None:
    parser = dl._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_cli_defaults_to_val_with_annotations() -> None:
    args = dl._build_parser().parse_args(["--data-root", "/data/coco"])
    assert args.splits == ["val"]
    assert args.annotations is True
    assert args.force is False


def test_cli_parses_splits_and_no_annotations() -> None:
    args = dl._build_parser().parse_args(
        ["--data-root", "/data/coco", "--splits", "train", "val", "--no-annotations", "--keep-archives", "--quiet"]
    )
    assert args.splits == ["train", "val"]
    assert args.annotations is False
    assert args.keep_archives is True
    assert args.quiet is True


def test_cli_rejects_unknown_split() -> None:
    with pytest.raises(SystemExit):
        dl._build_parser().parse_args(["--data-root", "/data/coco", "--splits", "test"])


def test_main_returns_zero_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mapping = {"https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip": _val_archive_bytes()}
    monkeypatch.setattr(urllib.request, "urlopen", _serve(mapping))
    code = dl.main(["--data-root", str(tmp_path / "coco"), "--splits", "val", "--no-annotations", "--quiet"])
    assert code == 0


def test_main_returns_one_on_checksum_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mapping = {"https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip": _val_archive_bytes()}
    monkeypatch.setattr(urllib.request, "urlopen", _serve(mapping))
    code = dl.main(
        [
            "--data-root",
            str(tmp_path / "coco"),
            "--splits",
            "val",
            "--no-annotations",
            "--sha256",
            "val2017.zip=bad",
            "--quiet",
        ]
    )
    assert code == 1


def test_main_returns_one_on_bad_checksum_arg(tmp_path: Path) -> None:
    code = dl.main(["--data-root", str(tmp_path / "coco"), "--sha256", "malformed", "--quiet"])
    assert code == 1
