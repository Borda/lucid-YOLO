# SPDX-License-Identifier: Apache-2.0
"""Offline unit gate for the COCO 2017 downloader (``lucid_yolo.data.download``).

Every test here runs without network: the transfer path is exercised by
monkeypatching ``urllib.request.urlopen`` with an in-memory zip server, and the
idempotency path monkeypatches it to raise if it is ever called. The tests cover
the URL constants, the split/annotation-to-archive plan (and its check-data
layout compatibility), the zip-slip guard, resume/atomic/skip behaviour, the
SHA-256 verification hook, and the command that drives them.

That command is ``lucid-data download``. WP-110 removed the ``lucid-download``
alias these tests used to parse through, so the spellings asserted here are the
shipped underscored ones (``--data_root``, ``--splits '[val]'``, ``--force
true``) and the parser is the one a wheel installs, not a frozen local copy.
"""

from __future__ import annotations

import hashlib
import io
import json
import shlex
import urllib.request
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from lucid_yolo.cli import data as data_cli
from lucid_yolo.data import check as check_data
from lucid_yolo.data import download as dl

if TYPE_CHECKING:
    from jsonargparse import Namespace


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
    """Build an in-memory zip archive from ``member name -> bytes``.

    Examples:
        >>> import io, zipfile
        >>> data = _make_zip({"a.txt": b"hi"})
        >>> zipfile.ZipFile(io.BytesIO(data)).read("a.txt")
        b'hi'
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)
    return buffer.getvalue()


def _val_archive_bytes(num_images: int = 1) -> bytes:
    """A fake ``val2017.zip`` producing a check-data-compatible val split.

    Examples:
        >>> import io, zipfile
        >>> archive = zipfile.ZipFile(io.BytesIO(_val_archive_bytes(num_images=2)))
        >>> sorted(archive.namelist())
        ['annotations/instances_val2017.json', 'val2017/000000000000.jpg', 'val2017/000000000001.jpg']
    """
    images = [{"id": i, "file_name": f"{i:012d}.jpg", "height": 4, "width": 4} for i in range(num_images)]
    annotations = {"images": images, "annotations": [], "categories": [{"id": 1, "name": "thing"}]}
    members: dict[str, bytes] = {f"val2017/{i:012d}.jpg": b"jpegbytes" for i in range(num_images)}
    members["annotations/instances_val2017.json"] = json.dumps(annotations).encode("utf-8")
    return _make_zip(members)


def _serve(mapping: dict[str, bytes]) -> Any:
    """Return a fake ``urlopen`` serving ``url -> payload`` with Range resume.

    Examples:
        >>> class _Req:
        ...     full_url = "https://example.org/x.zip"
        ...     headers: dict[str, str] = {}
        >>> fake_urlopen = _serve({"https://example.org/x.zip": b"abc"})
        >>> with fake_urlopen(_Req()) as response:
        ...     response.read()
        b'abc'
    """

    def fake_urlopen(request: Any) -> _FakeResponse:
        url = request.full_url
        payload = mapping[url]
        range_header = request.headers.get("Range")
        resume_from = int(range_header.removeprefix("bytes=").rstrip("-")) if range_header else 0
        return _FakeResponse(payload, resume_from=resume_from)

    return fake_urlopen


def _raise_if_called(*_args: object, **_kwargs: object) -> Any:
    """A ``urlopen`` replacement asserting the network is never touched.

    Examples:
        >>> try:
        ...     _raise_if_called()
        ... except AssertionError as exc:
        ...     str(exc)
        'network access attempted during an offline test'
    """
    raise AssertionError("network access attempted during an offline test")


def test_url_constants_use_official_https_host() -> None:
    """The download URLs point at the official S3 host, not a mirror.

    A wrong host here would either fail loudly at request time or, worse, resolve
    to something that looks like COCO but silently isn't, so the four constants
    the rest of this module builds requests from are pinned exactly.
    """
    assert dl.COCO_ZIP_BASE == "https://s3.amazonaws.com/images.cocodataset.org/zips"
    assert dl.COCO_ANNOTATION_BASE == "https://s3.amazonaws.com/images.cocodataset.org/annotations"
    assert dl.SPLIT_ARCHIVES == {"train": "train2017.zip", "val": "val2017.zip"}
    assert dl.ANNOTATIONS_ARCHIVE == "annotations_trainval2017.zip"


class TestPlanArchives:
    """Tests for ``dl._plan_archives``."""

    def test_val_only_default(self) -> None:
        """A val-only plan lists the val archive plus the shared annotations archive.

        Each planned archive's sentinel and URL are asserted individually rather than
        just its name, since a sentinel that drifted from the archive it belongs to
        would make ``download_coco`` think a split was already present when it isn't.
        """
        archives = dl._plan_archives(["val"], annotations=True)
        names = [a.name for a in archives]
        sentinels = [a.sentinel for a in archives]
        assert names == ["val2017.zip", "annotations_trainval2017.zip"]
        assert sentinels == ["val2017", "annotations"]
        assert archives[0].url == "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip"
        expected_annotations_url = (
            "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip"
        )
        assert archives[1].url == expected_annotations_url

    def test_train_and_val(self) -> None:
        """Requesting both splits orders the plan train, then val, then annotations.

        The annotations archive is shared by both splits and must appear once, last,
        rather than once per split -- a caller downloading both would otherwise fetch
        and extract the same archive twice.
        """
        archives = dl._plan_archives(["train", "val"], annotations=True)
        assert [a.sentinel for a in archives] == ["train2017", "val2017", "annotations"]

    def test_no_annotations(self) -> None:
        """``annotations=False`` drops the shared archive from the plan entirely.

        An operator who already has annotations, or wants images only, must not pay
        for or extract the annotations archive at all -- not just skip using it.
        """
        archives = dl._plan_archives(["val"], annotations=False)
        assert [a.name for a in archives] == ["val2017.zip"]

    def test_sentinels_match_check_data_layout(self, tmp_path: Path) -> None:
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

    def test_unknown_split_raises(self) -> None:
        """A split name outside ``{train, val}`` raises, naming the bad value.

        COCO's own test split shares no split name with these two, so a typo'd or
        aspirational ``"test"`` must fail at planning time rather than resolve to an
        empty archive list that silently downloads nothing.
        """
        with pytest.raises(ValueError, match="unknown split 'test'"):
            dl._plan_archives(["test"], annotations=False)


class TestSafeExtract:
    """Tests for ``dl._safe_extract`` (the zip-slip guard)."""

    def test_rejects_parent_traversal(self, tmp_path: Path) -> None:
        """A member path escaping the destination via ``../`` is refused, not extracted.

        This is the classic zip-slip: an archive fetched from a URL constant this
        module controls should be safe, but the guard is exercised anyway since a
        silent write outside ``dest`` is the failure mode a checksum cannot catch.
        """
        archive = tmp_path / "evil.zip"
        archive.write_bytes(_make_zip({"../escape.txt": b"pwn"}))
        with pytest.raises(ValueError, match="unsafe member path"):
            dl._safe_extract(archive, tmp_path / "dest")

    def test_rejects_absolute_member(self, tmp_path: Path) -> None:
        """An absolute-looking member that still resolves outside ``dest`` is refused.

        zipfile stores an absolute member name with its leading slash stripped, so a
        naive guard checking only for a leading ``/`` would miss this; the member is
        crafted to still escape via ``../../`` once that slash is gone.
        """
        archive = tmp_path / "evil.zip"
        archive.write_bytes(_make_zip({"../../etc/pwned": b"pwn"}))
        with pytest.raises(ValueError, match="unsafe member path"):
            dl._safe_extract(archive, tmp_path / "dest")

    def test_accepts_normal_members(self, tmp_path: Path) -> None:
        """Members that stay inside ``dest`` extract normally, the common case.

        The two traversal tests above only prove the guard rejects; this proves it
        does not also reject the ordinary archive layout ``download_coco`` produces.
        """
        dest = tmp_path / "dest"
        dest.mkdir()
        archive = tmp_path / "ok.zip"
        archive.write_bytes(_make_zip({"val2017/a.jpg": b"x", "annotations/i.json": b"{}"}))
        dl._safe_extract(archive, dest)
        assert (dest / "val2017" / "a.jpg").is_file()
        assert (dest / "annotations" / "i.json").is_file()


class TestVerifyChecksum:
    """Tests for ``dl._verify_checksum`` (the SHA-256 verification hook)."""

    def test_returns_digest_without_expected(self, tmp_path: Path) -> None:
        """With no expected digest, the function reports what it computed rather than pass/fail.

        This is the ``--sha256`` discovery path: an operator who wants to pin a
        checksum for later runs needs the real digest surfaced, not a boolean.
        """
        path = tmp_path / "blob.bin"
        path.write_bytes(b"hello coco")

        expected = hashlib.sha256(b"hello coco").hexdigest()
        assert dl._verify_checksum(path, None) == expected

    def test_accepts_matching_digest(self, tmp_path: Path) -> None:
        """An expected digest matches case-insensitively.

        ``--sha256`` values are operator-typed and hex case is not a meaningful
        distinction, so an uppercase expectation must not fail a lowercase-computed one.
        """
        path = tmp_path / "blob.bin"
        path.write_bytes(b"hello coco")

        expected = hashlib.sha256(b"hello coco").hexdigest().upper()  # case-insensitive
        assert dl._verify_checksum(path, expected)

    def test_mismatch_raises(self, tmp_path: Path) -> None:
        """A wrong expected digest raises, naming the file rather than just the mismatch.

        A corrupted or tampered download must stop the pipeline here, before
        extraction, and the error must be actionable -- which file, not just that
        something somewhere did not match.
        """
        path = tmp_path / "blob.bin"
        path.write_bytes(b"hello coco")
        with pytest.raises(ValueError, match=r"SHA-256 mismatch for blob\.bin"):
            dl._verify_checksum(path, "deadbeef")


class TestDownloadCoco:
    """Tests for ``dl.download_coco`` (end-to-end transfer, monkeypatched network)."""

    def test_produces_check_data_layout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A downloaded val split lands in the exact layout ``check_data`` expects.

        The download and the check are two independently maintained pieces of code;
        this is the seam where a layout drift between them would otherwise surface
        only as a confusing failure downstream, in training rather than here.
        """
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

    def test_removes_archive_by_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A finished download does not leave the source zip on disk by default.

        The extracted images and the archive that produced them would otherwise sit
        side by side, silently doubling disk use for something the caller only ever
        reads through the extracted tree.
        """
        mapping = {"https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip": _val_archive_bytes()}
        monkeypatch.setattr(urllib.request, "urlopen", _serve(mapping))
        root = dl.download_coco(tmp_path / "coco", ["val"], annotations=False, progress=False)
        assert not (root / "val2017.zip").exists()

    def test_keeps_archive_when_requested(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``keep_archives=True`` overrides the default cleanup and leaves the zip behind.

        An operator re-provisioning several roots from one archive needs this escape
        hatch; the default in the row above must not be the only path available.
        """
        mapping = {"https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip": _val_archive_bytes()}
        monkeypatch.setattr(urllib.request, "urlopen", _serve(mapping))
        root = dl.download_coco(tmp_path / "coco", ["val"], annotations=False, keep_archives=True, progress=False)
        assert (root / "val2017.zip").is_file()

    def test_verifies_checksum(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A correct checksum lets a download through; a wrong one raises before extraction.

        Both halves of the contract are exercised in one test since the interesting
        claim is that ``download_coco`` actually calls the verifier at all -- a
        checksum accepted with no verifier wired in would pass a matching-digest-only
        test just as happily.
        """
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
                tmp_path / "coco2",
                ["val"],
                annotations=False,
                checksums={"val2017.zip": "deadbeef"},
                progress=False,
            )

    def test_skips_extracted_split_without_network(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A split whose sentinel directory already exists is not re-fetched.

        ``urlopen`` is monkeypatched to raise on any call, so this test fails loudly
        if the sentinel check is ever bypassed -- re-running a download on a complete
        root must be free, not merely idempotent.
        """
        root = tmp_path / "coco"
        (root / "val2017").mkdir(parents=True)
        monkeypatch.setattr(urllib.request, "urlopen", _raise_if_called)
        # Must not raise: sentinel present, no download attempted.
        dl.download_coco(root, ["val"], annotations=False, progress=False)

    def test_force_redownloads(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``force=True`` re-fetches a split even though its sentinel already exists.

        The companion to the skip test above: the sentinel check has an override, and
        an operator suspecting a corrupted local split needs a way past it.
        """
        root = tmp_path / "coco"
        (root / "val2017").mkdir(parents=True)
        mapping = {"https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip": _val_archive_bytes()}
        monkeypatch.setattr(urllib.request, "urlopen", _serve(mapping))
        dl.download_coco(root, ["val"], annotations=False, force=True, progress=False)
        assert (root / "val2017" / "000000000000.jpg").is_file()

    def test_resumes_partial_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ``.part`` file from an interrupted download is resumed, not restarted.

        A large COCO archive over a flaky connection is exactly the case this exists
        for; re-fetching the whole file on every retry would make some networks never
        finish a train-split download at all.
        """
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


class TestParseChecksums:
    """Tests for ``dl._parse_checksums``."""

    def test_valid(self) -> None:
        """A well-formed ``name=digest`` list parses into an archive-name-keyed dict.

        This is the shape ``download_coco``'s ``checksums`` argument expects, so the
        parse result is asserted structurally rather than just checked for success.
        """
        assert dl._parse_checksums(["val2017.zip=abc123", "train2017.zip=def456"]) == {
            "val2017.zip": "abc123",
            "train2017.zip": "def456",
        }

    @pytest.mark.parametrize("item", ["novalue=", "=nokey", "noequals"])
    def test_invalid_raises(self, item: str) -> None:
        """A malformed ``--sha256`` item raises rather than parsing into a wrong pair.

        Three distinct malformations -- an empty digest, an empty name, and a missing
        ``=`` entirely -- are swept in one parametrized test, since silently accepting
        any of them would mean a typo'd flag verifies against the wrong (or an empty)
        expected digest instead of failing at parse time.
        """
        with pytest.raises(ValueError, match="invalid --sha256"):
            dl._parse_checksums([item])


def _parse_download(argv: list[str]) -> Namespace:
    """Parse ``lucid-data download`` arguments, returning that subcommand's namespace.

    Examples:
        >>> args = _parse_download(["--data_root", "/tmp/coco"])
        >>> list(args.splits)
        ['val']
    """
    return data_cli.build_parser().parse_args(["download", *argv]).download


class TestParseDownloadArgs:
    """Tests for the ``lucid-data download`` argument parser, via ``_parse_download``."""

    def test_requires_data_root(self) -> None:
        """``--data_root`` has no default, so omitting it fails the parse rather than the run."""
        with pytest.raises(SystemExit):
            _parse_download([])

    def test_defaults_to_val_with_annotations(self) -> None:
        """Bare ``--data_root`` fetches val plus annotations, the ~1 GB default."""
        args = _parse_download(["--data_root", "/data/coco"])
        assert list(args.splits) == ["val"]
        assert args.annotations is True
        assert args.force is False

    def test_parses_splits_and_no_annotations(self) -> None:
        """The underscored, list-valued spellings reach the operation function's parameters.

        This is the surface a copy-pasted command depends on, and the one that changed when
        the dashed alias went (WP-110): a list is ``'[train,val]'``, and a flag that used to
        be a bare switch now takes an explicit ``true``/``false``.
        """
        args = _parse_download(
            [
                "--data_root",
                "/data/coco",
                "--splits",
                "[train,val]",
                "--annotations",
                "false",
                "--keep_archives",
                "true",
                "--quiet",
                "true",
            ]
        )
        assert list(args.splits) == ["train", "val"]
        assert args.annotations is False
        assert args.keep_archives is True
        assert args.quiet is True


class TestMain:
    """Tests for ``data_cli.main`` (the ``download`` subcommand's exit-code contract)."""

    def test_rejects_unknown_split(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unknown split name fails the run, without reaching the network.

        The dashed alias rejected it in the parser (``choices=``); the shipped command derives
        its flags from :func:`download_dataset`'s signature, which has no such constraint, so
        :func:`_plan_archives` is the guard that matters and the exit code is what a caller
        sees. The difference is worth pinning: the check moved, it did not disappear.
        """
        monkeypatch.setattr(urllib.request, "urlopen", _raise_if_called)

        code = data_cli.main(["download", "--data_root", str(tmp_path), "--splits", "[test]", "--quiet", "true"])

        assert code == 1

    def test_returns_zero_on_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A clean end-to-end CLI download exits ``0``.

        This is the CLI's own contract, one layer above ``download_coco`` itself:
        a shell script or CI step checking the exit code needs this to hold even
        though every path it exercises is already covered piecemeal above.
        """
        mapping = {"https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip": _val_archive_bytes()}
        monkeypatch.setattr(urllib.request, "urlopen", _serve(mapping))
        code = data_cli.main(
            [
                "download",
                "--data_root",
                str(tmp_path / "coco"),
                "--splits",
                "[val]",
                "--annotations",
                "false",
                "--quiet",
                "true",
            ]
        )
        assert code == 0

    def test_returns_one_on_checksum_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A checksum mismatch surfaces at the CLI as exit ``1``, not an uncaught traceback.

        ``_verify_checksum`` raises ``ValueError`` internally; the CLI boundary is
        where that must become a script-friendly exit code instead of a stack trace.
        """
        mapping = {"https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip": _val_archive_bytes()}
        monkeypatch.setattr(urllib.request, "urlopen", _serve(mapping))
        code = data_cli.main(
            [
                "download",
                "--data_root",
                str(tmp_path / "coco"),
                "--splits",
                "[val]",
                "--annotations",
                "false",
                "--sha256",
                "[val2017.zip=bad]",
                "--quiet",
                "true",
            ]
        )
        assert code == 1

    def test_returns_one_on_bad_checksum_arg(self, tmp_path: Path) -> None:
        """A malformed ``--sha256`` value fails at argument parsing, before any network call.

        No ``monkeypatch`` on ``urlopen`` here, deliberately: a real network attempt
        on a bad flag would mean this validation runs too late to matter.
        """
        code = data_cli.main(
            ["download", "--data_root", str(tmp_path / "coco"), "--sha256", "[malformed]", "--quiet", "true"]
        )
        assert code == 1


def _seed_val_root(root: Path, annotated: int, present: int) -> None:
    """Seed ``root`` with a val split whose annotations list ``annotated`` images.

    Only the first ``present`` of those images are written to disk, so
    ``present < annotated`` reproduces an interrupted extraction.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     _seed_val_root(root, annotated=2, present=1)
        ...     sorted(p.name for p in (root / "val2017").iterdir())
        ['000000000000.jpg']
    """
    (root / "val2017").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    images = [{"id": i, "file_name": f"{i:012d}.jpg", "height": 4, "width": 4} for i in range(annotated)]
    (root / "annotations" / "instances_val2017.json").write_text(json.dumps({"images": images}), encoding="utf-8")
    for i in range(present):
        (root / "val2017" / f"{i:012d}.jpg").write_bytes(b"jpegbytes")


def _verify_only(root: Path) -> int:
    """Run ``lucid-data download --verify_only`` over ``root``'s val split.

    ``verify_only`` returns before the network path is ever reached (see
    :func:`~lucid_yolo.data.download.download_dataset`), so a complete root
    verifies to ``0`` with no monkeypatching needed to keep this example offline.
    ``--quiet`` silences the download progress bar only, not the verify report, so
    stdout is redirected here to keep the example deterministic.

    Examples:
        >>> import contextlib, io, tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp) / "coco"
        ...     _seed_val_root(root, annotated=1, present=1)
        ...     with contextlib.redirect_stdout(io.StringIO()):
        ...         code = _verify_only(root)
        >>> code
        0
    """
    return data_cli.main(
        ["download", "--data_root", str(root), "--splits", "[val]", "--verify_only", "true", "--quiet", "true"]
    )


def _hint_for_incomplete_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, str]:
    """Drive ``--verify`` over a skipped, incomplete val split; return code and streams.

    Examples:
        >>> callable(_hint_for_incomplete_split)  # needs live tmp_path/monkeypatch/capsys fixtures
        True
    """
    root = tmp_path / "coco"
    _seed_val_root(root, annotated=2, present=1)
    monkeypatch.setattr(urllib.request, "urlopen", _raise_if_called)

    code = data_cli.main(
        ["download", "--data_root", str(root), "--splits", "[val]", "--verify", "true", "--quiet", "true"]
    )
    captured = capsys.readouterr()
    return code, captured.out, captured.err


class TestCliVerify:
    """Tests for the ``download`` subcommand's ``--verify`` / ``--verify_only`` flags."""

    def test_only_passes_on_complete_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """`--verify_only` exits 0 on a complete root without touching the network."""
        root = tmp_path / "coco"
        _seed_val_root(root, annotated=2, present=2)
        monkeypatch.setattr(urllib.request, "urlopen", _raise_if_called)

        code = _verify_only(root)

        assert code == 0

    def test_only_fails_on_missing_images(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """`--verify_only` exits 1 and reports the shortfall on an incomplete root."""
        root = tmp_path / "coco"
        _seed_val_root(root, annotated=4, present=1)
        monkeypatch.setattr(urllib.request, "urlopen", _raise_if_called)

        code = _verify_only(root)

        assert code == 1

    def test_only_fails_on_missing_annotation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """`--verify_only` exits 1 when the split's annotation JSON is absent."""
        root = tmp_path / "coco"
        (root / "val2017").mkdir(parents=True)
        monkeypatch.setattr(urllib.request, "urlopen", _raise_if_called)

        code = _verify_only(root)

        assert code == 1

    def test_after_download_passes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """`--verify` after a fresh download of a complete split exits 0."""
        mapping = {"https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip": _val_archive_bytes(num_images=2)}
        monkeypatch.setattr(urllib.request, "urlopen", _serve(mapping))

        code = data_cli.main(
            [
                "download",
                "--data_root",
                str(tmp_path / "coco"),
                "--splits",
                "[val]",
                "--annotations",
                "false",
                "--verify",
                "true",
                "--quiet",
                "true",
            ]
        )

        assert code == 0

    def test_after_skipped_incomplete_split_hints_force(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`--verify` on a skipped, incomplete split exits 1 and prints a `--force` re-run hint."""
        code, out, err = _hint_for_incomplete_split(tmp_path, monkeypatch, capsys)

        assert code == 1
        assert "val2017" in out  # the verification report
        expected = f"lucid-data download --data_root {tmp_path / 'coco'} --splits '[val]' --force true"
        assert expected in err  # repair hint

    def test_the_repair_hint_names_a_command_that_still_parses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The printed repair hint is a command a caller can paste back into a shell.

        Asserting the hint's *text* only proves a string was printed; the alias removal
        (WP-110) is exactly the change that can leave a well-formed hint naming a command
        nothing installs. So the hint is split as a shell would split it and fed back through
        the shipped parser, which is what makes the assertion mean "runnable" rather than
        "spelled the way this test expects".
        """
        _, _, err = _hint_for_incomplete_split(tmp_path, monkeypatch, capsys)
        hint = next(line.strip() for line in err.splitlines() if line.strip().startswith("lucid-data"))

        command, *arguments = shlex.split(hint)
        config = data_cli.build_parser().parse_args(arguments).download

        assert command == "lucid-data"
        assert list(config.splits) == ["val"]
        assert config.force is True
        assert config.data_root == tmp_path / "coco"
