# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the synthetic fixture cache's invalidation key (M-45).

``tests/fixtures/synthetic.py`` caches its generated micro-datasets on disk, and what that cache is keyed on decides
whether a warm machine and a cold CI runner are looking at the same scene. Keyed on the annotation file merely existing,
they were not: a changed seed, image count, shape set or pinned ``fuse-augmentations`` commit produced a new scene on CI
(always cold, the directory is gitignored) while every fixture-consuming test on a contributor's machine kept running
against the previous one, silently. These tests pin the fingerprint key instead: a changed generation request
regenerates, an unchanged one is served without touching the generator, and ``make clean`` removes the cache directory.

``make test`` runs under pytest-xdist, so several processes reach one cold cache together. ``TestConcurrentPublishing``
starts real spawned processes behind a barrier and pins the other half of the cache's contract: however many publish at
once, none returns and none observes a half-written set. ``TestCacheLock`` orders one interleaving step by step instead:
a worker that judged a stale set stale, then stalled while another published the current set and returned it, must not
take that returned set away, even when its own rename fails.

Every set here is generated into ``tmp_path``, never the shared session cache.
"""

from __future__ import annotations

import errno
import fcntl
import json
import multiprocessing
import queue
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import synthetic

_SPLIT = "train"
_ANNOTATION = "_annotations.coco.json"
#: A seed that is simply not any of the three the module ships, so a regeneration shows.
_OTHER_SEED = 19700101
#: Repository root, three levels up from ``tests/fixtures/test_synthetic.py``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
#: Processes publishing one cache target at once: enough that several are always mid-generation together, few enough
#: that spawning them stays well inside a few seconds.
_RACE_PUBLISHERS = 6
#: Bound on every wait in the race, so a hung child fails the test instead of the CI job.
_RACE_TIMEOUT_S = 120
#: Pause between reader samples; paces the reader on the result queue, never orders anything.
_READER_POLL_S = 0.005
#: Bound on every wait in the lock interleavings, so a missed event fails the test instead of hanging it.
_STEP_TIMEOUT_S = 60


def _annotation_bytes(dataset_dir: Path) -> bytes:
    """Return the raw COCO annotation document of a generated set.

    Args:
        dataset_dir: A directory returned by one of the ``generate_*_fixtures`` helpers.

    Returns:
        The bytes of the ``train`` split's ``_annotations.coco.json``.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     len(_annotation_bytes(synthetic.generate_obb_fixtures(Path(tmp)))) > 0
        True
    """
    return (dataset_dir / _SPLIT / _ANNOTATION).read_bytes()


def _marker(dataset_dir: Path) -> str:
    """Return the fingerprint marker recorded beside a generated set.

    Args:
        dataset_dir: A directory returned by one of the ``generate_*_fixtures`` helpers.

    Returns:
        The text of the set's ``.fixture-fingerprint.json`` file.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     marker = _marker(synthetic.generate_obb_fixtures(Path(tmp)))
        >>> marker.startswith("{")
        True
    """
    return (dataset_dir / synthetic._FINGERPRINT_NAME).read_text(encoding="utf-8")


def _incomplete_reasons(dataset_dir: Path, num_images: int) -> list[str]:
    """Return why the set at ``dataset_dir`` is not a complete ``num_images`` set, or ``[]``.

    Complete means the annotation document parses, declares ``num_images`` images, every declared image exists and is
    non-empty, and the split directory holds no other ``.jpg``.

    Args:
        dataset_dir: A directory a ``generate_*_fixtures`` helper returned or is publishing.
        num_images: How many images the set is meant to hold.

    Returns:
        One message per defect found; a read failure (absent file, truncated JSON) is itself
        reported as the defect rather than raised.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     ds = synthetic.generate_obb_fixtures(Path(tmp))
        ...     complete = _incomplete_reasons(ds, synthetic.OBB_NUM_IMAGES)
        ...     short = _incomplete_reasons(ds, synthetic.OBB_NUM_IMAGES + 1)
        ...     absent = _incomplete_reasons(Path(tmp) / "absent", 1)
        >>> complete
        []
        >>> short[0].startswith("declares 8 images")
        True
        >>> absent[0].startswith("FileNotFoundError")
        True
    """
    split_dir = dataset_dir / _SPLIT
    try:
        document = json.loads((split_dir / _ANNOTATION).read_text(encoding="utf-8"))
        declared = sorted(entry["file_name"] for entry in document["images"])
        on_disk = sorted(path.name for path in split_dir.glob("*.jpg"))
        empty = [name for name in declared if (split_dir / name).stat().st_size == 0]
    except (OSError, ValueError, KeyError) as exc:
        return [f"{type(exc).__name__}: {exc}"]
    reasons = []
    if len(declared) != num_images:
        reasons.append(f"declares {len(declared)} images, expected {num_images}")
    if on_disk != declared:
        reasons.append(f"{len(on_disk)} jpg on disk do not match the {len(declared)} declared")
    if empty:
        reasons.append(f"empty image files: {empty[:3]}")
    return reasons


def _accepted_set_problems(dataset_dir: Path, fingerprint: str, num_images: int) -> list[str]:
    """Return the defects of ``dataset_dir`` once its marker says the set is finished, else ``[]``.

    This is the acceptance rule a consumer applies: the fingerprint file is the completion marker, so a set without it
    is "not there yet", not "broken". A set that carries the marker and is still incomplete is the failure the atomic
    publish exists to prevent.

    Args:
        dataset_dir: Directory the set is published at.
        fingerprint: The marker content that means "complete and current".
        num_images: How many images a complete set holds.

    Returns:
        The :func:`_incomplete_reasons` of an accepted set; ``[]`` when nothing is accepted yet.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     ds = synthetic.generate_obb_fixtures(Path(tmp))
        ...     marker = _marker(ds)
        ...     absent = _accepted_set_problems(Path(tmp) / "absent", marker, 8)
        ...     whole = _accepted_set_problems(ds, marker, 8)
        ...     (ds / _SPLIT / _ANNOTATION).unlink()
        ...     torn = _accepted_set_problems(ds, marker, 8)
        >>> absent, whole
        ([], [])
        >>> len(torn)
        1
    """
    try:
        accepted = _marker(dataset_dir) == fingerprint
    except OSError:
        return []
    return _incomplete_reasons(dataset_dir, num_images) if accepted else []


def _seed_stale_set(root: Path) -> None:
    """Leave a complete-looking set under ``root`` whose marker names another generation request.

    Args:
        root: Cache directory the ``detseg`` set is published beneath.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     _seed_stale_set(Path(tmp))
        ...     (Path(tmp) / "detseg" / synthetic._FINGERPRINT_NAME).read_text(encoding="utf-8")
        'stale'
    """
    dataset_dir = synthetic.generate_detseg_fixtures(root)
    (dataset_dir / synthetic._FINGERPRINT_NAME).write_text("stale", encoding="utf-8")
    for extra in range(3):  # surplus images that a refresh must not carry over
        (dataset_dir / _SPLIT / f"zz_surplus_{extra}.jpg").write_bytes(b"x")


def _publish_after_release(barrier: Any, root: Path, num_images: int, outbox: Any) -> None:
    """Child-process body: wait for the barrier, publish the ``detseg`` set, report its defects.

    Args:
        barrier: Barrier every racer waits at, so all start publishing together.
        root: Cache directory shared by every racer.
        num_images: How many images a complete set holds.
        outbox: Queue receiving one ``list[str]`` of defects per racer; any exception the publish raises is reported
            there as a defect instead of being lost with the child.

    Examples:
        >>> import tempfile, threading
        >>> outbox = queue.Queue()
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     _publish_after_release(threading.Barrier(1), Path(tmp), synthetic.DETSEG_NUM_IMAGES, outbox)
        >>> outbox.get_nowait()
        []
    """
    try:
        barrier.wait(_RACE_TIMEOUT_S)
        dataset_dir = synthetic.generate_detseg_fixtures(root)
        outbox.put(_incomplete_reasons(dataset_dir, num_images))
    except Exception as exc:
        outbox.put([f"publisher raised {type(exc).__name__}: {exc}"])


class _RaceOutcome(NamedTuple):
    """What one race left behind: defects the racers returned and defects a reader saw."""

    publisher_problems: list[str]
    reader_problems: list[str]


@pytest.fixture
def race_publishers(monkeypatch: pytest.MonkeyPatch) -> Callable[[Path, str], _RaceOutcome]:
    """Return a runner that publishes the ``detseg`` set from ``_RACE_PUBLISHERS`` spawned processes at once.

    The runner releases the racers from one barrier and, from the parent, keeps sampling the target the way a consumer
    does until every racer has reported. Spawned children unpickle this module's worker by name, so the repository root
    goes on ``sys.path`` for them.
    """
    monkeypatch.syspath_prepend(str(_REPO_ROOT))

    def run(root: Path, fingerprint: str) -> _RaceOutcome:
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(_RACE_PUBLISHERS + 1)
        outbox = context.Queue()
        num_images = synthetic.DETSEG_NUM_IMAGES
        racers = [
            context.Process(target=_publish_after_release, args=(barrier, root, num_images, outbox))
            for _ in range(_RACE_PUBLISHERS)
        ]
        for racer in racers:
            racer.start()
        reports: list[list[str]] = []
        reader_problems: list[str] = []
        deadline = time.monotonic() + _RACE_TIMEOUT_S
        try:
            barrier.wait(_RACE_TIMEOUT_S)
            while len(reports) < _RACE_PUBLISHERS:
                reader_problems += _accepted_set_problems(root / "detseg", fingerprint, num_images)
                try:
                    reports.append(outbox.get(timeout=_READER_POLL_S))
                except queue.Empty:
                    assert time.monotonic() < deadline, f"{len(reports)} of {_RACE_PUBLISHERS} publishers reported"
            for racer in racers:
                racer.join(_RACE_TIMEOUT_S)
        finally:
            for racer in racers:
                if racer.is_alive():
                    racer.terminate()
        return _RaceOutcome([reason for report in reports for reason in report], reader_problems)

    return run


class _Publisher(threading.Thread):
    """Thread making one ``generate_detseg_fixtures`` call and keeping what it returned or raised."""

    def __init__(self, name: str, root: Path, done: threading.Event | None = None) -> None:
        super().__init__(name=name, daemon=True)
        self.root = root
        self.done = done
        self.returned: Path | None = None
        self.marker: str | None = None
        self.error: Exception | None = None

    def run(self) -> None:
        """Publish under ``root``, record the result and its marker, then signal ``done``."""
        try:
            self.returned = synthetic.generate_detseg_fixtures(self.root)
            self.marker = _marker(self.returned)
        except Exception as exc:
            self.error = exc
        finally:
            if self.done is not None:
                self.done.set()


class _Interleaving(NamedTuple):
    """What one delayed-replacer interleaving left behind."""

    prompt: _Publisher
    delayed: _Publisher
    #: Defects of the target found after each rename the delayed worker made once the prompt one had returned.
    defects_after_return: list[str]


@pytest.fixture
def delayed_replacer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[[bool], _Interleaving]:
    """Return a runner pausing one worker between its stale inspection and its replacement.

    The cache starts with a stale ``detseg`` set. The *delayed* thread runs the publish path up to its first attempt to
    move the target aside, which only ever follows a judgement that the set there is stale, and parks. The *prompt*
    thread then publishes, and the delayed one is released once the prompt one has returned, or has reached the cache
    lock and so can do no more while the delayed one holds it. Each rename the delayed thread makes after the prompt one
    returned is followed by a completeness check of the target. ``fail_republish`` makes the delayed thread's renames of
    its own set onto the target fail once it has been released, standing in for a crash between moving the old set aside
    and renaming the new one in.

    Threads stand in for xdist workers: ``flock`` belongs to the open file description, so two threads that each open
    the lock file exclude each other as two processes do, and events rather than timing order every step.
    """
    cache = tmp_path / "cache"
    _seed_stale_set(cache)
    target = cache / "detseg"
    real_rename = Path.rename
    real_flock = fcntl.flock

    def run(fail_republish: bool) -> _Interleaving:
        paused, resume, prompt_progressed = threading.Event(), threading.Event(), threading.Event()
        delayed = _Publisher("delayed", cache)
        prompt = _Publisher("prompt", cache, done=prompt_progressed)
        defects: list[str] = []

        def rename(self: Path, destination: Any) -> Any:
            if threading.current_thread() is not delayed:
                return real_rename(self, destination)
            if self == target and not paused.is_set():
                paused.set()
                resume.wait(_STEP_TIMEOUT_S)
            if fail_republish and paused.is_set() and Path(destination) == target:
                raise OSError(errno.EIO, "injected republish failure", str(destination))
            moved = real_rename(self, destination)
            if prompt.returned is not None:
                defects.extend(_incomplete_reasons(target, synthetic.DETSEG_NUM_IMAGES))
            return moved

        def flock(fd: Any, operation: int) -> None:
            if threading.current_thread() is prompt:
                prompt_progressed.set()
            real_flock(fd, operation)

        monkeypatch.setattr(Path, "rename", rename)
        monkeypatch.setattr(fcntl, "flock", flock)
        delayed.start()
        try:
            assert paused.wait(_STEP_TIMEOUT_S), "the delayed worker never reached the replacement"
            prompt.start()
            assert prompt_progressed.wait(_STEP_TIMEOUT_S), "the prompt worker neither returned nor reached the lock"
        finally:
            resume.set()
        delayed.join(_STEP_TIMEOUT_S)
        prompt.join(_STEP_TIMEOUT_S)
        assert not (delayed.is_alive() or prompt.is_alive()), "a worker was still running after the interleaving"
        return _Interleaving(prompt, delayed, defects)

    return run


class TestFixtureCacheKey:
    """The cache is keyed on the generation request, not on the annotation file existing."""

    def test_changed_seed_regenerates_set(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Changing a seed replaces the cached scene instead of serving the previous one.

        This is the M-45 failure exactly: the seed constants are edited in the source, and every machine that has
        generated once before keeps the old images and annotations because the annotation file still exists. CI, which
        is always cold, sees the new ones — so the two disagree about what every fixture-consuming test is asserting
        against, and nothing says so.
        """
        before = _annotation_bytes(synthetic.generate_detseg_fixtures(tmp_path))
        monkeypatch.setattr(synthetic, "DETSEG_SEED", _OTHER_SEED)

        after = _annotation_bytes(synthetic.generate_detseg_fixtures(tmp_path))

        assert after != before, "a changed seed served the previously cached scene"

    def test_changed_image_count_regenerates_set(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Changing the image count replaces the cached scene, leaving no surplus images behind.

        Shrinking the set is the case a fingerprint check alone would not fix: regenerating into a directory that still
        holds the larger previous run leaves the extra images on disk, and a reader that walks the split directory
        rather than the annotation file would then see images the document never declares.
        """
        synthetic.generate_obb_fixtures(tmp_path)
        monkeypatch.setattr(synthetic, "OBB_NUM_IMAGES", synthetic.OBB_NUM_IMAGES - 1)

        dataset_dir = synthetic.generate_obb_fixtures(tmp_path)

        images = sorted((dataset_dir / _SPLIT).glob("*.jpg"))
        assert len(images) == synthetic.OBB_NUM_IMAGES

    def test_unchanged_request_served_from_cache(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """An identical second call returns without invoking the generator at all.

        The cache still has to *be* a cache: the fixture suite generates once per session and every consumer after that
        pays nothing. A fingerprint that matched only on a fresh run would turn every session into a full regeneration.
        """
        first = synthetic.generate_keypoints_fixtures(tmp_path)
        monkeypatch.setattr(synthetic, "generate_dataset", _refuse_to_generate)

        second = synthetic.generate_keypoints_fixtures(tmp_path)

        assert second == first

    def test_missing_fingerprint_regenerates_set(self, tmp_path: Path) -> None:
        """A set cached before fingerprints existed is refreshed once rather than trusted.

        Every warm machine at the time of this change holds exactly that: a complete set with no fingerprint beside it,
        generated by a request nothing recorded. Treating the absent key as a match would leave those machines on the
        stale scene indefinitely.
        """
        dataset_dir = synthetic.generate_detseg_fixtures(tmp_path)
        (dataset_dir / synthetic._FINGERPRINT_NAME).unlink()

        synthetic.generate_detseg_fixtures(tmp_path)

        assert (dataset_dir / synthetic._FINGERPRINT_NAME).is_file()

    def test_pinned_generator_commit_part_of_key(self) -> None:
        """The fingerprint names the exact upstream commit, not just its version string.

        ``fuse-augmentations`` is pinned to a git SHA and reports a dev version that does not move when that SHA does,
        so a bump to a generator that draws different scenes would leave the version identical and every cached set
        silently stale.
        """
        identity = synthetic._generator_identity()

        assert "direct_url" in identity, "a git-pinned generator must contribute its commit to the key"
        assert "commit_id" in identity["direct_url"]


class TestConcurrentPublishing:
    """Processes that reach one cache target together all end with the same complete set."""

    @pytest.mark.parametrize(
        "seed_cache",
        [
            pytest.param(lambda root: None, id="cold"),
            pytest.param(_seed_stale_set, id="stale"),
        ],
    )
    def test_racing_publishers_leave_one_complete_set(
        self,
        tmp_path: Path,
        race_publishers: Callable[[Path, str], _RaceOutcome],
        seed_cache: Callable[[Path], None],
    ) -> None:
        """Spawned processes publishing one target at once never return or expose a partial set.

        This is ``make test -n auto`` on a cold cache, and on a warm one whose set went stale: the fixture cache used to
        be regenerated in place, so one worker removed the tree another was still writing and consumers read truncated
        annotation JSON. Every racer must come back with a complete set, a reader accepting the target by its marker
        must never find a torn set behind it, the survivor must be the current one, and no staging directory may remain;
        only the lock file stays beside the set, by design.
        """
        reference = synthetic.generate_detseg_fixtures(tmp_path / "reference")
        fingerprint = _marker(reference)
        cache = tmp_path / "cache"
        seed_cache(cache)

        outcome = race_publishers(cache, fingerprint)

        published = cache / "detseg"
        assert outcome.publisher_problems == [], "a publisher returned a set that was not complete"
        assert outcome.reader_problems == [], "a reader accepted the target while the set behind it was torn"
        assert _incomplete_reasons(published, synthetic.DETSEG_NUM_IMAGES) == []
        assert _marker(published) == fingerprint
        assert sorted(path.name for path in cache.iterdir()) == sorted(
            ["detseg", synthetic._lock_path(published).name]
        ), "staging directories were left behind"


class TestCacheLock:
    """A set some caller has returned stays present and complete, whatever a late replacer does."""

    def test_current_set_served_while_lock_held(self, tmp_path: Path) -> None:
        """A set whose fingerprint matches is returned without waiting for the cache lock.

        Readers of a valid set must never queue behind a worker that is regenerating: the warm session cache is read by
        every fixture-consuming test on every worker. The lock is held here for the whole call, so a lookup that took it
        would wait out the join timeout.
        """
        dataset_dir = synthetic.generate_detseg_fixtures(tmp_path)
        reader = _Publisher("reader", tmp_path)

        with synthetic._exclusive(dataset_dir):
            reader.start()
            reader.join(_STEP_TIMEOUT_S)
            blocked = reader.is_alive()

        assert not blocked, "a lookup of a current set waited for the cache lock"
        assert reader.error is None
        assert reader.returned == dataset_dir

    def test_delayed_replacer_leaves_returned_set_complete(
        self, delayed_replacer: Callable[[bool], _Interleaving]
    ) -> None:
        """A worker that judged the set stale before another published it never takes the returned set away.

        This is the stale-cache window WP203-C-001 names: without a lock, the delayed worker moved the set the prompt
        one had just published and returned out of the way, and the path stayed absent until the delayed worker's own
        copy was renamed in. Any consumer reading it then failed.
        """
        outcome = delayed_replacer(False)

        assert (outcome.prompt.error, outcome.delayed.error) == (None, None)
        assert outcome.defects_after_return == [], "the returned set went missing or torn after it was returned"
        assert outcome.prompt.returned is not None
        assert _incomplete_reasons(outcome.prompt.returned, synthetic.DETSEG_NUM_IMAGES) == []
        assert _marker(outcome.prompt.returned) == outcome.prompt.marker

    def test_failed_republish_never_strands_returned_path(
        self, delayed_replacer: Callable[[bool], _Interleaving]
    ) -> None:
        """A replacer whose rename fails mid-swap raises, and never leaves a returned path absent.

        A crash or a failed rename between moving the old set aside and renaming the new one in leaves the target empty.
        That is harmless only while no caller has returned the path, so a caller waiting behind the failed one must find
        the set missing and generate it, rather than return the empty path or never learn the replacement failed.
        """
        outcome = delayed_replacer(True)

        assert isinstance(outcome.delayed.error, OSError)
        assert outcome.delayed.error.errno == errno.EIO
        assert outcome.prompt.error is None
        assert outcome.prompt.returned is not None
        assert _incomplete_reasons(outcome.prompt.returned, synthetic.DETSEG_NUM_IMAGES) == []


def _refuse_to_generate(*_args: object, **_kwargs: object) -> None:
    """A ``generate_dataset`` replacement asserting the cache was consulted first.

    Examples:
        >>> try:
        ...     _refuse_to_generate()
        ... except AssertionError as exc:
        ...     str(exc)
        'the fixture cache regenerated a set whose fingerprint already matched'
    """
    raise AssertionError("the fixture cache regenerated a set whose fingerprint already matched")


def test_make_clean_removes_generated_fixture_cache() -> None:
    """``make clean`` clears ``tests/fixtures/_generated`` along with the other caches.

    The directory is gitignored, so it survives every checkout and every branch switch; a ``clean`` that left it in
    place meant "start from nothing" named two different states depending on whether the machine had ever run the suite.
    """
    recipe = subprocess.run(
        ["make", "--dry-run", "--no-print-directory", "clean"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        check=True,
    ).stdout

    assert "tests/fixtures/_generated" in recipe
