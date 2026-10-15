# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the oriented acceptance instrument (WP-095).

Runs offline against a tiny tiled layout built by :mod:`lucid_yolo.data.tiles` from a
synthetic DOTA split, so the instrument is exercised over exactly the on-disk shape the
tier run feeds it — an untrained module scores near zero, which is fine: what is under
test is that ground truth, predictions and the accumulator meet correctly, not that a
random network detects anything.

The scoring contract itself is pinned separately and without a model: feeding the split's
own ground truth back in as perfect predictions must score 1.0, which is what says the
targets reach :func:`~lucid_yolo.eval.dota_eval.evaluate_rotated_map` in the frame and
the label space it expects. A wrong frame or a shifted label space fails that test while
"it ran and produced a float" passes.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from lucid_yolo.data import tiles as build
from lucid_yolo.data.layout import resolve_split
from lucid_yolo.eval import rotated_eval as evaluate
from lucid_yolo.eval.dota_eval import evaluate_rotated_map
from lucid_yolo.eval.tile_merge import TileIndex, load_tile_index
from lucid_yolo.ptl import datamodule as datamodule_module
from lucid_yolo.ptl.module import DetectionLitModule

#: Letterbox side and tile side used throughout: divisible by the level-32 stride.
IMG_SIZE = 64
#: First stride-32 side where the public val policy selects two workers at batch two.
PARALLEL_IMG_SIZE = 416
#: Source image side, tiled into four windows at the patch/overlap below.
SIDE = 96
PATCH = 64
OVERLAP = 32
#: The shared-memory mount whose presence decides which branch the worker policy takes.
DEV_SHM = Path("/dev/shm")
#: A roomy ``/dev/shm``: 1 GiB free, 262144 blocks of 4096 bytes. The cap is read but never
#: binds here, since the largest queue these tests build (two workers, two prefetched batches
#: of six 416 px images) is about 50 MB against a 512 MiB budget. The cap itself is pinned
#: in ``tests/data/test_coco.py``.
ROOMY_DEV_SHM = os.statvfs_result((4096, 4096, 0, 262144, 262144, 0, 0, 0, 0, 255))


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Seed the global RNG before each test (the module is randomly initialized)."""
    torch.manual_seed(0)


@pytest.fixture
def worker_host(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin what the worker policy reads off the host: two cores and, by default, ``/dev/shm``.

    The policy branches on whether ``/dev/shm`` exists, so left to the runner a Linux cell
    and a macOS cell walk different code under one assertion. Parametrized indirectly with
    ``False``, the host has no ``/dev/shm``. Both patches are path-selective: every other path
    keeps its real ``exists`` and ``statvfs``, because these tests also resolve the tiled
    layout through ``exists``, and a blanket answer would leak into it.
    """
    has_dev_shm: bool = getattr(request, "param", True)
    real_exists = datamodule_module.Path.exists
    real_statvfs = datamodule_module.os.statvfs

    def exists(path: Path, *args: Any, **kwargs: Any) -> bool:
        return has_dev_shm if path == DEV_SHM else real_exists(path, *args, **kwargs)

    def statvfs(path: Any) -> os.statvfs_result:
        return ROOMY_DEV_SHM if path == DEV_SHM else real_statvfs(path)

    monkeypatch.setattr(datamodule_module.os, "cpu_count", lambda: 2)
    monkeypatch.setattr(datamodule_module.Path, "exists", exists)
    monkeypatch.setattr(datamodule_module.os, "statvfs", statvfs)


@pytest.fixture
def requested_worker_counts(monkeypatch: pytest.MonkeyPatch) -> list[int | None]:
    """Record the ``num_workers`` every ``build_datamodule`` call asks for, then build it for real.

    The call is bound against the real signature rather than a copy of it, so the spy keeps
    up with the helper's parameters and a call leaving the count to its default records it.
    """
    real = evaluate.build_datamodule
    signature = inspect.signature(real)
    requested: list[int | None] = []

    def spy(*args: Any, **kwargs: Any) -> datamodule_module.DetectionDataModule:
        call = signature.bind(*args, **kwargs)
        call.apply_defaults()
        requested.append(call.arguments["num_workers"])
        return real(*args, **kwargs)

    monkeypatch.setattr(evaluate, "build_datamodule", spy)
    return requested


@pytest.fixture
def build_tiled_root(tmp_path: Path) -> Callable[[int], Path]:
    """Return a builder tiling a synthetic DOTA split of N source images into a layout.

    Every source image is the same one, carrying one oriented object, and tiles into four
    windows at the patch and overlap above, so the layout holds four tiles per image. One
    layout per test: the builder writes into the test's own ``tmp_path``.
    """

    def build_layout(images: int) -> Path:
        split_dir = tmp_path / "dota" / "val"
        (split_dir / "images").mkdir(parents=True)
        (split_dir / "labelTxt").mkdir(parents=True)
        image = (torch.arange(3 * SIDE * SIDE, dtype=torch.uint8) % 251).reshape(3, SIDE, SIDE)
        for index in range(images):
            stem = f"P{index + 1:04d}"
            build.write_png(image, str(split_dir / "images" / f"{stem}.png"))
            (split_dir / "labelTxt" / f"{stem}.txt").write_text("10 10 30 10 30 20 10 20 plane 0\n")

        build.convert_split(split_dir, tmp_path / "tiles", "val", patch=PATCH, overlap=OVERLAP)
        return tmp_path / "tiles"

    return build_layout


@pytest.fixture
def tiled_root(build_tiled_root: Callable[[int], Path]) -> Path:
    """Build a one-image, one-class tiled layout with a single oriented object."""
    return build_tiled_root(1)


@pytest.fixture
def parallel_tiled_root(build_tiled_root: Callable[[int], Path]) -> Path:
    """Build three source images so the worker pool receives more than one batch per worker."""
    return build_tiled_root(3)


def _datamodule(root: Path) -> object:
    """A val-only datamodule over the built split, set up and ready to iterate.

    Examples:
        >>> callable(_datamodule)  # needs the tiled_root fixture's on-disk tiled split
        True
    """
    datamodule = evaluate.build_datamodule(root, "val", img_size=IMG_SIZE, batch_size=2, variant="n")
    datamodule.setup("validate")
    return datamodule


def test_eval_package_not_reach_up_into_training_layer() -> None:
    """Importing ``lucid_yolo.eval`` must not drag in ``lucid_yolo.ptl``.

    ``eval.checkpoint`` needs :class:`DetectionLitModule`, and that module imports
    ``eval.dota_eval`` — so re-exporting the loader from the eval package closes a cycle
    and every test module that imports both dies at collection with a partially
    initialized module. Re-exporting it is the natural tidying move, which is why the
    direction of the dependency is asserted rather than left to a comment.
    """
    probe = "import lucid_yolo.eval, sys; print(any(m.startswith('lucid_yolo.ptl') for m in sys.modules))"

    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "False"


def test_detection_checkpoint_refused_rather_than_scored(tiled_root: Path) -> None:
    """A module with no angle stem has no oriented number, so it raises instead of reporting one."""
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=15).eval()

    with pytest.raises(ValueError, match="oriented checkpoint"):
        evaluate.score_split(module, _datamodule(tiled_root), torch.device("cpu"), img_size=IMG_SIZE)


def test_split_scored_tile_by_tile(tiled_root: Path) -> None:
    """Every tile of the split is scored, and the report counts the instances it saw."""
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=15, task="obb").eval()

    scoring = evaluate.score_split(module, _datamodule(tiled_root), torch.device("cpu"), img_size=IMG_SIZE)

    assert scoring.tiles == 4
    assert scoring.instances >= 1
    assert 0.0 <= scoring.per_tile["map_50"] <= 1.0


@pytest.mark.parametrize(
    ("limit", "tiles"),
    [pytest.param(2, 2, id="a-prefix"), pytest.param(9, 4, id="past-the-end-of-the-split")],
)
def test_limit_stops_early_without_changing_frame(tiled_root: Path, limit: int, tiles: int) -> None:
    """``--limit`` scores a prefix of the split rather than a resampled subset.

    A limit past the end of the four-tile split is not an error: the pass runs out of tiles
    before it reaches the limit and scores every one of them.
    """
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=15, task="obb").eval()

    scoring = evaluate.score_split(module, _datamodule(tiled_root), torch.device("cpu"), img_size=IMG_SIZE, limit=limit)

    assert scoring.tiles == tiles


@pytest.mark.usefixtures("worker_host")
def test_complete_and_limited_runs_request_chosen_worker_routes(
    tiled_root: Path, tmp_path: Path, requested_worker_counts: list[int | None]
) -> None:
    """Full scoring uses the shared auto policy while a short diagnostic avoids its startup cost.

    At 64 px the policy resolves one worker, which the split's two batches outnumber, so the
    complete run keeps the datamodule it asked the policy for and nothing is rebuilt.
    """
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=15, task="obb").eval()
    common = {
        "module": module,
        "info": {"ema": False},
        "data_root": tiled_root,
        "split": "val",
        "variant": "n",
        "img_size": IMG_SIZE,
        "batch_size": 2,
        "device_name": "cpu",
    }

    assert evaluate.run(**common, limit=0, output=tmp_path / "full.json") == 0  # type: ignore[arg-type]
    assert evaluate.run(**common, limit=1, output=tmp_path / "limited.json") == 0  # type: ignore[arg-type]

    assert requested_worker_counts == [None, 0]


@pytest.mark.parametrize("worker_host", [True, False], indirect=True)
@pytest.mark.parametrize(
    ("num_workers", "expected"),
    [pytest.param(0, 0, id="low-startup"), pytest.param(None, 2, id="policy")],
)
def test_each_route_resolves_one_val_worker_count_with_or_without_dev_shm(
    tiled_root: Path, worker_host: None, num_workers: int | None, expected: int
) -> None:
    """Either host gives zero workers on the low-startup route and the policy's two on the other.

    The shared fixture pins ``/dev/shm`` present everywhere else; this is the direct check
    of a host without it as well. The zero matters on both: before it was forwarded as the
    val loader's own count, the shared-memory floor raised it to one worker wherever
    ``/dev/shm`` exists, so the low-startup route was in-process on macOS alone.
    """
    datamodule = evaluate.build_datamodule(
        tiled_root, "val", img_size=PARALLEL_IMG_SIZE, batch_size=2, variant="n", num_workers=num_workers
    )
    datamodule.setup("validate")

    assert datamodule.val_dataloader().num_workers == expected


@pytest.mark.usefixtures("worker_host")
def test_parallel_validation_preserves_images_targets_and_order(tiled_root: Path) -> None:
    """The auto-worker route emits exactly the low-startup route's OBB inputs in the same order."""

    def snapshot(workers: int | None) -> tuple[int, list[tuple[torch.Tensor, list[tuple[torch.Tensor, ...]]]]]:
        datamodule = evaluate.build_datamodule(
            tiled_root,
            "val",
            img_size=PARALLEL_IMG_SIZE,
            batch_size=2,
            variant="n",
            num_workers=workers,
        )
        datamodule.setup("validate")
        loader = datamodule.val_dataloader()
        batches = []
        for batch in loader:
            images, targets, _ = datamodule.on_after_batch_transfer(batch, 0)
            batches.append(
                (
                    images,
                    [(target.boxes, target.labels, target.rboxes, target.difficult) for target in targets],
                )
            )
        return loader.num_workers, batches

    low_startup_workers, low_startup = snapshot(0)
    parallel_workers, parallel = snapshot(None)

    assert low_startup_workers == 0
    assert parallel_workers == 2
    assert len(low_startup) == len(parallel)
    for (low_startup_images, low_startup_targets), (parallel_images, parallel_targets) in zip(
        low_startup, parallel, strict=True
    ):
        assert torch.equal(low_startup_images, parallel_images)
        assert len(low_startup_targets) == len(parallel_targets)
        for low_startup_target, parallel_target in zip(low_startup_targets, parallel_targets, strict=True):
            assert all(
                torch.equal(left, right) for left, right in zip(low_startup_target, parallel_target, strict=True)
            )


@pytest.mark.usefixtures("worker_host")
def test_tiny_complete_run_keeps_low_startup_route(tiled_root: Path, tmp_path: Path) -> None:
    """A complete custom split does not start a pool with at most one batch per worker."""
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=15, task="obb").eval()
    output = tmp_path / "tiny-complete.json"

    assert (
        evaluate.run(
            module,
            {"ema": False},
            data_root=tiled_root,
            split="val",
            variant="n",
            img_size=PARALLEL_IMG_SIZE,
            batch_size=2,
            device_name="cpu",
            limit=0,
            output=output,
        )
        == 0
    )
    assert json.loads(output.read_text())["info"]["workers"] == 0


@pytest.mark.usefixtures("worker_host")
@pytest.mark.parametrize(
    ("batch_size", "requested", "workers"),
    [
        pytest.param(6, [None, 0], 0, id="one-batch-per-worker-rebuilds-in-process"),
        pytest.param(4, [None], 2, id="a-second-batch-for-a-worker-keeps-the-pool"),
    ],
)
def test_complete_run_rebuilds_in_process_only_when_no_worker_gets_second_batch(
    parallel_tiled_root: Path,
    tmp_path: Path,
    requested_worker_counts: list[int | None],
    batch_size: int,
    requested: list[int | None],
    workers: int,
) -> None:
    """A complete run swaps the policy's datamodule for an in-process one only when no worker gets two batches.

    Twelve tiles at 416 px, where the pinned host's policy resolves two workers. Batch six
    makes two batches, one per worker, so the probe datamodule is set aside and a second is
    built with zero workers; batch four makes three, so the probe's pool is kept and nothing
    is rebuilt. The scored tile count is what shows the rebuilt datamodule was set up and
    read the whole split rather than some prefix of it.
    """
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=15, task="obb").eval()
    output = tmp_path / "complete.json"

    code = evaluate.run(
        module,
        {"ema": False},
        data_root=parallel_tiled_root,
        split="val",
        variant="n",
        img_size=PARALLEL_IMG_SIZE,
        batch_size=batch_size,
        device_name="cpu",
        limit=0,
        output=output,
    )

    report = json.loads(output.read_text())
    assert code == 0
    assert requested_worker_counts == requested
    assert report["info"]["workers"] == workers
    assert report["tiles"] == 12


@pytest.mark.parametrize(
    ("batches", "workers", "expected"),
    [
        pytest.param(1, 1, True, id="one-batch-one-worker"),
        pytest.param(2, 1, False, id="one-worker-gets-a-second-batch"),
        pytest.param(2, 2, True, id="one-batch-per-worker"),
        pytest.param(3, 2, False, id="more-batches-than-workers"),
        pytest.param(0, 2, True, id="empty-split"),
        pytest.param(5, 0, False, id="already-in-process"),
    ],
)
def test_complete_run_sets_pool_aside_until_batches_outnumber_workers(
    batches: int, workers: int, expected: bool
) -> None:
    """The tiny-split fallback fires for any nonzero pool with no more batches than workers.

    Pinned on the pure decision rather than through ``run``, so it needs no model, no
    tiles and no host-dependent worker policy. The one-worker rows are the boundary the
    guard once missed: a policy resolving to a single worker for a one-batch split started
    that worker on hosts where the low-startup route starts none. Zero workers is already the
    in-process route, so there is nothing to set aside however many batches there are.
    """
    assert evaluate._keeps_low_startup_route(batches=batches, workers=workers) is expected


@pytest.mark.usefixtures("worker_host")
def test_run_through_real_workers_matches_run_through_low_startup_route(
    parallel_tiled_root: Path, tmp_path: Path
) -> None:
    """The public complete run keeps its report under a reused real worker pool.

    A 416 px side is the first stride-32 size where the public image-size policy
    selects two workers at batch two. Twelve tiles give each worker three batches;
    a positive limit covers the same complete input through the low-startup route.
    """
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=15, task="obb").eval()
    common = {
        "module": module,
        "info": {"ema": False},
        "data_root": parallel_tiled_root,
        "split": "val",
        "variant": "n",
        "img_size": PARALLEL_IMG_SIZE,
        "batch_size": 2,
        "device_name": "cpu",
    }
    parallel_output = tmp_path / "parallel.json"
    low_startup_output = tmp_path / "low-startup.json"

    assert evaluate.run(**common, limit=0, output=parallel_output) == 0  # type: ignore[arg-type]
    assert evaluate.run(**common, limit=12, output=low_startup_output) == 0  # type: ignore[arg-type]
    parallel_report = json.loads(parallel_output.read_text())
    low_startup_report = json.loads(low_startup_output.read_text())

    assert parallel_report["info"]["workers"] == 2
    assert low_startup_report["info"]["workers"] == 0
    for report in (parallel_report, low_startup_report):
        report.pop("seconds")
        report["info"].pop("workers")
    assert parallel_report == low_startup_report


@pytest.mark.usefixtures("worker_host")
def test_mispaired_index_fails_run_under_real_worker_pool(
    parallel_tiled_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    requested_worker_counts: list[int | None],
) -> None:
    """A pairing refusal raised while two workers feed the pass still ends the run in ``FAIL:`` and status 1.

    The pairing tests walk the refusal through an in-process loader; this walks it through
    the policy's pool. Twelve tiles at batch two give the pinned host's two workers six
    batches, so the pool is kept, and swapping two of the index's windows makes the first
    batch raise in the main process mid-iteration. When the pool shuts down afterwards is
    the loader's timing rather than this command's contract, so it is not asserted.
    """
    index = load_tile_index(resolve_split(parallel_tiled_root, "val")[1])
    assert index is not None
    windows = list(index.windows)
    windows[1], windows[2] = windows[2], windows[1]
    mispaired = dataclasses.replace(index, windows=tuple(windows))
    monkeypatch.setattr(evaluate, "load_tile_index", lambda _ann_file: mispaired)
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=15, task="obb").eval()
    output = tmp_path / "mispaired.json"

    code = evaluate.run(
        module,
        {"ema": False},
        data_root=parallel_tiled_root,
        split="val",
        variant="n",
        img_size=PARALLEL_IMG_SIZE,
        batch_size=2,
        device_name="cpu",
        limit=0,
        output=output,
    )

    assert code == 1
    assert "FAIL: the val dataset's tile 1 has COCO image id" in capsys.readouterr().out
    assert requested_worker_counts == [None]
    assert not output.exists()


def test_report_written_into_directory_not_exist_yet(tiled_root: Path, tmp_path: Path) -> None:
    """``--output`` creates its parent rather than raising after the scoring pass (WP-105).

    The failure this pins is expensive and silent in the worst way: the evaluation
    completes, prints its numbers, and then dies writing them, so the only durable form of
    minutes of GPU work is a traceback. Observed on the first oriented tier evaluation,
    into a `.experiments/obb_smoke/` that no run had created yet.
    """
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=15, task="obb").eval()
    output = tmp_path / "reports" / "nested" / "obb.json"

    code = evaluate.run(
        module,
        {"ema": False},
        data_root=tiled_root,
        split="val",
        variant="n",
        img_size=IMG_SIZE,
        batch_size=2,
        device_name="cpu",
        limit=0,
        output=output,
    )

    assert code == 0
    assert json.loads(output.read_text())["metrics"]["map_50"] >= 0.0


def _run_report(module: DetectionLitModule, root: Path, output: Path) -> dict[str, object]:
    """Score a layout end to end and return the written report.

    Args:
        module: The oriented module to score with.
        root: Root of the tiled layout.
        output: Where the report is written.

    Returns:
        The parsed report payload.

    Examples:
        >>> callable(_run_report)  # needs the tiled_root fixture's on-disk tiled split
        True
    """
    assert (
        evaluate.run(
            module,
            {"ema": False},
            data_root=root,
            split="val",
            variant="n",
            img_size=IMG_SIZE,
            batch_size=2,
            device_name="cpu",
            limit=0,
            output=output,
        )
        == 0
    )
    return json.loads(output.read_text())


def test_report_names_both_figures_rather_than_quoting_one(tiled_root: Path, tmp_path: Path) -> None:
    """WP-107: the per-tile and whole-image numbers are separate, labelled report entries.

    The confusion this package exists to end is a report that quotes an oriented mAP
    without saying which of the two it is — that is how 0.3.0's per-tile figures came to
    be read as comparable to published ones. The four tiles of this layout merge back to
    one source image, and the merged block names the rule and the cap it was scored at.
    """
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=15, task="obb").eval()

    payload = _run_report(module, tiled_root, tmp_path / "obb.json")

    assert payload["info"]["per_tile"] is True  # type: ignore[call-overload,index]
    assert payload["tiles"] == 4
    assert payload["whole_image"]["source_images"] == 1  # type: ignore[call-overload,index]
    assert payload["whole_image"]["max_detections"] is None  # type: ignore[call-overload,index]
    assert 0.0 <= payload["whole_image"]["metrics"]["map_50"] <= 1.0  # type: ignore[call-overload,index]


class TestMergePairing:
    """Which window a tile's detections are translated by is checked against the loader (M-13).

    The merge indexes ``windows[first + position]``, so an index paired to the wrong
    tiles — or a loader that yields the split in any order other than the index's —
    translates every tile's detections by another tile's origin. That failure is silent
    by construction: detections and ground truth move by different amounts, so the
    whole-image figure simply drops.

    Two checks answer it and the cases below separate them. The tile's own COCO image id
    asserts the index-to-dataset pairing outright, including for tiles whose annotation
    sets are byte-identical, where nothing read off the instances could tell them apart.
    The annotation sets are the only reading taken from the yielded batch, so they are
    the only thing that can see the loader's own iteration order.
    """

    @staticmethod
    def _index(root: Path) -> TileIndex:
        """The split's tile index, which every test in this group scores against."""
        index = load_tile_index(resolve_split(root, "val")[1])
        assert index is not None
        return index

    @staticmethod
    def _module() -> DetectionLitModule:
        """An oriented module; what it detects is irrelevant to the pairing."""
        return DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=15, task="obb").eval()

    def test_loader_in_index_order_accepted(self, tiled_root: Path) -> None:
        """The ordinary unshuffled val loader satisfies the check and scores the whole split.

        The refusals below are only worth having if the real path passes them, so the
        untouched loader is scored with the same index the reorder tests corrupt.
        """
        index = self._index(tiled_root)

        scoring = evaluate.score_split(
            self._module(),
            _datamodule(tiled_root),
            torch.device("cpu"),
            img_size=IMG_SIZE,
            windows=index.windows,
            expected_ground_truth=index.ground_truth,
        )

        assert scoring.tiles == 4
        assert len(scoring.source_predictions) == 4

    def test_reordered_loader_refused_rather_than_merged_by_position(
        self, tiled_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A loader yielding the split back-to-front is refused, naming the position and tile.

        This is the mechanism the finding describes: a shuffle, a different sort or any
        other loader change reorders the tiles while the merge keeps indexing windows by
        position. Reversing the sampler reproduces it exactly.
        """
        datamodule = _datamodule(tiled_root)
        index = self._index(tiled_root)
        original = datamodule.val_dataloader()  # type: ignore[attr-defined]
        reversed_loader = DataLoader(
            original.dataset,
            batch_size=2,
            sampler=list(reversed(range(len(original.dataset)))),
            collate_fn=original.collate_fn,
        )
        monkeypatch.setattr(datamodule, "val_dataloader", lambda: reversed_loader)

        with pytest.raises(ValueError, match="does not carry the ground truth the window index"):
            evaluate.score_split(
                self._module(),
                datamodule,
                torch.device("cpu"),
                img_size=IMG_SIZE,
                windows=index.windows,
                expected_ground_truth=index.ground_truth,
            )

    def test_two_tiles_with_identical_annotations_swapped_refused(self, tiled_root: Path) -> None:
        """Windows swapped between two tiles carrying the same annotation set are refused by id.

        This is the pairing failure the annotation-set comparison cannot see. Three of this
        layout's four tiles hold no instances at all, so their labels and difficult flags
        are byte-identical: exchanging two of their windows leaves every annotation-set
        comparison satisfied while every detection of both tiles is translated by the other
        one's origin. The tiles' own COCO image ids differ, and that is what refuses it.
        """
        index = self._index(tiled_root)
        assert index.ground_truth[1]["labels"].tolist() == index.ground_truth[2]["labels"].tolist()
        assert index.ground_truth[1]["difficult"].tolist() == index.ground_truth[2]["difficult"].tolist()
        windows = list(index.windows)
        windows[1], windows[2] = windows[2], windows[1]

        with pytest.raises(ValueError, match="COCO image id"):
            evaluate.score_split(
                self._module(),
                _datamodule(tiled_root),
                torch.device("cpu"),
                img_size=IMG_SIZE,
                windows=tuple(windows),
                expected_ground_truth=index.ground_truth,
            )

    def test_reader_publishing_no_image_ids_still_scores_split(
        self, tiled_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dataset with no COCO ids keeps the annotation-set check and skips the id one.

        The id comparison must not become a requirement the whole-image path imposes on
        every reader: :class:`~lucid_yolo.data.yolo.YoloDetectionDataset` numbers nothing,
        pairing an image to its labels by file stem, so a split it read would have no ids
        to offer. ``Subset`` stands in for such a reader here — it wraps the same tiles in
        the same order and publishes no ``image_ids`` — and the pass must complete on the
        remaining check rather than refuse for want of an identity nobody recorded.
        """
        datamodule = _datamodule(tiled_root)
        index = self._index(tiled_root)
        original = datamodule.val_dataloader()  # type: ignore[attr-defined]
        unnumbered = Subset(original.dataset, list(range(len(original.dataset))))
        assert not hasattr(unnumbered, "image_ids")
        monkeypatch.setattr(
            datamodule,
            "val_dataloader",
            lambda: DataLoader(unnumbered, batch_size=2, collate_fn=original.collate_fn),
        )

        scoring = evaluate.score_split(
            self._module(),
            datamodule,
            torch.device("cpu"),
            img_size=IMG_SIZE,
            windows=index.windows,
            expected_ground_truth=index.ground_truth,
        )

        assert scoring.tiles == 4

    def test_loader_drops_last_batch_refused(self, tiled_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A short pass over a full split is refused, because the merge needs every tile.

        A dropped last batch disturbs no pairing before it, so the per-tile check cannot
        see it: the tiles that did arrive are all correctly paired and the source image is
        merged from a subset of itself, reporting a recall the pipeline never attempted.
        """
        datamodule = _datamodule(tiled_root)
        index = self._index(tiled_root)
        original = datamodule.val_dataloader()  # type: ignore[attr-defined]
        short_loader = DataLoader(original.dataset, batch_size=3, drop_last=True, collate_fn=original.collate_fn)
        monkeypatch.setattr(datamodule, "val_dataloader", lambda: short_loader)

        with pytest.raises(ValueError, match="yielded 3 tiles for a split whose window index describes 4"):
            evaluate.score_split(
                self._module(),
                datamodule,
                torch.device("cpu"),
                img_size=IMG_SIZE,
                windows=index.windows,
                expected_ground_truth=index.ground_truth,
            )


def test_layout_without_window_provenance_reports_no_whole_image_figure(tiled_root: Path, tmp_path: Path) -> None:
    """A53's keys are non-schema, so a plain COCO layout has nothing to merge on.

    The alternative — inventing an offset from the tile file names — would make the naming
    a load-bearing interface, which A53 exists to avoid. Saying the figure is unavailable
    is the honest answer, and it must be said rather than left as a missing key.
    """
    annotations = tiled_root / "annotations" / "instances_val.json"
    payload = json.loads(annotations.read_text())
    for record in payload["images"]:
        del record["source_image"], record["window"]
    annotations.write_text(json.dumps(payload))
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=15, task="obb").eval()

    report = _run_report(module, tiled_root, tmp_path / "obb.json")

    assert report["whole_image"] is None
    assert report["metrics"]["map_50"] >= 0.0  # type: ignore[call-overload,index]


def test_perfect_predictions_score_one(tiled_root: Path) -> None:
    """The split's own ground truth, fed back as detections, scores 1.0.

    This is what pins the frame and the label space: the targets the instrument collects
    are scored by the same accumulator the instrument uses, so a shifted label space or a
    ground truth left in tile coordinates while predictions are letterboxed would fail
    here while any "it produced a float" assertion passed.
    """
    datamodule = _datamodule(tiled_root)
    ground_truth: list[dict[str, torch.Tensor]] = []
    for batch in datamodule.val_dataloader():  # type: ignore[attr-defined]
        _, targets, _ = datamodule.on_after_batch_transfer(batch, 0)  # type: ignore[attr-defined]
        ground_truth.extend(
            {"rboxes": t.rboxes, "labels": t.labels.to(torch.long), "difficult": t.difficult} for t in targets
        )
    predictions = [
        {"rboxes": target["rboxes"], "labels": target["labels"], "scores": torch.ones(target["labels"].shape[0])}
        for target in ground_truth
    ]

    metrics = evaluate_rotated_map(predictions, ground_truth)

    assert metrics["map_50"] == pytest.approx(1.0)
