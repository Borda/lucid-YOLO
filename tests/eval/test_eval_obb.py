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

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from lucid_yolo.data import tiles as build
from lucid_yolo.data.layout import resolve_split
from lucid_yolo.eval import rotated_eval as evaluate
from lucid_yolo.eval.dota_eval import evaluate_rotated_map
from lucid_yolo.eval.tile_merge import TileIndex, load_tile_index
from lucid_yolo.ptl.module import DetectionLitModule

#: Letterbox side and tile side used throughout: divisible by the level-32 stride.
IMG_SIZE = 64
#: Source image side, tiled into four windows at the patch/overlap below.
SIDE = 96
PATCH = 64
OVERLAP = 32


@pytest.fixture(autouse=True)
def reset_random_seeds() -> None:
    """Seed the global RNG before each test (the module is randomly initialized)."""
    torch.manual_seed(0)


@pytest.fixture
def tiled_root(tmp_path: Path) -> Path:
    """Build a one-image, one-class tiled layout with a single oriented object."""
    split_dir = tmp_path / "dota" / "val"
    (split_dir / "images").mkdir(parents=True)
    (split_dir / "labelTxt").mkdir(parents=True)
    image = (torch.arange(3 * SIDE * SIDE, dtype=torch.uint8) % 251).reshape(3, SIDE, SIDE)
    build.write_png(image, str(split_dir / "images" / "P0001.png"))
    (split_dir / "labelTxt" / "P0001.txt").write_text("10 10 30 10 30 20 10 20 plane 0\n")

    build.convert_split(split_dir, tmp_path / "tiles", "val", patch=PATCH, overlap=OVERLAP)
    return tmp_path / "tiles"


def _datamodule(root: Path) -> object:
    """A val-only datamodule over the built split, set up and ready to iterate.

    Examples:
        >>> callable(_datamodule)  # needs the tiled_root fixture's on-disk tiled split
        True
    """
    datamodule = evaluate.build_datamodule(root, "val", img_size=IMG_SIZE, batch_size=2, variant="n")
    datamodule.setup("validate")
    return datamodule


def test_the_eval_package_does_not_reach_up_into_the_training_layer() -> None:
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


def test_a_detection_checkpoint_is_refused_rather_than_scored(tiled_root: Path) -> None:
    """A module with no angle stem has no oriented number, so it raises instead of reporting one."""
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=15).eval()

    with pytest.raises(ValueError, match="oriented checkpoint"):
        evaluate.score_split(module, _datamodule(tiled_root), torch.device("cpu"), img_size=IMG_SIZE)


def test_the_split_is_scored_tile_by_tile(tiled_root: Path) -> None:
    """Every tile of the split is scored, and the report counts the instances it saw."""
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=15, task="obb").eval()

    scoring = evaluate.score_split(module, _datamodule(tiled_root), torch.device("cpu"), img_size=IMG_SIZE)

    assert scoring.tiles == 4
    assert scoring.instances >= 1
    assert 0.0 <= scoring.per_tile["map_50"] <= 1.0


def test_the_limit_stops_early_without_changing_the_frame(tiled_root: Path) -> None:
    """``--limit`` scores a prefix of the split rather than a resampled subset."""
    module = DetectionLitModule(depth=0.34, width=0.25, max_channels=256, num_classes=15, task="obb").eval()

    scoring = evaluate.score_split(module, _datamodule(tiled_root), torch.device("cpu"), img_size=IMG_SIZE, limit=2)

    assert scoring.tiles == 2


def test_the_report_is_written_into_a_directory_that_does_not_exist_yet(tiled_root: Path, tmp_path: Path) -> None:
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


def test_the_report_names_both_figures_rather_than_quoting_one(tiled_root: Path, tmp_path: Path) -> None:
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

    The merge indexes ``windows[first + position]``, so a loader that yields the split in
    any order other than the index's translates every tile's detections by another tile's
    origin. That failure is silent by construction — detections and ground truth move by
    different amounts, so the whole-image figure simply drops — which is why the pairing
    is asserted here rather than left to the prose that used to state it.
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

    def test_the_loader_in_index_order_is_accepted(self, tiled_root: Path) -> None:
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

    def test_a_reordered_loader_is_refused_rather_than_merged_by_position(
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

    def test_a_loader_that_drops_its_last_batch_is_refused(
        self, tiled_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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


def test_a_layout_without_window_provenance_reports_no_whole_image_figure(tiled_root: Path, tmp_path: Path) -> None:
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
