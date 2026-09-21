# SPDX-License-Identifier: Apache-2.0
"""Overfit-100 integration test: the full train loop memorizes a fixed slice (WP-040).

Exercises :mod:`scripts.overfit_micro` end to end. One accelerator-gated fixture
runs the whole pipeline once — generate the ~100-image synthetic slice, train the
``n``-scale detector, and score train-set recall at IoU 0.5 — and the ``gpu``/``data``
tests assert the WP-040 DoD against it (recall clears the ``>= 0.95`` floor, the
integer counts are exact, and the achieved recall matches the frozen golden within
tolerance). A parametrized sibling fixture does the same for the other three gates, so
segmentation, oriented detection and keypoints each have their published floor asserted
by something other than a human reading the CLI's output (WP-167 H-01).

The mark-free tests run in the offline unit gate and need no accelerator: one pins the
offline-harness contract (the frozen goldens live under ``goldens/gpu/`` and are
therefore *excluded* from the default ``scripts/check_goldens.py`` discovery), two stub
training to exercise the producer's own floor branch in both directions, and one covers
the freeze-time tolerance clamp that keeps a stored band above its floor (WP-167 H-02).
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import check_goldens, overfit_micro  # noqa: E402  (path arranged just above)

#: The exact integer-valued metrics the overfit slice must always produce.
_EXPECTED_COUNTS = (100.0, 592.0, 4.0, 100.0, 320.0)


def test_overfit_golden_excluded_from_offline_harness() -> None:
    """The frozen golden is discovered only under gpu/, never by the default harness."""
    goldens_dir = check_goldens.DEFAULT_GOLDENS_DIR
    assert overfit_micro._GOLDEN_PATH.is_file()
    assert overfit_micro._GOLDEN_PATH in check_goldens.discover_gpu_goldens(goldens_dir)
    assert overfit_micro._GOLDEN_PATH not in check_goldens.discover_goldens(goldens_dir)


@pytest.mark.parametrize(
    ("documented", "key"),
    [
        pytest.param("detect", "det", id="detect"),
        pytest.param("segment", "seg", id="segment"),
        pytest.param("obb", "obb", id="obb"),
        pytest.param("keypoints", "kp", id="keypoints"),
    ],
)
def test_the_documented_task_names_resolve_to_a_wired_gate(documented: str, key: str) -> None:
    """Every ``--task`` spelling TRAINING.md writes names a gate in ``TASK_SPECS``.

    Three of the four documented commands exited 1 with "unsupported task" before WP-190,
    because the lookup knew only the short keys; the alias table is what closed that.
    """
    assert overfit_micro.resolve_task(documented) == key
    assert key in overfit_micro.TASK_SPECS


@pytest.fixture(scope="module")
def overfit_metrics() -> dict[str, float]:
    """Run the full overfit pipeline once and share its metrics across the gpu tests."""
    return overfit_micro.run_overfit()


@pytest.mark.gpu
@pytest.mark.data
def test_overfit_recall_clears_floor(overfit_metrics: dict[str, float]) -> None:
    """Train-set recall at IoU 0.5 reaches the WP-040 DoD floor (>= 0.95)."""
    assert overfit_metrics["train_recall_at_050"] >= overfit_micro._RECALL_FLOOR


@pytest.mark.gpu
@pytest.mark.data
def test_overfit_integer_counts_are_exact(overfit_metrics: dict[str, float]) -> None:
    """The slice's integer metrics (images, instances, classes, epochs, size) are fixed."""
    counts = (
        overfit_metrics["num_images"],
        overfit_metrics["num_instances"],
        overfit_metrics["num_classes"],
        overfit_metrics["epochs"],
        overfit_metrics["img_size"],
    )
    assert counts == _EXPECTED_COUNTS


@pytest.mark.gpu
@pytest.mark.data
def test_overfit_recall_matches_frozen_golden(overfit_metrics: dict[str, float]) -> None:
    """The recomputed recall matches the frozen golden within its tolerance."""
    golden = json.loads(overfit_micro._GOLDEN_PATH.read_text())
    expected = golden["values"]["train_recall_at_050"]
    tolerance = golden["tolerances"]["train_recall_at_050"]
    assert abs(overfit_metrics["train_recall_at_050"] - expected) <= tolerance


@pytest.fixture(
    scope="module",
    params=[pytest.param("seg", id="seg"), pytest.param("obb", id="obb"), pytest.param("kp", id="kp")],
)
def non_detection_metrics(request: pytest.FixtureRequest) -> tuple[str, dict[str, float]]:
    """Run the overfit pipeline once per non-detection task and share its metrics."""
    task = str(request.param)
    return task, overfit_micro.run_overfit(task)


@pytest.mark.gpu
@pytest.mark.data
def test_non_detection_score_clears_floor(non_detection_metrics: tuple[str, dict[str, float]]) -> None:
    """Segmentation mask IoU, oriented mAP50 and keypoint OKS AP each clear their own DoD floor.

    Detection was the only task whose floor any automated caller enforced; the other
    three were compared once, inside the CLI's ``main()``, so a below-floor seg/obb/kp
    run passed every test and could still be frozen by hand. These cases pin the
    published floor per task (WP-167 H-01).
    """
    task, metrics = non_detection_metrics
    spec = overfit_micro.TASK_SPECS[task]
    assert metrics[spec.metric] >= spec.floor


@pytest.mark.parametrize(
    ("task", "score", "expected_tolerance"),
    [
        pytest.param("kp", 0.335667, 0.035667, id="kp-clamped-to-headroom"),
        pytest.param("obb", 0.938966, 0.038966, id="obb-clamped-to-headroom"),
        pytest.param("det", 0.994932, 0.04, id="det-keeps-nominal-band"),
        pytest.param("seg", 0.814623, 0.05, id="seg-keeps-nominal-band"),
    ],
)
def test_written_band_never_dips_below_the_floor(
    tmp_path: Path, task: str, score: float, expected_tolerance: float
) -> None:
    """Freezing narrows the stored tolerance to the headroom above the floor.

    The stored bands accepted an OKS AP of 0.285667 against a 0.30 floor and a
    rotated mAP50 of 0.888966 against a 0.90 floor — values the published DoD
    rejects. The clamp lives in the writer, so re-freezing produces a correct band
    (WP-167 H-02). The two tasks with headroom wider than their nominal band keep it.
    """
    spec = dataclasses.replace(overfit_micro.TASK_SPECS[task], golden_path=tmp_path / f"overfit_micro_{task}.json")
    metrics = {"num_images": 100.0, spec.metric: score}

    overfit_micro.write_golden(metrics, spec)

    written = json.loads(spec.golden_path.read_text())
    assert written["tolerances"][spec.metric] == expected_tolerance
    # Six decimals is the precision the goldens themselves store; binary float dust
    # otherwise puts the exactly-clamped obb edge one ulp under its floor.
    assert round(score - written["tolerances"][spec.metric], 6) >= spec.floor


def test_main_prints_probe_for_a_capped_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--epochs`` below the tuned budget exits 0 with a PROBE line and no PASS line.

    The CLI must not print PASS for a run the floor was never compared against.
    """
    monkeypatch.setattr(
        overfit_micro, "run_overfit", lambda task, epochs, batches: {"num_instances": 5.0, "train_oks_ap": 0.02}
    )

    status = overfit_micro.main(["--task", "keypoints", "--epochs", "1", "--batches", "4"])

    out = capsys.readouterr().out
    assert status == 0
    assert "PROBE: 1 of" in out
    assert "PASS" not in out


class TestRunOverfit:
    """The producer: floor enforcement, its probe override, and the budget it refuses."""

    def test_returns_metrics_when_score_clears_floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A run scoring above its floor returns the metric mapping with the score rounded.

        Stands in for the accelerator: training is stubbed so the producer's own
        floor branch — not the trainer — is what the case exercises.
        """
        spec = overfit_micro.TASK_SPECS["kp"]
        monkeypatch.setattr(overfit_micro, "generate_slice", lambda *_args: Path("unused"))
        monkeypatch.setattr(overfit_micro, "_train_and_score", lambda *_args: (0.3456789, 548, 1))

        metrics = overfit_micro.run_overfit("kp")

        assert metrics[spec.metric] == 0.345679
        assert metrics["num_instances"] == 548.0

    def test_raises_when_score_below_floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The producer itself rejects a below-floor run, so no caller can accept one.

        The keypoint gate is the narrowest of the four floors and the one whose frozen
        band was widest relative to it, so it is the case that would have gone unnoticed.
        """
        spec = overfit_micro.TASK_SPECS["kp"]
        monkeypatch.setattr(overfit_micro, "generate_slice", lambda *_args: Path("unused"))
        monkeypatch.setattr(overfit_micro, "_train_and_score", lambda *_args: (spec.floor - 0.01, 548, 1))

        with pytest.raises(overfit_micro.FloorNotMet) as excinfo:
            overfit_micro.run_overfit("kp")

        assert excinfo.value.floor == spec.floor
        assert excinfo.value.score == pytest.approx(spec.floor - 0.01)

    def test_reports_a_capped_run_without_the_floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A probe budget below the tuned one returns its score instead of raising ``FloorNotMet``.

        The development-gates notebook walks every task at one epoch on a CPU; that run is
        a wiring check and its score is nowhere near the floor by design, so the floor
        applies only at the budget it was tuned against.
        """
        spec = overfit_micro.TASK_SPECS["kp"]
        seen: dict[str, int] = {}
        monkeypatch.setattr(overfit_micro, "generate_slice", lambda *_args: Path("unused"))
        monkeypatch.setattr(
            overfit_micro,
            "_train_and_score",
            lambda recipe, *args: (
                seen.update(epochs=recipe.max_epochs, mosaic=recipe.close_mosaic, batches=args[-1]) or (0.01, 548, 1)
            ),
        )

        metrics = overfit_micro.run_overfit("kp", epochs=1, batches=4)

        assert metrics[spec.metric] == 0.01
        assert metrics["epochs"] == 1.0
        assert seen == {"epochs": 1, "mosaic": 1, "batches": 4}

    @pytest.mark.parametrize("epochs", [0, overfit_micro._EPOCHS + 1])
    def test_rejects_a_budget_outside_the_tuned_range(self, epochs: int) -> None:
        """Zero epochs and more than the tuned budget are refused before any training starts.

        Above the budget would silently gate a stronger run against the same floor.
        """
        with pytest.raises(ValueError, match="epochs must be in"):
            overfit_micro.run_overfit("kp", epochs=epochs)
