# SPDX-License-Identifier: Apache-2.0
"""Offline contract tests for the synthetic-shapes generalization golden (WP-083)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import shapes_regression


def test_load_recipe_reads_packaged_n_scale_recipe() -> None:
    """The producer loads its run knobs from the packaged n-scale recipe."""
    recipe = shapes_regression.load_recipe()

    assert recipe.variant == "n"
    assert recipe.max_epochs == 6
    assert recipe.close_mosaic == 1


def test_metric_values_pin_counts_and_val_map_contract() -> None:
    """The producer exposes exact count fields and the three val mAP fields."""
    metrics = shapes_regression._metric_values(
        num_train_images=36,
        num_val_images=4,
        num_val_instances=9,
        num_classes=4,
        epochs=1,
        report={
            "nms": {"map": 0.125, "map_50": 0.25},
            "e2e": {"map": 0.0625},
        },
    )

    assert metrics == {
        "num_train_images": 36,
        "num_val_images": 4,
        "num_val_instances": 9,
        "num_classes": 4,
        "epochs": 1,
        "img_size": 320,
        "val_map_50_95_nms": 0.125,
        "val_map_50_95_e2e": 0.0625,
        "val_map_50_nms": 0.25,
    }


def test_write_golden_round_trips_harness_schema(tmp_path: Path) -> None:
    """``write_golden`` emits the producer, tolerance, and value mappings."""
    path = tmp_path / "shapes_regression_det.json"
    metrics = {
        "num_train_images": 36,
        "num_val_images": 4,
        "num_val_instances": 9,
        "num_classes": 4,
        "epochs": 1,
        "img_size": 320,
        "val_map_50_95_nms": 0.125,
        "val_map_50_95_e2e": 0.0625,
        "val_map_50_nms": 0.25,
    }

    shapes_regression.write_golden(metrics, path)

    assert json.loads(path.read_text()) == {
        "producer": "scripts.shapes_regression:shapes_regression_det",
        "tolerances": {
            "val_map_50_95_nms": shapes_regression._VAL_MAP_50_95_NMS_TOLERANCE,
            "val_map_50_95_e2e": shapes_regression._VAL_MAP_50_95_E2E_TOLERANCE,
            "val_map_50_nms": shapes_regression._VAL_MAP_50_NMS_TOLERANCE,
        },
        "values": metrics,
    }


def test_main_rejects_non_detection_task(capsys) -> None:
    """The task guard fails before dataset generation or training starts."""
    assert shapes_regression.main(["--task", "seg"]) == 1
    assert "unsupported task" in capsys.readouterr().out
