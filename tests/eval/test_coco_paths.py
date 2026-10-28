# SPDX-License-Identifier: Apache-2.0
"""COCO evaluation reads custom paths independently and retains legacy defaults."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torchvision.io import write_png

from lucid_yolo.eval import detect_eval, pose_eval
from lucid_yolo.ptl.module import DetectionLitModule


@pytest.mark.parametrize("task", ["detect", "keypoints"])
@pytest.mark.parametrize("custom_annotations", [False, True])
@pytest.mark.parametrize("custom_images", [False, True])
def test_coco_evaluation_reads_selected_paths(
    task: str,
    custom_annotations: bool,
    custom_images: bool,
    tmp_path: Path,
) -> None:
    """A real image and annotations score even when either legacy path does not exist."""
    ann_name = "person_keypoints_val2017.json" if task == "keypoints" else "instances_val2017.json"
    ann_file = tmp_path / "subset.json" if custom_annotations else tmp_path / "annotations" / ann_name
    images_dir = tmp_path / ("subset" if custom_images else "val2017")
    ann_file.parent.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir()
    write_png(torch.full((3, 32, 32), 128, dtype=torch.uint8), str(images_dir / "image.png"))
    annotation: dict[str, Any] = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [4, 4, 16, 16],
        "area": 256,
        "iscrowd": 0,
    }
    if task == "keypoints":
        annotation.update(keypoints=[10, 10, 2] * 17, num_keypoints=17)
    ann_file.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "image.png", "height": 32, "width": 32}],
                "annotations": [annotation],
                "categories": [{"id": 1, "name": "person"}],
            }
        )
    )
    module = DetectionLitModule(
        depth=0.34,
        width=0.25,
        max_channels=64,
        num_classes=1,
        task=task,
        num_keypoints=17 if task == "keypoints" else None,
    ).eval()
    kwargs: dict[str, Any] = {}
    if custom_annotations:
        kwargs["ann_file"] = ann_file
    if custom_images:
        kwargs["images_dir"] = images_dir
    if task == "detect":
        kwargs["masks"] = False
    output = tmp_path / "report.json"
    protocol = pose_eval if task == "keypoints" else detect_eval
    assert (
        protocol.run(
            module,
            {},
            data_root=tmp_path,
            img_size=32,
            batch_size=1,
            device_name="cpu",
            limit=0,
            output=output,
            **kwargs,
        )
        == 0
    )
    payload = json.loads(output.read_text())
    assert payload["images"] == 1
    assert set(payload["report"]) == {"e2e", "nms"}
    for stats in payload["report"].values():
        assert 0 <= stats["map"] <= 1
        if task == "keypoints":
            assert "oks_AP_all" in stats
