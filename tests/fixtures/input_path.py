# SPDX-License-Identifier: Apache-2.0
"""Rich ragged targets shared by input-path equality tests."""

import torch
from torch import Tensor

from lucid_yolo.data.targets import Targets


def input_samples(counts: tuple[int, ...], size: int = 32) -> list[tuple[Tensor, Targets]]:
    """Build deterministic samples carrying boxes, polygons, rotated boxes and 17 points.

    Examples:
        >>> [len(t.labels) for _, t in input_samples((0, 2, 1))]
        [0, 2, 1]
    """
    samples = []
    for count in counts:
        boxes = torch.tensor([[2.0, 3.0, 12.0, 15.0]]).repeat(count, 1)
        rings = [torch.tensor([[2.0, 3.0], [12.0, 3.0], [12.0, 15.0], [2.0, 15.0]]) for _ in range(count)]
        target = (
            Targets(
                boxes=boxes,
                labels=torch.arange(count) % 4,
                polygons=rings,
                rboxes=torch.tensor([[7.0, 9.0, 12.0, 10.0, 0.2]]).repeat(count, 1),
                keypoints=torch.arange(count * 17 * 2, dtype=torch.float32).reshape(count, 17, 2) % size,
                keypoint_vis=torch.arange(count * 17).reshape(count, 17) % 3,
            )
            if count
            else Targets.empty()
        )
        samples.append((torch.full((3, size, size), 0.5), target))
    return samples
