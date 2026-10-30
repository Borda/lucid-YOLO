# SPDX-License-Identifier: Apache-2.0
"""Input builders for the rotated mAP50-95 unit gate (``tests/eval/test_dota_eval.py``).

Every case of that gate hands :func:`~lucid_yolo.eval.dota_eval.evaluate_rotated_map` the same two shapes of dict: one
prediction dict (``rboxes``, ``scores``, ``labels``) and one ground-truth dict (``rboxes``, ``labels``, ``difficult``)
per image. :func:`rotated_prediction` and :func:`rotated_target` build them with the defaults nearly every case wants
(class 0, nothing difficult), so a case spells out only the field that makes it unique. The larger exact-recall and
axis-aligned cross-check fixtures sit beside them, so the test module holds scenarios rather than scaffolding.
"""

from __future__ import annotations

import torch
from torch import Tensor

#: Shape of the cross-check fixture in :func:`axis_aligned_fixture`. Large enough that the matching is exercised across
#: many curves rather than one lucky one.
_FIXTURE_IMAGES = 24
_FIXTURE_CLASSES = 4
#: The class skipped in every other image, so at least one curve saturates below recall 1.
_STARVED_CLASS = 3


def rotated_prediction(
    rboxes: Tensor | list[list[float]], scores: Tensor, labels: Tensor | None = None
) -> dict[str, Tensor]:
    """Return one image's rotated detections as the dict the accumulator consumes.

    Args:
        rboxes: Detections as an ``(N, 5)`` tensor or a list of ``[cx, cy, w, h, theta]`` rows.
        scores: Confidence per detection, shape ``(N,)``.
        labels: Class per detection; every detection is class 0 when omitted.

    Returns:
        A dict with ``rboxes`` (float32), ``scores`` and ``labels`` (int64).

    Examples:
        >>> pred = rotated_prediction([[0.0, 0.0, 4.0, 2.0, 0.0]], torch.tensor([0.9]))
        >>> sorted(pred), pred["labels"].tolist(), pred["rboxes"].dtype
        (['labels', 'rboxes', 'scores'], [0], torch.float32)
    """
    boxes = torch.as_tensor(rboxes, dtype=torch.float32)
    if labels is None:
        labels = torch.zeros(boxes.shape[0], dtype=torch.long)
    return {"rboxes": boxes, "scores": scores, "labels": labels}


def rotated_target(
    rboxes: Tensor | list[list[float]], labels: Tensor | None = None, difficult: Tensor | None = None
) -> dict[str, Tensor]:
    """Return one image's rotated ground truth as the dict the accumulator consumes.

    Args:
        rboxes: Ground truths as an ``(N, 5)`` tensor or a list of ``[cx, cy, w, h, theta]`` rows.
        labels: Class per ground truth; every instance is class 0 when omitted.
        difficult: Difficult flag per ground truth; no instance is difficult when omitted.

    Returns:
        A dict with ``rboxes`` (float32), ``labels`` (int64) and ``difficult`` (bool).

    Examples:
        >>> target = rotated_target([[0.0, 0.0, 4.0, 2.0, 0.0]], difficult=torch.tensor([True]))
        >>> sorted(target), target["labels"].tolist(), target["difficult"].tolist()
        (['difficult', 'labels', 'rboxes'], [0], [True])
    """
    boxes = torch.as_tensor(rboxes, dtype=torch.float32)
    if labels is None:
        labels = torch.zeros(boxes.shape[0], dtype=torch.long)
    if difficult is None:
        difficult = torch.zeros(boxes.shape[0], dtype=torch.bool)
    return {"rboxes": boxes, "labels": labels, "difficult": difficult}


def perfect_case(count: int, match_index: int) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    """Build one image of ``count`` class-0 detections where only ``match_index`` is correct.

    Scores descend with the row index, so ``match_index`` doubles as the rank of the only detection that can become a
    true positive.

    Examples:
        >>> preds, targets = perfect_case(count=4, match_index=2)
        >>> preds[0]["rboxes"][2, :2].tolist()
        [100.0, 100.0]
        >>> targets[0]["rboxes"].tolist()
        [[100.0, 100.0, 20.0, 10.0, 0.20000000298023224]]
    """
    target = torch.tensor([[100.0, 100.0, 20.0, 10.0, 0.2]])
    boxes = torch.tensor([[1000.0 + row, 1000.0, 4.0, 2.0, 0.0] for row in range(count)])
    boxes[match_index] = target[0]
    return [rotated_prediction(boxes, torch.linspace(1.0, 0.01, count))], [rotated_target(target)]


def axis_aligned_fixture() -> tuple[
    list[dict[str, Tensor]], list[dict[str, Tensor]], list[dict[str, Tensor]], list[dict[str, Tensor]]
]:
    """Build one detection problem in both the rotated and the axis-aligned vocabularies.

    Every angle is zero, so the two descriptions denote the same rectangles. Each image detects a strict subset of its
    ground truths (leaving some unmatched), repeats its first detection (a duplicate the one-to-one rule must charge),
    and adds two detections of nothing. Class :data:`_STARVED_CLASS` is skipped entirely in every other image, so at
    least one class saturates well below full recall rather than every curve running to 1. Randomness comes from the
    autouse seed, so the fixture is fixed run to run.

    Examples:
        >>> torch.manual_seed(0)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> rotated_preds, rotated_targets, box_preds, box_targets = axis_aligned_fixture()
        >>> len(rotated_preds), len(rotated_targets), len(box_preds), len(box_targets)
        (24, 24, 24, 24)
        >>> sorted(box_targets[0].keys())
        ['boxes', 'labels']
    """
    rotated_preds, rotated_targets, box_preds, box_targets = [], [], [], []
    for image in range(_FIXTURE_IMAGES):
        count = int(torch.randint(3, 10, (1,)))
        centres = torch.rand(count, 2) * 400.0 + 50.0
        extents = torch.rand(count, 2) * 60.0 + 20.0
        labels = torch.randint(0, _FIXTURE_CLASSES, (count,))
        ground_truth = torch.cat([centres, extents, torch.zeros(count, 1)], dim=1)

        picked = torch.randperm(count)[: max(1, count - 2)]
        if image % 2 == 0:
            picked = picked[labels[picked] != _STARVED_CLASS]
        if picked.numel() == 0:
            picked = torch.randperm(count)[:1]

        jittered = centres[picked] + (torch.rand(picked.numel(), 2) - 0.5) * 24.0
        resized = extents[picked] * (0.85 + torch.rand(picked.numel(), 2) * 0.3)
        picked_labels = labels[picked].clone()

        noise_centres = torch.rand(2, 2) * 400.0 + 50.0
        noise_extents = torch.rand(2, 2) * 40.0 + 15.0
        noise_labels = torch.randint(0, _FIXTURE_CLASSES, (2,))

        all_centres = torch.cat([jittered, jittered[:1] + 1.0, noise_centres])
        all_extents = torch.cat([resized, resized[:1], noise_extents])
        predicted_labels = torch.cat([picked_labels, picked_labels[:1], noise_labels])
        scores = torch.rand(all_centres.shape[0])
        predicted = torch.cat([all_centres, all_extents, torch.zeros(all_centres.shape[0], 1)], dim=1)

        rotated_preds.append(rotated_prediction(predicted, scores, predicted_labels))
        rotated_targets.append(rotated_target(ground_truth, labels))
        box_preds.append({"boxes": to_xyxy(predicted), "scores": scores, "labels": predicted_labels})
        box_targets.append({"boxes": to_xyxy(ground_truth), "labels": labels})
    return rotated_preds, rotated_targets, box_preds, box_targets


def boundary_targets(found: int, positives: int) -> tuple[Tensor, Tensor]:
    """Return ``(ground truth, detections)`` rotated boxes for an exact-recall fixture.

    ``positives`` well-separated ground truths of one class, of which the first ``found`` are returned as exact copies,
    so every detection is a true positive and the attained recall is exactly ``found / positives``.

    Examples:
        >>> ground_truth, detections = boundary_targets(found=2, positives=3)
        >>> ground_truth.shape, detections.shape
        (torch.Size([3, 5]), torch.Size([2, 5]))
        >>> torch.equal(detections, ground_truth[:2])
        True
    """
    ground_truth = torch.tensor([[100.0 * index, 100.0, 20.0, 10.0, 0.0] for index in range(positives)])
    return ground_truth, ground_truth[:found].clone()


def boundary_case(found: int, positives: int) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    """Return rotated prediction and target dicts whose recall is exactly ``found/positives``.

    Examples:
        >>> preds, targets = boundary_case(found=2, positives=3)
        >>> preds[0]["rboxes"].shape, targets[0]["rboxes"].shape
        (torch.Size([2, 5]), torch.Size([3, 5]))
        >>> preds[0]["scores"].tolist()
        [0.8999999761581421, 0.5]
    """
    ground_truth, detections = boundary_targets(found, positives)
    return [rotated_prediction(detections, torch.linspace(0.9, 0.5, found))], [rotated_target(ground_truth)]


def boundary_case_axis_aligned(found: int, positives: int) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    """Return the same exact-recall fixture in the axis-aligned vocabulary.

    Examples:
        >>> preds, targets = boundary_case_axis_aligned(found=2, positives=3)
        >>> preds[0]["boxes"].shape, targets[0]["boxes"].shape
        (torch.Size([2, 4]), torch.Size([3, 4]))
    """
    ground_truth, detections = boundary_targets(found, positives)
    preds = [
        {
            "boxes": to_xyxy(detections),
            "scores": torch.linspace(0.9, 0.5, found),
            "labels": torch.zeros(found, dtype=torch.long),
        }
    ]
    targets = [{"boxes": to_xyxy(ground_truth), "labels": torch.zeros(positives, dtype=torch.long)}]
    return preds, targets


def to_xyxy(rboxes: Tensor) -> Tensor:
    """Return the ``xyxy`` corners of zero-angle rotated boxes.

    Examples:
        >>> to_xyxy(torch.tensor([[10.0, 10.0, 4.0, 2.0, 0.0]])).tolist()
        [[8.0, 9.0, 12.0, 11.0]]
    """
    half = rboxes[:, 2:4] / 2
    return torch.cat([rboxes[:, :2] - half, rboxes[:, :2] + half], dim=1)
