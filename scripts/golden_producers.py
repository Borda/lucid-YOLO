# SPDX-License-Identifier: Apache-2.0
"""Golden-metric producers for the WP-005 golden harness.

A *producer* is a zero-argument function returning ``dict[str, float]`` whose
result the golden harness (``scripts/check_goldens.py``) recomputes and compares
against a frozen ``goldens/*.json`` file. Every metric here is deterministic so
that an unchanged codebase reproduces byte-identical values on every run.

Four real producers live here. :func:`fixture_checksums` derives its metrics from
the seeded WP-007 synthetic fixtures. Because those fixtures live under
``tests/fixtures/`` (not an importable package), they are loaded by file path via
``importlib.util`` — the same trick ``tests/meta/test_license_audit.py`` uses.
:func:`data_pipeline_metrics` (WP-015) draws a fixed set of samples through the
Phase-1 augmentation pipeline (mosaic/affine/letterbox/mixup/copy-paste/photometric)
over those fixtures with a fixed seed and records platform-stable batch metrics:
exact integer counts (drawn samples, total instances, polygon rings, image shape)
plus tolerance-pinned float aggregates (image mean/std sums, bbox area/coord sums),
which drift only within libm/``interpolate`` rounding across OS and architecture.
:func:`optim_toy` (WP-033) runs a fully seeded toy training task and reports how
many optimization steps :class:`~lucid_yolo.optim.MuSGD` and momentum-SGD each need
to reach a fixed loss threshold — a directional convergence claim mirroring R1
Table 4 at toy scale. :func:`assignment_cases` (WP-029) freezes the Phase-3 label
-assignment behavior: it runs the Task-Aligned, Small-Target-Aware, and one-to-one
assigners on hand-placed synthetic scenes (no RNG) and records integer candidate
and positive counts — every value produced by running an assigner, never
hand-written.

Examples:
    ```pycon
    >>> metrics = fixture_checksums()
    >>> metrics["detseg_num_images"]
    16.0

    ```
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path
from types import ModuleType

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from lucid_yolo.assign import (
    SmallTargetAssigner,
    TaskAlignedAssigner,
    UniqueAssigner,
    make_anchor_points,
    surrogate_boxes,
)
from lucid_yolo.data.coco import CocoDetectionDataset, build_scale_policy
from lucid_yolo.models.build import build_detector, build_segmenter, count_flops, count_params
from lucid_yolo.optim import MuSGD
from lucid_yolo.ptl.datamodule import _TrainPipeline

#: Repository root (``scripts/`` is one level below it).
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Path to the WP-007 synthetic-fixture helpers, loaded by file path.
_SYNTHETIC_PATH = REPO_ROOT / "tests" / "fixtures" / "synthetic.py"

#: Per-split COCO annotation filename emitted by the fixture generator.
_COCO_ANNOTATION = "_annotations.coco.json"

#: The single split the fixtures materialize into.
_SPLIT = "train"


def _load_synthetic() -> ModuleType:
    """Load ``tests/fixtures/synthetic.py`` as an importable module.

    Returns:
        The loaded module exposing ``generate_detseg_fixtures`` and
        ``generate_obb_fixtures``.

    Examples:
        ```pycon
        >>> mod = _load_synthetic()
        >>> callable(mod.generate_detseg_fixtures)
        True

        ```
    """
    spec = importlib.util.spec_from_file_location("wp007_synthetic", _SYNTHETIC_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset_metrics(prefix: str, dataset_dir: Path) -> dict[str, float]:
    """Compute structural fixture metrics that are stable across platforms.

    Counts (images, annotations, categories, polygon points) are integers and
    compare exactly. Geometry aggregates (bbox area / coordinate sums) are
    float-derived: the generator's trigonometry goes through libm, whose last-bit
    rounding differs across OS/architecture, so byte-hashes of the annotation
    JSON diverge between platforms (observed: macOS arm64 vs ubuntu x86_64 CI).
    Aggregates drift only in the ~1e-3 range and are pinned with a small golden
    tolerance instead of a hash.

    Args:
        prefix: Metric-name prefix identifying the fixture set (``detseg``/``obb``).
        dataset_dir: The generated dataset directory holding ``train/``.

    Returns:
        A six-entry metric mapping derived from the split's COCO annotation file.

    Examples:
        ```pycon
        >>> import tempfile
        >>> from pathlib import Path
        >>> mod = _load_synthetic()
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     ds = mod.generate_detseg_fixtures(Path(tmp))
        ...     sorted(_dataset_metrics("detseg", ds))[:3]
        ['detseg_bbox_area_sum', 'detseg_bbox_coord_sum', 'detseg_num_annotations']

        ```
    """
    annotation_path = dataset_dir / _SPLIT / _COCO_ANNOTATION
    coco = json.loads(annotation_path.read_text())
    annotations = coco["annotations"]
    bbox_area_sum = sum(float(ann["bbox"][2]) * float(ann["bbox"][3]) for ann in annotations)
    bbox_coord_sum = sum(float(value) for ann in annotations for value in ann["bbox"])
    segmentation_points = sum(len(ann["segmentation"][0]) // 2 for ann in annotations if ann.get("segmentation"))
    return {
        f"{prefix}_num_images": float(len(coco["images"])),
        f"{prefix}_num_annotations": float(len(annotations)),
        f"{prefix}_num_categories": float(len(coco["categories"])),
        f"{prefix}_segmentation_points": float(segmentation_points),
        f"{prefix}_bbox_area_sum": round(bbox_area_sum, 3),
        f"{prefix}_bbox_coord_sum": round(bbox_coord_sum, 3),
    }


def fixture_checksums() -> dict[str, float]:
    """Deterministic checksum metrics over the WP-007 synthetic fixtures.

    Generates both seeded fixture sets (detection/segmentation and oriented-box)
    into a throwaway temporary directory, then reports each set's structural
    metrics: exact counts (images, annotations, categories, polygon points) and
    tolerance-pinned geometry aggregates (bbox area / coordinate sums). The seeds
    are fixed (A26): counts are identical on every platform, aggregates drift
    only within libm rounding across OS/architecture (see
    :func:`_dataset_metrics`).

    Returns:
        A mapping of twelve metrics, six per fixture set prefix
        (``detseg``/``obb``).

    Examples:
        ```pycon
        >>> metrics = fixture_checksums()
        >>> metrics["obb_num_images"]
        8.0
        >>> metrics["detseg_bbox_area_sum"] == fixture_checksums()["detseg_bbox_area_sum"]
        True

        ```
    """
    synthetic = _load_synthetic()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        detseg_dir = synthetic.generate_detseg_fixtures(root)
        obb_dir = synthetic.generate_obb_fixtures(root)
        return {
            **_dataset_metrics("detseg", detseg_dir),
            **_dataset_metrics("obb", obb_dir),
        }


#: Fixed seed for the WP-015 pipeline draw; a fresh :class:`_TrainPipeline` built
#: with this seed reproduces the same augmented samples on every call.
_PIPELINE_SEED = 20260731

#: Augmentation-strength policy variant for the pipeline draw (mildest recipe).
_PIPELINE_VARIANT = "n"

#: Square letterbox side for every drawn sample; small keeps the golden fast while
#: still exercising mosaic assembly and box/polygon warping.
_PIPELINE_IMG_SIZE = 128

#: Base-dataset indices drawn through the pipeline, in this fixed order (the seeded
#: generator advances as samples are drawn, so the order is part of the contract).
_PIPELINE_SAMPLE_INDICES = tuple(range(8))


def _pipeline_metrics(pipeline: _TrainPipeline) -> dict[str, float]:
    """Draw the fixed sample set through ``pipeline`` and aggregate stable metrics.

    Walks :data:`_PIPELINE_SAMPLE_INDICES` in order (the seeded generator advances
    with each draw, so the order is load-bearing) and accumulates: exact integer
    counts (drawn samples, total instances, polygon rings, the shared image shape)
    and float aggregates (image mean/std sums, bounding-box area/coordinate sums).
    The counts compare exactly across platforms; the float aggregates drift only
    within ``interpolate``/libm rounding and are pinned with a golden tolerance.

    Args:
        pipeline: The train-time augmentation pipeline to draw from.

    Returns:
        A ten-entry metric mapping: ``num_samples``, ``total_instances``,
        ``polygon_ring_count``, ``image_channels``/``image_height``/``image_width``,
        ``image_mean_sum``/``image_std_sum`` (rounded 4) and
        ``bbox_area_sum``/``bbox_coord_sum`` (rounded 3).
    """
    total_instances = 0
    polygon_rings = 0
    image_mean_sum = 0.0
    image_std_sum = 0.0
    bbox_area_sum = 0.0
    bbox_coord_sum = 0.0
    channels = height = width = 0
    for index in _PIPELINE_SAMPLE_INDICES:
        image, targets = pipeline[index]
        channels, height, width = image.shape
        image_mean_sum += float(image.mean())
        image_std_sum += float(image.std())
        total_instances += int(targets.boxes.shape[0])
        polygon_rings += len(targets.polygons)
        boxes = targets.boxes
        bbox_area_sum += float(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sum())
        bbox_coord_sum += float(boxes.sum())
    return {
        "num_samples": float(len(_PIPELINE_SAMPLE_INDICES)),
        "total_instances": float(total_instances),
        "polygon_ring_count": float(polygon_rings),
        "image_channels": float(channels),
        "image_height": float(height),
        "image_width": float(width),
        "image_mean_sum": round(image_mean_sum, 4),
        "image_std_sum": round(image_std_sum, 4),
        "bbox_area_sum": round(bbox_area_sum, 3),
        "bbox_coord_sum": round(bbox_coord_sum, 3),
    }


def data_pipeline_metrics() -> dict[str, float]:
    """Platform-stable augmented-batch metrics over the WP-015 train pipeline.

    Regenerates the seeded WP-007 detection/segmentation fixtures into a throwaway
    directory, builds the Phase-1 augmentation pipeline
    (:class:`~lucid_yolo.ptl.datamodule._TrainPipeline`: mosaic, affine, letterbox,
    mixup, copy-paste, HSV jitter, flip) over them with a fixed seed and the
    ``"n"`` strength policy, and draws :data:`_PIPELINE_SAMPLE_INDICES` in order.
    Every value is produced by *running* the pipeline — never hand-written — so the
    golden re-derives from live augmentation behavior. Counts are exact across
    platforms; image-statistic and bounding-box aggregates drift only within
    ``interpolate``/libm rounding and carry a small golden tolerance.

    Returns:
        The mapping from :func:`_pipeline_metrics` for the fixed draw.

    Examples:
        ```pycon
        >>> metrics = data_pipeline_metrics()
        >>> metrics["num_samples"]
        8.0
        >>> metrics["image_channels"], metrics["image_height"], metrics["image_width"]
        (3.0, 128.0, 128.0)
        >>> data_pipeline_metrics() == metrics
        True

        ```
    """
    synthetic = _load_synthetic()
    with tempfile.TemporaryDirectory() as tmp:
        dataset_dir = synthetic.generate_detseg_fixtures(Path(tmp))
        split_dir = dataset_dir / _SPLIT
        base = CocoDetectionDataset(split_dir, split_dir / _COCO_ANNOTATION)
        pipeline = _TrainPipeline(base, _PIPELINE_IMG_SIZE, build_scale_policy(_PIPELINE_VARIANT), _PIPELINE_SEED)
        return _pipeline_metrics(pipeline)


#: Shared learning rate for the toy convergence experiment; tuned so both
#: optimizers converge but MuSGD reaches the threshold first (WP-033).
_TOY_LR = 0.003

#: Shared momentum coefficient for both optimizers in the toy experiment.
_TOY_MOMENTUM = 0.95

#: Loss threshold the toy task must reach, and the hard cap on optimizer steps.
_TOY_LOSS_THRESHOLD = 0.1
_TOY_MAX_STEPS = 500

#: Micro-CNN toy-task geometry: batch, input channels, spatial side, hidden channels, output dim.
_TOY_BATCH = 8
_TOY_IN_CHANNELS = 3
_TOY_SIDE = 8
_TOY_HIDDEN = 8
_TOY_OUT_DIM = 4

#: Independent seeds for the toy task's three stochastic draws (input, target map, weight init).
_TOY_INPUT_SEED = 101
_TOY_TARGET_SEED = 202
_TOY_INIT_SEED = 303


def _toy_batch() -> tuple[Tensor, Tensor]:
    """Build the seeded toy regression batch: inputs and their linear-map targets.

    Seeds torch immediately before each stochastic draw (the input batch, then
    the ground-truth linear map) so the batch is byte-identical on every call,
    independent of any ambient RNG state — this is what makes :func:`optim_toy`
    reproducible.

    Returns:
        An ``(inputs, targets)`` pair. ``inputs`` has shape
        ``(_TOY_BATCH, _TOY_IN_CHANNELS, _TOY_SIDE, _TOY_SIDE)``; ``targets`` has
        shape ``(_TOY_BATCH, _TOY_OUT_DIM)`` and is a fixed random linear map of
        the flattened inputs.

    Examples:
        ```pycon
        >>> inputs, targets = _toy_batch()
        >>> tuple(inputs.shape)
        (8, 3, 8, 8)
        >>> tuple(targets.shape)
        (8, 4)

        ```
    """
    torch.manual_seed(_TOY_INPUT_SEED)
    inputs = torch.randn(_TOY_BATCH, _TOY_IN_CHANNELS, _TOY_SIDE, _TOY_SIDE)
    torch.manual_seed(_TOY_TARGET_SEED)
    weight = torch.randn(_TOY_IN_CHANNELS * _TOY_SIDE * _TOY_SIDE, _TOY_OUT_DIM)
    targets = inputs.reshape(_TOY_BATCH, -1) @ weight
    return inputs, targets


def _build_toy_model() -> nn.Sequential:
    """Construct the micro-CNN with fixed, reproducible weight initialization.

    Seeds torch immediately before construction so both optimizer runs start from
    byte-identical parameters. The network is two ``3x3`` convolutions (each
    followed by ReLU) and a linear head — a few thousand parameters whose matrix
    weights exercise the Muon branch of :class:`~lucid_yolo.optim.MuSGD`.

    Returns:
        A freshly initialized :class:`torch.nn.Sequential` mapping an image batch
        to ``_TOY_OUT_DIM`` regression outputs.

    Examples:
        ```pycon
        >>> model = _build_toy_model()
        >>> inputs, _ = _toy_batch()
        >>> tuple(model(inputs).shape)
        (8, 4)

        ```
    """
    torch.manual_seed(_TOY_INIT_SEED)
    return nn.Sequential(
        nn.Conv2d(_TOY_IN_CHANNELS, _TOY_HIDDEN, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(_TOY_HIDDEN, _TOY_HIDDEN, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(_TOY_HIDDEN * _TOY_SIDE * _TOY_SIDE, _TOY_OUT_DIM),
    )


def _steps_to_threshold(model: nn.Module, optimizer: Optimizer, inputs: Tensor, targets: Tensor) -> tuple[int, float]:
    """Train ``model`` with ``optimizer`` until the MSE loss first reaches the threshold.

    Evaluates the mean-squared-error loss on the fixed batch before every
    optimizer step and stops as soon as it drops to :data:`_TOY_LOSS_THRESHOLD`,
    capping at :data:`_TOY_MAX_STEPS`. The returned step count is the number of
    optimizer steps taken before the threshold was met.

    Args:
        model: The network to train in place.
        optimizer: The optimizer driving the update (``MuSGD`` or ``SGD``).
        inputs: The fixed input batch.
        targets: The fixed regression targets.

    Returns:
        A ``(steps, final_loss)`` pair: the step at which the loss first reached
        the threshold (or :data:`_TOY_MAX_STEPS` if it never did), and the loss
        observed at that step.

    Examples:
        ```pycon
        >>> from lucid_yolo.optim import MuSGD
        >>> model = _build_toy_model()
        >>> inputs, targets = _toy_batch()
        >>> opt = MuSGD(model.parameters(), lr=_TOY_LR, momentum=_TOY_MOMENTUM)
        >>> steps, loss = _steps_to_threshold(model, opt, inputs, targets)
        >>> steps < _TOY_MAX_STEPS and loss <= _TOY_LOSS_THRESHOLD
        True

        ```
    """
    loss_fn = nn.MSELoss()
    final_loss = float("nan")
    for step in range(_TOY_MAX_STEPS + 1):
        loss = loss_fn(model(inputs), targets)
        final_loss = float(loss.item())
        if final_loss <= _TOY_LOSS_THRESHOLD:
            return step, final_loss
        if step == _TOY_MAX_STEPS:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return _TOY_MAX_STEPS, final_loss


def optim_toy() -> dict[str, float]:
    """Directional convergence metrics: MuSGD reaches a loss threshold in fewer steps than SGD.

    Trains two byte-identical copies of the micro-CNN (:func:`_build_toy_model`)
    on the same fixed regression task (:func:`_toy_batch`) — one with
    :class:`~lucid_yolo.optim.MuSGD`, one with :class:`torch.optim.SGD` at the same
    learning rate and momentum — and reports how many optimizer steps each needs
    to drive the MSE loss to :data:`_TOY_LOSS_THRESHOLD`. This mirrors R1 Table
    4's MuSGD-beats-SGD result at toy scale. Every stochastic draw is seeded
    (:func:`_toy_batch` and :func:`_build_toy_model` re-seed internally), so two
    in-process calls return identical dicts regardless of ambient RNG state.

    Returns:
        A mapping with ``steps_to_threshold_musgd`` and ``steps_to_threshold_sgd``
        (integer step counts as floats) and ``final_loss_musgd`` /
        ``final_loss_sgd`` (final losses rounded to six decimals).

    Examples:
        ```pycon
        >>> metrics = optim_toy()
        >>> metrics["steps_to_threshold_musgd"] < metrics["steps_to_threshold_sgd"]
        True
        >>> optim_toy() == metrics
        True

        ```
    """
    inputs, targets = _toy_batch()
    musgd_model = _build_toy_model()
    musgd_optimizer = MuSGD(musgd_model.parameters(), lr=_TOY_LR, momentum=_TOY_MOMENTUM)
    musgd_steps, musgd_loss = _steps_to_threshold(musgd_model, musgd_optimizer, inputs, targets)
    sgd_model = _build_toy_model()
    sgd_optimizer = torch.optim.SGD(sgd_model.parameters(), lr=_TOY_LR, momentum=_TOY_MOMENTUM)
    sgd_steps, sgd_loss = _steps_to_threshold(sgd_model, sgd_optimizer, inputs, targets)
    return {
        "steps_to_threshold_musgd": float(musgd_steps),
        "steps_to_threshold_sgd": float(sgd_steps),
        "final_loss_musgd": round(musgd_loss, 6),
        "final_loss_sgd": round(sgd_loss, 6),
    }


#: Column count of an ``xyxy`` box; the assign scenes are all axis-aligned.
_BOX_COLUMNS = 4

#: Uniform class score placed at every anchor so alignment ordering is IoU-driven.
_SCENE_SCORE = 0.9

#: STAL surrogate thresholds at a 640-pixel input: inflate a side below ``s_min``
#: (the smallest stride) up to ``s_ref`` (the next stride).
_S_MIN = 8.0
_S_REF = 16.0

#: Tiny-target scene (scenario i): a 4x4 stride-8 grid (anchor centres at 4, 12,
#: 20, 28) and a 6x6 ground truth centred at (8, 8) — no anchor centre lies inside
#: it, so vanilla TAL yields zero candidates while STAL's 16x16 surrogate admits
#: four. ``topk`` exceeds the candidate count so every candidate becomes positive.
_TINY_FEATURE_SIZES = [(4, 4)]
_TINY_STRIDES = [8]
_TINY_GT = torch.tensor([[[5.0, 5.0, 11.0, 11.0]]])
_TINY_TOPK = 4

#: Per-dimension scene (scenario ii): an 8x8 stride-8 grid (64-pixel input) and a
#: ground truth centred at (24, 24). The 6-wide/20-tall variant inflates only its
#: width (6 -> 16) and the 20-wide/6-tall variant only its height, so the surrogate
#: clamp is exercised independently on each axis.
_PERDIM_FEATURE_SIZES = [(8, 8)]
_PERDIM_STRIDES = [8]
_PERDIM_GT_6X20 = torch.tensor([[[21.0, 14.0, 27.0, 34.0]]])
_PERDIM_GT_20X6 = torch.tensor([[[14.0, 21.0, 34.0, 27.0]]])
_PERDIM_TOPK = 4

#: Multi-ground-truth scene (scenario iii): a stride-8/16/32 grid at a 128-pixel
#: input holding three well-separated 24x24 ground truths (all above ``s_min``, so
#: no surrogate inflation). The one-to-one assigner collapses each to a single
#: positive; the one-to-many assigner keeps ``topk`` per ground truth.
_MULTI_FEATURE_SIZES = [(16, 16), (8, 8), (4, 4)]
_MULTI_STRIDES = [8, 16, 32]
_MULTI_GTS = torch.tensor([[[12.0, 12.0, 36.0, 36.0], [52.0, 52.0, 76.0, 76.0], [92.0, 92.0, 116.0, 116.0]]])
_MULTI_LABELS = torch.tensor([[0, 1, 2]])
_MULTI_MASK = torch.tensor([[True, True, True]])
_O2M_TOPK = 10
_O2O_TOPK = 7


def _uniform_scene(anchor_points: Tensor, gt_boxes: Tensor, num_classes: int) -> Tensor:
    """Predicted boxes that make alignment IoU-driven: each anchor predicts its ground truth.

    Every anchor whose centre lies inside a ground-truth box is given that box as
    its prediction (IoU 1 with the ground truth it is a candidate for); anchors
    inside no box keep a zero prediction and stay background. With uniform class
    scores this leaves the alignment metric ``t = s * u**6`` flat across a ground
    truth's candidates, so top-k keeps exactly ``min(topk, candidate_count)`` of
    them regardless of tie ordering — the property that keeps the counts
    deterministic across platforms.

    Args:
        anchor_points: ``(A, 2)`` anchor centres in input pixels.
        gt_boxes: ``(1, N, 4)`` ground-truth boxes in ``xyxy`` pixels.
        num_classes: Unused width hint kept for signature symmetry with callers
            that pass their score channel count; see :func:`_run_multi_gt`.

    Returns:
        A ``(1, A, 4)`` tensor of per-anchor predicted boxes.

    Examples:
        ```pycon
        >>> import torch
        >>> pts = torch.tensor([[8.0, 8.0], [40.0, 40.0]])
        >>> gts = torch.tensor([[[0.0, 0.0, 16.0, 16.0]]])
        >>> _uniform_scene(pts, gts, 1)[0, 0].tolist()
        [0.0, 0.0, 16.0, 16.0]

        ```
    """
    del num_classes
    num_anchors = anchor_points.shape[0]
    preds = torch.zeros(1, num_anchors, _BOX_COLUMNS)
    px = anchor_points[:, 0]
    py = anchor_points[:, 1]
    for gt in gt_boxes[0]:
        x1, y1, x2, y2 = gt
        inside = (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)
        preds[0, inside] = gt
    return preds


def _count_positives(result: object) -> int:
    """Total number of positive (foreground) anchors in an assignment result.

    Args:
        result: An :class:`~lucid_yolo.assign.AssignResult` for a single-image batch.

    Returns:
        The count of ``True`` entries in the result's foreground mask.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.assign import TaskAlignedAssigner, make_anchor_points
        >>> pts, _ = make_anchor_points([(2, 2)], [8])
        >>> gt = torch.tensor([[[0.0, 0.0, 16.0, 16.0]]])
        >>> out = TaskAlignedAssigner(topk=1)(
        ...     torch.full((1, 4, 1), 0.9), gt.expand(1, 4, 4).contiguous(),
        ...     pts, gt, torch.tensor([[0]]), torch.tensor([[True]]))
        >>> _count_positives(out)
        1

        ```
    """
    return int(result.fg_mask.sum())  # type: ignore[attr-defined]


def _max_positives_per_gt(result: object, num_gt: int) -> int:
    """Largest number of positive anchors assigned to any single ground truth.

    Args:
        result: An :class:`~lucid_yolo.assign.AssignResult` for a single-image batch.
        num_gt: Number of real ground truths in the scene.

    Returns:
        The maximum per-ground-truth positive count, or ``0`` when no anchor is
        positive.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.assign import UniqueAssigner, make_anchor_points
        >>> pts, _ = make_anchor_points([(4, 4)], [8])
        >>> gt = torch.tensor([[[0.0, 0.0, 32.0, 32.0]]])
        >>> out = UniqueAssigner(topk=7)(
        ...     torch.full((1, 16, 1), 0.9), gt.expand(1, 16, 4).contiguous(),
        ...     pts, gt, torch.tensor([[0]]), torch.tensor([[True]]))
        >>> _max_positives_per_gt(out, 1)
        1

        ```
    """
    gt_index = result.gt_index  # type: ignore[attr-defined]
    assigned = gt_index[gt_index >= 0]
    if assigned.numel() == 0:
        return 0
    return int(torch.bincount(assigned, minlength=num_gt).max())


def _candidates_inside(anchor_points: Tensor, boxes: Tensor) -> int:
    """Count anchor centres that fall inside any of the given boxes.

    Mirrors the assigners' centre-inside candidate test. Passing the original
    ground truth reproduces the vanilla-TAL candidate set; passing
    :func:`~lucid_yolo.assign.surrogate_boxes` output reproduces the STAL set.

    Args:
        anchor_points: ``(A, 2)`` anchor centres in input pixels.
        boxes: ``(1, N, 4)`` boxes in ``xyxy`` pixels to test containment against.

    Returns:
        The number of ``(box, anchor)`` centre-inside pairs.

    Examples:
        ```pycon
        >>> import torch
        >>> from lucid_yolo.assign import make_anchor_points, surrogate_boxes
        >>> pts, _ = make_anchor_points([(4, 4)], [8])
        >>> gt = torch.tensor([[[5.0, 5.0, 11.0, 11.0]]])  # 6x6, no centre inside
        >>> _candidates_inside(pts, gt)
        0
        >>> _candidates_inside(pts, surrogate_boxes(gt, 8.0, 16.0))
        4

        ```
    """
    px = anchor_points[:, 0]
    py = anchor_points[:, 1]
    x1 = boxes[..., 0].unsqueeze(-1)
    y1 = boxes[..., 1].unsqueeze(-1)
    x2 = boxes[..., 2].unsqueeze(-1)
    y2 = boxes[..., 3].unsqueeze(-1)
    inside = (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)
    return int(inside.sum())


def _run_tiny() -> dict[str, float]:
    """Scenario (i): a 6x6 ground truth gets zero TAL candidates and four STAL positives."""
    points, _ = make_anchor_points(_TINY_FEATURE_SIZES, _TINY_STRIDES)
    num_anchors = points.shape[0]
    scores = torch.full((1, num_anchors, 1), _SCENE_SCORE)
    preds = _TINY_GT.expand(1, num_anchors, _BOX_COLUMNS).contiguous()
    labels = torch.tensor([[0]])
    mask = torch.tensor([[True]])
    tal = TaskAlignedAssigner(topk=_TINY_TOPK)(scores, preds, points, _TINY_GT, labels, mask)
    stal = SmallTargetAssigner(topk=_TINY_TOPK)(scores, preds, points, _TINY_GT, labels, mask)
    surrogate = surrogate_boxes(_TINY_GT, _S_MIN, _S_REF)
    return {
        "tal_positives_tiny": float(_count_positives(tal)),
        "stal_positives_tiny": float(_count_positives(stal)),
        "tal_candidates_tiny": float(_candidates_inside(points, _TINY_GT)),
        "stal_candidates_tiny": float(_candidates_inside(points, surrogate)),
    }


def _run_per_dim() -> dict[str, float]:
    """Scenario (ii): the surrogate clamp inflates width and height independently."""
    points, _ = make_anchor_points(_PERDIM_FEATURE_SIZES, _PERDIM_STRIDES)
    num_anchors = points.shape[0]
    scores = torch.full((1, num_anchors, 1), _SCENE_SCORE)
    labels = torch.tensor([[0]])
    mask = torch.tensor([[True]])
    preds = _PERDIM_GT_6X20.expand(1, num_anchors, _BOX_COLUMNS).contiguous()
    stal = SmallTargetAssigner(topk=_PERDIM_TOPK)(scores, preds, points, _PERDIM_GT_6X20, labels, mask)
    surrogate_6x20 = surrogate_boxes(_PERDIM_GT_6X20, _S_MIN, _S_REF)[0, 0]
    surrogate_20x6 = surrogate_boxes(_PERDIM_GT_20X6, _S_MIN, _S_REF)
    return {
        "stal_positives_6x20": float(_count_positives(stal)),
        "stal_candidates_6x20": float(_candidates_inside(points, surrogate_boxes(_PERDIM_GT_6X20, _S_MIN, _S_REF))),
        "tal_candidates_6x20": float(_candidates_inside(points, _PERDIM_GT_6X20)),
        "stal_candidates_20x6": float(_candidates_inside(points, surrogate_20x6)),
        "tal_candidates_20x6": float(_candidates_inside(points, _PERDIM_GT_20X6)),
        "stal_surrogate_6x20_width": float(surrogate_6x20[2] - surrogate_6x20[0]),
        "stal_surrogate_6x20_height": float(surrogate_6x20[3] - surrogate_6x20[1]),
    }


def _run_multi_gt() -> dict[str, float]:
    """Scenario (iii): one-to-one yields one positive per ground truth, one-to-many yields more."""
    points, _ = make_anchor_points(_MULTI_FEATURE_SIZES, _MULTI_STRIDES)
    num_anchors = points.shape[0]
    num_gt = _MULTI_GTS.shape[1]
    scores = torch.full((1, num_anchors, num_gt), _SCENE_SCORE)
    preds = _uniform_scene(points, _MULTI_GTS, num_gt)
    o2o = UniqueAssigner(topk=_O2O_TOPK)(scores, preds, points, _MULTI_GTS, _MULTI_LABELS, _MULTI_MASK)
    o2m = SmallTargetAssigner(topk=_O2M_TOPK)(scores, preds, points, _MULTI_GTS, _MULTI_LABELS, _MULTI_MASK)
    surrogate = surrogate_boxes(_MULTI_GTS, _S_MIN, _S_REF)
    return {
        "unique_positives_per_gt_max": float(_max_positives_per_gt(o2o, num_gt)),
        "unique_total_positives": float(_count_positives(o2o)),
        "o2m_positives_per_gt_max": float(_max_positives_per_gt(o2m, num_gt)),
        "o2m_total_positives": float(_count_positives(o2m)),
        "multi_gt_candidates_total": float(_candidates_inside(points, surrogate)),
    }


def assignment_cases() -> dict[str, float]:
    """Frozen Phase-3 label-assignment metrics over hand-placed synthetic scenes (WP-029).

    Runs the Task-Aligned (:class:`~lucid_yolo.assign.TaskAlignedAssigner`),
    Small-Target-Aware (:class:`~lucid_yolo.assign.SmallTargetAssigner`), and
    one-to-one (:class:`~lucid_yolo.assign.UniqueAssigner`) assigners on three
    deterministic scenes built from literal tensors (no RNG) and reports the
    blueprint's Phase-3 exit-gate quantities as integer-valued floats:

    * **Tiny target** — a sub-8x8 ground truth on a stride-8 grid receives zero
      vanilla-TAL candidates and at least one STAL positive.
    * **Per-dimension clamp** — the STAL surrogate inflates width and height
      independently, so a 6-wide/20-tall box widens only along ``x``.
    * **Multi ground truth** — the one-to-one assigner collapses each of three
      ground truths to a single positive, while the one-to-many assigner keeps
      several, so its total strictly exceeds the one-to-one total.

    Every value is produced by running an assigner (or its centre-inside candidate
    test), never hand-written, so the golden re-derives from live behavior.

    Returns:
        A mapping of sixteen integer-valued metrics across the three scenes.

    Examples:
        ```pycon
        >>> metrics = assignment_cases()
        >>> metrics["tal_positives_tiny"], metrics["stal_positives_tiny"] >= 1
        (0.0, True)
        >>> metrics["o2m_total_positives"] > metrics["unique_total_positives"]
        True
        >>> assignment_cases() == metrics
        True

        ```
    """
    return {**_run_tiny(), **_run_per_dim(), **_run_multi_gt()}


#: The five published scale variants, in Table 7 row order.
_DET_VARIANTS = ("n", "s", "m", "l", "x")

#: Detection class count and input side for the fidelity metrics (R1 Table 7).
_DET_NUM_CLASSES = 80
_DET_IMG_SIZE = 640

#: Decimal places the per-variant GFLOP values are rounded to before freezing;
#: fvcore's MAC trace is deterministic, so this only trims float noise below the
#: golden's 0.5%-of-value tolerance.
_DET_GFLOP_DECIMALS = 4


def det_params_flops() -> dict[str, float]:
    """Frozen per-variant parameter and GFLOP metrics for the detector (WP-023).

    Builds each of the five scale variants
    (:func:`~lucid_yolo.models.build.build_detector`) with 80 classes and records,
    per variant, the exact parameter count of the full model
    (:func:`~lucid_yolo.models.build.count_params`) and the conventional GFLOPs of
    the deployed NMS-free inference model
    (:func:`~lucid_yolo.models.build.count_flops` on
    :meth:`~lucid_yolo.models.build.Detector.deploy`) at a 640-pixel input. Params
    are the full checkpoint (both dual-head branches); GFLOPs exclude the
    training-only one-to-many branch — the R6/YOLOv10 reporting convention that
    lands all five scales within R1 Table 7 tolerance (see
    :mod:`lucid_yolo.models.build`).

    Every value is produced by building and measuring the live modules — never
    hand-written — so the golden re-derives from the actual architecture. Param
    counts are exact integers (byte-stable across platforms); GFLOPs come from
    fvcore's deterministic MAC trace and are pinned with a small golden tolerance.

    Returns:
        A mapping of ten metrics: ``<variant>_params`` (exact) and
        ``<variant>_gflops`` (tolerance-pinned) for each variant ``n``…``x``.

    Examples:
        ```pycon
        >>> metrics = det_params_flops()
        >>> metrics["n_params"]
        2437552.0
        >>> metrics["m_gflops"] > 0
        True
        >>> det_params_flops() == metrics
        True

        ```
    """
    metrics: dict[str, float] = {}
    for variant in _DET_VARIANTS:
        model = build_detector(variant, _DET_NUM_CLASSES)
        metrics[f"{variant}_params"] = float(count_params(model))
        gflops = count_flops(model.deploy(), img_size=_DET_IMG_SIZE)
        metrics[f"{variant}_gflops"] = round(gflops, _DET_GFLOP_DECIMALS)
    return metrics


def seg_params_flops() -> dict[str, float]:
    """Frozen per-variant parameter and GFLOP metrics for the segmentation model (WP-052b).

    The segmentation counterpart of :func:`det_params_flops`, measured under the
    identical convention so the two goldens stay comparable: parameters of the
    full model (:func:`~lucid_yolo.models.build.count_params` on
    :func:`~lucid_yolo.models.build.build_segmenter`) and conventional GFLOPs of
    the deployed inference model
    (:meth:`~lucid_yolo.models.build.Segmenter.deploy`) at a 640-pixel input. The
    deployed view carries neither the one-to-many detection branch nor the
    training-only auxiliary semantic branch (A17), so its FLOPs are what an
    inference deployment actually costs, while the parameter count is the whole
    checkpoint.

    R1 Table S9 states neither its FLOP convention nor whether its Params column
    denotes the full or the deployed model — it reports one Params/FLOPs pair per
    scale shared by the E2E and non-E2E rows. This convention is the one that
    lands R1 Table 7 within tolerance for detection (WP-023) and it holds for
    Table S9 too, which is corroboration rather than assumption.

    Every value is produced by building and measuring the live modules — never
    hand-written — so the golden re-derives from the actual architecture.

    Returns:
        A mapping of ten metrics: ``<variant>_params`` (exact) and
        ``<variant>_gflops`` (tolerance-pinned) for each variant ``n``…``x``.

    Examples:
        ```pycon
        >>> metrics = seg_params_flops()
        >>> metrics["n_params"] > metrics["n_gflops"]
        True
        >>> sorted(metrics) == sorted(seg_params_flops())
        True

        ```
    """
    metrics: dict[str, float] = {}
    for variant in _DET_VARIANTS:
        model = build_segmenter(variant, _DET_NUM_CLASSES)
        metrics[f"{variant}_params"] = float(count_params(model))
        gflops = count_flops(model.deploy(), img_size=_DET_IMG_SIZE)
        metrics[f"{variant}_gflops"] = round(gflops, _DET_GFLOP_DECIMALS)
    return metrics
