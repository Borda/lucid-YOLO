# SPDX-License-Identifier: Apache-2.0
"""Golden-metric producers for the WP-005 golden harness.

A *producer* is a zero-argument function returning ``dict[str, float]`` whose
result the golden harness (``scripts/check_goldens.py``) recomputes and compares
against a frozen ``goldens/*.json`` file. Every metric here is deterministic so
that an unchanged codebase reproduces byte-identical values on every run.

Two real producers live here. :func:`fixture_checksums` derives its metrics from
the seeded WP-007 synthetic fixtures. Because those fixtures live under
``tests/fixtures/`` (not an importable package), they are loaded by file path via
``importlib.util`` — the same trick ``tests/meta/test_license_audit.py`` uses.
:func:`optim_toy` (WP-033) runs a fully seeded toy training task and reports how
many optimization steps :class:`~lit_yolo.optim.MuSGD` and momentum-SGD each need
to reach a fixed loss threshold — a directional convergence claim mirroring R1
Table 4 at toy scale.

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

from lit_yolo.optim import MuSGD

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
        >>> metrics["detseg_annotation_sha"] == fixture_checksums()["detseg_annotation_sha"]
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
    weights exercise the Muon branch of :class:`~lit_yolo.optim.MuSGD`.

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
        >>> from lit_yolo.optim import MuSGD
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
    :class:`~lit_yolo.optim.MuSGD`, one with :class:`torch.optim.SGD` at the same
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
