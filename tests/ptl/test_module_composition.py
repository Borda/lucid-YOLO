# SPDX-License-Identifier: Apache-2.0
"""Anti-drift gate for the two model composition sites (WP-087).

The detector is assembled in two places: :class:`~lucid_yolo.models.build.Detector`,
which the R1 Table 7 / Table S9 fidelity gates measure, and
:class:`~lucid_yolo.ptl.module.DetectionLitModule`, which training actually runs.
WP-087 routed both through :func:`~lucid_yolo.models.build.build_detection_stages`
so they cannot diverge; these tests are what keeps that true, because a silent
divergence would leave the fidelity gate certifying a model nobody trains while
every published number still looked right.

Two invariants are pinned here:

*Nothing moved.* ``prechange_detect_state_dict_keys.txt`` holds the 714 keys a
default detection module emitted **before** the refactor, captured from the
pre-change source. It is a frozen snapshot rather than a live comparison against
the previous git revision on purpose: once this work lands, ``HEAD`` holds the new
file and such a comparison would compare the new code against itself. The keys
matter because the accepted Det-smoke checkpoint (run v8) is keyed on them — holding a
:class:`~lucid_yolo.models.build.Detector` inside the module instead of the three
flat stages would prefix every key with ``model.`` and invalidate it.

*Both sites agree.* Per-stage parameter counts are compared between the module and
the composite models across all five scales, for the detection stages and for the
segmentation branches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lucid_yolo.models.build import Detector, Segmenter, count_params
from lucid_yolo.models.registry import VARIANTS
from lucid_yolo.ptl.module import DetectionLitModule

#: Frozen pre-WP-087 state-dict key sequence of a default detection module.
_PRECHANGE_KEYS_FILE = Path(__file__).parent / "prechange_detect_state_dict_keys.txt"

#: Pre-WP-087 parameter count of the same module.
_PRECHANGE_PARAMETER_COUNT = 2_414_600

#: Multipliers the pre-change snapshot was taken at (the module's own doctest scale).
_SNAPSHOT_DEPTH = 0.34
_SNAPSHOT_WIDTH = 0.25
_SNAPSHOT_MAX_CHANNELS = 1024
_SNAPSHOT_NUM_CLASSES = 4

#: Class count used for the cross-site comparisons; matches the composite models' default.
_NUM_CLASSES = 80

#: Attributes only a segmentation module carries.
_SEGMENTATION_ATTRIBUTES = ("proto_fusion", "protonet", "semantic")

_VARIANT_PARAMS = [pytest.param(name, id=name) for name in VARIANTS]


def _snapshot_module() -> DetectionLitModule:
    """Build a detection module at the exact scale the pre-change snapshot was taken at."""
    return DetectionLitModule(
        depth=_SNAPSHOT_DEPTH,
        width=_SNAPSHOT_WIDTH,
        max_channels=_SNAPSHOT_MAX_CHANNELS,
        num_classes=_SNAPSHOT_NUM_CLASSES,
    )


def _module_for(variant: str, task: str = "detect") -> DetectionLitModule:
    """Build a Lightning module from a named variant's raw multipliers."""
    spec = VARIANTS[variant]
    return DetectionLitModule(
        depth=spec.depth,
        width=spec.width,
        max_channels=spec.max_channels,
        num_classes=_NUM_CLASSES,
        task=task,
    )


def test_detection_state_dict_keys_are_unmoved() -> None:
    """The detection module's state-dict keys are byte-identical to the pre-WP-087 snapshot.

    Catches the checkpoint-invalidating refactor: nesting the three stages under a
    :class:`~lucid_yolo.models.build.Detector` attribute, reordering the
    ``backbone``/``neck``/``head`` assignments, or letting the segmentation
    branches be constructed for the detection task would each rewrite or reorder
    these keys and silently break loading of the accepted Det-smoke checkpoint.
    """
    expected = tuple(_PRECHANGE_KEYS_FILE.read_text().split())

    keys = tuple(_snapshot_module().state_dict())

    assert keys == expected
    assert len(keys) == 714


def test_detection_parameter_count_is_unchanged() -> None:
    """The detection module's parameter count is exactly the pre-WP-087 total.

    Key equality alone would not catch a stage rebuilt at different multipliers or
    a head given coefficient stems, since both keep the key names identical while
    changing tensor shapes.
    """
    assert count_params(_snapshot_module()) == _PRECHANGE_PARAMETER_COUNT


@pytest.mark.parametrize("variant", _VARIANT_PARAMS)
def test_module_detection_stages_match_the_detector(variant: str) -> None:
    """At every scale the module's backbone/neck/head match the Detector's, stage by stage.

    This is the anti-drift gate the R1 Table 7 fidelity tests rely on: those
    measure :class:`~lucid_yolo.models.build.Detector` only, so if the Lightning
    module's stack were built differently — a stale multiplier, a different
    channel cap — the published parameter and FLOP numbers would describe a model
    that was never trained, with nothing in the numbers looking wrong.

    Compared per stage rather than in total so a compensating error (one stage
    larger, another smaller) cannot cancel out.
    """
    module = _module_for(variant)
    detector = Detector(variant, num_classes=_NUM_CLASSES)

    assert count_params(module.backbone) == count_params(detector.backbone)
    assert count_params(module.neck) == count_params(detector.neck)
    assert count_params(module.head) == count_params(detector.head)


@pytest.mark.parametrize("variant", _VARIANT_PARAMS)
def test_segment_module_branches_match_the_segmenter(variant: str) -> None:
    """A segment module's three mask branches match the Segmenter's, at every scale.

    The mask-side half of the same anti-drift gate. The segmentation fidelity work
    measures :class:`~lucid_yolo.models.build.Segmenter`; a prototype stack built
    from a different fused width, or an auxiliary branch given the wrong class
    count, would leave training and the measured model disagreeing. The segmenter's
    total minus its detection stages is used as the expectation so the comparison
    also proves nothing else was added to it.
    """
    module = _module_for(variant, task="segment")
    segmenter = Segmenter(variant, num_classes=_NUM_CLASSES)

    for attribute in _SEGMENTATION_ATTRIBUTES:
        assert hasattr(module, attribute)
    module_branches = sum(count_params(getattr(module, name)) for name in _SEGMENTATION_ATTRIBUTES)
    segmenter_branches = count_params(segmenter) - sum(
        count_params(stage) for stage in (segmenter.backbone, segmenter.neck, segmenter.head)
    )
    assert module_branches == segmenter_branches


def test_detect_module_has_no_segmentation_branches() -> None:
    """A detection module does not merely zero the mask branches — it never holds them.

    Constructing them and leaving them unsupervised would add parameters to the
    optimizer and keys to the checkpoint for a task that has no use for them.
    Absence is asserted by attribute lookup rather than by a flag, so a
    ``proto_fusion = None`` placeholder fails here too.
    """
    module = _snapshot_module()

    assert not any(hasattr(module, attribute) for attribute in _SEGMENTATION_ATTRIBUTES)
