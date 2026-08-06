# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-047 mask-coefficient branch.

Covers opt-in coefficient shapes, tanh activation, anchor flattening, disabled
detection-head identity, and the separation of the two branches' coefficient
parameters. Prototype construction and mask assembly are intentionally outside
this module's scope.
"""

from __future__ import annotations

import torch

from lucid_yolo.assign.grid import make_anchor_points
from lucid_yolo.models import DualDetectionHead
from lucid_yolo.models.heads.detect import _flatten_level

_STRIDES = [8, 16, 32]
_CHANNELS = (16, 32, 64)
_PRECHANGE_PARAMETER_COUNT = 205600
_PRECHANGE_STEM_SUFFIXES = (
    "0.0.conv.conv.weight",
    "0.0.conv.bn.weight",
    "0.0.conv.bn.bias",
    "0.0.conv.bn.running_mean",
    "0.0.conv.bn.running_var",
    "0.0.conv.bn.num_batches_tracked",
    "0.1.conv.weight",
    "0.1.bn.weight",
    "0.1.bn.bias",
    "0.1.bn.running_mean",
    "0.1.bn.running_var",
    "0.1.bn.num_batches_tracked",
    "1.0.conv.conv.weight",
    "1.0.conv.bn.weight",
    "1.0.conv.bn.bias",
    "1.0.conv.bn.running_mean",
    "1.0.conv.bn.running_var",
    "1.0.conv.bn.num_batches_tracked",
    "1.1.conv.weight",
    "1.1.bn.weight",
    "1.1.bn.bias",
    "1.1.bn.running_mean",
    "1.1.bn.running_var",
    "1.1.bn.num_batches_tracked",
    "2.weight",
    "2.bias",
)


def _features(
    input_size: int,
    channels: tuple[int, int, int] = _CHANNELS,
    batch: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build stride-8/16/32 neck features for an ``input_size`` square image."""
    return tuple(
        torch.randn(batch, level_channels, input_size // stride, input_size // stride)
        for level_channels, stride in zip(channels, _STRIDES, strict=True)
    )


def _prechange_state_dict_keys() -> tuple[str, ...]:
    """Return the fixed WP-046 state-dict key sequence for the n-scale head."""
    return tuple(
        f"{branch}.{stem}.{level}.{suffix}"
        for branch in ("o2o", "o2m")
        for stem in ("box_stems", "cls_stems")
        for level in range(3)
        for suffix in _PRECHANGE_STEM_SUFFIXES
    )


def _prechange_branch_output(
    branch: torch.nn.Module,
    features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact WP-046 box/class prediction loop for one branch."""
    box_stems = branch.box_stems
    cls_stems = branch.cls_stems
    cls_levels = []
    box_levels = []
    for feature, box_stem, cls_stem in zip(features, box_stems, cls_stems, strict=True):
        box_levels.append(_flatten_level(box_stem(feature)))
        cls_levels.append(_flatten_level(cls_stem(feature)))
    return torch.cat(cls_levels, dim=1), torch.cat(box_levels, dim=1)


def test_coeff_shapes() -> None:
    """Both opt-in branches emit 32 tanh-bounded coefficients per dense anchor."""
    torch.manual_seed(0)
    head = DualDetectionHead(in_channels=_CHANNELS, num_classes=4, num_coeffs=32).eval()
    for branch in (head.o2m, head.o2o):
        assert branch.coeff_stems is not None
        for stem in branch.coeff_stems:
            output = stem[-1]
            assert isinstance(output, torch.nn.Conv2d)
            assert output.bias is not None
            output.weight.data.zero_()
            output.bias.data.fill_(2.0)

    with torch.no_grad():
        out = head(_features(128))

    assert out.o2m_coeff is not None
    assert out.o2o_coeff is not None
    for coefficients, classes in ((out.o2m_coeff, out.o2m_cls), (out.o2o_coeff, out.o2o_cls)):
        assert coefficients.shape == (2, classes.shape[1], 32)
        assert torch.all(coefficients >= -1.0)
        assert torch.all(coefficients <= 1.0)
        assert torch.all(coefficients > 0.9)


def test_coeff_anchor_order_agreement() -> None:
    """The head concatenates coefficient levels in the anchor-grid's own order.

    Drives the real head rather than the flattening helper: each level's
    coefficient stem is collapsed to a constant output (zero weights, a distinct
    bias per level), so the dense coefficient tensor must carry those constants
    over exactly the index ranges ``make_anchor_points`` assigns to each level.
    A branch that concatenated levels in another order would still produce the
    right shape and so pass every other test in this file.
    """
    torch.manual_seed(0)
    input_size = 128
    head = DualDetectionHead(in_channels=_CHANNELS, num_classes=4, num_coeffs=32).eval()
    biases = (0.25, 0.5, 0.75)
    for branch in (head.o2m, head.o2o):
        for stem, bias in zip(branch.coeff_stems, biases, strict=True):
            stem[-1].weight.data.zero_()
            stem[-1].bias.data.fill_(bias)

    with torch.no_grad():
        coefficients = head(_features(input_size)).o2o_coeff

    feature_sizes = [(input_size // stride, input_size // stride) for stride in _STRIDES]
    anchor_points, strides = make_anchor_points(feature_sizes, _STRIDES)
    assert coefficients.shape[1] == anchor_points.shape[0]

    start = 0
    for (height, width), stride, bias in zip(feature_sizes, _STRIDES, biases, strict=True):
        stop = start + height * width
        assert torch.all(strides[start:stop] == stride), f"level at stride {stride} occupies another index range"
        assert torch.allclose(coefficients[:, start:stop, :], torch.tanh(torch.tensor(bias)))
        start = stop
    assert start == coefficients.shape[1]


def test_coeffs_are_off_by_default_without_detection_head_changes() -> None:
    """The default head retains the exact WP-046 module/state/parameter contract."""
    torch.manual_seed(0)
    head = DualDetectionHead(in_channels=(64, 128, 256), num_classes=80).eval()
    features = _features(128, channels=(64, 128, 256))

    with torch.no_grad():
        expected_o2m_cls, expected_o2m_box = _prechange_branch_output(head.o2m, features)
        expected_o2o_cls, expected_o2o_box = _prechange_branch_output(head.o2o, features)
        out = head(features)

    assert tuple(head.state_dict()) == _prechange_state_dict_keys()
    assert sum(parameter.numel() for parameter in head.parameters()) == _PRECHANGE_PARAMETER_COUNT
    assert not any("coeff" in name for name, _module in head.named_modules())
    assert torch.equal(out.o2m_cls, expected_o2m_cls)
    assert torch.equal(out.o2m_box, expected_o2m_box)
    assert torch.equal(out.o2o_cls, expected_o2o_cls)
    assert torch.equal(out.o2o_box, expected_o2o_box)
    assert out.o2m_coeff is None
    assert out.o2o_coeff is None


def test_coefficient_stems_are_disjoint_between_branches() -> None:
    """One-to-one and one-to-many coefficient stems own separate parameters."""
    head = DualDetectionHead(in_channels=_CHANNELS, num_classes=4, num_coeffs=32)
    o2m_ids = {id(parameter) for parameter in head.o2m.coeff_stems.parameters()}
    o2o_ids = {id(parameter) for parameter in head.o2o.coeff_stems.parameters()}

    assert o2m_ids and o2o_ids
    assert o2m_ids.isdisjoint(o2o_ids)
