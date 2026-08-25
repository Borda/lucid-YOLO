# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-122 keypoint stems and coordinate decode.

The keypoint path is a generic ``K``-point extension of the dual detection head,
not a human-pose specialization. These tests pin the extension at its two
boundaries: the opt-in head emits point-major raw coordinate offsets and raw
per-axis uncertainty on both disjoint branches, while the pure decode composes
only the coordinates with the existing anchor-centre and stride convention.

Assumption A65 deliberately leaves uncertainty unbounded until WP-123 defines
the RLE loss. The regression test therefore drives the last convolution to
distinct negative and positive sigma values and requires those values to arrive
unchanged; a sigmoid, tanh, softplus, absolute value, exponential, or clamp would
all fail that check.

The composite gates at the end (WP-139) cover the same extension one level up:
that :class:`~lucid_yolo.models.KeypointDetector` runs the stems it builds, that
its deployed view is a parameter-sharing single-branch view rather than a copy,
that the triple it returns feeds the decode unadapted, and that ``K`` cannot be
defaulted into existence. The params/FLOPs golden is a separate gate and stays
outside this module.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from lucid_yolo.assign.grid import make_anchor_points
from lucid_yolo.models import KeypointDetector, build_keypoint_detector, count_params
from lucid_yolo.models.heads import decode_keypoints
from lucid_yolo.models.heads.detect import DualDetectionHead, _flatten_level

_CHANNELS = (16, 24, 32)
_NUM_CLASSES = 4
_NUM_KEYPOINTS = 3

#: The head's level strides, in the (8, 16, 32) order the neck emits.
_STRIDES = [8, 16, 32]

#: Square input side for the composite gates; divisible by 32, so every level's
#: grid is exact. Its 336 anchors differ from both the class and the point count,
#: so no shape assertion can pass by matching the wrong axis.
_IMG_SIZE = 128


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Seed torch so module initialization and input features are deterministic."""
    torch.manual_seed(0)


def _anchor_grid(input_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the shared anchor points and per-anchor strides for a square input.

    Examples:
        >>> points, strides = _anchor_grid(128)
        >>> points.shape, strides.shape
        (torch.Size([336, 2]), torch.Size([336]))
    """
    feature_sizes = [(input_size // stride, input_size // stride) for stride in _STRIDES]
    return make_anchor_points(feature_sizes, _STRIDES)


@pytest.fixture
def features() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return small neck-style feature maps with 21 anchors in total."""
    return (
        torch.randn(2, _CHANNELS[0], 4, 4),
        torch.randn(2, _CHANNELS[1], 2, 2),
        torch.randn(2, _CHANNELS[2], 1, 1),
    )


class TestKeypointHead:
    """Tests for the opt-in keypoint stem set on both detection branches."""

    def test_emits_coordinate_and_sigma_tensors_per_keypoint(
        self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """Both branches emit coordinate and sigma tensors shaped ``(B, A, K, 2)``.

        A three-level head with 21 anchors and generic ``K=3`` points exercises
        the channel-to-point reshape and the level concatenation on both paths.
        """
        head = DualDetectionHead(_CHANNELS, num_classes=_NUM_CLASSES, num_keypoints=_NUM_KEYPOINTS).eval()

        with torch.no_grad():
            output = head(features)

        expected_shape = (2, 21, _NUM_KEYPOINTS, 2)
        assert output.o2o_keypoints is not None
        assert output.o2m_keypoints is not None
        assert output.o2o_keypoint_sigma is not None
        assert output.o2m_keypoint_sigma is not None
        assert output.o2o_keypoints.shape == expected_shape
        assert output.o2m_keypoints.shape == expected_shape
        assert output.o2o_keypoint_sigma.shape == expected_shape
        assert output.o2m_keypoint_sigma.shape == expected_shape

    def test_is_absent_by_default_without_detection_output_changes(
        self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """The default head owns no keypoint modules and preserves class/box outputs.

        The scenario recomputes the accepted box/class loop directly before
        invoking the extended forward, so a default-off keypoint path cannot
        perturb the existing tensors while merely returning ``None`` extras.
        """
        head = DualDetectionHead(_CHANNELS, num_classes=_NUM_CLASSES).eval()
        expected: list[tuple[torch.Tensor, torch.Tensor]] = []
        with torch.no_grad():
            for branch in (head.o2m, head.o2o):
                cls_levels: list[torch.Tensor] = []
                box_levels: list[torch.Tensor] = []
                for feature, box_stem, cls_stem in zip(features, branch.box_stems, branch.cls_stems, strict=True):
                    box_levels.append(_flatten_level(box_stem(feature)))
                    cls_levels.append(_flatten_level(cls_stem(feature)))
                expected.append((torch.cat(cls_levels, dim=1), torch.cat(box_levels, dim=1)))
            output = head(features)

        assert head.o2o.keypoint_stems is None
        assert head.o2m.keypoint_stems is None
        assert not any("keypoint" in name for name, _module in head.named_modules())
        assert output.o2o_keypoints is None
        assert output.o2m_keypoints is None
        assert output.o2o_keypoint_sigma is None
        assert output.o2m_keypoint_sigma is None
        assert torch.equal(output.o2m_cls, expected[0][0])
        assert torch.equal(output.o2m_box, expected[0][1])
        assert torch.equal(output.o2o_cls, expected[1][0])
        assert torch.equal(output.o2o_box, expected[1][1])

    def test_stems_are_disjoint_from_both_branches_and_other_outputs(self) -> None:
        """Keypoint parameters share with neither branch nor any existing stem set.

        Enabling coefficients, angles, and keypoints together makes every
        possible accidental sharing surface present in one module tree.
        """
        head = DualDetectionHead(
            _CHANNELS,
            num_classes=_NUM_CLASSES,
            num_coeffs=8,
            predict_angle=True,
            num_keypoints=_NUM_KEYPOINTS,
        )
        assert head.o2o.keypoint_stems is not None
        assert head.o2m.keypoint_stems is not None

        o2o_keypoint_ids = {id(parameter) for parameter in head.o2o.keypoint_stems.parameters()}
        o2m_keypoint_ids = {id(parameter) for parameter in head.o2m.keypoint_stems.parameters()}
        existing_ids: set[int] = set()
        for branch in (head.o2o, head.o2m):
            for stems in (branch.box_stems, branch.cls_stems, branch.coeff_stems, branch.angle_stems):
                assert stems is not None
                existing_ids.update(id(parameter) for parameter in stems.parameters())

        assert o2o_keypoint_ids and o2m_keypoint_ids
        assert o2o_keypoint_ids.isdisjoint(o2m_keypoint_ids)
        assert o2o_keypoint_ids.isdisjoint(existing_ids)
        assert o2m_keypoint_ids.isdisjoint(existing_ids)

    def test_sigma_output_remains_raw_unbounded_and_point_major(
        self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """Distinct negative and positive sigma logits arrive unchanged from the 1x1.

        Each point's final-convolution channels are programmed as
        ``(x, y, sigma_x, sigma_y)``. Exact recovery catches both a forbidden
        range transform and a reshape that silently uses another channel layout.
        """
        num_keypoints = 2
        head = DualDetectionHead(_CHANNELS, num_classes=_NUM_CLASSES, num_keypoints=num_keypoints).eval()
        assert head.o2o.keypoint_stems is not None
        expected_sigma = torch.tensor([[-100.0, -50.0], [50.0, 100.0]])
        with torch.no_grad():
            for stem in head.o2o.keypoint_stems:
                output_conv = stem[-1]
                assert isinstance(output_conv, nn.Conv2d)
                assert output_conv.bias is not None
                output_conv.weight.zero_()
                output_conv.bias.zero_()
                output_conv.bias.view(num_keypoints, 4)[:, 2:].copy_(expected_sigma)
            output = head(features)

        assert output.o2o_keypoints is not None
        assert output.o2o_keypoint_sigma is not None
        assert torch.equal(output.o2o_keypoints, torch.zeros_like(output.o2o_keypoints))
        assert torch.equal(
            output.o2o_keypoint_sigma,
            expected_sigma.view(1, 1, num_keypoints, 2).expand_as(output.o2o_keypoint_sigma),
        )


class TestDecodeKeypoints:
    """Tests for composing raw coordinate offsets with anchors and strides."""

    def test_matches_a_hand_computable_anchor_offset(self) -> None:
        """The decode adds each stride-scaled offset to its anchor centre exactly.

        One anchor at ``(10, 20)`` with stride 4 and two signed point offsets
        makes every expected pixel coordinate explicit.
        """
        raw_coords = torch.tensor([[[[1.0, -2.0], [-0.5, 0.25]]]])
        anchor_points = torch.tensor([[10.0, 20.0]])
        strides = torch.tensor([4.0])

        decoded = decode_keypoints(raw_coords, anchor_points, strides)

        expected = torch.tensor([[[[14.0, 12.0], [8.0, 21.0]]]])
        assert torch.equal(decoded, expected)

    def test_preserves_batch_anchor_and_generic_point_axes(self) -> None:
        """A multi-batch, multi-anchor, ``K=4`` input keeps every structural axis.

        Distinct anchor centres and strides are broadcast over only the batch and
        point axes, preventing one anchor's geometry from leaking into another.
        """
        raw_coords = torch.arange(2 * 3 * 4 * 2, dtype=torch.float32).reshape(2, 3, 4, 2) / 10
        anchor_points = torch.tensor([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]])
        strides = torch.tensor([2.0, 4.0, 8.0])

        decoded = decode_keypoints(raw_coords, anchor_points, strides)

        expected = anchor_points.view(1, 3, 1, 2) + raw_coords * strides.view(1, 3, 1, 1)
        assert decoded.shape == raw_coords.shape
        assert torch.equal(decoded, expected)


class TestKeypointDetector:
    """Tests for the WP-139 composite keypoint model and its deploy-time view."""

    def test_forward_emits_points_and_sigma_on_both_branches(self) -> None:
        """The composite wires the point branch through to both branches' outputs.

        Catches a model that builds the stems but never runs them — the parameters
        would be exported and counted while the points stayed unpredicted. The
        class count, the point count, and the anchor count are mutually distinct,
        so no shape assertion can pass by matching the wrong axis.
        """
        model = KeypointDetector("n", num_classes=_NUM_CLASSES, num_keypoints=_NUM_KEYPOINTS).eval()
        anchor_points, _ = _anchor_grid(_IMG_SIZE)

        with torch.no_grad():
            out = model(torch.zeros(1, 3, _IMG_SIZE, _IMG_SIZE))

        num_anchors = anchor_points.shape[0]
        assert out.o2o_cls.shape == (1, num_anchors, _NUM_CLASSES)
        assert out.o2o_keypoints is not None
        assert out.o2m_keypoints is not None
        assert out.o2o_keypoints.shape == (1, num_anchors, _NUM_KEYPOINTS, 2)
        assert out.o2m_keypoints.shape == (1, num_anchors, _NUM_KEYPOINTS, 2)
        assert out.o2o_keypoint_sigma is not None
        assert out.o2o_keypoint_sigma.shape == (1, num_anchors, _NUM_KEYPOINTS, 2)

    def test_deployed_view_drops_the_one_to_many_branch_entirely(self) -> None:
        """The deployed model holds no one-to-many parameter, point stems included.

        The R6 convention the keypoint golden's GFLOP column is read under. A
        deployed view that still referenced the training-only branch would export
        it, count it, and make the NMS-free claim false for the artifact that
        ships — and the point stems are the easiest thing to leave behind, since
        they hang off the branch rather than the head. Parameter identity proves
        the view shares the trained weights instead of copying them.
        """
        model = KeypointDetector("n", num_classes=_NUM_CLASSES, num_keypoints=_NUM_KEYPOINTS).eval()

        deployed = model.deploy()

        o2m_ids = {id(parameter) for parameter in model.head.o2m.parameters()}
        deployed_ids = {id(parameter) for parameter in deployed.parameters()}
        assert o2m_ids.isdisjoint(deployed_ids)
        assert count_params(deployed) <= count_params(model)
        assert count_params(deployed) == count_params(model) - count_params(model.head.o2m)
        assert model.head.o2o is deployed.o2o
        assert id(next(deployed.backbone.parameters())) == id(next(model.backbone.parameters()))

    def test_deployed_triple_decodes_into_dense_point_coordinates(self) -> None:
        """The deployed triple flows through the point decode without adaptation.

        The end-to-end proof that the head's outputs and the decode's inputs agree
        on layout: a transposed point axis or a mismatched anchor ordering would
        raise here rather than surfacing as quietly displaced points during a later
        OKS evaluation run.
        """
        model = build_keypoint_detector("n", num_classes=_NUM_CLASSES, num_keypoints=_NUM_KEYPOINTS).eval()
        deployed = model.deploy()
        anchor_points, strides = _anchor_grid(_IMG_SIZE)

        with torch.no_grad():
            cls, box, keypoints = deployed(torch.zeros(1, 3, _IMG_SIZE, _IMG_SIZE))
            decoded = decode_keypoints(keypoints, anchor_points, strides)

        num_anchors = anchor_points.shape[0]
        assert cls.shape == (1, num_anchors, _NUM_CLASSES)
        assert box.shape == (1, num_anchors, 4)
        assert decoded.shape == (1, num_anchors, _NUM_KEYPOINTS, 2)
        assert torch.isfinite(decoded).all()

    @pytest.mark.parametrize(
        "factory",
        [
            pytest.param(KeypointDetector, id="class"),
            pytest.param(build_keypoint_detector, id="builder"),
        ],
    )
    def test_rejects_an_omitted_point_count(self, factory: Callable[..., KeypointDetector]) -> None:
        """Building without ``num_keypoints`` raises instead of picking a count.

        ``K`` is a property of the dataset's annotation schema, so a default would
        be a silent claim about data the model has never seen — the same refusal
        :class:`~lucid_yolo.ptl.module.DetectionLitModule` makes for
        ``task="keypoints"``. Keyword-only and required is what turns that refusal
        into a construction-time error rather than a wrongly shaped checkpoint.
        """
        with pytest.raises(TypeError):
            factory("n", num_classes=_NUM_CLASSES)
