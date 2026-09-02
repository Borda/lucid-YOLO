# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-012 mixup blend and copy-paste assembly (blueprint section 5.9).

Covers mixup — that ``p=1`` blends pixels by the sampled ``lam`` (recovered from a
known pixel), that both label sets concatenate with counts preserved, ``p=0``
identity, the same-size guard, and seeded determinism — and copy-paste — that a
crafted square-polygon instance lands its pixels and its box/label/polygon on the
destination, that polygon-less source instances are skipped, ``p=0`` identity, the
rotated-box guard, and seeded determinism — plus a direct rasteriser pixel-count
case.

Keypoints (WP-132) get a class per assembly. Neither displaces geometry, so no point is
ever warped and A70 does not arise here; what is pinned instead is that points survive
both merges, that a paste selects point rows by the same index as its boxes, and that
mixed point presence is rejected rather than silently half-annotated.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from lucid_yolo.data import CopyPaste, Mixup, Targets
from lucid_yolo.data.mixup import _rasterize_polygon

#: Square image side used across the suite.
_SIDE = 8
#: Number of images each assembly consumes.
_PAIR = 2


@pytest.fixture(autouse=True)
def reset_random_seeds() -> Iterator[None]:
    """Seed every RNG source before each test so generated geometry is deterministic."""
    torch.manual_seed(0)
    yield


def _generator(seed: int = 1234) -> torch.Generator:
    """Return a CPU generator seeded to ``seed`` for reproducible sampling.

    Examples:
        >>> torch.rand(1, generator=_generator()).item()
        0.028979241847991943
    """
    return torch.Generator().manual_seed(seed)


def _square_ring(x0: float, y0: float, x1: float, y1: float) -> torch.Tensor:
    """Return the four-point ring of an axis-aligned square ``(x0, y0)-(x1, y1)``.

    Examples:
        >>> _square_ring(1.0, 1.0, 5.0, 5.0).tolist()
        [[1.0, 1.0], [5.0, 1.0], [5.0, 5.0], [1.0, 5.0]]
    """
    return torch.tensor([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def _polygon_source(label: int = 5) -> Targets:
    """Return a source instance: one square polygon with its matching box and label.

    Examples:
        >>> source = _polygon_source(label=3)
        >>> source.boxes.tolist()
        [[1.0, 1.0, 5.0, 5.0]]
        >>> source.labels.tolist()
        [3]
        >>> len(source.polygons)
        1
    """
    ring = _square_ring(1.0, 1.0, 5.0, 5.0)
    return Targets(boxes=torch.tensor([[1.0, 1.0, 5.0, 5.0]]), labels=torch.tensor([label]), polygons=[ring])


def _boxes_only(count: int) -> Targets:
    """Return ``count`` boxes with labels but no polygons (never a paste candidate).

    Examples:
        >>> targets = _boxes_only(2)
        >>> targets.boxes.tolist()
        [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]]
        >>> targets.labels.tolist()
        [0, 0]
        >>> targets.polygons
        []
    """
    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0]]).repeat(count, 1)
    return Targets(boxes=boxes, labels=torch.zeros(count, dtype=torch.int64))


def _posed_polygon_source(label: int, x0: float, points: list[list[float]], visibility: list[int]) -> Targets:
    """Return a polygon-carrying instance that also carries a K-point set.

    Examples:
        >>> _posed_polygon_source(5, 1.0, [[2.0, 2.0]], [2]).keypoints.shape
        torch.Size([1, 1, 2])
    """
    ring = _square_ring(x0, 1.0, x0 + 4.0, 5.0)
    return Targets(
        boxes=torch.tensor([[x0, 1.0, x0 + 4.0, 5.0]]),
        labels=torch.tensor([label]),
        polygons=[ring],
        keypoints=torch.tensor([points]),
        keypoint_vis=torch.tensor([visibility]),
    )


class TestMixupKeypoints:
    """Mixup moves no geometry, so both inputs' points concatenate untouched."""

    def test_both_point_sets_survive_the_blend(self) -> None:
        """Blending concatenates both images' keypoints at their original coordinates.

        Mixup blends pixels and merges label sets; it displaces nothing, so a point must
        arrive on the far side with the coordinate it went in with. Both images' scenes are
        genuinely present in the blended pixels, so both point sets remain true.
        """
        mixup = Mixup(p=1.0, generator=_generator())
        first = _posed_polygon_source(1, 1.0, [[2.0, 2.0]], [2])
        second = _posed_polygon_source(2, 1.0, [[3.0, 4.0]], [1])

        _, out = mixup([(torch.ones(3, _SIDE, _SIDE), first), (torch.zeros(3, _SIDE, _SIDE), second)])

        assert out.keypoints.tolist() == [[[2.0, 2.0]], [[3.0, 4.0]]]
        assert out.keypoint_vis.tolist() == [[2], [1]]

    def test_keypoint_free_blend_keeps_the_canonical_empty(self) -> None:
        """A detection-only blend returns the canonical empty keypoint pair.

        Mixup fires on the train path of every task at the scale-aware probability, so the
        point-free case has to stay byte-identical for the frozen goldens.
        """
        mixup = Mixup(p=1.0, generator=_generator())
        pair = [(torch.ones(3, _SIDE, _SIDE), _boxes_only(1)), (torch.zeros(3, _SIDE, _SIDE), _boxes_only(1))]

        _, out = mixup(pair)

        assert out.keypoints.shape == (0, 0, 2)
        assert out.keypoint_vis.shape == (0, 0)


class TestCopyPasteKeypoints:
    """A pasted instance brings its points across with its box, selected by the same index."""

    def test_pasted_instance_carries_its_points(self) -> None:
        """The pasted source instance's keypoints and visibility are appended to the destination.

        A paste transplants an instance between images without moving it inside one — the
        masked pixels land at the coordinates they already occupied — so the landmark
        coordinates transfer verbatim. Dropping them would leave the pasted box supervised
        for detection but blank for pose.
        """
        copy_paste = CopyPaste(p=1.0, generator=_generator())
        destination = _posed_polygon_source(9, 1.0, [[2.0, 2.0]], [2])
        source = _posed_polygon_source(5, 1.0, [[3.0, 3.0]], [1])

        _, out = copy_paste([(torch.zeros(3, _SIDE, _SIDE), destination), (torch.ones(3, _SIDE, _SIDE), source)])

        assert out.labels.tolist() == [9, 5]
        assert out.keypoints.tolist() == [[[2.0, 2.0]], [[3.0, 3.0]]]
        assert out.keypoint_vis.tolist() == [[2], [1]]

    def test_points_follow_the_selection_not_the_source_order(self) -> None:
        """Only the selected instances' point rows are appended, aligned with their boxes.

        The paste draw runs per candidate, so the pasted subset is generally not the whole
        source. Selecting points by anything other than the boxes' own index would pair a
        pasted box with a landmark set belonging to an instance that was never pasted.
        """
        copy_paste = CopyPaste(p=1.0, max_paste=1, generator=_generator())
        source = Targets(
            boxes=torch.tensor([[1.0, 1.0, 3.0, 3.0], [4.0, 4.0, 6.0, 6.0]]),
            labels=torch.tensor([5, 6]),
            polygons=[_square_ring(1.0, 1.0, 3.0, 3.0), _square_ring(4.0, 4.0, 6.0, 6.0)],
            keypoints=torch.tensor([[[2.0, 2.0]], [[5.0, 5.0]]]),
            keypoint_vis=torch.tensor([[2], [1]]),
        )
        destination = (torch.zeros(3, _SIDE, _SIDE), Targets.empty())

        _, out = copy_paste([destination, (torch.ones(3, _SIDE, _SIDE), source)])

        assert copy_paste.last_pasted == 1
        assert out.labels.tolist() == [5]
        assert out.keypoints.tolist() == [[[2.0, 2.0]]]

    def test_keypoint_free_paste_keeps_the_canonical_empty(self) -> None:
        """A segmentation paste with no points returns the canonical empty keypoint pair.

        Copy-paste needs polygon rings, which is the segmentation path — the one that
        brings no landmarks. That combination is the common case and must stay a no-op.
        """
        copy_paste = CopyPaste(p=1.0, generator=_generator())
        pair = [(torch.zeros(3, _SIDE, _SIDE), Targets.empty()), (torch.ones(3, _SIDE, _SIDE), _polygon_source())]

        _, out = copy_paste(pair)

        assert out.boxes.shape[0] == 1
        assert out.keypoints.shape == (0, 0, 2)
        assert out.keypoint_vis.shape == (0, 0)

    def test_mixed_point_presence_is_rejected(self) -> None:
        """Pasting a point-carrying instance onto point-free targets raises rather than merging.

        The merged set would claim landmarks for some instances and none for others, which
        the container cannot represent and no consumer could interpret. Rejecting mirrors
        the rule ``Targets.concat`` already applies to polygons.
        """
        copy_paste = CopyPaste(p=1.0, generator=_generator())
        destination = _polygon_source(label=9)
        source = _posed_polygon_source(5, 1.0, [[3.0, 3.0]], [1])

        with pytest.raises(ValueError, match="keypoints count"):
            copy_paste([(torch.zeros(3, _SIDE, _SIDE), destination), (torch.ones(3, _SIDE, _SIDE), source)])


class TestMixupBlend:
    """A triggered mixup convex-blends pixels and concatenates both label sets."""

    def test_blend_recovers_lam_from_known_pixel(self) -> None:
        """With a=ones, b=zeros the blended pixel equals lam, so lam is recoverable."""
        mixup = Mixup(p=1.0, generator=_generator())
        a = (torch.ones(3, _SIDE, _SIDE), Targets.empty())
        b = (torch.zeros(3, _SIDE, _SIDE), Targets.empty())

        out_image, _ = mixup([a, b])

        assert mixup.last_lam is not None
        assert torch.allclose(out_image, torch.full_like(out_image, mixup.last_lam))

    def test_targets_concatenated_with_counts_preserved(self) -> None:
        """Both images' boxes survive the blend: 2 + 3 boxes concatenate to 5."""
        mixup = Mixup(p=1.0, generator=_generator())
        a = (torch.ones(3, _SIDE, _SIDE), _boxes_only(2))
        b = (torch.zeros(3, _SIDE, _SIDE), _boxes_only(3))

        _, out_targets = mixup([a, b])

        assert out_targets.boxes.shape[0] == 5
        assert out_targets.labels.shape[0] == 5


class TestMixupPassThrough:
    """An untriggered mixup returns the first pair untouched."""

    def test_p_zero_is_identity_on_first_pair(self) -> None:
        """p=0 never blends: the first image and targets return unchanged, lam is None."""
        mixup = Mixup(p=0.0, generator=_generator())
        a_image = torch.ones(3, _SIDE, _SIDE)
        a = (a_image, _boxes_only(1))
        b = (torch.zeros(3, _SIDE, _SIDE), _boxes_only(1))

        out_image, out_targets = mixup([a, b])

        assert torch.equal(out_image, a_image)
        assert out_targets.boxes.shape[0] == 1
        assert mixup.last_lam is None


class TestMixupGuards:
    """Wrong image sizes and wrong item counts are rejected."""

    def test_mismatched_sizes_raise(self) -> None:
        """A triggered blend of differently sized images raises ValueError."""
        mixup = Mixup(p=1.0, generator=_generator())
        a = (torch.ones(3, _SIDE, _SIDE), Targets.empty())
        b = (torch.zeros(3, _SIDE, _SIDE + 1), Targets.empty())

        with pytest.raises(ValueError, match="same-size"):
            mixup([a, b])

    def test_wrong_item_count_raises(self) -> None:
        """A single pair (not two) raises ValueError naming the required count."""
        mixup = Mixup(p=1.0)

        with pytest.raises(ValueError, match="exactly 2"):
            mixup([(torch.ones(3, _SIDE, _SIDE), Targets.empty())])


class TestCopyPastePaste:
    """A pasted instance lands its pixels and its box/label/polygon on the destination."""

    def test_square_instance_pixels_and_targets_land(self) -> None:
        """A source square overwrites destination pixels and appends one instance."""
        copy_paste = CopyPaste(p=1.0, generator=_generator())
        destination = (torch.zeros(3, _SIDE, _SIDE), Targets.empty())
        source = (torch.ones(3, _SIDE, _SIDE), _polygon_source(label=5))

        out_image, out_targets = copy_paste([destination, source])

        assert copy_paste.last_pasted == 1
        assert out_targets.labels.tolist() == [5]
        assert len(out_targets.polygons) == 1
        assert torch.equal(out_targets.boxes, source[1].boxes)
        assert out_image[0, 2, 2] == 1.0  # interior of the pasted square
        assert out_image[0, 0, 0] == 0.0  # outside the mask stays destination

    def test_pasted_pixel_count_matches_rasterised_mask(self) -> None:
        """The number of overwritten pixels equals the polygon's rasterised area."""
        copy_paste = CopyPaste(p=1.0, generator=_generator())
        destination = (torch.zeros(3, _SIDE, _SIDE), Targets.empty())
        source = (torch.ones(3, _SIDE, _SIDE), _polygon_source())

        out_image, _ = copy_paste([destination, source])

        mask = _rasterize_polygon(_square_ring(1.0, 1.0, 5.0, 5.0), _SIDE, _SIDE)
        assert int((out_image[0] == 1.0).sum()) == int(mask.sum())


class TestCopyPasteSkips:
    """Instances without a polygon are never pasted, and p=0 is identity."""

    def test_non_polygon_source_instances_skipped(self) -> None:
        """A source with boxes but no polygons pastes nothing even at p=1."""
        copy_paste = CopyPaste(p=1.0, generator=_generator())
        dest_image = torch.zeros(3, _SIDE, _SIDE)
        destination = (dest_image, Targets.empty())
        source = (torch.ones(3, _SIDE, _SIDE), _boxes_only(3))

        out_image, out_targets = copy_paste([destination, source])

        assert copy_paste.last_pasted == 0
        assert out_targets.boxes.shape[0] == 0
        assert torch.equal(out_image, dest_image)

    def test_p_zero_is_identity(self) -> None:
        """p=0 pastes nothing: the destination image and targets return unchanged."""
        copy_paste = CopyPaste(p=0.0, generator=_generator())
        dest_image = torch.zeros(3, _SIDE, _SIDE)
        destination = (dest_image, Targets.empty())
        source = (torch.ones(3, _SIDE, _SIDE), _polygon_source())

        out_image, out_targets = copy_paste([destination, source])

        assert copy_paste.last_pasted == 0
        assert out_targets.boxes.shape[0] == 0
        assert torch.equal(out_image, dest_image)


class TestCopyPasteGuards:
    """Rotated boxes and mismatched sizes are rejected."""

    def test_rboxes_raise_not_implemented(self) -> None:
        """A non-empty rboxes on either input is rejected: the paste unit is a polygon mask."""
        rboxes = torch.tensor([[4.0, 4.0, 3.0, 2.0, 0.2]])
        with_rbox = Targets(boxes=torch.zeros((0, 4)), labels=torch.zeros(0, dtype=torch.int64), rboxes=rboxes)
        copy_paste = CopyPaste(p=1.0)

        with pytest.raises(NotImplementedError, match="polygon"):
            copy_paste([(torch.zeros(3, _SIDE, _SIDE), Targets.empty()), (torch.ones(3, _SIDE, _SIDE), with_rbox)])

    def test_mismatched_sizes_raise(self) -> None:
        """Differently sized destination and source images raise ValueError."""
        copy_paste = CopyPaste(p=1.0)
        destination = (torch.zeros(3, _SIDE, _SIDE), Targets.empty())
        source = (torch.ones(3, _SIDE, _SIDE + 1), _polygon_source())

        with pytest.raises(ValueError, match="same-size"):
            copy_paste([destination, source])


class TestRasterizer:
    """The polygon rasteriser fills the expected pixel block for an axis-aligned square."""

    def test_square_fills_expected_pixel_count(self) -> None:
        """A 4x4 square rasterises to a half-open 4x4 pixel block (16 pixels)."""
        mask = _rasterize_polygon(_square_ring(2.0, 2.0, 6.0, 6.0), 10, 10)

        assert mask.dtype == torch.bool
        assert mask.shape == (10, 10)
        assert int(mask.sum()) == 16
