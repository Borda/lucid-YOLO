# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-012 mixup blend and copy-paste assembly (blueprint section 5.9).

Covers mixup — that ``p=1`` blends pixels by the sampled ``lam`` (recovered from a
known pixel), that both label sets concatenate with counts preserved, ``p=0``
identity, the same-size guard, and seeded determinism — and copy-paste — that a
crafted square-polygon instance lands its pixels and its box/label/polygon on the
destination, that polygon-less source instances are skipped, ``p=0`` identity, the
rotated-box guard, and seeded determinism — plus a direct rasteriser pixel-count
case.
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


class TestMixupDeterminism:
    """Equal-seed generators produce byte-identical blends."""

    def test_same_seed_identical_outputs(self) -> None:
        """Two identically seeded mixups give equal images, boxes and lam."""
        a = (torch.rand(3, _SIDE, _SIDE), _boxes_only(1))
        b = (torch.rand(3, _SIDE, _SIDE), _boxes_only(1))
        mixup_a = Mixup(p=0.5, generator=_generator(99))
        mixup_b = Mixup(p=0.5, generator=_generator(99))

        image_a, targets_a = mixup_a([a, b])
        image_b, targets_b = mixup_b([a, b])

        assert torch.equal(image_a, image_b)
        assert torch.equal(targets_a.boxes, targets_b.boxes)
        assert mixup_a.last_lam == mixup_b.last_lam


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


class TestCopyPasteDeterminism:
    """Equal-seed generators produce byte-identical paste decisions and pixels."""

    def test_same_seed_identical_outputs(self) -> None:
        """Two identically seeded copy-pastes give equal images, boxes and paste counts."""
        ring_a = _square_ring(1.0, 1.0, 3.0, 3.0)
        ring_b = _square_ring(4.0, 4.0, 6.0, 6.0)
        source = Targets(
            boxes=torch.tensor([[1.0, 1.0, 3.0, 3.0], [4.0, 4.0, 6.0, 6.0]]),
            labels=torch.tensor([1, 2]),
            polygons=[ring_a, ring_b],
        )
        destination = (torch.zeros(3, _SIDE, _SIDE), Targets.empty())
        items = [destination, (torch.ones(3, _SIDE, _SIDE), source)]
        copy_paste_a = CopyPaste(p=0.5, generator=_generator(7))
        copy_paste_b = CopyPaste(p=0.5, generator=_generator(7))

        image_a, targets_a = copy_paste_a(items)
        image_b, targets_b = copy_paste_b(items)

        assert torch.equal(image_a, image_b)
        assert torch.equal(targets_a.boxes, targets_b.boxes)
        assert copy_paste_a.last_pasted == copy_paste_b.last_pasted


class TestRasterizer:
    """The polygon rasteriser fills the expected pixel block for an axis-aligned square."""

    def test_square_fills_expected_pixel_count(self) -> None:
        """A 4x4 square rasterises to a half-open 4x4 pixel block (16 pixels)."""
        mask = _rasterize_polygon(_square_ring(2.0, 2.0, 6.0, 6.0), 10, 10)

        assert mask.dtype == torch.bool
        assert mask.shape == (10, 10)
        assert int(mask.sum()) == 16
