# SPDX-License-Identifier: Apache-2.0
"""Unit gate for the WP-012 mixup blend and copy-paste assembly (blueprint section 5.9).

Covers mixup — that ``p=1`` blends pixels by the sampled ``lam`` (recovered from a known pixel), that both label sets
concatenate with counts preserved, ``p=0`` identity, the same-size guard, and seeded determinism — and copy-paste —
that a crafted square-polygon instance lands its pixels and its box/label/polygon on the destination, that
polygon-less source instances are skipped, ``p=0`` identity, the rotated-box guard, and seeded determinism — plus a
direct rasteriser pixel-count case.

Keypoints (WP-132) get a class per assembly. Neither displaces geometry, so no point is ever warped and A70 does not
arise here; what is pinned instead is that points survive both merges, that a paste selects point rows by the same
index as its boxes, and that mixed point presence is rejected rather than silently half-annotated.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import pytest
import torch
from tensors import seeded_generator

from lucid_yolo.data import CopyPaste, Mixup, Targets
from lucid_yolo.data.mixup import CopyPasteParams, _rasterize_polygon

#: Square image side used across the suite.
_SIDE = 8
#: Number of images each assembly consumes.
_PAIR = 2


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


def _rotated_only() -> Targets:
    """Return targets holding one rotated box and no axis-aligned boxes (never accepted by copy-paste).

    Examples:
        >>> _rotated_only().rboxes.tolist()
        [[4.0, 4.0, 3.0, 2.0, 0.20000000298023224]]
    """
    rboxes = torch.tensor([[4.0, 4.0, 3.0, 2.0, 0.2]])
    return Targets(boxes=torch.zeros((0, 4)), labels=torch.zeros(0, dtype=torch.int64), rboxes=rboxes)


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


def _difficult_items() -> list[tuple[torch.Tensor, Targets]]:
    """Return a difficult one-instance destination and a two-instance source (easy, then difficult), both with rings.

    Examples:
        >>> items = _difficult_items()
        >>> [item[1].difficult.tolist() for item in items]
        [[True], [False, True]]
    """
    destination = Targets(
        boxes=torch.tensor([[1.0, 1.0, 3.0, 3.0]]),
        labels=torch.tensor([9]),
        polygons=[_square_ring(1.0, 1.0, 3.0, 3.0)],
        difficult=torch.tensor([True]),
    )
    source = Targets(
        boxes=torch.tensor([[4.0, 4.0, 6.0, 6.0], [1.0, 4.0, 3.0, 6.0]]),
        labels=torch.tensor([5, 6]),
        polygons=[_square_ring(4.0, 4.0, 6.0, 6.0), _square_ring(1.0, 4.0, 3.0, 6.0)],
        difficult=torch.tensor([False, True]),
    )
    return [(torch.zeros(3, _SIDE, _SIDE), destination), (torch.ones(3, _SIDE, _SIDE), source)]


class TestMixupKeypoints:
    """Mixup moves no geometry, so both inputs' points concatenate untouched."""

    def test_both_point_sets_survive_blend(self) -> None:
        """Blending concatenates both images' keypoints at their original coordinates.

        Mixup blends pixels and merges label sets; it displaces nothing, so a point must arrive on the far side with
        the coordinate it went in with. Both images' scenes are genuinely present in the blended pixels, so both point
        sets remain true.
        """
        mixup = Mixup(p=1.0, generator=seeded_generator())
        first = _posed_polygon_source(1, 1.0, [[2.0, 2.0]], [2])
        second = _posed_polygon_source(2, 1.0, [[3.0, 4.0]], [1])

        _, out = mixup([(torch.ones(3, _SIDE, _SIDE), first), (torch.zeros(3, _SIDE, _SIDE), second)])

        assert out.keypoints.tolist() == [[[2.0, 2.0]], [[3.0, 4.0]]]
        assert out.keypoint_vis.tolist() == [[2], [1]]

    def test_keypoint_free_blend_keeps_canonical_empty(self) -> None:
        """A detection-only blend returns the canonical empty keypoint pair.

        Mixup fires on the train path of every task at the scale-aware probability, so the point-free case has to
        stay byte-identical for the frozen goldens.
        """
        mixup = Mixup(p=1.0, generator=seeded_generator())
        pair = [(torch.ones(3, _SIDE, _SIDE), _boxes_only(1)), (torch.zeros(3, _SIDE, _SIDE), _boxes_only(1))]

        _, out = mixup(pair)

        assert out.keypoints.shape == (0, 0, 2)
        assert out.keypoint_vis.shape == (0, 0)


class TestCopyPasteKeypoints:
    """A pasted instance brings its points across with its box, selected by the same index."""

    def test_pasted_instance_carries_points(self) -> None:
        """The pasted source instance's keypoints and visibility are appended to the destination.

        A paste transplants an instance between images without moving it inside one — the masked pixels land at the
        coordinates they already occupied — so the landmark coordinates transfer verbatim. Dropping them would leave
        the pasted box supervised for detection but blank for pose.
        """
        copy_paste = CopyPaste(p=1.0, generator=seeded_generator())
        destination = _posed_polygon_source(9, 1.0, [[2.0, 2.0]], [2])
        source = _posed_polygon_source(5, 1.0, [[3.0, 3.0]], [1])

        _, out = copy_paste([(torch.zeros(3, _SIDE, _SIDE), destination), (torch.ones(3, _SIDE, _SIDE), source)])

        assert out.labels.tolist() == [9, 5]
        assert out.keypoints.tolist() == [[[2.0, 2.0]], [[3.0, 3.0]]]
        assert out.keypoint_vis.tolist() == [[2], [1]]

    def test_points_follow_selection_not_source_order(self) -> None:
        """Only the selected instances' point rows are appended, aligned with their boxes.

        The paste draw runs per candidate, so the pasted subset is generally not the whole source. Selecting points by
        anything other than the boxes' own index would pair a pasted box with a landmark set belonging to an instance
        that was never pasted.
        """
        copy_paste = CopyPaste(p=1.0, max_paste=1, generator=seeded_generator())
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

    def test_keypoint_free_paste_keeps_canonical_empty(self) -> None:
        """A segmentation paste with no points returns the canonical empty keypoint pair.

        Copy-paste needs polygon rings, which is the segmentation path — the one that brings no landmarks. That
        combination is the common case and must stay a no-op.
        """
        copy_paste = CopyPaste(p=1.0, generator=seeded_generator())
        pair = [(torch.zeros(3, _SIDE, _SIDE), Targets.empty()), (torch.ones(3, _SIDE, _SIDE), _polygon_source())]

        _, out = copy_paste(pair)

        assert out.boxes.shape[0] == 1
        assert out.keypoints.shape == (0, 0, 2)
        assert out.keypoint_vis.shape == (0, 0)

    def test_mixed_point_presence_rejected(self) -> None:
        """Pasting a point-carrying instance onto point-free targets raises rather than merging.

        The merged set would claim landmarks for some instances and none for others, which the container cannot
        represent and no consumer could interpret. Rejecting mirrors the rule ``Targets.concat`` already applies to
        polygons.
        """
        copy_paste = CopyPaste(p=1.0, generator=seeded_generator())
        destination = _polygon_source(label=9)
        source = _posed_polygon_source(5, 1.0, [[3.0, 3.0]], [1])

        with pytest.raises(ValueError, match="keypoints count"):
            copy_paste([(torch.zeros(3, _SIDE, _SIDE), destination), (torch.ones(3, _SIDE, _SIDE), source)])


class TestMixupBlend:
    """A triggered mixup convex-blends pixels and concatenates both label sets."""

    def test_blend_recovers_lam_from_known_pixel(self) -> None:
        """With a=ones, b=zeros the blended pixel equals lam, so lam is recoverable.

        An all-ones image blended with an all-zeros one makes every output pixel equal the blend weight, which must
        match the weight the transform reports.
        """
        mixup = Mixup(p=1.0, generator=seeded_generator())
        a = (torch.ones(3, _SIDE, _SIDE), Targets.empty())
        b = (torch.zeros(3, _SIDE, _SIDE), Targets.empty())

        out_image, _ = mixup([a, b])

        assert mixup.last_lam is not None
        assert torch.allclose(out_image, torch.full_like(out_image, mixup.last_lam))

    def test_targets_concatenated_with_counts_preserved(self) -> None:
        """Both images' boxes survive the blend: 2 + 3 boxes concatenate to 5."""
        mixup = Mixup(p=1.0, generator=seeded_generator())
        a = (torch.ones(3, _SIDE, _SIDE), _boxes_only(2))
        b = (torch.zeros(3, _SIDE, _SIDE), _boxes_only(3))

        _, out_targets = mixup([a, b])

        assert out_targets.boxes.shape[0] == 5
        assert out_targets.labels.shape[0] == 5


class TestMixupPassThrough:
    """An untriggered mixup returns the first pair untouched."""

    def test_p_zero_identity_on_first_pair(self) -> None:
        """p=0 never blends: the first image and targets return unchanged, lam is None.

        With the trigger probability at zero the second pair must be ignored entirely, not blended at weight zero.
        """
        mixup = Mixup(p=0.0, generator=seeded_generator())
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
        """A triggered blend of differently sized images raises ValueError.

        The second image is one column wider than the first, so a pixelwise blend is impossible.
        """
        mixup = Mixup(p=1.0, generator=seeded_generator())
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
        """A source square overwrites destination pixels and appends one instance.

        An all-ones source over an all-zeros destination makes the pasted region visible: inside the square the pixel
        is 1, outside it the destination 0 stays.
        """
        copy_paste = CopyPaste(p=1.0, generator=seeded_generator())
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
        """The number of overwritten pixels equals the polygon's rasterised area.

        The expected count comes from rasterising the same ring directly, so the paste must write exactly the mask and
        nothing around it.
        """
        copy_paste = CopyPaste(p=1.0, generator=seeded_generator())
        destination = (torch.zeros(3, _SIDE, _SIDE), Targets.empty())
        source = (torch.ones(3, _SIDE, _SIDE), _polygon_source())

        out_image, _ = copy_paste([destination, source])

        mask = _rasterize_polygon(_square_ring(1.0, 1.0, 5.0, 5.0), _SIDE, _SIDE)
        assert int((out_image[0] == 1.0).sum()) == int(mask.sum())


class TestCopyPasteSkips:
    """Instances without a polygon are never pasted, and p=0 is identity."""

    @pytest.mark.parametrize(
        ("p", "source_targets"),
        [
            pytest.param(1.0, partial(_boxes_only, 3), id="non-polygon-source-instances-skipped"),
            pytest.param(0.0, _polygon_source, id="p-zero-identity"),
        ],
    )
    def test_nothing_pasted(self, p: float, source_targets: Callable[[], Targets]) -> None:
        """No paste happens: boxes-only sources even at p=1, or polygon sources at p=0, leave the destination as is.

        The destination image and (empty) targets must come back unchanged and ``last_pasted`` must be zero.
        """
        copy_paste = CopyPaste(p=p, generator=seeded_generator())
        dest_image = torch.zeros(3, _SIDE, _SIDE)
        destination = (dest_image, Targets.empty())
        source = (torch.ones(3, _SIDE, _SIDE), source_targets())

        out_image, out_targets = copy_paste([destination, source])

        assert copy_paste.last_pasted == 0
        assert out_targets.boxes.shape[0] == 0
        assert torch.equal(out_image, dest_image)


class TestCopyPasteGuards:
    """Rotated boxes and mismatched sizes are rejected."""

    def test_rboxes_raise_not_implemented(self) -> None:
        """A non-empty rboxes on either input is rejected: the paste unit is a polygon mask."""
        copy_paste = CopyPaste(p=1.0)

        with pytest.raises(NotImplementedError, match="polygon"):
            copy_paste(
                [(torch.zeros(3, _SIDE, _SIDE), Targets.empty()), (torch.ones(3, _SIDE, _SIDE), _rotated_only())]
            )

    def test_mismatched_sizes_raise(self) -> None:
        """Differently sized destination and source images raise ValueError.

        The source is one column wider than the destination, so its mask cannot be indexed into the destination.
        """
        copy_paste = CopyPaste(p=1.0)
        destination = (torch.zeros(3, _SIDE, _SIDE), Targets.empty())
        source = (torch.ones(3, _SIDE, _SIDE + 1), _polygon_source())

        with pytest.raises(ValueError, match="same-size"):
            copy_paste([destination, source])


class TestCopyPasteModalitiesThroughApply:
    """Every modality survives the ``apply`` seam WP-147 made public (WP-172).

    ``__call__`` is not the only entrance any more: a caller that wants a stated selection rather than a drawn one
    builds :class:`CopyPasteParams` and calls :meth:`CopyPaste.apply` directly. Each guard and each carried channel
    therefore has to hold on that path too, not merely on the sampling wrapper.
    """

    @pytest.mark.parametrize(
        ("rotated_slot", "selected"),
        [pytest.param(0, (0,), id="destination"), pytest.param(1, (), id="source")],
    )
    def test_apply_refuses_rotated_input(self, rotated_slot: int, selected: tuple[int, ...]) -> None:
        """A rotated destination or source is refused by ``apply``, not only by ``__call__``.

        The oriented path carries no polygons, so a rotated instance has nothing to paste and nothing to carry it:
        reaching the merge, its ``rboxes`` are dropped and the modality leaves silently. The refusal is what keeps that
        unreachable. ``_reject_rboxes`` inspects both inputs, so the seam must consult it rather than trusting a
        wrapper the direct caller never went through.
        """
        items = [(torch.zeros(3, _SIDE, _SIDE), Targets.empty()), (torch.ones(3, _SIDE, _SIDE), _polygon_source())]
        items[rotated_slot] = (items[rotated_slot][0], _rotated_only())

        with pytest.raises(NotImplementedError, match="polygon"):
            CopyPaste(p=1.0).apply(items, CopyPasteParams(selected=selected))

    @pytest.mark.parametrize(
        ("selected", "expected"),
        [
            pytest.param((1,), [True, True], id="selected-difficult-source-instance"),
            pytest.param((0,), [True, False], id="selected-easy-source-instance"),
        ],
    )
    def test_difficult_flag_follows_selected_instance(self, selected: tuple[int, ...], expected: list[bool]) -> None:
        """Both sides' R18 ``difficult`` flags reach the merged axis, a pasted flag indexed by its box's selection.

        A merge that rebuilds ``Targets`` without the channel does not raise — the container refills it with one
        ``False`` per instance — so a dropped flag is indistinguishable from an easy instance, and A48's discard rule
        silently turns every ignorable detection into a false positive (A48, A51). Selecting the source's easy
        instance must also yield an easy pasted instance: taking the flag column unindexed, or defaulting it, would
        score identically on the all-difficult selection and wrongly on this one.
        """
        _, out = CopyPaste(p=1.0).apply(_difficult_items(), CopyPasteParams(selected=selected))

        assert out.difficult.tolist() == expected


class TestCopyPasteUnion:
    """Overlapping rings copy one immutable source once without losing selected instances."""

    @pytest.mark.parametrize(
        "selected",
        [
            pytest.param((), id="none"),
            pytest.param((0,), id="single-ring"),
            pytest.param((0, 1), id="overlapping-pair"),
            pytest.param((1, 0, 1), id="repeated-index"),
            pytest.param((0, 1, 2), id="all-three"),
        ],
    )
    @pytest.mark.parametrize("transposed", [False, True])
    def test_matches_sequential_pixel_bytes_and_target_order(self, selected: tuple[int, ...], transposed: bool) -> None:
        """The original per-ring assignments are an independent oracle, including overlap.

        Whatever the selection order or repetition, and for a transposed (non-contiguous) source, the union copy must
        produce the same pixel bytes as assigning ring after ring, keep the labels in selection order, and leave both
        input images unmodified.
        """
        rings = [_square_ring(0.0, 0.0, 5.0, 5.0), _square_ring(3.0, 2.0, 8.0, 7.0), _square_ring(9.0, 9.0, 12.0, 12.0)]
        targets = Targets(
            boxes=torch.tensor([[0.0, 0.0, 5.0, 5.0], [3.0, 2.0, 8.0, 7.0], [9.0, 9.0, 12.0, 12.0]]),
            labels=torch.tensor([1, 3, 7]),
            polygons=rings,
        )
        source = torch.arange(3 * _SIDE * _SIDE, dtype=torch.float32).reshape(3, _SIDE, _SIDE)
        source[0, 0, 0] = -0.0
        if transposed:
            source = source.transpose(-1, -2)
        destination = torch.full_like(source, -1.0)
        expected = destination.clone()
        for index in selected:
            mask = _rasterize_polygon(rings[index], _SIDE, _SIDE)
            expected[:, mask] = source[:, mask]
        source_before = source.clone()
        got, got_targets = CopyPaste(p=1.0).apply(
            [(destination, Targets.empty()), (source, targets)], CopyPasteParams(selected=selected)
        )
        assert torch.equal(got.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
        assert got_targets.labels.tolist() == [int(targets.labels[index]) for index in selected]
        assert torch.equal(source.contiguous().view(torch.uint8), source_before.contiguous().view(torch.uint8))
        assert torch.equal(destination, torch.full_like(destination, -1.0))

    def test_multiple_rings_issue_one_image_assignment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returning silently to a copy per ring fails on the observable removed operation.

        Three identical rings are pasted in one call and the writes into the destination are counted through a
        ``Tensor.__setitem__`` wrapper; the union must reach the destination with exactly one assignment.
        """
        source = torch.ones(3, _SIDE, _SIDE)
        destination = torch.zeros_like(source)
        ring = _square_ring(1.0, 1.0, 6.0, 6.0)
        targets = Targets(
            boxes=torch.tensor([[1.0, 1.0, 6.0, 6.0]] * 3),
            labels=torch.zeros(3, dtype=torch.long),
            polygons=[ring, ring.clone(), ring.clone()],
        )
        calls: list[object] = []
        incumbent = torch.Tensor.__setitem__

        def counted(tensor: torch.Tensor, key: object, value: object) -> None:
            if tensor is destination:
                calls.append(key)
            incumbent(tensor, key, value)

        monkeypatch.setattr(torch.Tensor, "__setitem__", counted)
        CopyPaste._paste_selected(destination, source, targets, [0, 1, 2])
        assert len(calls) == 1


class TestRasterizer:
    """The polygon rasteriser fills the expected pixel block for an axis-aligned square."""

    def test_square_fills_expected_pixel_count(self) -> None:
        """A 4x4 square rasterises to a half-open 4x4 pixel block (16 pixels)."""
        mask = _rasterize_polygon(_square_ring(2.0, 2.0, 6.0, 6.0), 10, 10)

        assert mask.dtype == torch.bool
        assert mask.shape == (10, 10)
        assert int(mask.sum()) == 16
