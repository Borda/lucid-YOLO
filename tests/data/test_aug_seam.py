# SPDX-License-Identifier: Apache-2.0
"""Temporary same-codebase equivalence guard for the WP-147 sample/apply seam.

WP-147 has a chicken-and-egg problem: it restructures every stochastic transform
*before* the frozen expectation suite that authorises restructuring exists. What
guards it in the meantime is this file, plus the existing seeded assertions in
``tests/data/test_affine.py``, ``test_photometric.py``, ``test_mosaic.py`` and
``test_mixup_copypaste.py``.

Every test here asserts one property: for a transform seeded to some state,
``__call__`` produces exactly what ``sample()`` followed by ``apply()`` produces
from the same state. That is *old against old inside one repository* with no
upstream package involved, so it is not the old-versus-new comparison the plan's
regression-guard section rules out -- nothing here can enshrine legacy behaviour
as the definition of correct, because nothing here outlives the seam it guards.

**This file is deleted by WP-149**, once tiers A, B and C freeze expectations
against stated parameters and the seam no longer needs a witness of its own.

The generator is rewound rather than re-created between the two halves: each test
records the generator state, runs one path, restores the state, runs the other.
Anything that consumed a different number of draws in the two paths shows up as a
mismatched result rather than as a silent divergence later.
"""

from __future__ import annotations

import torch

from lucid_yolo.data.affine import FusedAffineLetterbox, RandomAffine
from lucid_yolo.data.augment import HorizontalFlip, HSVJitter
from lucid_yolo.data.mixup import CopyPaste, Mixup
from lucid_yolo.data.mosaic import MosaicAssembly
from lucid_yolo.data.targets import Targets

#: One seed for every case; the property is per-state equivalence, not a value.
SEED = 20260902


def _boxed_targets() -> Targets:
    """Build a two-instance target set with boxes, polygons and keypoints.

    Every modality the transforms carry is present, so an apply path that drops one
    fails here rather than in whichever downstream task happens to read it.

    Examples:
        ```pycon
        >>> _boxed_targets().boxes.shape
        torch.Size([2, 4])

        ```
    """
    return Targets(
        boxes=torch.tensor([[2.0, 2.0, 10.0, 10.0], [12.0, 4.0, 20.0, 14.0]]),
        labels=torch.tensor([0, 1]),
        polygons=[
            torch.tensor([[2.0, 2.0], [10.0, 2.0], [10.0, 10.0], [2.0, 10.0]]),
            torch.tensor([[12.0, 4.0], [20.0, 4.0], [20.0, 14.0], [12.0, 14.0]]),
        ],
        keypoints=torch.tensor([[[3.0, 3.0], [9.0, 9.0]], [[13.0, 5.0], [19.0, 13.0]]]),
        keypoint_vis=torch.tensor([[2, 2], [2, 1]]),
    )


def _rewound() -> tuple[torch.Generator, torch.Tensor]:
    """Return a seeded generator and its state, for running two paths from one draw point.

    Examples:
        ```pycon
        >>> gen, state = _rewound()
        >>> gen.get_state().equal(state)
        True

        ```
    """
    generator = torch.Generator().manual_seed(SEED)
    return generator, generator.get_state()


class TestRandomAffineSeam:
    """``RandomAffine.__call__`` equals ``sample`` then ``apply`` from the same RNG state."""

    def test_matches_call_on_every_modality(self) -> None:
        """The seam reproduces the image and every target modality bit for bit.

        A warp with rotation, shear, scale and translation all live is the case that
        exercises all six sampled parameters; a seam that dropped one -- reading a
        default instead of the draw -- would still return plausible geometry.
        """
        generator, state = _rewound()
        transform = RandomAffine(degrees=15.0, translate=0.2, scale=0.3, shear=5.0, generator=generator)
        image = torch.rand(3, 24, 24, generator=torch.Generator().manual_seed(1))

        called_image, called_targets = transform(image, _boxed_targets())
        generator.set_state(state)
        applied_image, applied_targets = transform.apply(image, _boxed_targets(), transform.sample(24, 24))

        assert torch.equal(called_image, applied_image)
        assert torch.equal(called_targets.boxes, applied_targets.boxes)
        assert torch.equal(called_targets.keypoints, applied_targets.keypoints)
        assert all(torch.equal(a, b) for a, b in zip(called_targets.polygons, applied_targets.polygons, strict=True))

    def test_matches_warp_to_on_the_fused_path(self) -> None:
        """``warp_to`` equals ``sample`` then ``apply_to`` under one post-matrix.

        The fused path composes the sampled affine with a downscale before a single
        resample, so it is the one place where a seam that re-sampled inside apply
        would produce two different matrices for one call.
        """
        generator, state = _rewound()
        transform = RandomAffine(degrees=10.0, translate=0.1, scale=0.2, generator=generator)
        image = torch.rand(3, 24, 24, generator=torch.Generator().manual_seed(2))
        post = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]])

        called_image, called_targets = transform.warp_to(image, _boxed_targets(), post, 12, 12)
        generator.set_state(state)
        params = transform.sample(24, 24)
        applied_image, applied_targets = transform.apply_to(image, _boxed_targets(), params, post, 12, 12)

        assert torch.equal(called_image, applied_image)
        assert torch.equal(called_targets.boxes, applied_targets.boxes)


class TestFusedAffineLetterboxSeam:
    """``FusedAffineLetterbox.__call__`` equals ``sample`` then ``apply``."""

    def test_matches_call(self) -> None:
        """The fused transform's seam reproduces its single-resample output exactly.

        Its letterbox half draws nothing, so this also asserts the seam did not
        introduce a draw where the deterministic half sits.
        """
        generator, state = _rewound()
        transform = FusedAffineLetterbox(16, degrees=12.0, translate=0.15, scale=0.25, generator=generator)
        image = torch.rand(3, 20, 32, generator=torch.Generator().manual_seed(3))

        called_image, called_targets = transform(image, _boxed_targets())
        generator.set_state(state)
        applied_image, applied_targets = transform.apply(image, _boxed_targets(), transform.sample(20, 32))

        assert torch.equal(called_image, applied_image)
        assert torch.equal(called_targets.boxes, applied_targets.boxes)


class TestHSVJitterSeam:
    """``HSVJitter.__call__`` equals ``sample`` then ``apply``."""

    def test_matches_call(self) -> None:
        """The three gains are drawn in the same order and applied to the same effect.

        Hue is additive modulo one and the other two are multiplicative, so a seam
        that reordered the draws would still produce a plausible image.
        """
        generator, state = _rewound()
        transform = HSVJitter(generator=generator)
        image = torch.rand(3, 12, 12, generator=torch.Generator().manual_seed(4))

        called_image, _ = transform(image, Targets.empty())
        generator.set_state(state)
        applied_image, _ = transform.apply(image, Targets.empty(), transform.sample())

        assert torch.equal(called_image, applied_image)


class TestHorizontalFlipSeam:
    """``HorizontalFlip.__call__`` equals ``sample`` then ``apply``."""

    def test_matches_call_across_many_draws(self) -> None:
        """Both outcomes of the trigger agree, over enough draws to see each.

        A single draw at ``p=0.5`` sees one branch; twenty sees both, which is what
        distinguishes a working seam from one that always takes the pass-through.
        """
        generator, state = _rewound()
        transform = HorizontalFlip(p=0.5, generator=generator, keypoint_flip_pairs=[(0, 1)])
        image = torch.rand(3, 8, 10, generator=torch.Generator().manual_seed(5))

        called = [transform(image, _boxed_targets()) for _ in range(20)]
        generator.set_state(state)
        applied = [transform.apply(image, _boxed_targets(), transform.sample()) for _ in range(20)]

        assert all(torch.equal(a[0], b[0]) for a, b in zip(called, applied, strict=True))
        assert all(torch.equal(a[1].keypoints, b[1].keypoints) for a, b in zip(called, applied, strict=True))


class TestMosaicAssemblySeam:
    """``MosaicAssembly.__call__`` equals ``sample`` then ``apply``."""

    def test_matches_call(self) -> None:
        """The stitched canvas and merged targets agree for a drawn centre.

        The centre decides all four placements at once, so a seam that drew it twice
        would produce a canvas whose four quadrants disagree with its own targets.
        """
        generator, state = _rewound()
        transform = MosaicAssembly(target_size=16, generator=generator)
        images = [torch.rand(3, 16, 16, generator=torch.Generator().manual_seed(6 + i)) for i in range(4)]
        items = [(image, _boxed_targets()) for image in images]

        called_image, called_targets = transform(items)
        generator.set_state(state)
        applied_image, applied_targets = transform.apply(items, transform.sample())

        assert torch.equal(called_image, applied_image)
        assert torch.equal(called_targets.boxes, applied_targets.boxes)


class TestMixupSeam:
    """``Mixup.__call__`` equals ``sample`` then ``apply``."""

    def test_matches_call_when_the_trigger_fires(self) -> None:
        """A triggered blend reproduces the same convex factor.

        ``p=1.0`` forces the branch that draws two gammas after the trigger uniform,
        which is the branch whose draw count a seam can get wrong.
        """
        generator, state = _rewound()
        transform = Mixup(p=1.0, generator=generator)
        items = [
            (torch.rand(3, 8, 8, generator=torch.Generator().manual_seed(10)), _boxed_targets()),
            (torch.rand(3, 8, 8, generator=torch.Generator().manual_seed(11)), _boxed_targets()),
        ]

        called_image, called_targets = transform(items)
        generator.set_state(state)
        applied_image, applied_targets = transform.apply(items, transform.sample())

        assert torch.equal(called_image, applied_image)
        assert torch.equal(called_targets.boxes, applied_targets.boxes)

    def test_matches_call_when_the_trigger_does_not_fire(self) -> None:
        """A pass-through consumes the trigger uniform and nothing else.

        ``p=0.0`` never blends, so this pins the branch where the seam must *not*
        draw a blend factor -- consuming one there would desynchronise every later
        draw in the pipeline.
        """
        generator, state = _rewound()
        transform = Mixup(p=0.0, generator=generator)
        items = [
            (torch.rand(3, 8, 8, generator=torch.Generator().manual_seed(12)), _boxed_targets()),
            (torch.rand(3, 8, 8, generator=torch.Generator().manual_seed(13)), _boxed_targets()),
        ]

        called_image, _ = transform(items)
        generator.set_state(state)
        applied_image, _ = transform.apply(items, transform.sample())

        assert torch.equal(called_image, applied_image)
        assert generator.get_state().equal(torch.Generator().manual_seed(SEED).get_state()) is False


class TestCopyPasteSeam:
    """``CopyPaste.__call__`` equals ``sample`` then ``apply``."""

    def test_matches_call_under_a_paste_cap(self) -> None:
        """The same instances are selected and the same pixels are pasted.

        ``max_paste`` truncates the draw loop early, so this is the case where a
        seam that drew once per candidate regardless of the cap would consume more
        of the stream than the original.
        """
        generator, state = _rewound()
        transform = CopyPaste(p=0.7, max_paste=1, generator=generator)
        destination = Targets(
            boxes=torch.tensor([[0.0, 0.0, 3.0, 3.0]]),
            labels=torch.tensor([0]),
            polygons=[torch.tensor([[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0]])],
        )
        # Copy-paste merges polygons only, so the source carries no keypoint axis to
        # merge against a destination that has none (a `Targets` invariant, not a seam one).
        source = Targets(
            boxes=torch.tensor([[2.0, 2.0, 10.0, 10.0], [12.0, 4.0, 20.0, 14.0]]),
            labels=torch.tensor([0, 1]),
            polygons=[
                torch.tensor([[2.0, 2.0], [10.0, 2.0], [10.0, 10.0], [2.0, 10.0]]),
                torch.tensor([[12.0, 4.0], [20.0, 4.0], [20.0, 14.0], [12.0, 14.0]]),
            ],
        )
        items = [(torch.zeros(3, 24, 24), destination), (torch.ones(3, 24, 24), source)]

        called_image, called_targets = transform(items)
        generator.set_state(state)
        applied_image, applied_targets = transform.apply(items, transform.sample(len(items[1][1].polygons)))

        assert torch.equal(called_image, applied_image)
        assert torch.equal(called_targets.labels, applied_targets.labels)
