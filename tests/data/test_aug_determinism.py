# SPDX-License-Identifier: Apache-2.0
"""Tier D: seeded determinism, scoped to whichever sampler is live (WP-150).

Every assertion here is keyed to a **seed**, which makes it categorically
different from the other three tiers and is why it lives apart from them.

- Tier A (``test_aug_contract.py``) states derived results and proves the code
  right.
- Tier B (``test_aug_frozen.py``) states parameters and pins behaviour.
- Tier C (``test_aug_contract.py``) states parameter-free invariants tying the
  modalities to each other.
- **Tier D, here**, states that a sampler is reproducible: the same seed twice
  gives byte-identical output, and different seeds give different output.

What tier D pins is the sampler that happens to be installed. It is **re-frozen
at every implementation swap and never carried across one**: a replacement engine
will not draw the same numbers from the same seed -- different call order,
different distributions, different consumption of the stream -- so these values
moving at a swap is expected and says nothing about correctness. Tiers A, B and C
moving at a swap is a defect.

Keeping the two apart is the whole point of this file. Mixed in with the
behaviour tiers, a tier-D failure at swap time reads as a regression, and the
natural response -- re-freeze it -- is the same keystroke that would silently
destroy a tier-B guard. Separated, the response is obvious from the file the
failure is in.

No value in this file appears in ``goldens/``. Reproducibility is a property of a
run, not a number worth freezing across releases.
"""

from __future__ import annotations

import torch

from lucid_yolo.data.affine import RandomAffine
from lucid_yolo.data.augment import HorizontalFlip, HSVJitter
from lucid_yolo.data.mixup import CopyPaste, Mixup
from lucid_yolo.data.mosaic import MosaicAssembly
from lucid_yolo.data.targets import Targets

#: Mosaic combines exactly four images.
MOSAIC_COUNT = 4
#: Canvas side for the single-image cases.
SIDE = 32
#: Number of flip draws taken before comparing two seeded sequences. One draw at
#: p=0.5 sees one branch; sixteen sees both.
FLIP_DRAWS = 16


def _generator(seed: int) -> torch.Generator:
    """Build a CPU generator seeded to ``seed``.

    Args:
        seed: The seed to set.

    Examples:
        ```pycon
        >>> _generator(7).initial_seed()
        7

        ```
    """
    return torch.Generator().manual_seed(seed)


def _image(seed: int = 0) -> torch.Tensor:
    """Draw a fixed pseudo-random image, independent of any transform's generator.

    The image's own seed is separate from the transform's on purpose: these cases
    compare two transforms over the *same* input, so the input must not vary.

    Args:
        seed: Seed for the image draw.

    Examples:
        ```pycon
        >>> _image().shape
        torch.Size([3, 32, 32])

        ```
    """
    return torch.rand(3, SIDE, SIDE, generator=_generator(seed))


def _polygon_targets() -> Targets:
    """Build a one-instance box-and-polygon target set.

    Examples:
        ```pycon
        >>> len(_polygon_targets().polygons)
        1

        ```
    """
    return Targets(
        boxes=torch.tensor([[6.0, 6.0, 18.0, 18.0]]),
        labels=torch.tensor([0]),
        polygons=[torch.tensor([[6.0, 6.0], [18.0, 6.0], [18.0, 18.0], [6.0, 18.0]])],
    )


class TestRandomAffineDeterminism:
    """Tier D -- ``RandomAffine``'s six draws are reproducible from a seed."""

    def test_equal_seeds_give_equal_outputs(self) -> None:
        """Two affines sharing a seed produce equal images, boxes and rings.

        Reproducibility is what makes a failed training run diagnosable at all: an
        augmentation pipeline that varied per process would make every anomaly
        unrepeatable, which is the failure WP-079 was opened by.
        """
        image = _image()
        kwargs = {"degrees": 15.0, "translate": 0.1, "scale": 0.2, "shear": 5.0}
        affine_a = RandomAffine(**kwargs, generator=_generator(99))
        affine_b = RandomAffine(**kwargs, generator=_generator(99))

        image_a, targets_a = affine_a(image.clone(), _polygon_targets())
        image_b, targets_b = affine_b(image.clone(), _polygon_targets())

        assert torch.equal(image_a, image_b)
        assert torch.equal(targets_a.boxes, targets_b.boxes)
        assert all(torch.equal(a, b) for a, b in zip(targets_a.polygons, targets_b.polygons, strict=True))

    def test_different_seeds_give_different_outputs(self) -> None:
        """Two affines on different seeds draw different transforms.

        The other half of reproducibility, and the half that catches a generator
        that was silently ignored: a transform reading the global RNG instead of its
        own would still pass the equal-seeds case whenever the global state happened
        to line up.
        """
        image = _image()
        kwargs = {"degrees": 15.0, "translate": 0.1, "scale": 0.2, "shear": 5.0}

        image_a, _ = RandomAffine(**kwargs, generator=_generator(1))(image.clone(), _polygon_targets())
        image_b, _ = RandomAffine(**kwargs, generator=_generator(2))(image.clone(), _polygon_targets())

        assert not torch.equal(image_a, image_b)


class TestAffineOntoLetterboxCanvasDeterminism:
    """Tier D -- the fused path samples once and reproduces from a seed."""

    def test_equal_seeds_give_equal_outputs(self) -> None:
        """Two fused transforms sharing a seed produce equal images, boxes and rings."""
        image = torch.rand(3, 180, 240, generator=_generator(3))
        kwargs = {"degrees": 15.0, "translate": 0.1, "scale": 0.2, "shear": 5.0}
        fused_a = RandomAffine(**kwargs, generator=_generator(99), letterbox=128)
        fused_b = RandomAffine(**kwargs, generator=_generator(99), letterbox=128)

        image_a, targets_a = fused_a(image.clone(), _polygon_targets())
        image_b, targets_b = fused_b(image.clone(), _polygon_targets())

        assert torch.equal(image_a, image_b)
        assert torch.equal(targets_a.boxes, targets_b.boxes)
        assert all(torch.equal(a, b) for a, b in zip(targets_a.polygons, targets_b.polygons, strict=True))


class TestHSVJitterDeterminism:
    """Tier D -- three photometric gains, reproducible from a seed."""

    def test_equal_seeds_give_equal_outputs(self) -> None:
        """Two jitters sharing a seed produce a byte-identical image."""
        image = _image(4)

        first, _ = HSVJitter(generator=_generator(7))(image, Targets.empty())
        second, _ = HSVJitter(generator=_generator(7))(image, Targets.empty())

        assert torch.equal(first, second)

    def test_different_seeds_give_different_outputs(self) -> None:
        """Two jitters on different seeds draw different gains."""
        image = _image(4)

        first, _ = HSVJitter(generator=_generator(7))(image, Targets.empty())
        second, _ = HSVJitter(generator=_generator(8))(image, Targets.empty())

        assert not torch.equal(first, second)


class TestHorizontalFlipDeterminism:
    """Tier D -- the flip trigger is reproducible over a sequence of draws."""

    def test_equal_seeds_give_equal_decision_sequences(self) -> None:
        """Sixteen draws from two equally seeded flips agree decision for decision.

        One draw would see one branch and pass for a flip that always returned the
        same answer; a sequence sees both branches and their order.
        """
        image = _image(5)
        flip_a = HorizontalFlip(p=0.5, generator=_generator(21))
        flip_b = HorizontalFlip(p=0.5, generator=_generator(21))

        decisions_a = [(flip_a(image, Targets.empty()), flip_a.last_flipped)[1] for _ in range(FLIP_DRAWS)]
        decisions_b = [(flip_b(image, Targets.empty()), flip_b.last_flipped)[1] for _ in range(FLIP_DRAWS)]

        assert decisions_a == decisions_b
        assert all(isinstance(decision, bool) for decision in decisions_a)
        assert len(set(decisions_a)) == 2


class TestMosaicDeterminism:
    """Tier D -- the stitch centre is reproducible from a seed."""

    def test_equal_seeds_give_equal_outputs(self) -> None:
        """Two mosaics sharing a seed produce equal canvases, boxes and centres."""
        images = [_image(index) for index in range(MOSAIC_COUNT)]
        items_a = [(image.clone(), _polygon_targets()) for image in images]
        items_b = [(image.clone(), _polygon_targets()) for image in images]
        mosaic_a = MosaicAssembly(target_size=SIDE, generator=_generator(99))
        mosaic_b = MosaicAssembly(target_size=SIDE, generator=_generator(99))

        image_a, targets_a = mosaic_a(items_a)
        image_b, targets_b = mosaic_b(items_b)

        assert torch.equal(image_a, image_b)
        assert torch.equal(targets_a.boxes, targets_b.boxes)
        assert mosaic_a.last_center == mosaic_b.last_center


class TestMixupDeterminism:
    """Tier D -- the trigger and the blend factor are reproducible from a seed."""

    def test_equal_seeds_give_equal_outputs(self) -> None:
        """Two mixups sharing a seed produce equal images, boxes and blend factors."""
        pair = [(_image(6), _polygon_targets()), (_image(7), _polygon_targets())]
        mixup_a = Mixup(p=0.5, generator=_generator(99))
        mixup_b = Mixup(p=0.5, generator=_generator(99))

        image_a, targets_a = mixup_a(pair)
        image_b, targets_b = mixup_b(pair)

        assert torch.equal(image_a, image_b)
        assert torch.equal(targets_a.boxes, targets_b.boxes)
        assert mixup_a.last_lam == mixup_b.last_lam


class TestCopyPasteDeterminism:
    """Tier D -- the per-candidate paste draws are reproducible from a seed."""

    def test_equal_seeds_give_equal_outputs(self) -> None:
        """Two copy-pastes sharing a seed select the same instances and write the same pixels."""
        source = Targets(
            boxes=torch.tensor([[1.0, 1.0, 3.0, 3.0], [4.0, 4.0, 6.0, 6.0]]),
            labels=torch.tensor([1, 2]),
            polygons=[
                torch.tensor([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]]),
                torch.tensor([[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0]]),
            ],
        )
        items = [(torch.zeros(3, SIDE, SIDE), Targets.empty()), (torch.ones(3, SIDE, SIDE), source)]
        copy_paste_a = CopyPaste(p=0.5, generator=_generator(7))
        copy_paste_b = CopyPaste(p=0.5, generator=_generator(7))

        image_a, targets_a = copy_paste_a(items)
        image_b, targets_b = copy_paste_b(items)

        assert torch.equal(image_a, image_b)
        assert torch.equal(targets_a.boxes, targets_b.boxes)
        assert copy_paste_a.last_pasted == copy_paste_b.last_pasted
