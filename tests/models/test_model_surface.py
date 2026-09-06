# SPDX-License-Identifier: Apache-2.0
"""Unit gates on the models package's public surface and its constructor contracts.

Three properties that live between modules rather than inside one, which is why none of
the per-module suites owns them:

* what :mod:`lucid_yolo.models` re-exports -- a task decoder reachable from
  ``lucid_yolo.models.heads`` and not from the package above it is a surface that
  contradicts itself, and the omission is invisible from either module alone;
* that the scale registry cannot be edited through the name every model reads it by;
* that a module refuses a hyperparameter its task builds no stem for.

**Placement note (WP audit T6/L-25).** The third group tests
:class:`~lucid_yolo.ptl.module.DetectionLitModule`, whose natural mirror is
``tests/ptl/test_module_composition.py``. It sits here because the change it guards was
made under an ownership boundary that did not include ``tests/ptl/``. It should move
there when the two are next touched together; nothing about the assertion depends on
living in this file.
"""

from __future__ import annotations

import pytest

import lucid_yolo.models as models_package
from lucid_yolo.models import VARIANTS
from lucid_yolo.models.registry import ScaleSpec
from lucid_yolo.ptl.module import DetectionLitModule

#: The four task decoders, one per task. They are siblings in every sense that matters to
#: a caller -- same package, same role -- so the package surface either carries all four
#: or documents why not.
_TASK_DECODERS = ("decode_ltrb", "decode_rboxes", "decode_keypoints", "assemble_masks")

#: Smallest registry row, expanded; the constructor gates under test are scale-independent.
_N_SCALE = VARIANTS["n"]


class TestPackageExports:
    """What ``lucid_yolo.models`` re-exports from ``lucid_yolo.models.heads``."""

    @pytest.mark.parametrize("name", [pytest.param(n, id=n) for n in _TASK_DECODERS])
    def test_every_task_decoder_is_importable_from_the_package(self, name: str) -> None:
        """Each task's decoder is reachable from ``lucid_yolo.models``, not only from ``.heads``.

        ``decode_keypoints`` was the one that was not. Its three siblings were imported
        and listed in ``__all__``; the keypoint task arrived later and reached
        ``heads/__init__.py`` without reaching the package above it, so a caller who had
        found ``decode_ltrb`` where the others were had no reason to suspect the fourth
        lived one module deeper.
        """
        assert hasattr(models_package, name), f"{name} is missing from the package namespace"
        assert name in models_package.__all__, f"{name} is importable but undeclared in __all__"


class TestVariantRegistry:
    """The compound-scaling table's readability and its immutability."""

    def test_reading_a_row_is_unchanged(self) -> None:
        """Subscript, membership and iteration all still work on the exported table.

        The immutability below is only worth having if it costs nothing to read, and the
        registry is read by subscript in the module builders and iterated when
        ``scale_spec`` composes its error message.
        """
        assert VARIANTS["n"] == ScaleSpec(depth=0.50, width=0.25, max_channels=1024)
        assert "x" in VARIANTS
        assert list(VARIANTS) == ["n", "s", "m", "l", "x"]

    def test_a_row_cannot_be_rebound_through_the_exported_name(self) -> None:
        """Assigning into ``VARIANTS`` raises rather than redefining a published variant.

        The registry module calls itself "the one place the multipliers live", and while
        it was a plain ``dict`` that was a courtesy: any importer could rebind a row and
        every model built afterwards in that process -- including the ones the WP-023
        fidelity gate measures -- would scale by the new numbers, with the gate's own
        source of truth having moved underneath it.
        """
        with pytest.raises(TypeError):
            VARIANTS["n"] = ScaleSpec(depth=9.0, width=9.0, max_channels=9)  # type: ignore[index]

    def test_a_new_variant_cannot_be_added_through_the_exported_name(self) -> None:
        """Inserting an unknown key raises too, so the table cannot grow a row at runtime.

        The rebinding case above covers overwriting a published row; this covers the
        other direction, where a caller registers a variant that the goldens, the config
        schema and ``scale_spec``'s error message all know nothing about.
        """
        with pytest.raises(TypeError):
            VARIANTS["xxl"] = ScaleSpec(depth=2.0, width=2.0, max_channels=1024)  # type: ignore[index]


class TestKeypointHyperparameterContract:
    """``DetectionLitModule``'s two-directional gate on ``num_keypoints``."""

    def test_a_point_count_is_refused_for_a_task_with_no_point_stem(self) -> None:
        """``num_keypoints`` on a non-keypoint task raises instead of being stored and ignored.

        The constructor already refused the missing case -- ``task='keypoints'`` without
        a count -- and was silent on the contradictory one. ``save_hyperparameters``
        records every argument, so a ``detect`` module built with ``num_keypoints=17``
        wrote a point schema into its checkpoint that its head has no stem to predict,
        leaving a later reader a documented ``K`` and no points.
        """
        with pytest.raises(ValueError, match="meaningless for task='detect'"):
            DetectionLitModule(
                depth=_N_SCALE.depth,
                width=_N_SCALE.width,
                max_channels=_N_SCALE.max_channels,
                num_classes=4,
                num_keypoints=17,
            )

    def test_the_keypoints_task_still_requires_a_point_count(self) -> None:
        """The pre-existing missing-count refusal is untouched by the added one.

        Both directions are errors and they must stay distinguishable: this one says the
        count is needed, the other says it is meaningless, and a caller acts on which.
        """
        with pytest.raises(ValueError, match="needs an explicit num_keypoints"):
            DetectionLitModule(
                depth=_N_SCALE.depth,
                width=_N_SCALE.width,
                max_channels=_N_SCALE.max_channels,
                num_classes=1,
                task="keypoints",
            )

    def test_the_keypoints_task_accepts_its_own_point_count(self) -> None:
        """The normal keypoint construction is unaffected -- the gate rejects only the mismatch.

        Without this the two refusals above would pass against a constructor that refused
        every ``num_keypoints``, which is the failure mode a pair of error-path tests
        cannot see on its own.
        """
        module = DetectionLitModule(
            depth=_N_SCALE.depth,
            width=_N_SCALE.width,
            max_channels=_N_SCALE.max_channels,
            num_classes=1,
            task="keypoints",
            num_keypoints=17,
        )

        assert module.task == "keypoints"
        assert module.head.num_keypoints == 17
