# SPDX-License-Identifier: Apache-2.0
"""LightningCLI entry for detection experiments (WP-038).

:func:`main` builds a :class:`~pytorch_lightning.cli.LightningCLI` over
:class:`~lit_yolo.ptl.module.DetectionLitModule` and
:class:`~lit_yolo.ptl.datamodule.DetectionDataModule`, so a run is launched
entirely from YAML::

    python -m lit_yolo.ptl.cli fit --config configs/det_tier_a_n.yaml

or, once the package is installed, through the ``lit-yolo`` console script.

Reproducibility contract:
    The default :class:`~pytorch_lightning.cli.SaveConfigCallback` is left
    enabled, so every run writes its **fully resolved** config (``config.yaml``)
    next to the checkpoints — the config that reproduces the run byte-for-byte,
    including the values the ``variant`` link expands (below). Trainer defaults
    are kept minimal and accelerator-agnostic: only ``deterministic=True`` is
    forced (D12c leaves accelerator selection to Lightning's auto-detection), and
    ``seed_everything`` defaults to ``0``.

Variant link (ADR-001):
    The module constructor takes the three raw compound-scaling multipliers
    (``depth``/``width``/``max_channels``) rather than a variant name, so that the
    topology stays typed Python (:mod:`lit_yolo.models.registry`) and never a
    config field. To keep configs from repeating multiplier triples,
    :class:`DetectionCLI` adds a single top-level ``variant`` argument and
    ``link_arguments`` it — through :func:`~lit_yolo.models.registry.scale_spec`
    ``compute_fn`` hooks — onto ``model.depth``/``model.width``/
    ``model.max_channels`` (and, unmodified, onto ``data.variant`` for the
    augmentation-strength policy). A config therefore names a registry **row**
    (``variant: n``); it can never state a topology. Because those three targets
    become link-computed, jsonargparse forbids setting them directly, which
    enforces the ADR-001 boundary at parse time rather than by convention.

Provenance: D9/ADR-001, D12c. Assumptions: A8.
"""

from __future__ import annotations

from pytorch_lightning.cli import ArgsType, LightningArgumentParser, LightningCLI

from lit_yolo.models.registry import scale_spec
from lit_yolo.ptl.datamodule import DetectionDataModule
from lit_yolo.ptl.module import DetectionLitModule

__all__ = ["DetectionCLI", "main"]

#: Default scale variant when a config omits ``variant`` (the n-scale debug row, D3).
_DEFAULT_VARIANT = "n"


class DetectionCLI(LightningCLI):
    """LightningCLI wiring the detection module/datamodule with a ``variant`` link.

    Adds one top-level ``variant`` argument and links it, via
    :func:`~lit_yolo.models.registry.scale_spec`, onto the module's three raw
    compound-scaling multipliers and the datamodule's augmentation-policy letter,
    so a config selects a registry row by name instead of restating the topology
    (see the module docstring for the ADR-001 rationale).
    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add the ``variant`` argument and link it onto the model and datamodule.

        The three ``model`` links carry ``compute_fn`` hooks that expand the
        variant letter into its :class:`~lit_yolo.models.registry.ScaleSpec`
        field; the ``data.variant`` link is a plain pass-through of the same
        letter. Registering these targets as link-computed also makes
        jsonargparse reject any attempt to set them directly, enforcing ADR-001.

        Args:
            parser: The CLI parser to extend (populated by LightningCLI with the
                ``model``/``data``/``trainer`` argument groups already added).
        """
        parser.add_argument("--variant", type=str, default=_DEFAULT_VARIANT)
        parser.link_arguments("variant", "model.depth", compute_fn=lambda variant: scale_spec(variant).depth)
        parser.link_arguments("variant", "model.width", compute_fn=lambda variant: scale_spec(variant).width)
        parser.link_arguments(
            "variant", "model.max_channels", compute_fn=lambda variant: scale_spec(variant).max_channels
        )
        parser.link_arguments("variant", "data.variant")


def main(args: ArgsType = None) -> DetectionCLI:
    """Run the detection LightningCLI.

    Builds a :class:`DetectionCLI` with ``deterministic=True`` and a default seed
    of ``0``; every other setting comes from the CLI/config. With no ``args`` the
    CLI reads ``sys.argv`` (the console-script and ``python -m`` path), so a
    subcommand such as ``fit`` and a ``--config`` are supplied there.

    Args:
        args: Explicit arguments to parse instead of ``sys.argv`` — a list of
            option strings or a mapping — used by the tests to dry-parse each
            config. ``None`` (the default) reads ``sys.argv``.

    Returns:
        The constructed :class:`DetectionCLI`; when invoked with a run subcommand
        the fit/validate loop has already executed by the time it is returned.

    Examples:
        >>> from lit_yolo.ptl.cli import main
        >>> cli = main(  # doctest: +SKIP
        ...     ["fit", "--config", "configs/det_tier_a_n.yaml"]
        ... )
    """
    return DetectionCLI(
        DetectionLitModule,
        DetectionDataModule,
        trainer_defaults={"deterministic": True},
        seed_everything_default=0,
        args=args,
    )


if __name__ == "__main__":
    main()
