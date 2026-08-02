# SPDX-License-Identifier: Apache-2.0
"""LightningCLI entry for detection experiments (WP-038).

:func:`main` builds a :class:`~pytorch_lightning.cli.LightningCLI` over
:class:`~open_yolos.ptl.module.DetectionLitModule` and
:class:`~open_yolos.ptl.datamodule.DetectionDataModule`, so a run is launched
entirely from YAML::

    python -m open_yolos.ptl.cli fit --config configs/det_tier_a_n.yaml

or, once the package is installed, through the ``open-yolos`` console script.

Reproducibility contract:
    The default :class:`~pytorch_lightning.cli.SaveConfigCallback` is left
    enabled, so every run writes its **fully resolved** config (``config.yaml``)
    next to the checkpoints — the config that reproduces the run byte-for-byte,
    including the values the ``variant`` link expands (below). Trainer defaults
    are kept minimal and accelerator-agnostic: accelerator selection stays with
    Lightning's auto-detection (D12c — the best available device is picked
    automatically), ``seed_everything`` defaults to ``0``, and determinism is
    :func:`default_determinism`: strict ``True`` on CPU/CUDA, ``"warn_only"``
    when MPS is the auto-picked accelerator, because MPS lacks deterministic
    kernels for some backward ops (``index_put_with_accumulate``) and strict
    mode would abort the run. A config or CLI flag can still override it
    explicitly.

Variant link (ADR-001):
    The module constructor takes the three raw compound-scaling multipliers
    (``depth``/``width``/``max_channels``) rather than a variant name, so that the
    topology stays typed Python (:mod:`open_yolos.models.registry`) and never a
    config field. To keep configs from repeating multiplier triples,
    :class:`DetectionCLI` adds a single top-level ``variant`` argument and
    ``link_arguments`` it — through :func:`~open_yolos.models.registry.scale_spec`
    ``compute_fn`` hooks — onto ``model.depth``/``model.width``/
    ``model.max_channels`` (and, unmodified, onto ``data.variant`` for the
    augmentation-strength policy). A config therefore names a registry **row**
    (``variant: n``); it can never state a topology. Because those three targets
    become link-computed, jsonargparse forbids setting them directly, which
    enforces the ADR-001 boundary at parse time rather than by convention.

Provenance: D9/ADR-001, D12c. Assumptions: A8.
"""

from __future__ import annotations

import torch
from pytorch_lightning.cli import ArgsType, LightningArgumentParser, LightningCLI

from open_yolos.models.registry import scale_spec
from open_yolos.ptl.datamodule import DetectionDataModule
from open_yolos.ptl.module import DetectionLitModule

__all__ = ["DetectionCLI", "default_determinism", "main"]


def default_determinism() -> bool | str:
    """Return the strictest determinism setting the auto-picked accelerator supports.

    Lightning's ``accelerator="auto"`` prefers MPS on Apple silicon, and MPS is
    missing deterministic implementations for some backward kernels (for example
    ``index_put_with_accumulate``, hit by the assignment/loss backward), so
    strict ``deterministic=True`` aborts mid-step there. ``"warn_only"`` keeps
    every op deterministic where a deterministic kernel exists and downgrades
    the rest to a warning; CPU and CUDA runs keep strict ``True``.

    Returns:
        ``"warn_only"`` when MPS is available (and therefore auto-picked),
        ``True`` otherwise.

    Examples:
        >>> default_determinism() in (True, "warn_only")
        True
    """
    return "warn_only" if torch.backends.mps.is_available() else True


#: Default scale variant when a config omits ``variant`` (the n-scale debug row, D3).
_DEFAULT_VARIANT = "n"


class DetectionCLI(LightningCLI):
    """LightningCLI wiring the detection module/datamodule with a ``variant`` link.

    Adds one top-level ``variant`` argument and links it, via
    :func:`~open_yolos.models.registry.scale_spec`, onto the module's three raw
    compound-scaling multipliers and the datamodule's augmentation-policy letter,
    so a config selects a registry row by name instead of restating the topology
    (see the module docstring for the ADR-001 rationale).
    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add the ``variant`` argument and link it onto the model and datamodule.

        The three ``model`` links carry ``compute_fn`` hooks that expand the
        variant letter into its :class:`~open_yolos.models.registry.ScaleSpec`
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

    Builds a :class:`DetectionCLI` with :func:`default_determinism` and a default seed
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
        >>> from open_yolos.ptl.cli import main
        >>> cli = main(  # doctest: +SKIP
        ...     ["fit", "--config", "configs/det_tier_a_n.yaml"]
        ... )
    """
    return DetectionCLI(
        DetectionLitModule,
        DetectionDataModule,
        trainer_defaults={"deterministic": default_determinism()},
        seed_everything_default=0,
        args=args,
    )


if __name__ == "__main__":
    main()
