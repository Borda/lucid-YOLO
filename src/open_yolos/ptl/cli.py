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

import sys
from pathlib import Path

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

#: Packaged config loaded as the parser's defaults when no ``--config`` is given
#: (the Det-A reference recipe); any user config or CLI flag overrides per key.
_DEFAULT_CONFIG = "det_tier_a_n"


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


def packaged_config(name: str) -> Path:
    """Return the path of a config shipped inside the installed package.

    Args:
        name: Config file name with or without the ``.yaml`` suffix
            (e.g. ``"det_tier_a_n"`` or ``"det_tier_a_n.yaml"``).

    Returns:
        Absolute path of ``open_yolos/configs/<name>.yaml``. The path is
        returned without an existence check — the CLI parser reports a missing
        file with its usual error.

    Examples:
        >>> packaged_config("det_tier_a_n").name
        'det_tier_a_n.yaml'
    """
    if not name.endswith((".yaml", ".yml")):
        name = f"{name}.yaml"
    return Path(__file__).resolve().parents[1] / "configs" / name


def _resolve_config_args(args: list[str]) -> list[str]:
    """Rewrite ``--config`` values naming packaged configs onto their real paths.

    A value following ``--config``/``-c`` (or embedded as ``--config=NAME``)
    that does not exist on disk but matches a file under the packaged
    ``open_yolos/configs`` tree is replaced by that packaged path, so an
    installed wheel runs ``open-yolos fit --config det_tier_a_n.yaml`` with no
    checkout and no absolute path. Existing paths always win untouched, and
    unknown names pass through unchanged for the parser's normal error.
    """
    resolved: list[str] = []
    expect_value = False
    for token in args:
        if expect_value:
            expect_value = False
            candidate = packaged_config(token)
            resolved.append(str(candidate) if not Path(token).exists() and candidate.is_file() else token)
            continue
        if token in ("--config", "-c"):
            expect_value = True
            resolved.append(token)
            continue
        if token.startswith("--config="):
            value = token.removeprefix("--config=")
            candidate = packaged_config(value)
            rewritten = f"--config={candidate}" if not Path(value).exists() and candidate.is_file() else token
            resolved.append(rewritten)
            continue
        resolved.append(token)
    return resolved


def main(args: ArgsType = None) -> DetectionCLI:
    """Run the detection LightningCLI.

    Builds a :class:`DetectionCLI` with :func:`default_determinism` and a default seed
    of ``0``; every other setting comes from the CLI/config. With no ``args`` the
    CLI reads ``sys.argv`` (the console-script and ``python -m`` path), so a
    subcommand such as ``fit`` and a ``--config`` are supplied there.

    ``--config`` values that name a **packaged** config (with or without the
    ``.yaml`` suffix) are resolved onto the installed ``open_yolos/configs``
    tree when no such file exists locally, so a bare
    ``open-yolos fit --config det_tier_a_n.yaml`` works from a wheel install
    (see :func:`packaged_config`).

    Args:
        args: Explicit arguments to parse instead of ``sys.argv`` — a list of
            option strings or a mapping — used by the tests to dry-parse each
            config. ``None`` (the default) reads ``sys.argv``.

    Returns:
        The constructed :class:`DetectionCLI`; when invoked with a run subcommand
        the fit/validate loop has already executed by the time it is returned.

    Examples:
        >>> from open_yolos.ptl.cli import main
        >>> cli = main(["fit", "--config", "det_tier_a_n"])  # doctest: +SKIP
    """
    if args is None:
        args = sys.argv[1:]
    if isinstance(args, list) and all(isinstance(token, str) for token in args):
        args = _resolve_config_args(args)
    return DetectionCLI(
        DetectionLitModule,
        DetectionDataModule,
        trainer_defaults={"deterministic": default_determinism()},
        seed_everything_default=0,
        parser_kwargs={
            subcommand: {"default_config_files": [str(packaged_config(_DEFAULT_CONFIG))]}
            for subcommand in ("fit", "validate", "test", "predict")
        },
        args=args,
    )


if __name__ == "__main__":
    main()
