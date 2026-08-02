# SPDX-License-Identifier: Apache-2.0
"""Tests for the detection LightningCLI and the experiment configs (WP-038).

Covers the DoD: every file under ``configs/`` dry-parses through the CLI parser
(``LightningCLI(run=False)`` -- classes are instantiated, ``fit`` never runs) and
the resolved config survives a YAML round trip unchanged
(``test_yaml_roundtrip``). Alongside those, the ADR-001 boundary is asserted from
both sides: the ``variant`` link expands a registry-row name into the module's
compound-scaling multipliers and the datamodule augmentation policy (``variant:
n`` -> n-row multipliers + 7.5 box gain; ``variant: s`` -> s-row multipliers),
and a topology-like key the module does not accept (``model.layers``) -- as well
as a direct write to a link-computed multiplier (``model.depth``) -- is rejected
by the parser.

The CLI is always built with ``run=False`` so instantiation is exercised without
launching training; the configs carry only placeholder data paths, which the
datamodule stores without touching disk (loading happens in ``setup``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import open_yolos
from open_yolos.data.coco import build_scale_policy
from open_yolos.models.registry import scale_spec
from open_yolos.ptl.cli import DetectionCLI, default_determinism
from open_yolos.ptl.datamodule import DetectionDataModule
from open_yolos.ptl.module import DetectionLitModule

#: Packaged configs tree (``open_yolos/configs``; the repo-root ``configs`` symlinks here).
_CONFIGS_DIR = Path(open_yolos.__file__).resolve().parent / "configs"
#: Every experiment config, sorted for stable parametrization ids.
_CONFIG_PATHS = sorted(_CONFIGS_DIR.glob("*.yaml"))


def _build_cli(*args: str) -> DetectionCLI:
    """Build the detection CLI in non-running mode with the given extra args.

    Mirrors :func:`open_yolos.ptl.cli.main` (same ``deterministic``/seed defaults)
    but forces ``run=False`` so the model and datamodule are instantiated without
    launching ``fit``.
    """
    return DetectionCLI(
        DetectionLitModule,
        DetectionDataModule,
        trainer_defaults={"deterministic": default_determinism()},
        seed_everything_default=0,
        run=False,
        args=list(args),
    )


def test_default_determinism_matches_accelerator() -> None:
    """Strict determinism everywhere except MPS, which only supports warn_only."""

    expected = "warn_only" if torch.backends.mps.is_available() else True
    assert default_determinism() == expected


def test_configs_leave_determinism_to_cli_default() -> None:
    """No shipped config pins trainer.deterministic; the accelerator-aware default rules."""
    for path in _CONFIG_PATHS:
        assert "deterministic:" not in path.read_text(encoding="utf-8")


def _config_cli(path: Path, *extra: str) -> DetectionCLI:
    """Build the CLI from a config file plus any extra override args."""
    return _build_cli("--config", str(path), *extra)


def test_configs_dir_is_non_empty() -> None:
    """The three shipped configs are discovered (guards against an empty glob)."""
    names = {path.name for path in _CONFIG_PATHS}
    assert {"det_tier_a_n.yaml", "det_tier_b_s.yaml", "overfit_100.yaml"} <= names


@pytest.mark.parametrize("config_path", _CONFIG_PATHS, ids=lambda path: path.name)
def test_config_dry_parses(config_path: Path) -> None:
    """Every config instantiates the module and datamodule via the CLI parser."""
    cli = _config_cli(config_path)
    assert isinstance(cli.model, DetectionLitModule)
    assert isinstance(cli.datamodule, DetectionDataModule)


def test_tier_a_resolves_variant_n_multipliers_and_gains() -> None:
    """Det-A (``variant: n``) expands to the n-row multipliers and reference gains."""
    cli = _config_cli(_CONFIGS_DIR / "det_tier_a_n.yaml")
    spec = scale_spec("n")
    assert cli.model.hparams.depth == spec.depth
    assert cli.model.hparams.width == spec.width
    assert cli.model.hparams.max_channels == spec.max_channels
    assert cli.model.hparams.box_gain == 7.5
    assert cli.model.hparams.cls_gain == 0.5
    assert cli.model.hparams.l1_gain == 6.0
    assert cli.model.hparams.lr == 0.01
    assert cli.model.hparams.weight_decay == 0.0005
    assert cli.model.hparams.alpha_init == 0.8
    assert cli.model.hparams.alpha_final == 0.1
    # The same ``variant`` also drove the datamodule augmentation policy.
    assert cli.datamodule._policy == build_scale_policy("n")


def test_variant_s_resolves_registry_multipliers() -> None:
    """Det-B (``variant: s``) expands to the s-row multipliers and s-policy."""
    cli = _config_cli(_CONFIGS_DIR / "det_tier_b_s.yaml")
    spec = scale_spec("s")
    assert cli.model.hparams.depth == spec.depth
    assert cli.model.hparams.width == spec.width
    assert cli.model.hparams.max_channels == spec.max_channels
    assert cli.datamodule._policy == build_scale_policy("s")


@pytest.mark.parametrize("config_path", _CONFIG_PATHS, ids=lambda path: path.name)
def test_yaml_roundtrip(config_path: Path) -> None:
    """Resolved config -> YAML -> re-parse -> YAML is a fixed point for each config.

    The dumped config is the reproducibility artifact
    (:class:`~pytorch_lightning.cli.SaveConfigCallback` writes it per run); parsing
    it back through the same parser and re-dumping must reproduce it byte-for-byte.
    """
    cli = _config_cli(config_path)
    dumped = cli.parser.dump(cli.config, format="yaml")
    reparsed = cli.parser.parse_string(dumped)
    assert cli.parser.dump(reparsed, format="yaml") == dumped
    # The resolved config names the registry row, never the topology (ADR-001).
    assert "variant:" in dumped


def test_unknown_topology_key_is_rejected() -> None:
    """A topology-like key the module does not accept aborts parsing (ADR-001)."""
    with pytest.raises(SystemExit):
        _config_cli(_CONFIGS_DIR / "det_tier_a_n.yaml", "--model.layers=5")


def test_direct_multiplier_override_is_rejected() -> None:
    """A link-computed multiplier cannot be set directly; ``variant`` is the seam."""
    with pytest.raises(SystemExit):
        _config_cli(_CONFIGS_DIR / "det_tier_a_n.yaml", "--model.depth=0.9")
