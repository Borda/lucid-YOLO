# SPDX-License-Identifier: Apache-2.0
"""Tests for the detection LightningCLI and the experiment configs (WP-038).

Covers the DoD: every file under ``lucid_yolo/configs/`` dry-parses through the CLI parser
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

import lucid_yolo
from lucid_yolo.cli.train import DetectionCLI, _resolve_config_args, default_determinism, main, packaged_config
from lucid_yolo.data.coco import build_scale_policy
from lucid_yolo.models.registry import scale_spec
from lucid_yolo.ptl.datamodule import DetectionDataModule
from lucid_yolo.ptl.module import DetectionLitModule

#: Packaged configs tree.
_CONFIGS_DIR = Path(lucid_yolo.__file__).resolve().parent / "configs"
#: Every experiment config, sorted for stable parametrization ids.
_CONFIG_PATHS = sorted(_CONFIGS_DIR.glob("*.yaml"))


def _build_cli(*args: str) -> DetectionCLI:
    """Build the detection CLI in non-running mode with the given extra args.

    Mirrors :func:`lucid_yolo.cli.train.main` (same ``deterministic``/seed defaults)
    but forces ``run=False`` so the model and datamodule are instantiated without
    launching ``fit``.

    Examples:
        >>> cli = _build_cli("--config", str(packaged_config("det_nano_smoke")))
        >>> isinstance(cli.model, DetectionLitModule)
        True
    """
    return DetectionCLI(
        DetectionLitModule,
        DetectionDataModule,
        trainer_defaults={"deterministic": default_determinism()},
        seed_everything_default=0,
        run=False,
        args=list(args),
    )


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        pytest.param((), "TQDMProgressBar", id="default-tqdm"),
        pytest.param(("--progress_bar", "rich"), "RichProgressBar", id="rich"),
        pytest.param(("--progress_bar", "none"), None, id="none"),
    ],
)
def test_progress_bar_choice_selects_callback(flags: tuple[str, ...], expected: str | None) -> None:
    """--progress_bar picks the bar flavour; the default is the notebook-safe tqdm bar."""
    cli = _build_cli("--config", str(packaged_config("det_nano_smoke")), *flags)
    bar = cli.trainer.progress_bar_callback
    assert (None if bar is None else type(bar).__name__) == expected


def test_default_loggers_are_tensorboard_plus_csv(tmp_path: Path) -> None:
    """An unset trainer.logger yields TensorBoard + CSV sharing one version directory."""
    cli = _build_cli(
        "--config",
        str(packaged_config("det_nano_smoke")),
        "--trainer.default_root_dir",
        str(tmp_path),
    )
    names = [type(logger).__name__ for logger in cli.trainer.loggers]
    assert names == ["TensorBoardLogger", "CSVLogger"]
    assert cli.trainer.loggers[0].log_dir == cli.trainer.loggers[1].log_dir


def test_logger_false_disables_default_loggers(tmp_path: Path) -> None:
    """An explicit trainer.logger=false wins over the TensorBoard + CSV default."""
    cli = _build_cli(
        "--config",
        str(packaged_config("det_nano_smoke")),
        "--trainer.default_root_dir",
        str(tmp_path),
        "--trainer.logger=false",
    )
    assert cli.trainer.loggers == []


def test_default_determinism_matches_accelerator() -> None:
    """Strict determinism everywhere except MPS, which only supports warn_only."""
    expected = "warn_only" if torch.backends.mps.is_available() else True
    assert default_determinism() == expected


def test_configs_leave_determinism_to_cli_default() -> None:
    """No shipped config pins trainer.deterministic; the accelerator-aware default rules."""
    for path in _CONFIG_PATHS:
        assert "deterministic:" not in path.read_text(encoding="utf-8")


def _config_cli(path: Path, *extra: str) -> DetectionCLI:
    """Build the CLI from a config file plus any extra override args.

    Examples:
        >>> cli = _config_cli(packaged_config("det_nano_smoke"))
        >>> isinstance(cli.model, DetectionLitModule)
        True
    """
    return _build_cli("--config", str(path), *extra)


def test_configs_dir_is_non_empty() -> None:
    """Every shipped config is discovered (guards against an empty glob)."""
    names = {path.name for path in _CONFIG_PATHS}
    assert {
        "det_nano_smoke.yaml",
        "det_small_ablations.yaml",
        "det_nano_overfit_100.yaml",
        "seg_nano_smoke.yaml",
        "obb_nano_smoke.yaml",
    } <= names


@pytest.mark.parametrize("config_path", _CONFIG_PATHS, ids=lambda path: path.name)
def test_config_dry_parses(config_path: Path) -> None:
    """Every config instantiates the module and datamodule via the CLI parser."""
    cli = _config_cli(config_path)
    assert isinstance(cli.model, DetectionLitModule)
    assert isinstance(cli.datamodule, DetectionDataModule)


def test_smoke_tier_resolves_variant_n_multipliers_and_gains() -> None:
    """Det-smoke (``variant: n``) expands to the n-row multipliers and reference gains."""
    cli = _config_cli(_CONFIGS_DIR / "det_nano_smoke.yaml")
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
    """Det-ablations (``variant: s``) expands to the s-row multipliers and s-policy."""
    cli = _config_cli(_CONFIGS_DIR / "det_small_ablations.yaml")
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
        _config_cli(_CONFIGS_DIR / "det_nano_smoke.yaml", "--model.layers=5")


def test_direct_multiplier_override_is_rejected() -> None:
    """A link-computed multiplier cannot be set directly; ``variant`` is the seam."""
    with pytest.raises(SystemExit):
        _config_cli(_CONFIGS_DIR / "det_nano_smoke.yaml", "--model.depth=0.9")


def test_packaged_config_resolves_name_with_and_without_suffix() -> None:
    """packaged_config maps bare names onto the installed configs tree."""
    with_suffix = packaged_config("det_nano_smoke.yaml")
    without_suffix = packaged_config("det_nano_smoke")
    assert with_suffix == without_suffix
    assert with_suffix.is_file()
    assert with_suffix.parent == _CONFIGS_DIR


@pytest.mark.parametrize(
    ("argv", "expected_value"),
    [
        pytest.param(["fit", "--config", "det_nano_smoke.yaml"], None, id="separate-token"),
        pytest.param(["fit", "--config", "det_nano_smoke"], None, id="bare-name"),
        pytest.param(["fit", "--config=det_nano_smoke"], None, id="equals-form"),
    ],
)
def test_resolve_config_args_rewrites_packaged_names(argv, expected_value) -> None:
    """A --config value naming a packaged config is rewritten onto its real path."""
    del expected_value
    resolved = " ".join(_resolve_config_args(argv))
    assert str(packaged_config("det_nano_smoke")) in resolved


def test_resolve_config_args_leaves_existing_and_unknown_paths_alone(tmp_path) -> None:
    """Existing local paths and unknown names pass through untouched."""
    local = tmp_path / "det_nano_smoke.yaml"
    local.write_text("variant: n\n", encoding="utf-8")
    assert _resolve_config_args(["fit", "--config", str(local)]) == ["fit", "--config", str(local)]
    assert _resolve_config_args(["fit", "--config", "no_such_config"]) == ["fit", "--config", "no_such_config"]


def test_fit_without_config_defaults_to_packaged_recipe(capsys) -> None:
    """Bare fit (no --config) loads the packaged Det-smoke recipe as parser defaults."""
    with pytest.raises(SystemExit):
        main(["fit", "--print_config"])
    out = capsys.readouterr().out
    assert "max_epochs: 50" in out
    assert "gradient_clip_val: 10.0" in out
    assert "variant: n" in out


@pytest.mark.parametrize(
    ("config_name", "expected"),
    [
        pytest.param("seg_nano_smoke.yaml", True, id="segment-rasterises-in-the-loader"),
        pytest.param("det_nano_smoke.yaml", False, id="detect-does-not"),
    ],
)
def test_mask_targets_follows_the_model_task(config_name: str, expected: bool) -> None:
    """``model.task`` decides whether the loader rasterises mask targets, with no second knob.

    A detection loader that rasterised would raise on the first box without a ring;
    a segmentation loader that did not would silently put ~1 s per step of CPU work
    back on the training process's critical path, which is a performance regression
    no test asserting correctness would ever see.
    """
    cli = _config_cli(_CONFIGS_DIR / config_name)

    assert cli.datamodule._mask_targets is expected


@pytest.mark.parametrize(
    ("config_name", "expected"),
    [
        pytest.param("obb_nano_smoke.yaml", True, id="obb-reads-rotated-targets"),
        pytest.param("det_nano_smoke.yaml", False, id="detect-does-not"),
    ],
)
def test_rotated_targets_follows_the_model_task(config_name: str, expected: bool) -> None:
    """``model.task`` decides whether the loader reads rotated boxes, with no second knob.

    The failure this guards is silent in exactly the WP-088 way: a loader left in
    axis-aligned mode hands an ``obb`` run empty ``rboxes``, which is not a shape error
    anywhere — the step would simply raise on the padding check, or worse, a future
    fallback would train the plain detection objective while the angle stems idled.
    """
    cli = _config_cli(_CONFIGS_DIR / config_name)

    assert cli.datamodule._rotated_targets is expected
