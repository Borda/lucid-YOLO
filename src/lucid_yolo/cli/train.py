# SPDX-License-Identifier: Apache-2.0
"""LightningCLI entry for detection experiments (WP-038).

:func:`main` builds a :class:`~pytorch_lightning.cli.LightningCLI` over
:class:`~lucid_yolo.ptl.module.DetectionLitModule` and
:class:`~lucid_yolo.ptl.datamodule.DetectionDataModule`, so a run is launched
entirely from YAML::

    python -m lucid_yolo.cli.train fit --config det_nano_smoke.yaml

or, once the package is installed, through the ``lucid-yolo`` console script.

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
    topology stays typed Python (:mod:`lucid_yolo.models.registry`) and never a
    config field. To keep configs from repeating multiplier triples,
    :class:`DetectionCLI` adds a single top-level ``variant`` argument and
    ``link_arguments`` it — through :func:`~lucid_yolo.models.registry.scale_spec`
    ``compute_fn`` hooks — onto ``model.depth``/``model.width``/
    ``model.max_channels`` (and, unmodified, onto ``data.variant`` for the
    augmentation-strength policy). A config therefore names a registry **row**
    (``variant: n``); it can never state a topology. Because those three targets
    become link-computed, jsonargparse forbids setting them directly, which
    enforces the ADR-001 boundary at parse time rather than by convention.

Provenance: D9/ADR-001, D12c. Assumptions: A8.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar, RichProgressBar, TQDMProgressBar
from pytorch_lightning.cli import ArgsType, LightningArgumentParser, LightningCLI
from pytorch_lightning.loggers import CSVLogger, Logger, TensorBoardLogger

from lucid_yolo.models.registry import scale_spec
from lucid_yolo.ptl.datamodule import DetectionDataModule
from lucid_yolo.ptl.module import DetectionLitModule

__all__ = ["DetectionCLI", "default_determinism", "main"]


def _default_loggers(default_root_dir: str | None) -> list[Logger]:
    """Build the default logger pair: TensorBoard plus CSV in one version directory.

    Lightning's own default is a lone :class:`TensorBoardLogger`; a plain
    ``metrics.csv`` alongside the event file keeps every run inspectable without
    TensorBoard tooling. The CSV logger is pinned to the TensorBoard logger's
    freshly resolved version so both write into the same
    ``lightning_logs/version_N`` directory (checkpoints stay with the first
    logger, exactly where a TensorBoard-only run puts them).

    Args:
        default_root_dir: The trainer's ``default_root_dir`` (both loggers'
            ``save_dir``); ``None`` falls back to the working directory,
            matching the trainer's own default.

    Returns:
        ``[TensorBoardLogger, CSVLogger]`` sharing one version directory.
    """
    root = default_root_dir or os.getcwd()
    tensorboard = TensorBoardLogger(save_dir=root)
    return [tensorboard, CSVLogger(save_dir=root, version=tensorboard.version)]


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
#: (the Det-smoke reference recipe); any user config or CLI flag overrides per key.
_DEFAULT_CONFIG = "det_nano_smoke"

#: Default progress-bar flavour. ``tqdm`` — not Lightning's rich-when-available
#: auto-pick — because :class:`~pytorch_lightning.callbacks.TQDMProgressBar` goes
#: through ``tqdm.auto`` (an ipywidgets widget in notebooks), while
#: :class:`~pytorch_lightning.callbacks.RichProgressBar`'s live rendering prints
#: one line per refresh in Colab/Jupyter cell output.
_DEFAULT_PROGRESS_BAR = "tqdm"


def _checkpoint_filename(task: str, variant: str) -> str:
    """Return a checkpoint filename template naming the run's task and scale variant.

    Lightning's own unnamed-run default is ``"{epoch}-{step}"`` (rendered
    ``epoch=X-step=Y.ckpt``); this keeps that suffix — and the metric-templating
    machinery that fills it in — intact, and only prepends the run identity that
    was previously recoverable only by opening ``hparams.yaml`` (WP-127).

    Args:
        task: The module's supervision task (``"detect"``, ``"segment"``, or ``"obb"``).
        variant: The scale-registry row letter (e.g. ``"n"``, ``"s"``).

    Returns:
        A filename template, ``{epoch}``/``{step}`` still literal for
        :class:`~pytorch_lightning.callbacks.ModelCheckpoint` to fill in.

    Examples:
        >>> _checkpoint_filename("segment", "s")
        'segment_s_{epoch}-{step}'
    """
    return f"{task}_{variant}_{{epoch}}-{{step}}"


class DetectionCLI(LightningCLI):
    """LightningCLI wiring the detection module/datamodule with a ``variant`` link.

    Adds one top-level ``variant`` argument and links it, via
    :func:`~lucid_yolo.models.registry.scale_spec`, onto the module's three raw
    compound-scaling multipliers and the datamodule's augmentation-policy letter,
    so a config selects a registry row by name instead of restating the topology
    (see the module docstring for the ADR-001 rationale).
    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Add the ``variant`` argument and link it onto the model and datamodule.

        The three ``model`` links carry ``compute_fn`` hooks that expand the
        variant letter into its :class:`~lucid_yolo.models.registry.ScaleSpec`
        field; the ``data.variant`` link is a plain pass-through of the same
        letter. Registering these targets as link-computed also makes
        jsonargparse reject any attempt to set them directly, enforcing ADR-001.

        Args:
            parser: The CLI parser to extend (populated by LightningCLI with the
                ``model``/``data``/``trainer`` argument groups already added).
        """
        parser.add_argument(
            "--progress_bar",
            type=str,
            default=_DEFAULT_PROGRESS_BAR,
            choices=("tqdm", "rich", "none"),
            help="Progress bar flavour: tqdm (notebook-safe default), rich (terminal live view), none.",
        )
        parser.add_argument("--variant", type=str, default=_DEFAULT_VARIANT)
        parser.link_arguments("variant", "model.depth", compute_fn=lambda variant: scale_spec(variant).depth)
        parser.link_arguments("variant", "model.width", compute_fn=lambda variant: scale_spec(variant).width)
        parser.link_arguments(
            "variant", "model.max_channels", compute_fn=lambda variant: scale_spec(variant).max_channels
        )
        parser.link_arguments("variant", "data.variant")
        # A segmentation run rasterises its mask targets in the loader workers rather
        # than in the training process (WP-087 perf). The task already states whether
        # masks are supervised, so linking beats a second knob two configs could
        # disagree on: a loader that rasterised for a detection run would raise on the
        # first box without a ring, and one that did not for a segmentation run would
        # silently put that CPU work back on the critical path.
        parser.link_arguments("model.task", "data.mask_targets", compute_fn=lambda task: task == "segment")
        # The oriented counterpart (WP-088), linked for the same reason: the task already
        # states whether rotated boxes are supervised, and a loader that disagreed would
        # either raise on the first non-quadrilateral ring or hand an obb run empty
        # `rboxes` — which trains the plain detection objective and reports nothing wrong.
        parser.link_arguments("model.task", "data.rotated_targets", compute_fn=lambda task: task == "obb")
        # And the pose counterpart (WP-132). Same argument a third time: a loader that
        # did not read the point fields would hand a keypoints run boxes alone, which
        # `pad_keypoints` refuses rather than trains on — loud, but loud at the first
        # step of a configured run instead of at the line that configured it.
        parser.link_arguments("model.task", "data.keypoint_targets", compute_fn=lambda task: task == "keypoints")

    def _config_value(self, key: str, default: Any) -> Any:
        """Read a top-level key out of the instantiated config, unwrapping the subcommand.

        ``fit``/``validate``/``test``/``predict`` each nest the whole config one level deep
        under the subcommand name, while a subcommand-less invocation does not; this reads
        either shape. It stands in for :class:`~pytorch_lightning.cli.LightningCLI`'s own
        private ``_get`` — same two lookups, but written against
        :class:`jsonargparse.Namespace`'s public :meth:`~jsonargparse.Namespace.get`, so
        the CLI's behaviour is not pinned to an undocumented method that any Lightning
        release is free to rename or drop.

        Args:
            key: Top-level config key to read (e.g. ``"trainer"`` or ``"variant"``).
            default: Value to return when the key is absent.

        Returns:
            The configured value, or ``default`` when the key is not set.
        """
        config = self.config_init
        scoped = config if self.subcommand is None else config.get(self.subcommand, config)
        return scoped.get(key, default)

    def _add_trainer_default_callback(self, callback: Callback) -> None:
        """Append ``callback`` to ``trainer_defaults["callbacks"]`` without replacing it.

        The only injection channel that *extends* the config's callback list
        instead of replacing it outright.

        Args:
            callback: The callback instance to append.
        """
        defaults_callbacks = self.trainer_defaults.get("callbacks", [])
        if not isinstance(defaults_callbacks, list):
            defaults_callbacks = [defaults_callbacks]
        self.trainer_defaults = {**self.trainer_defaults, "callbacks": [*defaults_callbacks, callback]}

    def instantiate_trainer(self, **kwargs: Any) -> Trainer:
        """Instantiate the trainer with the ``--progress_bar`` choice and default loggers.

        Lightning's own default is rich-when-available, whose live rendering
        prints one line per refresh in notebook cell output (Colab/Jupyter), so
        the choice is made explicit here: the selected bar is appended via
        :meth:`_add_trainer_default_callback`, and ``none`` disables the bar
        entirely. A ``ProgressBar`` instance placed directly in
        ``trainer.callbacks`` by a user config still wins: Lightning rejects two
        bars, so the default injection is skipped in that case.

        The checkpoint filename gets the same treatment (WP-127): a bare
        ``lightning_logs/version_N`` numbers the *run*, not the checkpoint
        inside it, so identifying what a saved ``.ckpt`` trained meant opening
        its ``hparams.yaml``. Unless a config already places a
        :class:`~pytorch_lightning.callbacks.ModelCheckpoint` in
        ``trainer.callbacks``, one is injected with
        :func:`_checkpoint_filename` naming the task and scale variant;
        ``dirpath`` is left at Lightning's own default (``<version dir>/checkpoints``),
        so no already-written checkpoint moves or is renamed.

        When the config leaves ``trainer.logger`` at its default (``null`` or
        ``true`` — Lightning's TensorBoard-only auto-pick), the default is
        widened to **TensorBoard + CSV side by side** in the same
        ``lightning_logs/version_N`` directory (the CSV logger is pinned to the
        TensorBoard logger's version), so every run leaves both the event file
        and a plain ``metrics.csv``. An explicit ``logger: false`` or a concrete
        logger (list) in the config wins untouched.

        Args:
            kwargs: Extra trainer arguments forwarded to LightningCLI.

        Returns:
            The configured :class:`~pytorch_lightning.Trainer`.
        """
        choice = str(self._config_value("progress_bar", _DEFAULT_PROGRESS_BAR))
        trainer_config = self._config_value("trainer", {})
        trainer_callbacks = trainer_config.get("callbacks") or []
        user_bar = any(isinstance(callback, ProgressBar) for callback in trainer_callbacks)
        if choice == "none":
            kwargs.setdefault("enable_progress_bar", False)
        elif not user_bar:
            bar: ProgressBar = RichProgressBar() if choice == "rich" else TQDMProgressBar()
            self._add_trainer_default_callback(bar)
        user_checkpoint = any(isinstance(callback, ModelCheckpoint) for callback in trainer_callbacks)
        if not user_checkpoint:
            variant = str(self._config_value("variant", _DEFAULT_VARIANT))
            self._add_trainer_default_callback(ModelCheckpoint(filename=_checkpoint_filename(self.model.task, variant)))
        if trainer_config.get("logger") in (None, True) and "logger" not in kwargs:
            kwargs["logger"] = _default_loggers(trainer_config.get("default_root_dir"))
        return super().instantiate_trainer(**kwargs)


def packaged_config(name: str) -> Path:
    """Return the path of a config shipped inside the installed package.

    Args:
        name: Config file name with or without the ``.yaml`` suffix
            (e.g. ``"det_nano_smoke"`` or ``"det_nano_smoke.yaml"``).

    Returns:
        Absolute path of ``lucid_yolo/configs/<name>.yaml``. The path is
        returned without an existence check — the CLI parser reports a missing
        file with its usual error.

    Examples:
        >>> packaged_config("det_nano_smoke").name
        'det_nano_smoke.yaml'
    """
    if not name.endswith((".yaml", ".yml")):
        name = f"{name}.yaml"
    return Path(__file__).resolve().parents[1] / "configs" / name


def _resolve_config_args(args: list[str]) -> list[str]:
    """Rewrite ``--config`` values naming packaged configs onto their real paths.

    A value following ``--config``/``-c`` (or embedded as ``--config=NAME``)
    that does not exist on disk but matches a file under the packaged
    ``lucid_yolo/configs`` tree is replaced by that packaged path, so an
    installed wheel runs ``lucid-yolo fit --config det_nano_smoke.yaml`` with no
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
    ``.yaml`` suffix) are resolved onto the installed ``lucid_yolo/configs``
    tree when no such file exists locally, so a bare
    ``lucid-yolo fit --config det_nano_smoke.yaml`` works from a wheel install
    (see :func:`packaged_config`).

    Args:
        args: Explicit arguments to parse instead of ``sys.argv`` — a list of
            option strings or a mapping — used by the tests to dry-parse each
            config. ``None`` (the default) reads ``sys.argv``.

    Returns:
        The constructed :class:`DetectionCLI`; when invoked with a run subcommand
        the fit/validate loop has already executed by the time it is returned.

    Examples:
        >>> from lucid_yolo.cli.train import main
        >>> cli = main(["fit", "--config", "det_nano_smoke"])  # doctest: +SKIP
    """
    if args is None:
        # Rewrite sys.argv in place and keep args=None: passing an args list while
        # sys.argv also carries arguments makes LightningCLI warn about the overlap.
        sys.argv[1:] = _resolve_config_args(sys.argv[1:])
    elif isinstance(args, list) and all(isinstance(token, str) for token in args):
        args = _resolve_config_args(args)
    if torch.cuda.is_available():
        # Lightning's Tensor Core advisory: allow TF32 matmuls for the fp32 ops
        # AMP leaves untouched. CUDA-only effect; CPU/MPS numerics unchanged.
        torch.set_float32_matmul_precision("high")
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
