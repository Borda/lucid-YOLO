# SPDX-License-Identifier: Apache-2.0
"""Every command this package installs, in one place (WP-096).

Three entry points, one parser stack:

``lucid-yolo``
    Training and validation (:mod:`lucid_yolo.cli.train`) — a
    :class:`~pytorch_lightning.cli.LightningCLI`, so its flags are derived from the
    module and datamodule signatures.
``lucid-data``
    ``download`` · ``check`` · ``build-tiles`` (:mod:`lucid_yolo.cli.data`) — everything
    that happens to a dataset before a run.
``lucid-eval``
    Acceptance scoring (:mod:`lucid_yolo.cli.eval`), on the protocol the checkpoint's own
    task names.

Why one package:
    The wiring used to be spread across ``lucid_yolo/ptl/cli.py``, a console script in
    ``lucid_yolo/data/download.py``, and two files under ``scripts/`` that the wheel did
    not ship — so a remote tier run could install the package and still not reach two
    thirds of its own pipeline. Command surfaces now live here and the modules they drive
    stay where their logic belongs (:mod:`lucid_yolo.data.check`,
    :mod:`lucid_yolo.data.tiles`, :mod:`lucid_yolo.eval.detect_eval`,
    :mod:`lucid_yolo.eval.rotated_eval`).

Why jsonargparse:
    ``lucid-yolo`` is a LightningCLI, which is jsonargparse underneath, so the other two
    commands use it directly rather than argparse. Flags and help text are then derived
    from the function signatures and their Google docstrings — one description per
    argument, in the docstring the API already required — and every command accepts
    ``--config`` for free, which is how a tier run records what it was given. The
    consequence to know: jsonargparse spells flags with underscores (``--data_root``),
    exactly as ``--data.batch_size`` already does on the training CLI.

    The deprecated ``lucid-download`` alias keeps its original argparse parser and its
    dashed flags, frozen: it exists so published reproduction instructions keep running,
    and re-spelling its flags would defeat that.
"""

from lucid_yolo.cli.data import main as data_main
from lucid_yolo.cli.eval import main as eval_main
from lucid_yolo.cli.train import main as train_main

__all__ = ["data_main", "eval_main", "train_main"]
