# SPDX-License-Identifier: Apache-2.0
"""Every command this package installs, in one place (WP-096).

Four entry points, one parser stack:

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
``lucid-predict``
    Detections for a single image (:mod:`lucid_yolo.cli.predict`) — the other reading of
    a checkpoint. Where ``lucid-eval`` answers "how good is this model", this answers
    "what is in this picture", and reads its task from the same place: a checkpoint whose
    task is not ``detect`` is refused by name rather than run as a detector (WP-089).

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

    That consequence had one exemption and no longer does: the ``lucid-download`` alias
    kept its argparse parser and its dashed flags through 0.3.0 so published reproduction
    instructions kept running. It was removed in 0.4.0 as 0.3.0 said it would be, so
    ``--data_root`` is now the only spelling any shipped command answers to (WP-110).
"""

from lucid_yolo.cli.data import main as data_main
from lucid_yolo.cli.eval import main as eval_main
from lucid_yolo.cli.predict import main as predict_main
from lucid_yolo.cli.train import main as train_main

__all__ = ["data_main", "eval_main", "predict_main", "train_main"]
