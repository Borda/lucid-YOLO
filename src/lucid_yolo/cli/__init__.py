# SPDX-License-Identifier: Apache-2.0
"""Every command this package installs, in one place (WP-096).

Five entry points, one parser stack:

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
    "what is in this picture", and reads its task from the same place: ``detect`` supplies
    boxes, ``segment`` adds instance masks, ``obb`` supplies rotated boxes and ``keypoints``
    adds point sets. Unknown tasks are refused by name.
``lucid-export``
    One ONNX file from a checkpoint (:mod:`lucid_yolo.cli.export`) — the deployable
    reading. It too takes the task from the checkpoint and writes that task's
    suppression-free graph through :func:`~lucid_yolo.export.graph_from_module` and
    :func:`~lucid_yolo.export.export_graph`, at a static input shape. Writing the file
    needs the ``onnx`` package, which the ``export`` extra installs.

Why one package:
    The wiring used to be spread across ``lucid_yolo/ptl/cli.py``, a console script in
    ``lucid_yolo/data/download.py``, and two files under ``scripts/`` that the wheel did
    not ship — so a remote tier run could install the package and still not reach two
    thirds of its own pipeline. Command surfaces now live here and the modules they drive
    stay where their logic belongs (:mod:`lucid_yolo.data.check`,
    :mod:`lucid_yolo.data.tiles`, :mod:`lucid_yolo.eval.detect_eval`,
    :mod:`lucid_yolo.eval.rotated_eval`, :mod:`lucid_yolo.export`).

Why jsonargparse:
    ``lucid-yolo`` is a LightningCLI, which is jsonargparse underneath, so the other
    four commands — ``lucid-data``, ``lucid-eval``, ``lucid-predict`` and
    ``lucid-export`` — use it directly rather than argparse. Flags and help text are then derived
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

import warnings

# Imported ahead of the block below, never inside it: `catch_warnings` restores the filter
# list it found on entry, so the filters torch installs on its first import would be
# discarded with the block if that first import happened there.
import torch  # noqa: F401

# fvcore.nn.focal_loss scripts its loss functions while it is imported, which these imports
# reach through the model builder. The filter lives only as long as the block, so usage text
# prints without that dependency's FutureWarning while every later `torch.jit.script` call
# -- a library user's own included -- still warns.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"^`torch\.jit\.script` is deprecated\. Please switch to `torch\.compile` or `torch\.export`\.$",
        category=FutureWarning,
        module=r"^torch\.jit\._script$",
    )
    from lucid_yolo.cli.data import main as data_main
    from lucid_yolo.cli.eval import main as eval_main
    from lucid_yolo.cli.export import main as export_main
    from lucid_yolo.cli.predict import main as predict_main
    from lucid_yolo.cli.train import main as train_main

__all__ = ["data_main", "eval_main", "export_main", "predict_main", "train_main"]
