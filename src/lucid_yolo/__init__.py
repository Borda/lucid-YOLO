# SPDX-License-Identifier: Apache-2.0
"""lucid-yolo: independent PyTorch Lightning reproduction of the YOLO26 methods.

Implements the detection, instance-segmentation, and oriented-detection methods
published in arXiv:2606.03748 from the papers and their cited primary literature
only. See docs/PROVENANCE.md and docs/ASSUMPTIONS.md for the clean-room record.
"""

#: Single source of truth for the package version (pyproject reads this attribute
#: statically). Pre-release wheels for a tier run carry a PEP 440 ``.devN`` suffix,
#: which sorts *below* the release it leads to, so ``pip install lucid-yolo`` never
#: resolves to one by accident and a run's artifacts record exactly which build trained
#: them. ``N`` counts from zero. Bump it for every wheel that leaves this machine, even a rebuild of the same
#: tree: a 15-hour run once trained on a stale wheel because the intended version was
#: never published and pip silently resolved the newest that existed.
__version__ = "0.3.0"

__all__ = ["__version__"]
