# SPDX-License-Identifier: Apache-2.0
"""Content-pin audit: tracked files that must, or must not, carry a literal string.

Each pin is a structural claim about a file's text rather than about what the code does when run, so it belongs
to a hook rather than to pytest. Three pins live here, each moved out of a unit test that read the file as text:

* ``src/lucid_yolo/decode/topk_e2e.py`` names no suppression operator, IoU helper or ``torchvision``
  (case-insensitive). The score-based top-k decoder is defined by having no non-maximum suppression; a helper import
  or call creeping in would leave every functional test green while the module stopped being what its name claims.
* No packaged recipe under ``src/lucid_yolo/configs/`` pins ``trainer.deterministic``.
  The CLI picks the accelerator-aware default (strict everywhere, ``warn_only`` on MPS),
  and a recipe hard-coding it overrides that on the one backend that cannot honour it.
* Every packaged recipe that is not a detection one (``det_*``) states ``task:``. ``det_nano_smoke.yaml`` is a
  ``default_config_files`` entry for every run subcommand and jsonargparse merges per key, so a recipe omitting
  ``model.task`` inherits ``detect`` and trains a detector under its own name.

Examples:
    Command-line usage (exit status is the process return code)::

        $ python scripts/lint/audit_content_pins.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, so `scripts.lint` resolves under the hook
from scripts.lint._common import add_repo_root_argument, print_failure

#: Path of the suppression-free decoder, relative to the repository root.
TOPK_MODULE = Path("src/lucid_yolo/decode/topk_e2e.py")
#: Lower-cased substrings the suppression-free decoder may not contain.
TOPK_FORBIDDEN = ("nms", "box_iou", "torchvision")
#: Directory of the packaged training recipes, relative to the repository root.
CONFIGS_DIR = Path("src/lucid_yolo/configs")


def check_topk_is_suppression_free(repo_root: Path) -> list[str]:
    """Name every forbidden substring the top-k decoder's source contains.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     module = Path(tmp) / TOPK_MODULE
        ...     module.parent.mkdir(parents=True)
        ...     _ = module.write_text("from torchvision.ops import NMS\\n", encoding="utf-8")
        ...     check_topk_is_suppression_free(Path(tmp))
        ["src/lucid_yolo/decode/topk_e2e.py names 'nms'", "src/lucid_yolo/decode/topk_e2e.py names 'torchvision'"]
    """
    source = (repo_root / TOPK_MODULE).read_text(encoding="utf-8").lower()
    return [f"{TOPK_MODULE.as_posix()} names {word!r}" for word in TOPK_FORBIDDEN if word in source]


def check_configs_leave_determinism_to_cli(repo_root: Path) -> list[str]:
    """Name every packaged recipe that pins ``deterministic:``.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     configs = Path(tmp) / CONFIGS_DIR
        ...     configs.mkdir(parents=True)
        ...     _ = (configs / "det_x.yaml").write_text("trainer:\\n  deterministic: true\\n", encoding="utf-8")
        ...     check_configs_leave_determinism_to_cli(Path(tmp))
        ['src/lucid_yolo/configs/det_x.yaml pins trainer.deterministic']
    """
    configs = sorted((repo_root / CONFIGS_DIR).glob("*.yaml"))
    return [
        f"{(CONFIGS_DIR / path.name).as_posix()} pins trainer.deterministic"
        for path in configs
        if "deterministic:" in path.read_text(encoding="utf-8")
    ]


def check_non_detection_configs_state_task(repo_root: Path) -> list[str]:
    """Name every non-``det_`` packaged recipe that omits ``task:``, or report the directory missing.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     configs = Path(tmp) / CONFIGS_DIR
        ...     configs.mkdir(parents=True)
        ...     _ = (configs / "seg_x.yaml").write_text("model:\\n  num_classes: 2\\n", encoding="utf-8")
        ...     _ = (configs / "obb_x.yaml").write_text("model:\\n  task: obb\\n", encoding="utf-8")
        ...     check_non_detection_configs_state_task(Path(tmp))
        ['src/lucid_yolo/configs/seg_x.yaml omits model.task']
    """
    configs = sorted(p for p in (repo_root / CONFIGS_DIR).glob("*.yaml") if not p.name.startswith("det_"))
    if not configs:
        return [f"no non-detection recipe found under {CONFIGS_DIR.as_posix()}"]
    return [
        f"{(CONFIGS_DIR / path.name).as_posix()} omits model.task"
        for path in configs
        if "task:" not in path.read_text(encoding="utf-8")
    ]


_CHECKS = (
    check_topk_is_suppression_free,
    check_configs_leave_determinism_to_cli,
    check_non_detection_configs_state_task,
)


def find_violations(repo_root: Path) -> list[str]:
    """Run every content pin and collect their violation strings.

    A pinned file that is absent is itself a violation rather than a crash, so one
    missing file does not hide the other pins' findings.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     find_violations(Path(tmp))[0].startswith("check_topk_is_suppression_free:")
        True
    """
    violations = []
    for check in _CHECKS:
        try:
            violations.extend(check(repo_root))
        except FileNotFoundError as exc:
            violations.append(f"{check.__name__}: {exc}")
    return violations


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run every pin, and print a report.

    Args:
        argv: Command-line arguments; ``None`` uses ``sys.argv``.

    Returns:
        Process exit code: ``0`` clean, ``1`` when any pin is violated.
    """
    parser = argparse.ArgumentParser(description="Assert literal content pins on tracked files.")
    add_repo_root_argument(parser, "repository whose files to read")
    args = parser.parse_args(argv)

    violations = find_violations(args.repo_root)
    if violations:
        return print_failure("content-pins-audit", violations)
    print("content-pins-audit clean: every pinned file holds its claim")
    return 0


if __name__ == "__main__":
    sys.exit(main())
