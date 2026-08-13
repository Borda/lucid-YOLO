# SPDX-License-Identifier: Apache-2.0
"""``lucid-data`` — one command for everything that happens to a dataset (WP-096).

Three subcommands, in the order a tier run uses them::

    lucid-data download --data_root /data/coco --splits '[train,val]' --verify true
    lucid-data check --data_root /data/coco
    lucid-data build-tiles --root /data/dota --out /data/dota_tiles --overlap 512

They were three separate things before: a shipped ``lucid-download`` console script, an
unshipped ``scripts/check_data.py`` reachable only through ``make``, and an unshipped
``scripts/build_dota_tiles.py``. A remote tier run installs a wheel, so two thirds of its
own data pipeline were unavailable to it and the recipe named a build step nobody
installing the package could run.

Naming:
    Subcommands name the *operation* and a flag names the dataset — ``check --dataset``,
    ``build-tiles --source``. A command per dataset would grow a verb every time one is
    added and let each drift into its own flag spellings. ``build-tiles`` in particular
    is not ``build-dota-tiles``: what it writes is a plain COCO container with
    quadrilateral rings, and DOTA is the reader it happens to have.

    Flags and help come from the operation functions' signatures and Google docstrings
    (:func:`lucid_yolo.data.download.download_dataset`,
    :func:`lucid_yolo.data.check.check_dataset`,
    :func:`lucid_yolo.data.tiles.build_tiles`), so a flag cannot document itself
    differently from the function it sets.

What is deliberately *not* here:
    Neither check runs inside ``lucid-yolo fit``. :func:`~lucid_yolo.data.check.check_dataset`
    asserts the dataset's **published** totals, so wiring it into training would refuse a
    legitimate subset or smoke run at startup, and a preflight on every ``fit`` charges
    every run for a property of the disk that changes once. The dataset contract
    (AGENTS.md sec. 3) makes provisioning an explicit step; these are that step's tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jsonargparse import ArgumentParser

from lucid_yolo.data.check import check_dataset
from lucid_yolo.data.download import download_dataset
from lucid_yolo.data.tiles import build_tiles

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["SUBCOMMANDS", "build_parser", "main"]

#: Subcommand name to the function that implements it. The parser is built from these
#: signatures, so adding an operation is one entry here and one typed function.
SUBCOMMANDS: dict[str, Callable[..., int]] = {
    "download": download_dataset,
    "check": check_dataset,
    "build-tiles": build_tiles,
}


def build_parser() -> ArgumentParser:
    """Build the ``lucid-data`` parser from the subcommand functions' signatures.

    Returns:
        The configured parser.

    Examples:
        >>> parser = build_parser()
        >>> config = parser.parse_args(["check", "--data_root", "/data/coco"])
        >>> config.check.dataset
        'coco'
    """
    parser = ArgumentParser(prog="lucid-data", description="Download, validate and tile datasets.")
    subcommands = parser.add_subcommands(dest="command", required=True)
    for name, function in SUBCOMMANDS.items():
        # The summary line only: the full docstring would print its Examples section,
        # doctest directives and all, into the subcommand's help.
        subparser = ArgumentParser(description=_summary(function))
        subparser.add_function_arguments(function)
        subcommands.add_subcommand(name, subparser, help=_summary(function))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one ``lucid-data`` subcommand.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        The subcommand's exit code.

    Examples:
        >>> main(["check", "--data_root", "/nonexistent"])  # doctest: +ELLIPSIS
        check-data: /nonexistent
        ...
        1
    """
    config = build_parser().parse_args(argv)
    command = str(config.command)
    return SUBCOMMANDS[command](**config[command].as_dict())


def _summary(function: Callable[..., int]) -> str:
    """Return a function's one-line docstring summary, for the subcommand list."""
    return (function.__doc__ or "").strip().split("\n")[0]


if __name__ == "__main__":  # pragma: no cover - `python -m lucid_yolo.cli.data`
    raise SystemExit(main())
