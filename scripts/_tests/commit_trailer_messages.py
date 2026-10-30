# SPDX-License-Identifier: Apache-2.0
"""Build commit messages for the commit-trailer validator tests.

``test_check_commit_trailers.py`` needs a few dozen messages that differ in one trailer each; spelling every
one out as a string literal buries the trailer under test in boilerplate. ``conftest.py`` puts this directory
on ``sys.path``, which is how a test file reaches it.
"""

from __future__ import annotations


def make_message(
    subject: str = "fix(data): x",
    *,
    body: str | None = None,
    wp: str | None = "none",
    provenance: str = "R1",
    assumptions: str = "none",
    gate: str = "t",
) -> str:
    """Assemble a commit message from a subject and the four mandatory trailers.

    The defaults are the most common shape in the validator tests (a ``fix(data)`` subject, no body, ``WP: none``,
    ``Provenance: R1``, ``Assumptions: none``), so a case passes only the field it is about.

    Args:
        subject: Subject line.
        body: One body paragraph placed between subject and trailers; omitted when ``None``.
        wp: Value of the ``WP`` trailer; the whole trailer line is omitted when ``None``.
        provenance: Value of the ``Provenance`` trailer.
        assumptions: Value of the ``Assumptions`` trailer.
        gate: Value of the ``Gate`` trailer.

    Returns:
        The message, ending in a single newline.

    Examples:
        >>> make_message(wp="144").splitlines()[-4:]
        ['WP: 144', 'Provenance: R1', 'Assumptions: none', 'Gate: t']
        >>> "WP:" in make_message(wp=None)
        False
        >>> make_message("chore: x", body="text").splitlines()[:3]
        ['chore: x', '', 'text']
    """
    lines = [subject, ""]
    if body is not None:
        lines += [body, ""]
    if wp is not None:
        lines.append(f"WP: {wp}")
    lines += [f"Provenance: {provenance}", f"Assumptions: {assumptions}", f"Gate: {gate}"]
    return "\n".join(lines) + "\n"
