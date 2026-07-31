# SPDX-License-Identifier: Apache-2.0
"""Dependency-license audit: reject copyleft licenses anywhere in the tree.

Scans every distribution installed in the active environment and fails when a
GPL-family (AGPL/GPL/LGPL) license indicator is found, per the project policy
that every dependency must be Apache-compatible, including transitive ones.

Examples:
    Run against the active environment (exit 1 on violation)::

        python scripts/audit_licenses.py
"""

from __future__ import annotations

import re
import sys
from email.message import Message
from importlib import metadata
from typing import cast

#: Matches GPL-family identifiers (AGPL-3.0, GPLv2, LGPL, "GNU General Public License").
COPYLEFT_PATTERN = re.compile(r"\b(?:[AL]?GPL|GNU (?:Affero |Lesser )?General Public License)", re.IGNORECASE)

#: Distributions exempt from the copyleft check, with the reason documented.
#: Empty by design — any addition requires a DECISIONS.md entry first.
ALLOWLIST: dict[str, str] = {}

#: Free-text ``License`` fields longer than this are full license prose (some
#: packages embed third-party license texts that mention GPL), not an SPDX-style
#: identifier — they are skipped in favor of ``License-Expression`` and classifiers.
MAX_IDENTIFIER_LEN = 100


def license_indicators(dist: metadata.Distribution) -> list[str]:
    """Collect every license-bearing metadata field of a distribution.

    Args:
        dist: An installed distribution as yielded by ``metadata.distributions()``.

    Returns:
        Non-empty license strings from the ``License``, ``License-Expression``,
        and license-related ``Classifier`` fields.

    Examples:
        >>> dist = metadata.distribution("pytest")
        >>> any("MIT" in field for field in license_indicators(dist))
        True
    """
    meta = cast(Message, dist.metadata)
    license_field = meta.get("License") or ""
    if len(license_field) > MAX_IDENTIFIER_LEN:
        license_field = ""
    fields = [license_field, meta.get("License-Expression") or ""]
    fields += [value for value in meta.get_all("Classifier") or [] if value.startswith("License ::")]
    return [field for field in fields if field.strip()]


def find_copyleft_violations(dists: list[metadata.Distribution]) -> list[tuple[str, str]]:
    """Return (distribution name, offending license string) pairs.

    Args:
        dists: Distributions to audit.

    Returns:
        One tuple per violating distribution; empty when the tree is clean.

    Examples:
        >>> find_copyleft_violations(list(metadata.distributions()))
        []
    """
    violations = []
    for dist in dists:
        name = cast(Message, dist.metadata).get("Name") or "<unknown>"
        if name.lower() in {allowed.lower() for allowed in ALLOWLIST}:
            continue
        for field in license_indicators(dist):
            if COPYLEFT_PATTERN.search(field):
                violations.append((name, field))
                break
    return violations


def main() -> int:
    """Audit the active environment; print a verdict and return the exit code."""
    dists = list(metadata.distributions())
    violations = find_copyleft_violations(dists)
    if violations:
        print("LICENSE AUDIT FAILED — copyleft licenses found:")
        for name, field in sorted(violations):
            print(f"  {name}: {field}")
        return 1
    print(f"license audit clean: {len(dists)} distributions, no GPL-family licenses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
