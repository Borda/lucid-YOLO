# SPDX-License-Identifier: Apache-2.0
"""Dependency-license audit: reject copyleft licenses anywhere in the tree.

Scans every distribution installed in the active environment and fails when a
GPL-family (AGPL/GPL/LGPL) license indicator is found, per the project policy
that every dependency must be Apache-compatible, including transitive ones.

Two places are read, because a distribution's own metadata is not the whole
story. :func:`license_indicators` reads the ``License``, ``License-Expression``
and ``License ::`` classifier fields — what the package says about itself — and
:func:`bundled_license_indicators` reads the ``License-File`` documents the wheel
ships alongside them. A wheel may vendor a native library under a license its
metadata never mentions: ``shapely`` declares ``BSD 3-Clause`` and bundles GEOS
shared libraries under LGPLv2.1, which the metadata-only audit passed for as long
as it existed (found by WP-063, recorded as D15). Bundled findings carry their own
allowlist, since a vendored binary and a package's own license are different
exposures and are decided separately.

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

#: Distributions whose *bundled* license files may name a copyleft license, with
#: the reason. Separate from :data:`ALLOWLIST`, which exempts a package's own
#: license: vendoring an LGPL binary and being LGPL are different exposures.
#: Additions require a DECISIONS.md entry, exactly as for :data:`ALLOWLIST`.
BUNDLED_ALLOWLIST: dict[str, str] = {
    "numpy": (
        "D15: BSD-3-Clause itself; its wheel bundles libquadmath under LGPL-2.1-or-later, "
        "dynamically linked, as part of the GCC runtime every scientific-Python wheel carries. "
        "The sibling libgfortran/libgcc are covered by LICENSE_EXCEPTIONS rather than by this "
        "entry, so the exemption here is libquadmath alone."
    ),
    "shapely": (
        "D15: BSD-3-Clause itself; its wheel bundles GEOS shared libraries under LGPLv2.1. "
        "A dev dependency-group package only — src/ never imports it, GEOS is dynamically "
        "linked at test time, and neither is vendored here or present in any published artifact. "
        "It provides A24's independent polygon oracle."
    ),
}

#: License exceptions that make a GPL-family declaration permissive by construction,
#: so a match on one is not a finding for **any** package. The GCC Runtime Library
#: Exception exists precisely to let GPL-3 runtime objects (``libgfortran``,
#: ``libgcc``) be linked into programs under any license; treating it as copyleft
#: would flag every wheel built by gcc and teach the reader to ignore the gate.
#: Recognized by expression, not by package, so a new dependency carrying the same
#: exception needs no allowlist entry (D15).
LICENSE_EXCEPTIONS = re.compile(r"WITH\s+GCC-exception-3\.1", re.IGNORECASE)

#: Free-text ``License`` fields longer than this are full license prose (some
#: packages embed third-party license texts that mention GPL), not an SPDX-style
#: identifier — they are skipped in favor of ``License-Expression`` and classifiers.
MAX_IDENTIFIER_LEN = 100

#: Lines in a bundled license file that *declare* a license, as opposed to the
#: license prose itself. A full LGPL text mentions "GPL" on dozens of lines, so
#: matching the body would report every bundled copy of any license; the wheels
#: that vendor a binary announce it in a ``License: <name>`` or ``Name: <lib>``
#: header block instead (the pattern numpy, scipy and shapely all follow).
BUNDLED_DECLARATION = re.compile(r"^\s*(?:License|Licence)\s*:\s*(\S.*)$", re.IGNORECASE | re.MULTILINE)


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


def bundled_license_indicators(dist: metadata.Distribution) -> list[str]:
    """Collect license declarations from the license files a distribution ships.

    Reads each ``License-File`` the metadata names and returns the ``License: ...``
    declaration lines inside it, not the license prose — a bundled copy of the LGPL
    names "GPL" throughout its own text, so matching the body would flag every wheel
    that ships any license file at all.

    Args:
        dist: An installed distribution as yielded by ``metadata.distributions()``.

    Returns:
        One ``"<file>: <declared license>"`` string per declaration found, across
        every bundled license file. Files that are absent or unreadable are skipped:
        a missing document is not evidence of a copyleft one.

    Examples:
        >>> declared = bundled_license_indicators(metadata.distribution("shapely"))
        >>> any("LGPL" in entry for entry in declared)  # the vendored GEOS binaries
        True
    """
    meta = cast(Message, dist.metadata)
    found: list[str] = []
    for name in meta.get_all("License-File") or []:
        # Wheels built to PEP 639 file the documents under ``licenses/``; older ones
        # drop them beside METADATA. Both spellings are tried before giving up.
        for candidate in (f"licenses/{name}", name):
            try:
                text = dist.read_text(candidate)
            except (OSError, ValueError):
                text = None
            if text:
                found += [f"{name}: {match.group(1).strip()}" for match in BUNDLED_DECLARATION.finditer(text)]
                break
    return found


def find_bundled_violations(dists: list[metadata.Distribution]) -> list[tuple[str, str]]:
    """Return (distribution name, offending bundled declaration) pairs.

    The counterpart of :func:`find_copyleft_violations` for vendored code: a wheel
    may declare a permissive license for itself and ship a native library under a
    copyleft one, which the metadata fields never mention.

    Args:
        dists: Distributions to audit.

    Returns:
        One tuple per violating distribution, excluding those in
        :data:`BUNDLED_ALLOWLIST`; empty when nothing unallowed is vendored.

    Examples:
        >>> find_bundled_violations(list(metadata.distributions()))
        []
    """
    allowed = {name.lower() for name in BUNDLED_ALLOWLIST}
    violations = []
    for dist in dists:
        name = cast(Message, dist.metadata).get("Name") or "<unknown>"
        for declaration in bundled_license_indicators(dist):
            # The exception is tested before the allowlist, not after, so that a
            # package listed for one vendored library is not silently excused for
            # every other one it ships. numpy is the live case: libgfortran passes
            # here on the GCC exception, and only libquadmath reaches the allowlist.
            if LICENSE_EXCEPTIONS.search(declaration):
                continue
            if COPYLEFT_PATTERN.search(declaration) and name.lower() not in allowed:
                violations.append((name, declaration))
                break
    return violations


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
    declared = find_copyleft_violations(dists)
    bundled = find_bundled_violations(dists)
    if declared or bundled:
        print("LICENSE AUDIT FAILED — copyleft licenses found:")
        for name, field in sorted(declared):
            print(f"  {name}: {field}")
        for name, field in sorted(bundled):
            print(f"  {name} (bundled): {field}")
        return 1
    print(f"license audit clean: {len(dists)} distributions, no GPL-family licenses declared or bundled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
