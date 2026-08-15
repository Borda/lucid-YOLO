# SPDX-License-Identifier: Apache-2.0
"""Dependency-license audit: reject copyleft licenses anywhere in the tree.

Scans every distribution installed in the active environment and fails when a
GPL-family (AGPL/GPL/LGPL) license indicator is found, per the project policy
that every dependency must be Apache-compatible, including transitive ones.

Three places are read, because neither what a distribution says about itself nor
what it says in the documents beside it is the whole story.
:func:`license_indicators` reads the ``License``, ``License-Expression`` and
``License ::`` classifier fields — what the package says about itself.
:func:`bundled_license_indicators` reads the ``License-File`` documents the wheel
ships alongside them, because a wheel may vendor a native library under a license
its metadata never mentions: ``shapely`` declares ``BSD 3-Clause`` and bundles GEOS
shared libraries under LGPLv2.1, which the metadata-only audit passed for as long
as it existed (found by WP-063, recorded as D15). :func:`bundled_binary_indicators`
reads the wheel's own file list, because a wheel may vendor a copyleft binary and
declare it **nowhere at all**: ``av`` declares ``BSD-3-Clause``, ships a
``licenses/LICENSE.txt`` whose text contains no GPL mention anywhere in it, and
ships ``av/.dylibs/libx264.165.dylib`` — x264 is GPL. Both of the first two checks
pass it clean (WP-109). Bundled findings carry their own allowlists, since a
vendored binary and a package's own license are different exposures and are
decided separately.

**The binary scan narrows the hole rather than closing it, and should be read that
way.** It matches shipped filenames against :data:`COPYLEFT_BINARIES`, a table of
native libraries whose names identify a known copyleft project, so it catches the
libraries that table lists and nothing else. A wheel vendoring a copyleft library
this file has never heard of, or shipping one under a name that does not announce
it, passes exactly as ``av`` did before the table existed. A green run is evidence
that no *listed* library is present, and is not evidence that no copyleft binary
is. The table is documentation as much as data for the same reason: each entry
records the license and where it was verified, so the next reader can argue with
it instead of trusting it.

The file list comes from each distribution's ``RECORD`` — what the wheel declared
it installed — and not from walking site-packages, which keeps the check to the
question it can answer and off the disk. A distribution installed without a
``RECORD`` therefore ships nothing as far as this check can see; every one in the
current environment has one.

Examples:
    Run against the active environment (exit 1 on violation)::

        python scripts/audit_licenses.py
"""

from __future__ import annotations

import re
import sys
from email.message import Message
from importlib import metadata
from typing import NamedTuple, cast

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

#: Filenames a shipped native library carries, in every spelling the four platforms
#: use: ``.so``/``.dylib``/``.dll``/``.pyd`` plain, and with the version tag either
#: platform appends (``libx264.165.dylib``, ``libx264.so.164``).
NATIVE_LIBRARY_SUFFIX = re.compile(r"\.(?:so|dylib|dll|pyd)(?:\.\d+(?:\.\d+)*)?$", re.IGNORECASE)

#: The other version spelling, joined to the name rather than separated from it
#: (``avcodec-61.dll``). Stripped so that one table entry matches every platform's
#: build of the same library.
LIBRARY_VERSION_TAG = re.compile(r"[-_]\d+$")

#: The hash ``auditwheel`` grafts onto a vendored library's name when it repairs a
#: manylinux wheel: the ``libgeos_c.so.1.19.2`` that macOS ships verbatim arrives on
#: Linux as ``libgeos_c-abcdd5fa.so.1.19.2``. Eight lowercase hex characters, uniformly
#: — checked against all 20 vendored libraries in the ``shapely`` and ``pillow``
#: manylinux wheels. Stripping it is not cosmetic: without it the table matches on macOS
#: and goes blind on Linux, which is the platform CI actually runs on, and the blindness
#: would present as a green gate. ``test_live_shapely_ships_the_geos_binaries_the_table_names``
#: is what fails if the convention ever changes.
AUDITWHEEL_GRAFT = re.compile(r"-[0-9a-f]{8}$")


class CopyleftLibrary(NamedTuple):
    """A native library whose filename identifies a known copyleft project.

    Attributes:
        license: SPDX-style identifier for the license the project ships under.
        note: Why the entry is here and where the license was verified — the table
            is read by people deciding whether a finding is real, so an entry that
            states only an identifier cannot be checked or argued with.
    """

    license: str
    note: str


#: The FFmpeg libraries, listed as a family because they are built and shipped as one.
FFMPEG_LIBRARIES = ("avcodec", "avdevice", "avfilter", "avformat", "avutil", "postproc", "swresample", "swscale")

#: Two licenses rather than one, because which of them applies is a build-time choice
#: the shipped filename does not record. Stated as a range instead of picking a side.
FFMPEG_LICENSE = "LGPL-2.1-or-later, GPL-2.0-or-later if built --enable-gpl"

#: Shared by every entry in :data:`FFMPEG_LIBRARIES`: one library of the family arriving
#: in a wheel means the rest did too, so eight separate justifications would be one fact
#: written eight times.
FFMPEG_NOTE = (
    "FFmpeg's own `LICENSE.md` states most of it is under the LGPL v2.1 or later, and that passing "
    "`--enable-gpl` to activate its optional GPL-covered parts changes FFmpeg's license to GPL v2+. Which "
    "of the two a given `libavcodec` is cannot be read off its filename, so the family is listed at "
    "its floor: both spellings are copyleft, and only the severity is unknown. The realistic route "
    "in is a wheel vendoring FFmpeg for video I/O, which is exactly what `av` is (WP-109)."
)

#: Native libraries whose *filename* identifies a known copyleft project, keyed by the
#: normalized stem :func:`library_stem` produces. Deliberately a named table and not a
#: heuristic: `torch` alone ships hundreds of binaries, and a check that failed on every
#: unrecognized one would be silenced within a week. The cost of that choice is stated in
#: the module docstring — this catches what it lists, and nothing else.
COPYLEFT_BINARIES: dict[str, CopyleftLibrary] = {
    "x264": CopyleftLibrary(
        "GPL-2.0-or-later",
        "VideoLAN's x264 page states it is released under the terms of the GNU GPL and is "
        "separately available under a commercial license. The dual offer is why the filename "
        "cannot be read as reassurance: the copy inside a wheel is the GPL one unless somebody "
        "bought the other, and nothing in the wheel says which. The library that motivated this "
        "check — `av` ships `av/.dylibs/libx264.165.dylib` while declaring BSD-3-Clause (WP-109).",
    ),
    "x265": CopyleftLibrary(
        "GPL-2.0-or-later",
        "The x265 source tree's own COPYING is the GNU GPL version 2 text, and FFmpeg's `LICENSE.md` "
        "names libx264, libx265 and libxvid together as the GPL-v2 externals `--enable-gpl` admits. "
        "Listed alongside x264 rather than after an incident of its own, because the two travel "
        "together in every FFmpeg build that carries either.",
    ),
    "mp3lame": CopyleftLibrary(
        "LGPL",
        "The LAME project's own page states the encoder is licensed under the LGPL and names no "
        "version, so neither does this entry — a version read off nothing would be the one part of "
        "the row a reader could not check. `av` ships it beside libx264 (WP-109). Weak copyleft is "
        "still copyleft under a policy that admits Apache-compatible licenses only, and D15 already "
        "treats LGPL binaries as findings.",
    ),
    "geos": CopyleftLibrary(
        "LGPL-2.1",
        "`shapely`'s own bundled `LICENSE_GEOS` opens with `License: LGPLv2.1` over the LGPL text, "
        "read from the installed wheel rather than from a project page — the same exposure D15 recorded, reached "
        "independently by a second route. It is present in this environment and allowlisted "
        "below, which is what makes the table's behaviour observable rather than hypothetical.",
    ),
    "geos_c": CopyleftLibrary(
        "LGPL-2.1",
        "GEOS's stable C API, shipped beside `libgeos` by the same wheel under the same license.",
    ),
    **dict.fromkeys(FFMPEG_LIBRARIES, CopyleftLibrary(FFMPEG_LICENSE, FFMPEG_NOTE)),
}

#: Vendored binaries that :data:`COPYLEFT_BINARIES` names and that are permitted anyway,
#: keyed by ``(distribution name lowercased, library stem)``.
#:
#: Deliberately keyed on the *pair* where :data:`BUNDLED_ALLOWLIST` is keyed on the package
#: alone. That file's own rule is that a package listed for one vendored library must not be
#: excused for every other one it ships; a package-keyed entry here would break exactly that
#: rule, since allowing shapely's GEOS would also wave through a libx264 it started shipping
#: tomorrow. Additions require a DECISIONS.md entry, as for both other allowlists.
BUNDLED_BINARY_ALLOWLIST: dict[tuple[str, str], str] = {
    ("shapely", "geos"): (
        "D15: the LGPLv2.1 GEOS build shapely's wheel carries, already decided there for the "
        "license-file check and reached here by the file list instead. A dev dependency-group "
        "package that src/ never imports, dynamically linked at test time, vendored into no "
        "artifact this project publishes."
    ),
    ("shapely", "geos_c"): "D15: GEOS's C API, shipped by the same wheel under the same decision as `geos`.",
}


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


def library_stem(filename: str) -> str | None:
    """Reduce a shipped filename to the native library it identifies.

    Matching is on the stem rather than the whole filename because the same library
    is spelled several ways across platforms and rebuilt under a new version tag every
    release: ``libx264.165.dylib``, ``libx264.so.164`` and ``x264-165.dll`` are one
    entry in :data:`COPYLEFT_BINARIES`, not three that go stale. The wheel repair tools
    add a spelling of their own — see :data:`AUDITWHEEL_GRAFT`, which is the difference
    between this table working on Linux and only appearing to.

    Args:
        filename: The base name of a file a distribution ships, with no directory part.

    Returns:
        The lowercased stem with any ``lib`` prefix, version tag and repair-tool hash
        removed, or ``None`` when the file is not a native library, or when the stem is
        empty (``lib.cpython-311-darwin.so``, which shapely ships, is a Python extension
        module named ``lib`` rather than a library called nothing).

    Examples:
        >>> library_stem("libx264.165.dylib")
        'x264'
        >>> library_stem("libx264.so.164")
        'x264'
        >>> library_stem("libgeos_c.1.19.2.dylib")  # as macOS ships it
        'geos_c'
        >>> library_stem("libgeos_c-abcdd5fa.so.1.19.2")  # the same library, manylinux
        'geos_c'
        >>> library_stem("__init__.py") is None
        True
    """
    if not NATIVE_LIBRARY_SUFFIX.search(filename):
        return None
    stem = AUDITWHEEL_GRAFT.sub("", filename.lower().split(".", 1)[0])
    return LIBRARY_VERSION_TAG.sub("", stem).removeprefix("lib") or None


def _shipped_libraries(dist: metadata.Distribution) -> list[tuple[str, str]]:
    """Return ``(path, library stem)`` for every native library a distribution ships.

    Reads the distribution's ``RECORD`` through ``Distribution.files``, so what is
    scanned is what the wheel declared it installed. A distribution without a
    ``RECORD`` reports no files and is invisible here.
    """
    found = []
    for path in dist.files or ():
        stem = library_stem(path.name)
        if stem is not None:
            found.append((str(path), stem))
    return found


def _copyleft_libraries(dist: metadata.Distribution) -> list[tuple[str, str]]:
    """Return ``(library stem, description)`` for each known-copyleft binary shipped.

    The single place a binary finding is formatted, so the audit's report and its
    allowlist decision cannot describe the same file two different ways.
    """
    found = []
    for path, stem in _shipped_libraries(dist):
        entry = COPYLEFT_BINARIES.get(stem)
        if entry is not None:
            found.append((stem, f"{path}: {stem} ({entry.license})"))
    return found


def bundled_binary_indicators(dist: metadata.Distribution) -> list[str]:
    """Collect the known-copyleft native libraries a distribution ships.

    The third surface, reading neither the metadata fields nor the license documents
    but the wheel's own file list: a vendored binary can arrive with no license
    paperwork of any kind, which is how a GPL x264 passes both other checks.

    Args:
        dist: An installed distribution as yielded by ``metadata.distributions()``.

    Returns:
        One ``"<path>: <library> (<license>)"`` string per shipped file whose name
        matches :data:`COPYLEFT_BINARIES`, allowlist notwithstanding — the allowlist
        is applied by :func:`find_binary_violations`, so this stays a report of what
        is actually there. Empty for a distribution shipping no listed library, which
        is nearly all of them.

    Examples:
        >>> declared = bundled_binary_indicators(metadata.distribution("shapely"))
        >>> any("geos" in entry for entry in declared)  # the vendored GEOS binaries
        True
    """
    return [description for _, description in _copyleft_libraries(dist)]


def find_binary_violations(dists: list[metadata.Distribution]) -> list[tuple[str, str]]:
    """Return (distribution name, offending shipped binary) pairs.

    The counterpart of :func:`find_bundled_violations` for a vendored library that
    documents itself nowhere: the wheel's declared license is permissive, the license
    files it ships say nothing about it, and only the filename gives it away.

    Args:
        dists: Distributions to audit.

    Returns:
        One tuple per violating distribution — the first unallowed library each ships,
        since the resolution is dropping the dependency rather than deleting a file.
        Excludes ``(distribution, library)`` pairs in :data:`BUNDLED_BINARY_ALLOWLIST`.

    Examples:
        >>> find_binary_violations(list(metadata.distributions()))
        []
    """
    violations = []
    for dist in dists:
        name = cast(Message, dist.metadata).get("Name") or "<unknown>"
        for stem, description in _copyleft_libraries(dist):
            if (name.lower(), stem) in BUNDLED_BINARY_ALLOWLIST:
                continue
            violations.append((name, description))
            break
    return violations


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
    binaries = find_binary_violations(dists)
    if declared or bundled or binaries:
        print("LICENSE AUDIT FAILED — copyleft licenses found:")
        for name, field in sorted(declared):
            print(f"  {name}: {field}")
        for name, field in sorted(bundled):
            print(f"  {name} (bundled): {field}")
        for name, field in sorted(binaries):
            print(f"  {name} (binary): {field}")
        return 1
    # The binary count is reported rather than kept internal: a scan that silently
    # stopped seeing files would otherwise pass exactly as loudly as a clean tree.
    shipped = sum(len(_shipped_libraries(dist)) for dist in dists)
    print(
        f"license audit clean: {len(dists)} distributions, {shipped} shipped binaries, "
        "no GPL-family licenses declared, bundled or shipped"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
