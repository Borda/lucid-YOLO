# SPDX-License-Identifier: Apache-2.0
"""Dependency-license audit: reject copyleft licenses anywhere in the tree.

Scans every distribution installed in the active environment and fails when a
GPL-family (AGPL/GPL/LGPL) license indicator is found, per the project policy
that every dependency must be Apache-compatible, including transitive ones.

Four surfaces are read, because neither what a distribution says about itself nor
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

The fourth surface is the **absence** of a license, added by WP-115.
:func:`unreadable_license_reason` reports a distribution none of the three checks
above can read at all: the copyleft checks match a forbidden pattern against a
declaration, so a distribution declaring nothing matches nothing and passes in the
same run that prints a clean verdict. That inverts the reading rule the bundled
check follows. There, an absent or unreadable license document is not evidence of a
copyleft one and is skipped; here, an absent license *is* the finding, because what
cannot be read cannot be cleared. When no field declares anything, the shipped
``License-File`` documents are identified by their own opening prose against
:data:`PERMISSIVE_TEXTS`, and a document matching nothing in that table stays a
finding rather than being given the benefit of the doubt.

That fourth check is the one graded by tier (:func:`dependency_tiers`), and it is
the only one. A package in the ``[project.dependencies]`` closure is republished in
this project's own wheel metadata and installed by everyone who installs it, so an
unreadable license there is a failure; a package reachable only through the ``dev``
or ``docs`` dependency groups appears in no wheel metadata, is imported by ``src/``
never, and is vendored into no published artifact, so it is reported and the run
still passes. A package reachable from both takes the stricter tier, and so does one
the resolver cannot attribute at all — a gap in the walk must fail loudly rather
than quietly demote a shipped dependency. The copyleft checks are tier-blind and
stay that way: an AGPL dependency is a failure wherever it sits, and a distribution
whose only license document is copyleft prose is a failure for the same reason,
regardless of which group reached it.

**The examples below assert nothing about what is installed**, and neither does the test
suite (WP-115b). This script is the live check — it runs against the real environment as
a commit-time hook, which is the whole point of it. A *test* that made the same assertion
would fail for anyone whose environment holds something this repository never named, which
is a property of their machine rather than a defect in this file.

Examples:
    Run against the active environment (exit 1 on violation)::

        python scripts/audit_licenses.py
"""

from __future__ import annotations

import re
import sys
import tomllib
from collections.abc import Mapping
from email.message import Message
from importlib import metadata
from pathlib import Path
from typing import NamedTuple, cast

from packaging.requirements import Requirement
from packaging.utils import NormalizedName, canonicalize_name

#: The project's own metadata, located from this file rather than from installed
#: metadata: the tiers are a property of what the repository declares, and the docs
#: CI job installs the site toolchain without installing the project at all.
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

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

#: Distributions exempt from the *unreadable-license* check, with the reason. A fourth
#: allowlist and a fourth exposure: this one excuses a package whose license cannot be
#: read at all, which is neither a copyleft declaration (:data:`ALLOWLIST`) nor a
#: vendored copyleft library (:data:`BUNDLED_ALLOWLIST`, :data:`BUNDLED_BINARY_ALLOWLIST`).
#: Additions require a DECISIONS.md entry first, exactly as for the other three.
UNREADABLE_ALLOWLIST: dict[str, str] = {
    "cuda-toolkit": (
        "D17: an NVIDIA meta-package whose wheel dist-info holds only METADATA, WHEEL and "
        "RECORD — no License field, no License-Expression, no classifier, no license document "
        "of any kind, verified from the wheel itself. Its every requirement is extras-gated, so "
        "a bare install pulls nothing. Exempt because CUDA is the accelerator environment rather "
        "than a dependency of this project: `pyproject.toml` declares no CUDA package, the "
        "project redistributes none of the toolkit, and it publishes no trained weights (D14). "
        "The exemption is the metadata gap, not a licence finding — the NVIDIA CUDA EULA is "
        "proprietary and this row states that rather than implying otherwise."
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

#: The tier whose packages this project republishes: everything reachable from
#: ``[project.dependencies]`` lands in the wheel's own ``Requires-Dist`` and is
#: installed by every user. An unreadable license here is a failure.
TIER_SHIPPED = "base"

#: The dependency groups that ship nowhere. PEP 735 groups appear in no wheel
#: metadata at all, so a package reached only through one of them is a tool this
#: repository runs, not a dependency anyone inherits. Reported, not fatal.
TIER_GROUPS = ("dev", "docs")

#: How many lines of a license document :func:`recognize_license_text` reads. A
#: license names itself in its own header; further down, Apache-2.0's appendix
#: quotes a boilerplate notice and the GPL texts quote each other, so a recognizer
#: reading the whole body would identify documents by what they cite.
LICENSE_HEADER_LINES = 15


class LicenseText(NamedTuple):
    """A permissive license identified by a distinctive phrase in its own opening lines.

    Attributes:
        license: SPDX-style identifier for what the phrase identifies.
        note: Why the phrase is distinctive, and what it must not also match — the
            table decides whether a package with no license field passes, so an
            entry that states only a phrase cannot be argued with.
    """

    license: str
    note: str


#: Opening phrases that identify a permissive license text, keyed by the lowercased
#: phrase :func:`recognize_license_text` searches for. This table is reached only when
#: a distribution declares no license field at all, and identification *is* the verdict:
#: the copyleft pattern is deliberately not run over a recognized text as a second
#: opinion, because MPL-2.0 names the GNU General Public License in its own
#: Secondary-License definitions and would self-flag a license this environment already
#: carries through `certifi` and `pathspec`. Anything unrecognized stays a finding, so
#: the cost of a missing entry is a false alarm rather than a silent pass.
PERMISSIVE_TEXTS: dict[str, LicenseText] = {
    "apache license": LicenseText(
        "Apache-2.0",
        "The first line of the Apache-2.0 text, above `Version 2.0, January 2004`. Apache-1.1 "
        "opens `The Apache Software License, Version 1.1` and does not contain this phrase, so "
        "the entry does not quietly accept the older license. `faster-coco-eval` is the live "
        "case: PEP 639 metadata with no License, no License-Expression and no classifier, "
        "shipping `licenses/LICENSE` holding this text (WP-115).",
    ),
    "permission is hereby granted, free of charge": LicenseText(
        "MIT",
        "The opening grant of the MIT text, shared verbatim with the ISC-style and X11 variants "
        "that carry the same permissions. Named MIT because that is what the phrase identifies "
        "in practice; the distinction between the variants does not change the verdict.",
    ),
    "redistribution and use in source and binary forms": LicenseText(
        "BSD-2-Clause or BSD-3-Clause",
        "The first clause of every BSD text. Which of the two it is depends on whether a "
        "non-endorsement clause follows, which this recognizer does not read — both are "
        "permissive, so the ambiguity does not reach the verdict and is stated rather than "
        "resolved by guessing.",
    ),
    "permission to use, copy, modify, and/or distribute this software": LicenseText(
        "ISC",
        "The ISC grant, which differs from MIT's opening enough not to match the entry above.",
    ),
    "mozilla public license version 2.0": LicenseText(
        "MPL-2.0",
        "Weak file-level copyleft, not what this policy bans (AGPL/GPL/LGPL) and already present "
        "through `certifi` and `pathspec`. Listed so that an MPL document is identified rather "
        "than reported as unreadable — and listed by its header, since matching its body would "
        "hit the Secondary-License clause that names the GPL.",
    ),
    "python software foundation license": LicenseText(
        "PSF-2.0",
        "The PSF license header, carried by CPython-derived code that some wheels vendor.",
    ),
}

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
        >>> callable(license_indicators)  # a real call needs an installed distribution
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
        >>> callable(bundled_license_indicators)  # a real call needs an installed distribution
        True
    """
    return [
        f"{name}: {match.group(1).strip()}"
        for name, text in _license_documents(dist)
        for match in BUNDLED_DECLARATION.finditer(text)
    ]


def _license_documents(dist: metadata.Distribution) -> list[tuple[str, str]]:
    """Return ``(filename, text)`` for every ``License-File`` the distribution ships and this can read.

    Wheels built to PEP 639 file the documents under ``licenses/``; older ones drop
    them beside ``METADATA``. Both spellings are tried before giving up, and a document
    that is absent or unreadable is skipped — what that absence *means* is the caller's
    to decide, and the two callers decide it oppositely (see the module docstring).
    """
    meta = cast(Message, dist.metadata)
    found: list[tuple[str, str]] = []
    for name in meta.get_all("License-File") or []:
        for candidate in (f"licenses/{name}", name):
            try:
                text = dist.read_text(candidate)
            except (OSError, ValueError):
                text = None
            if text:
                found.append((name, text))
                break
    return found


def recognize_license_text(text: str) -> str | None:
    """Identify a license from the opening lines of its own text.

    Reached only for a distribution that declares no license field at all, where
    identification is the verdict: an unrecognized document is a finding, so this
    never has to decide that a text is *forbidden* — only whether it is one of the
    permissive texts :data:`PERMISSIVE_TEXTS` names.

    Args:
        text: The full contents of a shipped license document.

    Returns:
        The SPDX-style identifier of the first entry whose phrase appears in the
        document's first :data:`LICENSE_HEADER_LINES` lines, or ``None`` when no
        entry matches — including for every copyleft text, none of which is listed.

    Examples:
        >>> recognize_license_text("                    Apache License\\n   Version 2.0, January 2004\\n")
        'Apache-2.0'
        >>> recognize_license_text("MIT License\\n\\nPermission is hereby granted, free of charge, to any")
        'MIT'
        >>> recognize_license_text("GNU GENERAL PUBLIC LICENSE\\nVersion 3, 29 June 2007\\n") is None
        True
    """
    header = "\n".join(text.splitlines()[:LICENSE_HEADER_LINES]).lower()
    for phrase, entry in PERMISSIVE_TEXTS.items():
        if phrase in header:
            return entry.license
    return None


def unreadable_license_reason(dist: metadata.Distribution) -> str | None:
    """Say why a distribution's license cannot be read, or ``None`` when it can.

    The check the three copyleft surfaces cannot perform, because each of them
    matches a forbidden pattern against a declaration and a distribution declaring
    nothing matches nothing.

    Args:
        dist: An installed distribution as yielded by ``metadata.distributions()``.

    Returns:
        A sentence naming what was missing or which document could not be
        identified, or ``None`` when a license field exists or a shipped document
        was recognized.

    Examples:
        >>> callable(unreadable_license_reason)  # a real call needs an installed distribution
        True
    """
    if license_indicators(dist):
        return None
    documents = _license_documents(dist)
    if not documents:
        return "declares no License, License-Expression or License :: classifier, and ships no License-File"
    unidentified = [name for name, text in documents if recognize_license_text(text) is None]
    if len(unidentified) < len(documents):
        return None
    return (
        "declares no License, License-Expression or License :: classifier, and the "
        f"{', '.join(unidentified)} it ships matches no permissive license text this audit recognizes"
    )


def _ships_copyleft_prose(dist: metadata.Distribution) -> bool:
    """Whether an unidentified license document names a GPL-family license in its header.

    Only ever consulted for a distribution :func:`unreadable_license_reason` already
    reported, so no recognized permissive text reaches this — which matters, because
    MPL-2.0's body names the GNU General Public License and would flag itself. The
    header is what a license uses to name itself.
    """
    return any(
        COPYLEFT_PATTERN.search("\n".join(text.splitlines()[:LICENSE_HEADER_LINES]))
        for _, text in _license_documents(dist)
    )


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
        >>> callable(bundled_binary_indicators)  # a real call needs an installed distribution
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
        >>> find_binary_violations([])
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
        >>> find_bundled_violations([])
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


def _root_requirements(pyproject: Path) -> dict[str, list[str]]:
    """Read the requirement strings each tier starts from.

    Raises:
        ValueError: If a dependency group uses a ``{include-group = ...}`` entry.
            The walk below would silently drop it and under-attribute everything it
            reaches, which downgrades a package's tier without saying so; this
            repository uses no such entry today, and the check is what makes adding
            one a visible decision rather than a silent hole.
    """
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    groups = data.get("dependency-groups", {})
    roots = {TIER_SHIPPED: list(data["project"]["dependencies"])}
    for tier in TIER_GROUPS:
        entries = groups.get(tier, [])
        if not all(isinstance(entry, str) for entry in entries):
            raise ValueError(f"dependency group {tier!r} uses an include-group entry this walk cannot follow")
        roots[tier] = list(entries)
    return roots


def _requested(raw: str) -> tuple[NormalizedName, frozenset[str]]:
    """Normalize one requirement string to ``(canonical name, requested extras)``."""
    requirement = Requirement(raw)
    return canonicalize_name(requirement.name), frozenset(requirement.extras)


def _reachable(roots: list[str], installed: Mapping[NormalizedName, metadata.Distribution]) -> set[NormalizedName]:
    """Walk the installed requirement graph from a tier's roots.

    Extras are carried through the walk rather than dropped: a requirement gated on
    ``extra == "foo"`` is reached exactly when the parent was requested with ``foo``,
    and dropping every extra-markered requirement instead left 19 of 105 distributions
    unattributed when this was first measured.
    """
    seen: set[NormalizedName] = set()
    queue = [_requested(raw) for raw in roots]
    visited: set[tuple[NormalizedName, frozenset[str]]] = set()
    while queue:
        name, extras = queue.pop()
        if (name, extras) in visited:
            continue
        visited.add((name, extras))
        seen.add(name)
        dist = installed.get(name)
        if dist is None:
            continue
        contexts = [{"extra": extra} for extra in (extras or {""})]
        for raw in cast(Message, dist.metadata).get_all("Requires-Dist") or []:
            requirement = Requirement(raw)
            if requirement.marker and not any(requirement.marker.evaluate(context) for context in contexts):
                continue
            queue.append((canonicalize_name(requirement.name), frozenset(requirement.extras)))
    return seen


def dependency_tiers(dists: list[metadata.Distribution], pyproject: Path = PYPROJECT) -> dict[str, str]:
    """Map each installed distribution to the tier that reaches it first.

    Args:
        dists: Distributions to attribute.
        pyproject: The project metadata declaring the roots of each tier.

    Returns:
        Canonical distribution name to tier name. Only distributions a group reaches
        *without* the shipped closure reaching them appear under a group's tier, so a
        package both depend on is attributed to :data:`TIER_SHIPPED`. Distributions
        the walk never reaches are absent, and callers treat absence as
        :data:`TIER_SHIPPED` — a gap in the resolver must fail loudly rather than
        quietly demote something the wheel republishes.

    Examples:
        >>> dependency_tiers([])  # nothing installed, nothing to attribute
        {}
    """
    installed = {canonicalize_name(cast(Message, dist.metadata).get("Name") or ""): dist for dist in dists}
    roots = _root_requirements(pyproject)
    tiers = dict.fromkeys(_reachable(roots[TIER_SHIPPED], installed), TIER_SHIPPED)
    for tier in TIER_GROUPS:
        for name in _reachable(roots[tier], installed):
            tiers.setdefault(name, tier)
    return {name: tier for name, tier in tiers.items() if name in installed}


def find_unreadable_licenses(
    dists: list[metadata.Distribution], tiers: dict[str, str]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split distributions with no readable license into failures and flags.

    Args:
        dists: Distributions to audit.
        tiers: Attribution from :func:`dependency_tiers`; a name it omits is treated
            as :data:`TIER_SHIPPED`.

    Returns:
        ``(failures, flags)``, each a list of ``(distribution name, reason)``, excluding
        anything in :data:`UNREADABLE_ALLOWLIST`. A distribution is a failure when the
        shipped closure reaches it, when no tier reaches it at all, or when the only
        license document it ships is copyleft prose — the last case ignores the tier,
        because an AGPL dependency is a failure wherever it sits. The reason carries the
        tier, and an unattributed package says so rather than reading as a base one.

    Examples:
        >>> find_unreadable_licenses([], {})
        ([], [])
    """
    allowed = {name.lower() for name in UNREADABLE_ALLOWLIST}
    failures, flags = [], []
    for dist in dists:
        name = cast(Message, dist.metadata).get("Name") or "<unknown>"
        reason = unreadable_license_reason(dist)
        if reason is None or canonicalize_name(name) in allowed:
            continue
        attributed = tiers.get(canonicalize_name(name))
        # Spelled apart from a genuine base attribution, because the two are the same
        # verdict for opposite reasons: one says the wheel republishes this package, the
        # other says the walk never found it and is failing loudly rather than guessing.
        tier = attributed or f"unattributed, treated as {TIER_SHIPPED}"
        if attributed == TIER_SHIPPED or attributed is None or _ships_copyleft_prose(dist):
            failures.append((name, f"{reason} ({tier})"))
        else:
            flags.append((name, f"{reason} ({tier})"))
    return failures, flags


def find_copyleft_violations(dists: list[metadata.Distribution]) -> list[tuple[str, str]]:
    """Return (distribution name, offending license string) pairs.

    Args:
        dists: Distributions to audit.

    Returns:
        One tuple per violating distribution; empty when the tree is clean.

    Examples:
        >>> find_copyleft_violations([])
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


def _report_copyleft(
    declared: list[tuple[str, str]], bundled: list[tuple[str, str]], binaries: list[tuple[str, str]]
) -> None:
    """Print the copyleft findings under a header that describes them."""
    print("LICENSE AUDIT FAILED — copyleft licenses found:")
    for name, field in sorted(declared):
        print(f"  {name}: {field}")
    for name, field in sorted(bundled):
        print(f"  {name} (bundled): {field}")
    for name, field in sorted(binaries):
        print(f"  {name} (binary): {field}")


def main() -> int:
    """Audit the active environment; print a verdict and return the exit code."""
    dists = list(metadata.distributions())
    declared = find_copyleft_violations(dists)
    bundled = find_bundled_violations(dists)
    binaries = find_binary_violations(dists)
    unreadable, flagged = find_unreadable_licenses(dists, dependency_tiers(dists))
    # Printed before any verdict, and on a passing run too: a package this audit
    # cannot read is worth seeing whether or not its tier makes it fatal.
    for name, reason in sorted(flagged):
        print(f"license audit flag: {name} {reason} — not shipped, so not a failure")
    if declared or bundled or binaries:
        _report_copyleft(declared, bundled, binaries)
        return 1
    if unreadable:
        # A separate header, because filing these under the copyleft one would report
        # a license nobody could read as a license somebody read and rejected.
        print("LICENSE AUDIT FAILED — licenses that could not be read:")
        for name, reason in sorted(unreadable):
            print(f"  {name}: {reason}")
        return 1
    # The binary count is reported rather than kept internal: a scan that silently
    # stopped seeing files would otherwise pass exactly as loudly as a clean tree.
    shipped = sum(len(_shipped_libraries(dist)) for dist in dists)
    print(
        f"license audit clean: {len(dists)} distributions, {shipped} shipped binaries, "
        "no GPL-family licenses declared, bundled or shipped, none unreadable"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
