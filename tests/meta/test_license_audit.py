# SPDX-License-Identifier: Apache-2.0
"""Meta tests: dependency-license audit rejects copyleft (WP-004, WP-063).

Guards the Apache-compatibility policy: the copyleft pattern must catch the
GPL family and spare permissive licenses, and ``find_copyleft_violations`` must
flag an AGPL-classified distribution while leaving the live (clean) environment
untouched.

A second surface is guarded since WP-063: what a wheel *vendors*, as opposed to
what its metadata declares. ``shapely`` says ``BSD 3-Clause`` and ships GEOS under
LGPLv2.1; the metadata-only audit passed on it for as long as it existed. These
tests pin the three rules that make the bundled check meaningful — declarations are
read, prose is not; the GCC Runtime Library Exception is permissive for every
package; and the per-package allowlist excuses only what it names (D15).

A third surface is guarded since WP-109: what a wheel *ships*, when it documents the
vendored library nowhere at all. ``av`` declares ``BSD-3-Clause``, ships a license
document with no GPL mention in it, and ships a GPL x264 binary — both older checks
pass it. The tests here are two-sided on purpose, because either side alone would be
worthless: the synthetic ``av`` must be caught, and well-formed metadata must stay
clean. A check firing on everything gets disabled by the next person; a check firing on
nothing is decoration.

**Nothing here reads the installed environment, deliberately** (WP-115b). Earlier
versions of this file asserted over ``metadata.distributions()`` — the whole venv is
clean, ``shapely`` really does ship GEOS — and those assertions are not about this
code. A contributor who installs anything into their environment can fail them without
touching a line of the audit, and one did: ``cuda-toolkit``, a package no dependency
list here mentions. The live claim belongs to the pre-commit hook, which runs the audit
against the real environment on every commit and is the actual gate; the suite's job is
that the audit *decides correctly*, which synthetic distributions answer without
depending on what anyone happens to have installed. Every fixture below is written by
hand, and the shapes they are written to were read off real wheels once and recorded in
their docstrings.

A fourth surface is guarded since WP-115: the *absence* of a license. All three checks
above match a forbidden pattern against a declaration, so a distribution declaring
nothing matches nothing and passes in the same run that prints a clean verdict — which
is why WP-114b had to bound MkDocs with a version cap rather than a check. The tests
here pin both halves of the fix: a shipped license text is identified from its own
header, so a field-less but genuinely permissive wheel is cleared rather than failed
(``faster-coco-eval`` is the live case), and the finding is graded by dependency tier,
so a package the wheel republishes fails while one only the dev or docs groups reach is
reported and the run passes. Copyleft stays tier-blind: a document whose header names a
GPL-family license escalates past the flag wherever it sits.
"""

import importlib.util
from collections.abc import Callable, Sequence
from importlib import metadata
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "scripts" / "audit_licenses.py"


@pytest.fixture(scope="session")
def audit() -> ModuleType:
    """Load ``scripts/audit_licenses.py`` as an importable module.

    A fixture rather than a module-level import: ``scripts/`` is not a package, so the
    module has to be loaded by path, and doing that at import time makes collecting this
    file execute it. Session-scoped because the module is stateless and loading it per
    test would re-execute the tables for nothing.
    """
    spec = importlib.util.spec_from_file_location("audit_licenses", AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubMetadata:
    """Minimal stand-in for ``importlib.metadata`` message objects."""

    def __init__(
        self,
        name: str,
        classifiers: tuple[str, ...] = (),
        license_text: str = "",
        license_files: tuple[str, ...] = (),
    ) -> None:
        self._fields = {"Name": name, "License": license_text, "License-Expression": ""}
        self._classifiers = list(classifiers)
        self._license_files = list(license_files)

    def get(self, key: str, default: object = None) -> object:
        return self._fields.get(key, default)

    def get_all(self, key: str, default: object = None) -> object:
        if key == "Classifier":
            return list(self._classifiers)
        if key == "License-File":
            return list(self._license_files)
        return default


class _StubDist:
    """Minimal stand-in for an installed distribution."""

    def __init__(self, metadata_obj: _StubMetadata, documents: dict[str, str] | None = None) -> None:
        self.metadata = metadata_obj
        self._documents = documents or {}

    def read_text(self, path: str) -> str | None:
        """Return a bundled document's text, mimicking ``Distribution.read_text``."""
        return self._documents.get(path)


COPYLEFT_SAMPLES = ("AGPL-3.0", "GPLv2", "LGPL-2.1", "GNU General Public License")
PERMISSIVE_SAMPLES = (
    "Apache-2.0",
    "MIT",
    "BSD-3-Clause",
    "Mozilla Public License",
    "Python Software Foundation License",
)


def test_copyleft_pattern_matches_gpl_family(audit: ModuleType) -> None:
    """The copyleft pattern catches every GPL-family indicator."""
    for sample in COPYLEFT_SAMPLES:
        assert audit.COPYLEFT_PATTERN.search(sample), f"should match copyleft: {sample!r}"


def test_copyleft_pattern_spares_permissive(audit: ModuleType) -> None:
    """The copyleft pattern leaves permissive licenses alone."""
    for sample in PERMISSIVE_SAMPLES:
        assert not audit.COPYLEFT_PATTERN.search(sample), f"should not match permissive: {sample!r}"


def test_agpl_classifier_is_flagged(audit: ModuleType) -> None:
    """A distribution carrying an AGPL OSI classifier is reported as a violation."""
    stub = _StubDist(
        _StubMetadata(
            "evil-copyleft",
            classifiers=("License :: OSI Approved :: GNU Affero General Public License v3",),
        )
    )
    violations = audit.find_copyleft_violations([stub])
    assert len(violations) == 1
    name, field = violations[0]
    assert name == "evil-copyleft"
    assert "Affero" in field


def test_clean_stub_set_is_empty(audit: ModuleType) -> None:
    """A set of permissively licensed distributions yields no violation."""
    clean = [
        _StubDist(_StubMetadata("alpha", classifiers=("License :: OSI Approved :: MIT License",))),
        _StubDist(_StubMetadata("beta", classifiers=("License :: OSI Approved :: Apache Software License",))),
        _StubDist(_StubMetadata("gamma", classifiers=("License :: OSI Approved :: BSD License",))),
    ]
    assert audit.find_copyleft_violations(clean) == []


def test_long_prose_license_field_is_not_an_identifier(audit: ModuleType) -> None:
    """A License field holding full license prose (mentioning GPL) is not flagged when classifiers are permissive.

    Regression case: matplotlib embeds third-party license texts (FreeType,
    'GPL-2.0-or-later') in a 64 KB free-text License field while its actual
    classifier is PSF — prose must not be scanned as an identifier.
    """
    prose = "x" * 200 + " ... GNU General Public License ... " + "y" * 200
    stub = _StubDist(
        _StubMetadata(
            "prose-license",
            classifiers=("License :: OSI Approved :: Python Software Foundation License",),
            license_text=prose,
        )
    )
    assert audit.find_copyleft_violations([stub]) == []


def test_short_gpl_license_field_is_flagged(audit: ModuleType) -> None:
    """A short identifier-style License field naming GPL is still caught."""
    stub = _StubDist(_StubMetadata("short-gpl", license_text="GPL-3.0-only"))
    violations = audit.find_copyleft_violations([stub])
    assert len(violations) == 1


def _bundling_dist(name: str, body: str, filename: str = "LICENSE.txt") -> _StubDist:
    """A distribution declaring a permissive license while vendoring ``body``."""
    return _StubDist(
        _StubMetadata(
            name,
            classifiers=("License :: OSI Approved :: BSD License",),
            license_files=(filename,),
        ),
        documents={f"licenses/{filename}": body},
    )


def test_bundled_copyleft_is_flagged_though_the_metadata_is_permissive(audit: ModuleType) -> None:
    """A wheel declaring BSD while vendoring an LGPL binary is reported.

    The shapely case that motivated the check: every metadata field says BSD, so
    the declared-license audit passes and the LGPL sits in the environment unseen.
    """
    stub = _bundling_dist("vendors-lgpl", "Name: libthing\nFiles: libthing.so\nLicense: LGPL-2.1-or-later\n")

    violations = audit.find_bundled_violations([stub])

    assert violations == [("vendors-lgpl", "LICENSE.txt: LGPL-2.1-or-later")]


def test_bundled_license_prose_is_not_scanned(audit: ModuleType) -> None:
    """A vendored copy of the LGPL text itself is not a declaration.

    A full GPL text names "GPL" on dozens of lines. Scanning the body would flag
    every wheel shipping any license document, and a gate that fires on everything
    teaches its reader to bypass it.
    """
    stub = _bundling_dist("ships-the-text", "GNU LESSER GENERAL PUBLIC LICENSE\nVersion 2.1\n...prose...\n")

    assert audit.find_bundled_violations([stub]) == []


def test_gcc_runtime_exception_is_permissive_for_any_package(audit: ModuleType) -> None:
    """GPL-3 under the GCC Runtime Library Exception is not a finding, allowlist or not.

    The exception exists precisely to let GPL-3 runtime objects be linked into
    programs under any license, so it is recognized by expression rather than by
    package: a new dependency built by gcc needs no allowlist entry (D15).
    """
    stub = _bundling_dist(
        "not-on-any-allowlist",
        "Name: GCC runtime library\nFiles: libgfortran.dylib\nLicense: GPL-3.0-or-later WITH GCC-exception-3.1\n",
    )

    assert "not-on-any-allowlist" not in audit.BUNDLED_ALLOWLIST
    assert audit.find_bundled_violations([stub]) == []


def test_allowlisted_package_is_excused_only_for_bundled_findings(audit: ModuleType) -> None:
    """An allowlisted distribution's vendored copyleft passes; an unlisted one's does not."""
    body = "Name: libthing\nFiles: libthing.so\nLicense: LGPL-2.1-or-later\n"
    listed = next(iter(audit.BUNDLED_ALLOWLIST))

    assert audit.find_bundled_violations([_bundling_dist(listed, body)]) == []
    assert audit.find_bundled_violations([_bundling_dist("someone-else", body)]) != []


PERMISSIVE_LICENSE_DOCUMENT = (
    "Copyright (c) 2025 the authors\n\n"
    "Redistribution and use in source and binary forms, with or without modification,\n"
    "are permitted provided that the following conditions are met: ...\n"
)


@pytest.fixture
def synthetic_dist(tmp_path: Path) -> Callable[[str, Sequence[str]], metadata.Distribution]:
    """Build an installed distribution from a hand-written ``.dist-info``.

    ``av`` cannot be installed to test against — keeping it out of this tree is the
    whole point of the check — so the positive case is written by hand instead: a
    METADATA declaring a permissive license, a bundled license document that names no
    copyleft anywhere, and a RECORD listing the files the wheel ships. Going through
    ``Distribution.at`` rather than a stub object means the real ``RECORD`` parser runs,
    which is the part the check depends on.
    """

    def build(name: str, shipped: Sequence[str] = ()) -> metadata.Distribution:
        info = tmp_path / name / f"{name}-1.0.dist-info"
        info.mkdir(parents=True, exist_ok=True)
        info.joinpath("METADATA").write_text(
            "Metadata-Version: 2.4\n"
            f"Name: {name}\n"
            "Version: 1.0\n"
            "License-Expression: BSD-3-Clause\n"
            "Classifier: License :: OSI Approved :: BSD License\n"
            "License-File: LICENSE.txt\n"
        )
        info.joinpath("licenses").mkdir(exist_ok=True)
        info.joinpath("licenses", "LICENSE.txt").write_text(PERMISSIVE_LICENSE_DOCUMENT)
        records = [f"{path},," for path in shipped] + [f"{name}-1.0.dist-info/METADATA,,"]
        info.joinpath("RECORD").write_text("\n".join(records) + "\n")
        return metadata.Distribution.at(info)

    return build


@pytest.fixture
def av_lookalike(synthetic_dist: Callable[[str, Sequence[str]], metadata.Distribution]) -> metadata.Distribution:
    """The WP-109 case: a BSD-declaring wheel shipping a GPL x264 and an LGPL mp3lame."""
    return synthetic_dist("av", ("av/__init__.py", "av/.dylibs/libx264.165.dylib", "av/.dylibs/libmp3lame.0.dylib"))


def test_shipped_x264_is_flagged_though_nothing_declares_it(
    audit: ModuleType, av_lookalike: metadata.Distribution
) -> None:
    """A wheel shipping a GPL x264 binary is reported from its file list alone.

    The finding that motivated the check: ``supervision`` pulls ``av>=14.2``, whose
    wheel bundles ``libx264`` — GPL-2.0 — under BSD-3-Clause metadata. Nothing but the
    filename says so, so the filename is what the audit reads.
    """
    violations = audit.find_binary_violations([av_lookalike])

    assert violations == [("av", "av/.dylibs/libx264.165.dylib: x264 (GPL-2.0-or-later)")]


def test_the_older_two_checks_pass_the_same_wheel_clean(audit: ModuleType, av_lookalike: metadata.Distribution) -> None:
    """Neither the declared-license nor the license-file check sees the x264 binary.

    Pinned as a test rather than left as a claim in a docstring: this is the exact gap
    WP-109 closes, and if a later change made either older check catch it, the third
    check's justification would have quietly evaporated.
    """
    assert audit.find_copyleft_violations([av_lookalike]) == []
    assert audit.find_bundled_violations([av_lookalike]) == []


def test_shipped_x264_is_flagged_through_the_manylinux_spelling(
    audit: ModuleType,
    synthetic_dist: Callable[[str, Sequence[str]], metadata.Distribution],
) -> None:
    """The same wheel repaired for manylinux is caught under its grafted filename.

    The platform this matters on is the one no developer here runs: ``auditwheel``
    rewrites ``libx264.so.164`` to ``libx264-94059858.so.164`` when it vendors the
    library into a Linux wheel, and CI runs on Linux. A table that matched only the
    macOS spelling would pass every local check and see nothing on the runner.
    """
    linux_av = synthetic_dist("av", ("av/__init__.py", "av.libs/libx264-94059858.so.164"))

    violations = audit.find_binary_violations([linux_av])

    assert violations == [("av", "av.libs/libx264-94059858.so.164: x264 (GPL-2.0-or-later)")]


def test_unrecognized_binaries_are_not_findings(
    audit: ModuleType,
    synthetic_dist: Callable[[str, Sequence[str]], metadata.Distribution],
) -> None:
    """A wheel shipping ordinary native libraries produces nothing.

    ``torch`` alone ships hundreds of ``.dylib``/``.so`` files, so a check failing on
    anything it does not recognize would fire on every commit and be switched off within
    a week. The table names what is known to be copyleft; everything else passes.
    """
    ordinary = synthetic_dist(
        "well-behaved",
        (
            "pkg/_core.cpython-311-darwin.so",
            "pkg/lib/libtorch_cpu.dylib",
            "pkg/.dylibs/libomp.dylib",
            "pkg/.dylibs/libjpeg.62.4.0.dylib",
            "pkg/.dylibs/libfoo.so.3",
        ),
    )

    assert audit.find_binary_violations([ordinary]) == []


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        pytest.param("libx264.165.dylib", "x264", id="macos-version-suffix"),
        pytest.param("libx264.so.164", "x264", id="linux-soname-version"),
        pytest.param("x264.dll", "x264", id="windows-no-lib-prefix"),
        pytest.param("avcodec-61.dll", "avcodec", id="windows-dash-version-tag"),
        pytest.param("libgeos_c.1.19.2.dylib", "geos_c", id="underscore-is-part-of-the-name"),
        pytest.param("libgeos-3ef06f11.so.3.13.1", "geos", id="auditwheel-graft-shapely"),
        pytest.param("libgeos_c-abcdd5fa.so.1.19.2", "geos_c", id="auditwheel-graft-with-underscore"),
        pytest.param("libpng16-abb096d5.so.16.58.0", "png16", id="auditwheel-graft-pillow"),
        pytest.param("libx264-94059858.so.164", "x264", id="auditwheel-graft-of-a-listed-library"),
        pytest.param("_core.cpython-311-darwin.so", "_core", id="python-extension-module"),
        pytest.param("libpng16.16.dylib", "png16", id="digits-joined-to-the-name-are-kept"),
        pytest.param("LIBX264.DYLIB", "x264", id="case-insensitive"),
        pytest.param("__init__.py", None, id="not-a-library"),
        pytest.param("libx264.dylib.txt", None, id="suffix-must-end-the-name"),
        pytest.param("lib.cpython-311-darwin.so", None, id="empty-stem-identifies-nothing"),
    ],
)
def test_library_stem_normalizes_every_platform_spelling(
    audit: ModuleType, filename: str, expected: str | None
) -> None:
    """One table entry matches a library however the platform spelled its filename.

    The table is keyed on the stem precisely so that a rebuild bumping ``libx264.164``
    to ``libx264.165``, or a Windows wheel spelling it ``x264.dll``, does not walk out
    from under the entry that names it.

    The ``auditwheel-graft`` cases are the ones that matter most and are the reason this
    is parametrized rather than asserted on the live environment alone: the graft only
    appears on manylinux wheels, so a normalizer that handles the macOS spelling and not
    the Linux one passes every check a macOS developer can run while seeing nothing at
    all on the Linux runner CI uses. The four filenames are transcribed from the
    ``shapely`` and ``pillow`` manylinux wheels, not invented — except the ``libx264``
    one, which applies the same convention to a library this tree must never install.
    """
    assert audit.library_stem(filename) == expected


def test_binary_allowlist_excuses_the_named_library_only(
    audit: ModuleType,
    synthetic_dist: Callable[[str, Sequence[str]], metadata.Distribution],
) -> None:
    """An allowlisted pair passes; the same distribution's other copyleft binary does not.

    This is what the ``(distribution, library)`` key buys over the package-keyed
    allowlist beside it. ``shapely`` is allowed to ship GEOS and nothing else — if it
    started shipping x264 tomorrow the audit must still say so, which a package-keyed
    entry would have silently prevented.
    """
    excused = synthetic_dist("shapely", ("shapely/.dylibs/libgeos.3.13.1.dylib",))
    unrelated = synthetic_dist("shapely-but-x264", ("shapely/.dylibs/libx264.165.dylib",))

    assert audit.find_binary_violations([excused]) == []
    assert audit.find_binary_violations([unrelated]) != []


def test_every_binary_allowlist_entry_names_a_listed_library(audit: ModuleType) -> None:
    """No allowlist entry excuses a library the table does not name.

    An entry whose stem has drifted out of ``COPYLEFT_BINARIES`` — renamed, or removed
    when the table was edited — excuses nothing and reads as though it does, which is
    the failure mode of an allowlist nobody re-reads.
    """
    listed = {stem for _, stem in audit.BUNDLED_BINARY_ALLOWLIST}

    assert listed <= set(audit.COPYLEFT_BINARIES)


# ---------------------------------------------------------------------------
# WP-115: the fourth surface — a license nothing declares, graded by tier.
# ---------------------------------------------------------------------------

APACHE_DOCUMENT = (
    "                                 Apache License\n"
    "                           Version 2.0, January 2004\n\n"
    "   TERMS AND CONDITIONS\n"
)
MPL_DOCUMENT = (
    "Mozilla Public License Version 2.0\n"
    "==================================\n\n"
    "1.12. Secondary License\n"
    "  means either the GNU General Public License, Version 2.0, the GNU Lesser\n"
    "  General Public License, Version 2.1, or the GNU Affero General Public License.\n"
)
LGPL_DOCUMENT = (
    "GNU LESSER GENERAL PUBLIC LICENSE\n"
    "Version 2.1, February 1999\n\n"
    "Everyone is permitted to copy and distribute verbatim copies\n"
)
UNKNOWN_DOCUMENT = "Terms of use\n\nYou may use this software if you ask nicely and we agree in writing.\n"

COPYLEFT_HEADERS = (
    "GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n",
    "GNU LESSER GENERAL PUBLIC LICENSE\nVersion 2.1, February 1999\n",
    "GNU AFFERO GENERAL PUBLIC LICENSE\nVersion 3, 19 November 2007\n",
)


def _undeclared_dist(name: str, documents: dict[str, str] | None = None) -> _StubDist:
    """A distribution with no license field, optionally shipping license documents."""
    documents = documents or {}
    return _StubDist(
        _StubMetadata(name, license_files=tuple(documents)),
        {f"licenses/{filename}": text for filename, text in documents.items()},
    )


def _tiered_pyproject(tmp_path: Path, base: Sequence[str], dev: Sequence[str], docs: Sequence[str]) -> Path:
    """Write a minimal pyproject declaring one tier's roots each."""
    path = tmp_path / "pyproject.toml"
    path.write_text(
        "[project]\nname = 'x'\nversion = '0'\n"
        f"dependencies = {list(base)!r}\n\n"
        f"[dependency-groups]\ndev = {list(dev)!r}\ndocs = {list(docs)!r}\n".replace("'", '"'),
        encoding="utf-8",
    )
    return path


def test_a_distribution_declaring_nothing_at_all_is_a_finding(audit: ModuleType) -> None:
    """No license field and no license document is the case the copyleft checks cannot see.

    Each of the three older surfaces matches a forbidden pattern against a declaration,
    so a distribution declaring nothing matches nothing and passes in the same run that
    prints a clean verdict. This is the hole WP-114b had to reach for a version cap to
    work around.
    """
    reason = audit.unreadable_license_reason(_undeclared_dist("says-nothing"))

    assert reason is not None
    assert "ships no License-File" in reason


def test_an_apache_document_with_no_license_field_is_read_and_cleared(audit: ModuleType) -> None:
    """A shipped Apache-2.0 text resolves a distribution whose metadata declares nothing.

    The live shape this check had to accommodate: PEP 639 metadata may carry the license
    only as a file, so a check that demanded a field would fail on a genuinely permissive
    base dependency and be turned off within a week.
    """
    dist = _undeclared_dist("field-less-apache", {"LICENSE": APACHE_DOCUMENT})

    assert audit.unreadable_license_reason(dist) is None


def test_an_unrecognized_license_document_stays_a_finding(audit: ModuleType) -> None:
    """A document matching no known permissive text is not given the benefit of the doubt.

    The direction that makes the check worth having: bespoke terms are exactly what a
    licence audit exists to surface, and silence about them reads as approval.
    """
    dist = _undeclared_dist("bespoke-terms", {"LICENSE": UNKNOWN_DOCUMENT})

    reason = audit.unreadable_license_reason(dist)

    assert reason is not None
    assert "LICENSE" in reason


@pytest.mark.parametrize("header", COPYLEFT_HEADERS)
def test_the_recognizer_declines_every_copyleft_header(audit: ModuleType, header: str) -> None:
    """No GPL-family text is recognizable, so each one falls through to a finding.

    The table lists permissive texts only, which means the recognizer never has to
    decide that something is forbidden — an omission from the table costs a false
    alarm, never a silent pass.
    """
    assert audit.recognize_license_text(header) is None


def test_mpl_prose_is_identified_rather_than_self_flagged(audit: ModuleType) -> None:
    """MPL-2.0 names the GPL in its own Secondary-License clause and is still recognized.

    The reason the copyleft pattern is deliberately not run over a recognized text as a
    second opinion: this environment already carries MPL-2.0 through ``certifi`` and
    ``pathspec``, and a belt-and-braces re-scan would fail the audit on a license the
    policy tolerates.
    """
    assert audit.recognize_license_text(MPL_DOCUMENT) == "MPL-2.0"


def test_a_group_only_package_is_flagged_rather_than_failed(audit: ModuleType, tmp_path: Path) -> None:
    """A dev- or docs-only package with no readable license is reported, and the run passes.

    PEP 735 groups appear in no wheel metadata, so nothing a consumer installs is
    affected by what the repository's own tooling pulls; treating it as fatal would make
    the check about tidiness rather than about exposure.
    """
    dists = [_undeclared_dist("alpha"), _undeclared_dist("beta")]
    pyproject = _tiered_pyproject(tmp_path, base=["alpha"], dev=["beta"], docs=[])

    failures, flags = audit.find_unreadable_licenses(dists, audit.dependency_tiers(dists, pyproject))

    assert [name for name, _ in failures] == ["alpha"]
    assert [name for name, _ in flags] == ["beta"]


def test_a_package_both_tiers_reach_takes_the_shipped_tier(audit: ModuleType, tmp_path: Path) -> None:
    """Reachable from base and from a group means base: the stricter attribution wins.

    A shared transitive is republished in the wheel regardless of what else also pulls
    it, so letting the group attribution win would downgrade a genuinely shipped package.
    """
    dists = [_undeclared_dist("shared")]
    pyproject = _tiered_pyproject(tmp_path, base=["shared"], dev=["shared"], docs=[])

    failures, flags = audit.find_unreadable_licenses(dists, audit.dependency_tiers(dists, pyproject))

    assert [name for name, _ in failures] == ["shared"]
    assert flags == []


def test_an_unattributed_package_is_treated_as_shipped(audit: ModuleType, tmp_path: Path) -> None:
    """A distribution no tier reaches fails rather than passing quietly.

    The resolver walks installed metadata, so a gap in it — an unparsed marker, a
    requirement form it does not follow — must present as a loud failure. The opposite
    default would let a hole in the walk silently demote a shipped dependency to a flag.
    """
    dists = [_undeclared_dist("orphan")]
    pyproject = _tiered_pyproject(tmp_path, base=[], dev=[], docs=[])

    failures, _ = audit.find_unreadable_licenses(dists, audit.dependency_tiers(dists, pyproject))

    assert [name for name, _ in failures] == ["orphan"]


def test_copyleft_prose_fails_even_in_a_group_tier(audit: ModuleType, tmp_path: Path) -> None:
    """A group-only package whose only license document is LGPL text is a failure, not a flag.

    The tier grades unreadability, not copyleft. "No AGPL at any cost" is tier-blind, so a
    document that names a GPL-family license in its own header escalates past the flag.
    """
    dists = [_undeclared_dist("tooling", {"COPYING": LGPL_DOCUMENT})]
    pyproject = _tiered_pyproject(tmp_path, base=[], dev=["tooling"], docs=[])

    failures, flags = audit.find_unreadable_licenses(dists, audit.dependency_tiers(dists, pyproject))

    assert [name for name, _ in failures] == ["tooling"]
    assert flags == []


def test_an_include_group_entry_is_refused(audit: ModuleType, tmp_path: Path) -> None:
    """A ``{include-group = ...}`` entry raises instead of being walked past.

    The walk follows requirement strings; a group reference it silently dropped would
    under-attribute everything that group reaches, turning failures into flags without
    anyone choosing that.
    """
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[project]\nname = "x"\nversion = "0"\ndependencies = []\n\n'
        '[dependency-groups]\ndev = [{include-group = "docs"}]\ndocs = []\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="include-group"):
        audit.dependency_tiers([], path)


def test_an_allowlisted_distribution_is_not_a_finding(audit: ModuleType, tmp_path: Path) -> None:
    """A name in ``UNREADABLE_ALLOWLIST`` is skipped even in the shipped tier.

    The fourth allowlist and the fourth exposure: this one excuses metadata that says
    nothing at all, which is neither a copyleft declaration nor a vendored copyleft
    library. ``cuda-toolkit`` is the live entry — its wheel dist-info holds METADATA,
    WHEEL and RECORD and no licence document of any kind (D17).
    """
    listed = next(iter(audit.UNREADABLE_ALLOWLIST))
    dists = [_undeclared_dist(listed), _undeclared_dist("not-on-any-allowlist")]
    pyproject = _tiered_pyproject(tmp_path, base=[listed, "not-on-any-allowlist"], dev=[], docs=[])

    failures, flags = audit.find_unreadable_licenses(dists, audit.dependency_tiers(dists, pyproject))

    assert [name for name, _ in failures] == ["not-on-any-allowlist"]
    assert flags == []


def test_every_unreadable_allowlist_entry_cites_a_decision(audit: ModuleType) -> None:
    """Each entry names the DECISIONS.md row that admitted it.

    The convention the three older allowlists follow, pinned here so the fourth cannot
    quietly become the one where a package is excused by a reason nobody wrote down.
    """
    for name, reason in audit.UNREADABLE_ALLOWLIST.items():
        assert reason.startswith("D"), f"{name}: allowlist reason does not open with a decision id: {reason[:40]!r}"


def test_an_unattributed_package_says_so_rather_than_reading_as_base(audit: ModuleType, tmp_path: Path) -> None:
    """The reported tier distinguishes "the wheel ships this" from "the walk never found it".

    Both are failures and for opposite reasons, so printing them identically would send a
    reader looking for a dependency declaration that does not exist. This is the shape the
    live ``cuda-toolkit`` finding arrived in: reported as base, actually unattributed.
    """
    dists = [_undeclared_dist("orphan"), _undeclared_dist("declared")]
    pyproject = _tiered_pyproject(tmp_path, base=["declared"], dev=[], docs=[])

    failures, _ = audit.find_unreadable_licenses(dists, audit.dependency_tiers(dists, pyproject))
    reasons = dict(failures)

    assert "unattributed" in reasons["orphan"]
    assert reasons["declared"].endswith(f"({audit.TIER_SHIPPED})")


def test_well_formed_metadata_yields_no_unreadable_finding(audit: ModuleType, tmp_path: Path) -> None:
    """A set of distributions that each declare a license produces neither a failure nor a flag.

    The positive control for the fourth surface, and the half that keeps it honest: every
    other test here supplies metadata that is missing something, so without this one a
    check that reported *every* distribution would pass them all.
    """
    dists = [
        _StubDist(_StubMetadata("alpha", classifiers=("License :: OSI Approved :: MIT License",))),
        _StubDist(_StubMetadata("beta", license_text="Apache-2.0")),
        _undeclared_dist("gamma", {"LICENSE": APACHE_DOCUMENT}),
    ]
    pyproject = _tiered_pyproject(tmp_path, base=["alpha", "beta"], dev=["gamma"], docs=[])

    failures, flags = audit.find_unreadable_licenses(dists, audit.dependency_tiers(dists, pyproject))

    assert failures == []
    assert flags == []
