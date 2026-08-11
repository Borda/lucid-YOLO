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
"""

import importlib.util
from importlib import metadata
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "scripts" / "audit_licenses.py"


def _load_audit() -> ModuleType:
    """Load ``scripts/audit_licenses.py`` as an importable module."""
    spec = importlib.util.spec_from_file_location("audit_licenses", AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit()


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


def test_copyleft_pattern_matches_gpl_family() -> None:
    """The copyleft pattern catches every GPL-family indicator."""
    for sample in COPYLEFT_SAMPLES:
        assert audit.COPYLEFT_PATTERN.search(sample), f"should match copyleft: {sample!r}"


def test_copyleft_pattern_spares_permissive() -> None:
    """The copyleft pattern leaves permissive licenses alone."""
    for sample in PERMISSIVE_SAMPLES:
        assert not audit.COPYLEFT_PATTERN.search(sample), f"should not match permissive: {sample!r}"


def test_agpl_classifier_is_flagged() -> None:
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


def test_clean_stub_set_is_empty() -> None:
    """A set of permissively licensed distributions yields no violation."""
    clean = [
        _StubDist(_StubMetadata("alpha", classifiers=("License :: OSI Approved :: MIT License",))),
        _StubDist(_StubMetadata("beta", classifiers=("License :: OSI Approved :: Apache Software License",))),
        _StubDist(_StubMetadata("gamma", classifiers=("License :: OSI Approved :: BSD License",))),
    ]
    assert audit.find_copyleft_violations(clean) == []


def test_long_prose_license_field_is_not_an_identifier() -> None:
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


def test_short_gpl_license_field_is_flagged() -> None:
    """A short identifier-style License field naming GPL is still caught."""
    stub = _StubDist(_StubMetadata("short-gpl", license_text="GPL-3.0-only"))
    violations = audit.find_copyleft_violations([stub])
    assert len(violations) == 1


def test_live_environment_is_clean() -> None:
    """The active environment carries no GPL-family dependency."""
    assert audit.find_copyleft_violations(list(metadata.distributions())) == []


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


def test_bundled_copyleft_is_flagged_though_the_metadata_is_permissive() -> None:
    """A wheel declaring BSD while vendoring an LGPL binary is reported.

    The shapely case that motivated the check: every metadata field says BSD, so
    the declared-license audit passes and the LGPL sits in the environment unseen.
    """
    stub = _bundling_dist("vendors-lgpl", "Name: libthing\nFiles: libthing.so\nLicense: LGPL-2.1-or-later\n")

    violations = audit.find_bundled_violations([stub])

    assert violations == [("vendors-lgpl", "LICENSE.txt: LGPL-2.1-or-later")]


def test_bundled_license_prose_is_not_scanned() -> None:
    """A vendored copy of the LGPL text itself is not a declaration.

    A full GPL text names "GPL" on dozens of lines. Scanning the body would flag
    every wheel shipping any license document, and a gate that fires on everything
    teaches its reader to bypass it.
    """
    stub = _bundling_dist("ships-the-text", "GNU LESSER GENERAL PUBLIC LICENSE\nVersion 2.1\n...prose...\n")

    assert audit.find_bundled_violations([stub]) == []


def test_gcc_runtime_exception_is_permissive_for_any_package() -> None:
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


def test_allowlisted_package_is_excused_only_for_bundled_findings() -> None:
    """An allowlisted distribution's vendored copyleft passes; an unlisted one's does not."""
    body = "Name: libthing\nFiles: libthing.so\nLicense: LGPL-2.1-or-later\n"
    listed = next(iter(audit.BUNDLED_ALLOWLIST))

    assert audit.find_bundled_violations([_bundling_dist(listed, body)]) == []
    assert audit.find_bundled_violations([_bundling_dist("someone-else", body)]) != []


def test_live_environment_bundles_nothing_unallowed() -> None:
    """The active environment vendors no copyleft binary outside the allowlist."""
    assert audit.find_bundled_violations(list(metadata.distributions())) == []
