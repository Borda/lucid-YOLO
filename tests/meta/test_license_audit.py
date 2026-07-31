# SPDX-License-Identifier: Apache-2.0
"""Meta tests: dependency-license audit rejects copyleft (WP-004).

Guards the Apache-compatibility policy: the copyleft pattern must catch the
GPL family and spare permissive licenses, and ``find_copyleft_violations`` must
flag an AGPL-classified distribution while leaving the live (clean) environment
untouched.
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

    def __init__(self, name: str, classifiers: tuple[str, ...] = (), license_text: str = "") -> None:
        self._fields = {"Name": name, "License": license_text, "License-Expression": ""}
        self._classifiers = list(classifiers)

    def get(self, key: str, default: object = None) -> object:
        return self._fields.get(key, default)

    def get_all(self, key: str, default: object = None) -> object:
        if key == "Classifier":
            return list(self._classifiers)
        return default


class _StubDist:
    """Minimal stand-in for an installed distribution."""

    def __init__(self, metadata_obj: _StubMetadata) -> None:
        self.metadata = metadata_obj


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
