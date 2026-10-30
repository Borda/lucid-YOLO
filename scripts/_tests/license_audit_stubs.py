# SPDX-License-Identifier: Apache-2.0
"""Hand-written distribution stand-ins for the dependency-license audit tests.

``test_audit_licenses.py`` never reads the installed environment (WP-115b); every distribution it audits is built here
or by its own ``.dist-info`` fixture. The stubs answer exactly the ``importlib.metadata`` surface the audit calls --
``metadata.get``/``get_all`` and ``Distribution.read_text`` -- and nothing more, so a test cannot pass by leaning on a
behaviour the real objects do not have. ``conftest.py`` puts this directory on ``sys.path``, which is how a test file
reaches the module by its bare name.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


class StubMetadata:
    """Minimal stand-in for ``importlib.metadata`` message objects.

    Examples:
        >>> stub = StubMetadata("pkg", classifiers=("License :: OSI Approved :: MIT License",), license_text="MIT")
        >>> stub.get("Name"), stub.get("License")
        ('pkg', 'MIT')
        >>> stub.get_all("Classifier")
        ['License :: OSI Approved :: MIT License']
    """

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
        """Return a single-valued metadata field, or ``default`` when the stub has no such field."""
        return self._fields.get(key, default)

    def get_all(self, key: str, default: object = None) -> object:
        """Return the classifiers or ``License-File`` entries for those keys, else ``default``."""
        if key == "Classifier":
            return list(self._classifiers)
        if key == "License-File":
            return list(self._license_files)
        return default


class StubDist:
    """Minimal stand-in for an installed distribution.

    Examples:
        >>> dist = StubDist(StubMetadata("pkg"), {"licenses/LICENSE": "text"})
        >>> dist.read_text("licenses/LICENSE")
        'text'
        >>> dist.read_text("licenses/missing") is None
        True
    """

    def __init__(self, metadata_obj: StubMetadata, documents: dict[str, str] | None = None) -> None:
        self.metadata = metadata_obj
        self._documents = documents or {}

    def read_text(self, path: str) -> str | None:
        """Return a bundled document's text, mimicking ``Distribution.read_text``."""
        return self._documents.get(path)


def bundling_dist(name: str, body: str, filename: str = "LICENSE.txt") -> StubDist:
    """A distribution declaring a permissive license while vendoring ``body``.

    Examples:
        >>> dist = bundling_dist("vendors-lgpl", "License: LGPL-2.1-or-later\\n")
        >>> dist.metadata.get_all("License-File")
        ['LICENSE.txt']
        >>> dist.read_text("licenses/LICENSE.txt")
        'License: LGPL-2.1-or-later\\n'
    """
    return StubDist(
        StubMetadata(
            name,
            classifiers=("License :: OSI Approved :: BSD License",),
            license_files=(filename,),
        ),
        documents={f"licenses/{filename}": body},
    )


def undeclared_dist(name: str, documents: dict[str, str] | None = None) -> StubDist:
    """A distribution with no license field, optionally shipping license documents.

    Examples:
        >>> bare = undeclared_dist("says-nothing")
        >>> bare.metadata.get_all("License-File")
        []
        >>> shipped = undeclared_dist("has-doc", {"LICENSE": "Apache text"})
        >>> shipped.read_text("licenses/LICENSE")
        'Apache text'
    """
    documents = documents or {}
    return StubDist(
        StubMetadata(name, license_files=tuple(documents)),
        {f"licenses/{filename}": text for filename, text in documents.items()},
    )


def tiered_pyproject(tmp_path: Path, base: Sequence[str], dev: Sequence[str], docs: Sequence[str]) -> Path:
    """Write a minimal pyproject declaring one tier's roots each.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     path = tiered_pyproject(Path(tmp), base=["alpha"], dev=["beta"], docs=[])
        ...     print(path.read_text())
        [project]
        name = "x"
        version = "0"
        dependencies = ["alpha"]
        <BLANKLINE>
        [dependency-groups]
        dev = ["beta"]
        docs = []
        <BLANKLINE>
    """
    path = tmp_path / "pyproject.toml"
    path.write_text(
        "[project]\nname = 'x'\nversion = '0'\n"
        f"dependencies = {list(base)!r}\n\n"
        f"[dependency-groups]\ndev = {list(dev)!r}\ndocs = {list(docs)!r}\n".replace("'", '"'),
        encoding="utf-8",
    )
    return path
