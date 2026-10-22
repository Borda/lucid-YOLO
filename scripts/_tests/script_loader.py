# SPDX-License-Identifier: Apache-2.0
"""Load a ``scripts/`` module by file stem, for the suites that test a script's own functions.

The scripts are not a package, so a test cannot ``import`` them; each suite used to carry
its own ``importlib`` recipe, identical down to the module-name string. ``conftest.py``
puts this directory on ``sys.path``, which is how a test file reaches it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_SCRIPTS = Path(__file__).resolve().parents[1]


def load_script(name: str) -> ModuleType:
    """Execute ``scripts/lint/<name>.py`` (else ``scripts/<name>.py``) and return it as a module.

    The module is registered in ``sys.modules`` under ``name`` before it runs -- a ``@dataclass`` in
    it looks its own module up there -- and each call executes the file afresh.

    Args:
        name: File stem of the script, without ``.py``.

    Returns:
        The executed module, named ``name``.

    Raises:
        FileNotFoundError: When neither directory holds the script.

    Examples:
        >>> module = load_script("audit_wp_links")
        >>> module.__name__
        'audit_wp_links'
        >>> callable(module.find_stale)
        True
    """
    for directory in (_SCRIPTS / "lint", _SCRIPTS):
        path = directory / f"{name}.py"
        if path.is_file():
            break
    else:
        raise FileNotFoundError(f"no script named {name!r} under scripts/lint or scripts")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
