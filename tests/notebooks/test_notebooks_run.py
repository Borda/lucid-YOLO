# SPDX-License-Identifier: Apache-2.0
"""Smoke test over every notebook source: the anti-rot mechanism for ``notebooks/*.py`` (WP-188).

The site renders each notebook unexecuted (D25), so nothing about publishing a notebook
proves it still runs. This does: every ``notebooks/<name>.py`` jupytext source is
converted to an ``.ipynb`` and **executed through a Jupyter kernel**, from the repository
root, twice. A kernel rather than ``python <name>.py`` because the notebooks' commands are
``%%bash`` cells -- ``# %% language="bash"`` in the ``.py`` with the script commented
beneath, the way jupytext writes a cell magic -- and a script run reads those as comments
and proves nothing about them. The **fast** case sets ``LUCID_NOTEBOOK_FAST=1``, the contract
in ``notebooks/README.md`` under which a notebook swaps its dataset for the synthetic
slice, trains one epoch on a small batch, draws no figure and downloads nothing; it runs
on CPU under ``make test`` and so under ``make gate``. The **full** case is the same
command without the variable, marked ``gpu`` and ``data`` so it runs under ``make
test-gpu`` and the nightly accelerator gate, where the notebook does what it does for a
reader. The third layer -- the Colab run behind the badge -- is a human step once per
notebook and is not automated here.

**Which kernel.** ``jupytext --execute`` hands the notebook to ``nbconvert``'s
``ExecutePreprocessor`` with the kernelspec the notebook names, and the command names it
explicitly: ``--set-kernel python3``, the spec ``ipykernel``'s wheel installs into
``<sys.prefix>/share/jupyter/kernels/python3`` -- no ``ipykernel install`` step, nothing
under ``~``. That spec's ``argv[0]`` is the bare word ``python``, and
``jupyter_client.manager.KernelManager.format_kernel_cmd`` swaps a bare ``python`` for
the ``sys.executable`` of the process launching the kernel, which is this interpreter
because jupytext is run as ``[sys.executable, "-m", "jupytext", ...]``. So the kernel is
the venv the test runs in, with the editable ``lucid_yolo`` and its console scripts.
(``--set-kernel -``, jupytext's "match the current interpreter" spelling, does not work
here: it compares each spec's ``argv[0]`` to ``sys.executable`` with ``os.path.samefile``
and a bare ``python`` is not a file.) ``JUPYTER_PATH`` is set to that ``share/jupyter``
so it is searched first: a user-level ``python3`` spec (``~/Library/Jupyter/kernels`` on
macOS, ``~/.local/share/jupyter/kernels`` on Linux) otherwise precedes the venv's and
would run some other interpreter under the same name.

``--from py:percent`` is not decoration. Without it jupytext guesses the format, and an
indented ``    # !git clone`` inside the Colab ``if`` block makes the guess ``hydrogen``
-- a format that leaves ``# !`` lines as comments -- because the guesser's escaped-magic
test does not strip indentation. ``--run-path`` pins the kernel's working directory to
the repository root (jupytext's default is the output notebook's directory), which is
where a notebook's relative paths (``scripts/``, ``results/``, ``.cache/``) resolve when
a reader runs it from a checkout.

``MPLBACKEND=Agg`` on both: a notebook that draws under the full case must not need a
display, and a headless CI runner has none.

The parametrisation is the glob, so a repository with no sources yet collects one
skipped item per case and nothing else.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

#: The kernelspec ``ipykernel`` ships in its wheel; see the module docstring for why it,
#: and not ``--set-kernel -``, is how the executing interpreter becomes this one.
KERNEL_NAME = "python3"

#: The fast case's wall-clock ceiling, in seconds. The contract in ``notebooks/README.md``
#: is "well under a minute on CPU": a fast run that takes longer is a notebook whose
#: ``FAST`` branch is not capping what it should, and the timeout makes that a failure
#: naming the notebook rather than a slow gate nobody attributes. The kernel start and
#: the conversion are inside the budget; they cost about two seconds.
FAST_TIMEOUT_S = 60


def notebook_sources(notebooks_dir: Path) -> list[Path]:
    """Every jupytext source under ``notebooks_dir``, sorted so the parametrisation is stable.

    Takes the directory rather than reading the module constant so the answer can be
    checked on a directory whose contents are known -- the live one changes with every
    notebook WP. A directory that does not exist yields nothing, the same as an empty one.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     _ = (root / "b.py").write_text("# %%\\n", encoding="utf-8")
        ...     _ = (root / "a.py").write_text("# %%\\n", encoding="utf-8")
        ...     _ = (root / "README.md").write_text("# not a source\\n", encoding="utf-8")
        ...     [path.name for path in notebook_sources(root)]
        ...     notebook_sources(root / "absent")
        ['a.py', 'b.py']
        []
    """
    if not notebooks_dir.is_dir():
        return []
    return sorted(notebooks_dir.glob("*.py"))


def jupytext_command(source: Path, output: Path) -> list[str]:
    """The ``jupytext`` argv that converts ``source`` and executes it, writing ``output``.

    Every flag is load-bearing; the module docstring says why each one is there. The
    interpreter is ``sys.executable`` because that is what the kernel inherits.

    Examples:
        >>> argv = jupytext_command(Path("notebooks/demo.py"), Path("/tmp/out/demo.ipynb"))
        >>> argv[0] == sys.executable and argv[1:3] == ["-m", "jupytext"]
        True
        >>> argv[3:]  # doctest: +NORMALIZE_WHITESPACE
        ['--from', 'py:percent', '--to', 'ipynb', '--set-kernel', 'python3', '--execute',
         '--run-path', '...', '--output', '/tmp/out/demo.ipynb', 'notebooks/demo.py']
    """
    return [
        sys.executable,
        "-m",
        "jupytext",
        "--from",
        "py:percent",
        "--to",
        "ipynb",
        "--set-kernel",
        KERNEL_NAME,
        "--execute",
        "--run-path",
        str(REPO_ROOT),
        "--output",
        str(output),
        str(source),
    ]


def notebook_env(scratch: Path, *, fast: bool) -> dict[str, str]:
    """The environment a notebook run gets: the caller's, plus the four variables the run pins.

    ``LUCID_NOTEBOOK_FAST`` is set to ``1`` under ``fast`` or removed otherwise -- removed
    rather than left alone, so a developer's shell exporting the variable cannot turn the
    full case into a second fast one. ``MPLBACKEND=Agg`` is the headless backend, and
    ``JUPYTER_PATH`` puts this interpreter's kernelspecs ahead of any user-level ones.
    ``TMPDIR`` is ``scratch``, so what a fast run writes through ``tempfile`` -- its slice,
    its Lightning logs -- lands under the test's own directory, which pytest prunes; a
    kernel that nbclient shuts down runs no finalizer, so nothing else would remove it.

    Examples:
        >>> env = notebook_env(Path('/scratch'), fast=True)
        >>> env["LUCID_NOTEBOOK_FAST"], env["MPLBACKEND"]
        ('1', 'Agg')
        >>> env["JUPYTER_PATH"] == str(Path(sys.prefix) / "share" / "jupyter")
        True
        >>> env["TMPDIR"]
        '/scratch'
        >>> "LUCID_NOTEBOOK_FAST" in notebook_env(Path('/scratch'), fast=False)
        False
    """
    env = {key: value for key, value in os.environ.items() if key != "LUCID_NOTEBOOK_FAST"}
    env["MPLBACKEND"] = "Agg"
    env["JUPYTER_PATH"] = str(Path(sys.prefix) / "share" / "jupyter")
    env["TMPDIR"] = str(scratch)
    if fast:
        env["LUCID_NOTEBOOK_FAST"] = "1"
    return env


def run_notebook(source: Path, output_dir: Path, *, fast: bool, timeout: float | None = None) -> Path:
    """Convert and execute one notebook source through a kernel; raise if any cell fails.

    Returns the executed ``.ipynb``, written under ``output_dir`` with the source's stem.
    A failing cell -- an exception, a ``%%bash`` cell whose exit status is non-zero, a
    ``SystemExit`` -- stops the run and surfaces as a non-zero jupytext exit, so
    ``check=True`` turns it into ``CalledProcessError``; the cell's stderr is in jupytext's
    own stderr above the traceback.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     script = Path(tmp) / "hello.py"
        ...     _ = script.write_text(
        ...         "# %%\\nimport os\\nassert os.environ['MPLBACKEND'] == 'Agg'\\n"
        ...         "assert os.environ.get('LUCID_NOTEBOOK_FAST') == '1'\\n"
        ...         '# %% language="bash"\\n# test "$LUCID_NOTEBOOK_FAST" = 1\\n',
        ...         encoding="utf-8",
        ...     )
        ...     run_notebook(script, Path(tmp), fast=True).name
        'hello.ipynb'
    """
    output = output_dir / f"{source.stem}.ipynb"
    subprocess.run(
        jupytext_command(source, output),
        check=True,
        cwd=REPO_ROOT,
        env=notebook_env(output_dir, fast=fast),
        timeout=timeout,
        stdout=subprocess.DEVNULL,
    )
    return output


_SOURCES = [pytest.param(source, id=source.stem) for source in notebook_sources(NOTEBOOKS_DIR)]


@pytest.mark.parametrize("source", _SOURCES)
def test_runs_under_the_fast_contract(source: Path, tmp_path: Path) -> None:
    """Each notebook runs end to end on CPU with ``LUCID_NOTEBOOK_FAST=1``, within the ceiling.

    This is the run ``make gate`` pays for: it proves the code path the notebook walks
    still exists and still executes, on the synthetic slice, with no accelerator and no
    download. It proves nothing about the numbers the full run reports.
    """
    run_notebook(source, tmp_path, fast=True, timeout=FAST_TIMEOUT_S)


@pytest.mark.gpu
@pytest.mark.data
@pytest.mark.parametrize("source", _SOURCES)
def test_runs_in_full(source: Path, tmp_path: Path) -> None:
    """Each notebook runs end to end as a reader would, with no fast cap.

    Marked ``gpu`` and ``data`` because that is what a full run needs -- an accelerator
    and whatever dataset the notebook walks through -- so it is excluded from the
    offline gate by the same selection the other accelerator tests use, and runs under
    ``make test-gpu``.
    """
    run_notebook(source, tmp_path, fast=False)
