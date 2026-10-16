# 📦 WP-116b — `fix(deps): pin docformatter's hook to a compatible interpreter`

> Engineering log · [Phase 11 — Rolling, toward v0.5.0](https://borda.github.io/lucid-YOLO/engineering/phase-11-rolling/)
>
> 🔗 Depends on [WP-116](https://borda.github.io/lucid-YOLO/work_packages/wp-116/) · Needed by none
>
> ⬅️ [WP-116](https://borda.github.io/lucid-YOLO/work_packages/wp-116/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-117](https://borda.github.io/lucid-YOLO/work_packages/wp-117/) ➡️


## WP-116b — a hook pinned to no interpreter in particular

<a id="wp-116b"></a>

[WP-116](https://borda.github.io/lucid-YOLO/work_packages/wp-116/)'s `docformatter` hook carried no `language_version`, so pre-commit built its isolated environment against whatever `python3` resolved first on `PATH` at hook-creation time. That happened to be a 3.11 install then; a later `python3.10` framework install on this machine moved ahead of it in `PATH`, and the next `pre-commit run` silently rebuilt the hook's environment against 3.10 -- a version below this project's own `requires-python = ">=3.11"` floor.

docformatter's own config reader needs `tomllib`, stdlib only since 3.11, to parse `[tool.docformatter]` out of `pyproject.toml`. Under 3.10 that import is simply absent, and the hook failed with `NameError: name 'tomllib' is not defined` rather than anything naming the real cause. Every other hook in the file that needs a specific interpreter says so explicitly -- `mypy` and `license-audit` both pin `entry: .venv/bin/python`. `docformatter` is a third-party hook rather than a local one, so it takes `language_version: python3.11` instead: the same guarantee, expressed the way pre-commit resolves environments for repos it clones rather than ones this project owns.

This is the same fault as `cross-venv` above -- an environment silently reconstructed between two commands -- reached through a different door: there it was `uv run` rebuilding `.venv`, here it is `pre-commit` rebuilding one hook's env off unpinned `PATH` resolution. Neither the tree nor the hook's own config changed; only what interpreter answered to `python3` did.

