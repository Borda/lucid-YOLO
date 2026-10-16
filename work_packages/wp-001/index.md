# 📦 WP-001 — `chore(repo): scaffold src layout, pyproject, Makefile`

> Engineering log · [Phase 0 — Foundation](https://borda.github.io/lucid-YOLO/engineering/phase-00-foundation/)
>
> 🔗 Depends on none · Needed by WP-002, WP-003
>
> [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-004](https://borda.github.io/lucid-YOLO/work_packages/wp-004/) ➡️


## WP-001 — one place to write the version

<a id="wp-001"></a>

The version was declared twice and the two **had already drifted**: `pyproject.toml` said `0.1.0` while `lucid_yolo.__version__` said `0.0.1.dev0`, so a build published one number while every runtime consumer reported the other. `pyproject` now declares the version dynamic and reads the module attribute.

The attribute stays a plain string **literal** deliberately: setuptools resolves a literal by static analysis and only falls back to importing the package when it cannot, so a computed value would pull torch into every build and every source install. A meta test pins the literal form for that reason.

Also recorded — a test deliberately **not** written: `importlib.metadata` asserted against `__version__`. An editable install freezes its metadata at install time, so that assertion would fail after every bump until someone reinstalled, i.e. fail for a reason unrelated to what it claims to guard.


> Engineering log · [Phase 0 — Foundation](https://borda.github.io/lucid-YOLO/engineering/phase-00-foundation/)


## WP-001 — markdown formatting, and what it found

<a id="wp-001-mdformat"></a>

Directly relevant to why this file exists.

`--compact-tables` was added on evidence rather than preference. **Table padding is charged per column at its widest cell**, and this repository's governance rows run past 2,500 characters, so padded output grew the tracked markdown from **162 kB to 375 kB — all of it trailing spaces**. Compact output is size-neutral at 161 kB. A single very wide cell taxes every other row in its table.

The GFM plugin is required, not optional: core mdformat is CommonMark-only, reads a GFM table as a paragraph, and `--wrap=no` would then collapse every row onto one line.

The content-preservation check — strip whitespace, cell pipes and backslash escapes, compare the remaining characters before and after — **found a real defect**. A roadmap row held an unescaped pipe inside a code span, and a pipe opens a table cell wherever it appears: the row parsed as **eight cells against a six-column header**, so GitHub had been rendering its tail into the wrong columns and silently dropping the overflow. A sweep found no other row in the repository whose cell count disagreed with its header.

