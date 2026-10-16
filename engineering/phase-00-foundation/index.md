# 🧱 Phase 0 — Foundation


## WP-001 — one place to write the version

<a id="wp-001"></a>

The version was declared twice and the two **had already drifted**: `pyproject.toml` said `0.1.0` while `lucid_yolo.__version__` said `0.0.1.dev0`, so a build published one number while every runtime consumer reported the other. `pyproject` now declares the version dynamic and reads the module attribute.

The attribute stays a plain string **literal** deliberately: setuptools resolves a literal by static analysis and only falls back to importing the package when it cannot, so a computed value would pull torch into every build and every source install. A meta test pins the literal form for that reason.

Also recorded — a test deliberately **not** written: `importlib.metadata` asserted against `__version__`. An editable install freezes its metadata at install time, so that assertion would fail after every bump until someone reinstalled, i.e. fail for a reason unrelated to what it claims to guard.



## WP-001 — markdown formatting, and what it found

<a id="wp-001-mdformat"></a>

Directly relevant to why this file exists.

`--compact-tables` was added on evidence rather than preference. **Table padding is charged per column at its widest cell**, and this repository's governance rows run past 2,500 characters, so padded output grew the tracked markdown from **162 kB to 375 kB — all of it trailing spaces**. Compact output is size-neutral at 161 kB. A single very wide cell taxes every other row in its table.

The GFM plugin is required, not optional: core mdformat is CommonMark-only, reads a GFM table as a paragraph, and `--wrap=no` would then collapse every row onto one line.

The content-preservation check — strip whitespace, cell pipes and backslash escapes, compare the remaining characters before and after — **found a real defect**. A roadmap row held an unescaped pipe inside a code span, and a pipe opens a table cell wherever it appears: the row parsed as **eight cells against a six-column header**, so GitHub had been rendering its tail into the wrong columns and silently dropping the overflow. A sweep found no other row in the repository whose cell count disagreed with its header.



## WP-004 — auditing what a wheel *bundles*, not what it declares

<a id="wp-004"></a>

The copyleft audit read what a distribution says about itself — `License`, `License-Expression`, OSI classifiers — and nothing about what it ships. A wheel may vendor a native library under a licence its own metadata never mentions, which is precisely the case the gate exists to catch.

Reading the `License-File` documents too, it found something on the first run:

| package | declares | bundles |
| -- | -- | -- |
| numpy | BSD-3-Clause | libgfortran, libgcc — GPL-3.0-or-later **with GCC Runtime Library Exception**; libquadmath — LGPL-2.1-or-later |
| shapely | BSD-3-Clause | GEOS binaries — LGPL-2.1 |

numpy is a runtime dependency, not a test-only one, and no audit this project had run could ever have seen it.

Three rules keep the result meaningful rather than merely red:

- **Only declaration lines are read, never licence prose.** A bundled copy of the LGPL names the GPL on dozens of its own lines, and a gate that fires on every wheel shipping a licence file teaches its reader to bypass it.
- **A recognized licence exception is permissive by construction**, matched by expression rather than by name, so a future gcc-built dependency needs no entry.
- **The exception is tested before the allowlist, not after.** Written the other way round the allowlist short-circuits first, the exception never runs, and a per-package entry quietly covers whatever that package vendors *next*. Mutation testing is what caught the ordering: with numpy delisted the audit reports libquadmath alone, and with the exception disabled it reports the GPL-3 line — evidence the two mechanisms act independently rather than one masking the other.

A related gate lesson from the same package: the decisions test moved from a hardcoded `D1–D14` to **contiguity plus a floor**. The hardcoded upper bound failed the moment a decision was added, which is a test asking for maintenance rather than reporting a defect; contiguity alone would not notice the last row being deleted, since what remains stays contiguous. The roadmap's `_WP_FLOOR` ratchet is the same pattern.



## WP-005 — goldens that survive a second machine

<a id="wp-005"></a>

The first fixture golden byte-hashed the generated annotation JSON. Those hashes are per-platform stable only — see [Float reproducibility](https://borda.github.io/lucid-YOLO/engineering/cross-cutting/#cross-float) — so they were replaced with structural metrics: exact integer counts (images, annotations, categories, polygon points) plus bbox area and coordinate sums pinned with a small absolute tolerance.

The split that has held ever since: **integers exact, aggregates to tolerance**. Counts are platform-stable by construction; sums of floats are not. Every golden this project has frozen follows it.

