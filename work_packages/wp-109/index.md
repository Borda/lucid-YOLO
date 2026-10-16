# 📦 WP-109 — `fix(scripts): audit the binaries a wheel ships`

> Engineering log · [Phase 10 — Consolidation](https://borda.github.io/lucid-YOLO/engineering/phase-10-consolidation/)
>
> 🔗 Depends on [WP-004](https://borda.github.io/lucid-YOLO/work_packages/wp-004/) · Needed by none
>
> ⬅️ [WP-066](https://borda.github.io/lucid-YOLO/work_packages/wp-066/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-110](https://borda.github.io/lucid-YOLO/work_packages/wp-110/) ➡️


## WP-109 — a wheel can ship a GPL binary and declare it nowhere

<a id="wp-109"></a>

The package exists because of what sourcing [WP-067](https://borda.github.io/lucid-YOLO/work_packages/wp-067/)'s example turned up. `supervision`, the obvious drawing library for boxes, masks and rotated boxes, hard-requires `av>=14.2`. The `av` wheel declares `BSD-3-Clause`, ships a `licenses/LICENSE.txt` in which `grep -ci gpl` returns 0, and ships `av/.dylibs/libx264.165.dylib`. x264 is GPL-2.0. Running this repository's own audit functions against a throwaway install of it — never the project venv — returned `declared: ['BSD-3-Clause'] | bundled: []`: both checks clean, on a wheel carrying a GPL binary. The gate whose entire purpose is seeing this saw nothing, and the reason it saw nothing is that a vendored binary need not be documented anywhere at all. D15 was found by reading a license document the wheel happened to ship; this one had no document to read.

**The third check reads the file list, and the honest description of it is that it narrows the hole rather than closing it.** Filenames are matched against a named table of copyleft libraries — x264, x265, mp3lame, GEOS, the eight FFmpeg libraries — so a copyleft library the table has never heard of passes exactly as `av` did before the table existed. The alternative was considered and is not available: `torch` alone ships hundreds of legitimate native libraries, and a check failing on every unrecognized one would fire on every commit and be switched off within the week. What the table can honestly claim is that no *listed* library is present, and the module docstring says that rather than implying more.

Two things made the scan work at all, and both were near-misses.

**The normalizer has to know what `auditwheel` does, or the check works on macOS and goes blind on Linux.** The `libgeos_c.1.19.2.dylib` macOS ships verbatim arrives on manylinux as `libgeos_c-abcdd5fa.so.1.19.2`: the repair tool grafts eight hex characters onto the name. A table keyed on the plain stem matches every local run and nothing on the CI runner — and the failure presents as a green gate, which is the worst shape a licence check can fail in. Verified against all 20 vendored libraries in the `shapely` and `pillow` manylinux wheels, the graft is uniformly eight lowercase hex characters; it is stripped before matching, and a live test on `shapely` fails if the convention ever changes.

**The table's licence notes were the part most at risk of being written from memory.** Each entry claims a licence and where that licence was read, which is worth exactly as much as the reading actually done. So they were read: VideoLAN's x264 page states x264 is released under the GNU GPL and separately available commercially; x265's own `COPYING` is the GPL version 2 text; FFmpeg's `LICENSE.md` states most of FFmpeg is LGPL v2.1-or-later and that `--enable-gpl` changes its licence to GPL v2+, naming libx264, libx265 and libxvid together as the GPL-v2 externals that flag admits; the LAME page states LGPL and names no version, so the entry names none either. GEOS needed no page at all — `shapely`'s installed `LICENSE_GEOS` opens `License: LGPLv2.1`, which is the same exposure D15 recorded reached by a second route, and it is what makes the table's behaviour observable in this environment rather than hypothetical.

The allowlist is keyed on `(distribution, library)` rather than on the package, which is the one place this check deliberately differs from the bundled-document allowlist beside it. That file's own rule, from D15, is that a package excused for one vendored library must not be excused for every other one it ships — and a package-keyed entry here would break exactly that rule, since allowing `shapely`'s GEOS would also wave through a libx264 it started shipping tomorrow.

The tests are two-sided on purpose: the synthetic `av` must be caught, and the live environment — torch, numpy, pillow, onnxruntime, matplotlib, shapely, 340 native libraries between them — must stay clean. Either side alone is worthless. A check firing on the current venv gets disabled by the next person; a check firing on nothing is decoration. The positive case is a hand-written `.dist-info` built through `Distribution.at`, so the real `RECORD` parser runs, because `av` is the one package this tree must never install to test against.

Cost is 0.09 s to 0.24 s on the pre-commit hook, ~0.15 s of it walking 87 wheels' `RECORD` files. Every distribution in this environment has one; a distribution installed without a `RECORD` ships nothing as far as this check can see, which is stated where the reader will need it rather than left to be discovered.

`supervision` is refused on the strength of this (D16) and [WP-067](https://borda.github.io/lucid-YOLO/work_packages/wp-067/)'s example is drawn with `matplotlib`, already a dependency (R29). The finding is the more useful artifact than the example would have been.

