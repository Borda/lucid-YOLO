# 📦 WP-115 — `feat(scripts): an undeclared licence is a finding, per tier`

> Engineering log · [Phase 11 — Rolling, toward v0.5.0](https://borda.github.io/lucid-YOLO/engineering/phase-11-rolling/)
>
> 🔗 Depends on [WP-114b](https://borda.github.io/lucid-YOLO/work_packages/wp-114b/) · Needed by [WP-115b](https://borda.github.io/lucid-YOLO/work_packages/wp-115b/)
>
> ⬅️ [WP-114b](https://borda.github.io/lucid-YOLO/work_packages/wp-114b/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-115b](https://borda.github.io/lucid-YOLO/work_packages/wp-115b/) ➡️


## WP-115 — the check that could not see what it was capping

<a id="wp-115"></a>

`scripts/audit_licenses.py` read three surfaces and each of them matched a forbidden pattern against a declaration, which meant a distribution declaring nothing matched nothing and passed — in the same run that printed "license audit clean". [WP-114b](https://borda.github.io/lucid-YOLO/work_packages/wp-114b/) hit that wall directly: the exposure it wanted to prevent could only be bounded with a version cap, because the audit had no way to report it. The fourth surface is the absence of a licence.

**Reading a licence with no field to read it from.** One installed distribution declares nothing: `faster-coco-eval` 1.7.2, PEP 639 metadata with no `License`, no `License-Expression`, no `License ::` classifier, shipping `licenses/LICENSE` holding the Apache-2.0 text. So the naive rule — no field, no pass — would have failed on a genuinely Apache-2.0 base dependency on its first run, which is how a gate gets turned off. `bundled_license_indicators` was no help by design: it reads `License: ...` declaration lines rather than prose, because a bundled LGPL names GPL on dozens of its own lines, and the Apache text carries no such line at all. The recognizer added here identifies a document from a distinctive phrase in its own first 15 lines, and identification *is* the verdict.

**The copyleft pattern is deliberately not run over a recognized text as a second opinion.** MPL-2.0's own Secondary-License clause names the GNU General Public License, the GNU Lesser General Public License and the GNU Affero General Public License in consecutive lines. A belt-and-braces re-scan would therefore fail the audit on a licence this environment already carries through `certifi` and `pathspec`. The table lists permissive texts only, so the recognizer never has to decide that something is forbidden — an omission costs a false alarm, never a silent pass. The escalation runs the other way: a document that was *not* recognized and whose header names a GPL-family licence is a failure in any tier, because "no AGPL at any cost" does not care which group reached the package.

**Two tiers, and only the fourth check is graded.** A package in the `[project.dependencies]` closure is republished in this project's own wheel metadata and installed by everyone; an unreadable licence there fails. A package reachable only through `dev` or `docs` appears in no wheel metadata — PEP 735 groups are not published at all — is imported by `src/` never, and is vendored into no artifact; it is printed as a flag and the run passes. Both-tier packages take the stricter attribution, and so do packages the walk cannot attribute, so a hole in the resolver presents as a loud failure rather than as a quiet demotion.

**Attributing the tier was the larger half.** The audit walks a flat installed environment; the tiers are a property of `pyproject.toml`, so the closure has to be resolved the way an installer would. The first attempt dropped every requirement carrying an `extra` marker and left 19 of 105 distributions unattributed. Carrying the requested extras through the walk and evaluating each marker against them closes that: the residue is one distribution, `lucid-yolo` itself, which no dependency list mentions because it *is* the project. Final attribution: 56 base, 29 dev, 19 docs. `packaging` became a declared `dev` dependency in the process — it was already there transitively through both pytest and mkdocs, and a direct import is not a transitive dependency (R34).

The audit still prints one line on a clean run, and the flag lines only when there are any.

