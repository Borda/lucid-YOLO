# 🔁 Phase 11 — Rolling, toward v0.5.0


## WP-112 — two compositions of the same decode, and the one that can't filter

<a id="wp-112"></a>

[WP-066](https://borda.github.io/lucid-YOLO/work_packages/wp-066/) needed one importable `nn.Module` per task so `torch.onnx.export` had something to trace, and wrote three small wrappers private to the test file to get one. `predict.py` composes the same pieces — a task's `deploy()` view plus its E2E decoder — for single-image inference, independently. [WP-066](https://borda.github.io/lucid-YOLO/work_packages/wp-066/)'s own log entry named the risk and declined to fix it: nothing forced the two compositions to keep agreeing as either side changed, and nothing shipped needed a third caller to justify the extraction. Roadmap 112 is that extraction, now that something does.

`src/lucid_yolo/export.py` reproduces `predict_image`/`predict_segmentation`/`predict_oriented`'s `"e2e"` compositions call for call — same decoder calls, same argument order, same `decode_instance_masks(prototypes, coefficients, boxes, image_size=...)` order, same `decode_rboxes` -> `o2o_rotated_topk` pairing — so the exported graph and the single-image path can be shown to agree rather than merely resemble each other. `predict.py` itself is untouched: it calls a raw `DetectionLitModule` and decodes an unbatched result with confidence filtering, a genuinely different call convention from a traced graph's fixed-shape `nn.Module`, and unifying the two call conventions was never this row's scope.

One divergence is real and stays, documented rather than engineered away. `predict_segmentation` drops padding rows (`anchor_index == PAD_ANCHOR_INDEX`) before gathering mask coefficients, since a single-image caller wants a mask per real object and a boolean filter is free to shrink the output. A traced graph's output shape is fixed at trace time, so `SegmentExportGraph` gathers every one of its `k` rows, padding included — which only stays safe because a padding row's index is a valid position in the coefficient tensor exactly when the canvas's anchor count exceeds `k`. Below that, gathering at the padding sentinel (`-1`) raises. This is the same constraint the ONNX export test's 128 px canvas (336 anchors, above the 300 cap) was already built to satisfy; `export.py`'s module docstring restates it because the module now has callers that test fixture does not.



## WP-113 — an emoji changes the anchor a renderer derives

<a id="wp-113"></a>

Decoration with one load-bearing consequence. GitHub and Python-Markdown both slugify a heading by stripping what is not a word character and joining the rest with hyphens, so `## 🧠 How a modern YOLO works` no longer answers to `#how-a-modern-yolo-works` — the stripped emoji leaves the leading separator behind. The README's own audience table links to four of its H2 sections, and every one of those links would have gone dead in the same commit that made the page prettier, with nothing failing anywhere.

The fix is not to spell the new slug. The two renderers this project publishes through do not have to agree on what a slug becomes, and one that today keeps a leading hyphen is free to trim it tomorrow — so a link written against a derived anchor is a link written against a renderer's current behaviour. Explicit `<a id=>` tags above the four headings are what the research log already uses for exactly this reason: they are greppable, they survive a retitle, and they mean the same thing to both renderers.

**Two gates read heading text and had to stop.** `test_report_sections` pinned four `## `-prefixed literals from the reproduction report and `test_decisions_carry_all_ids` pinned `## ADR-00N`; both would have failed on a purely visual edit while reporting a missing section, which is a gate that misdescribes its own finding. They now compare undecorated titles — a lead token holding no ASCII alphanumeric is decoration, anything else is title — so the assertion is about which sections exist, which is what it was always for.

The emoji themselves are chosen per section and reused across files only where the subject is the same: 🎯 for detection, 🖌️ for segmentation, 🔄 for oriented detection, in the roadmap phase, the log phase, the model card and the training recipe alike. A palette walked in file order would have been faster to produce and would carry no information.



## WP-113b — a README cannot be relative in two places at once

<a id="wp-113b"></a>

`pyproject.toml` hands `README.md` to setuptools as the long description, and PyPI resolves its relative targets against `pypi.org`: the three training-curve figures render as broken images and every `docs/…` link 404s. The same file has to stay relative in the git tree, where relative is exactly what works. So the rewrite belongs to the moment of packaging rather than to the file, and it is opt-in per invocation — `--ref v0.4.0` or `LUCID_YOLO_RELEASE_REF` — so an ordinary `make build` or a `pip install .` cannot reach it.

**Two hosts, because the two link classes fail differently.** A figure served through `github.com/.../blob` renders an entire HTML page inside an `<img>`, which is a broken image; a document served through `raw.githubusercontent.com` hands the reader unrendered markdown. Images therefore resolve through `raw`, documents through `blob`, and the classifier keys on the markdown image bang *or* the target's suffix, since an HTML `<img src=>` carries no bang.

**A branch may not be pinned.** A released wheel's README is a snapshot of one tree, and a link into `main` describes whatever that branch holds when a reader clicks it — which is how a page for 0.4.0 ends up documenting code that shipped years later. Only a `v0.MINOR.PATCH` tag or a full commit sha is accepted.

**The slug is read from `project.urls.Homepage`, and reading it exposed a live defect.** The declared homepage was `Borda/lit-YOLOs`; the repository's canonical `full_name` is `Borda/lucid-YOLO`. GitHub redirects a renamed repository's HTML URLs, so the stale slug looks fine in a browser and every `blob` link would have worked — but `raw.githubusercontent.com` does not redirect, so precisely the three figures this package exists to publish would have 404'd, while the links beside them resolved. A rename is invisible until something reads the metadata programmatically.

What the tests cannot establish is whether the produced URLs resolve. The repository is private at the time of writing, so every generated link 404s for an anonymous reader regardless of correctness; the URLs become true when the repository is public and the tag exists. That is stated rather than asserted, because a test that fetched them would be a test of the repository's visibility settings.



## WP-114 — a default list that vanishes when you name one member

<a id="wp-114"></a>

The site publishes what the repository already holds: `mkdocs.yml` owns no prose beyond `docs/index.md`, no plugin generates a page from source, and every register stays a plain markdown file a reader can open on GitHub without the toolchain. The work was configuration, and configuration is where a wrong value reports success.

**Naming one member of a MkDocs default list replaces the whole list.** Supplying `plugins:` at all drops the implicit `search`, so a site whose search box finds nothing builds green; supplying `markdown_extensions:` drops `tables`, and this repository is wall-to-wall pipe tables — the assumption register, the roadmap, every acceptance table would render as literal pipe characters on a build that reports no error. Both are restated explicitly, and `tables` is asserted in `scripts/_tests/test_audit_docs_site.py` rather than trusted to stay listed.

**`--strict` is the whole value of building locally.** It fails on a link to a page that does not exist and on a page the nav never lists. A register nobody can navigate to is one nobody reads, and neither failure raises anything without the flag.

**A second config file needed a second hook instance, for the same reason `docs/` did.** Material's mermaid fence is enabled through a `!!python/name:` tag; MkDocs parses its config with `yaml.Loader` and constructs it, pre-commit's `check-yaml` uses `safe_load` and cannot. So `check-yaml` is now two instances — the default one excluding `mkdocs.yml`, and an `--unsafe` one scoped to it, which still validates the syntax of a 120-line nav without importing anything from it. `mdformat` split the same way and for the same shape of reason: `docs/` is now rendered by a dialect the rest of the tree is not written in, and running the mkdocs plugin over the README or `AGENTS.md` would apply that dialect to files no site renders.

**The audit has to run where the exposure enters.** The docs group is deliberately outside `make setup` and outside `dev`: no gate imports it, and a contributor who never builds the site never installs the tree. The consequence is that no other workflow's environment contains it, so the licence audit — which scans the installed environment rather than the diff (D15, D16) — runs in the docs job, ahead of the build. Measured over the installed tree: 106 distributions, 343 shipped binaries, no GPL-family licence declared, bundled or shipped. One transitive is not permissive and is named rather than buried: `certifi` is MPL-2.0, weak file-level copyleft, reached through `mkdocs-material` → `requests`; MPL-2.0 was already present via `pathspec`, which arrives with mypy.

**`configure-pages` reads by default, and cannot do otherwise with the token CI has.** Its `enablement` input defaults to `false`, and its own `action.yml` states that enabling requires a token other than `GITHUB_TOKEN`. So `pages: read` is the ceiling that job can use rather than a permission it is being denied, and the step is skipped on pull requests: it fails when Pages is not enabled, which would turn every PR red for a reason no PR can fix.



## WP-114b — an unlicensed dependency is not one the audit can see

<a id="wp-114b"></a>

Building the site prints a notice from the Material team about the upstream MkDocs 2.0 release. One line of it is a licence fact rather than an opinion about the release: "Currently unlicensed – unsuitable for production use". This project admits permissive licences only, and unlicensed is not a weaker version of permissive — it is stricter than the AGPL the audit bans, because the default with no declared licence is no grant at all.

**The audit would not have caught it.** `scripts/audit_licenses.py` matches `COPYLEFT_PATTERN` — a GPL-family regex — against what each installed distribution declares, and reports only what matches. A distribution that declares nothing matches nothing and passes silently, in the same run that reports the environment clean. The control this project relies on for licence exposure is shaped to catch a forbidden declaration, not an absent one, so the bound has to be a version cap: `mkdocs>=1.6,<2`, which keeps the resolution from happening rather than detecting it afterwards.

The cap is asserted in `scripts/_tests/test_audit_docs_site.py` because it is the kind of pin a routine dependency bump lifts without anyone deciding to. The assertion is keyed on the version operator directly after the name, since `mkdocs-material` shares the prefix and sits two lines away in the same list — without that, a deleted `mkdocs` pin would move the assertion silently onto its neighbour and keep passing. Resolution after the cap: mkdocs 1.6.1, mkdocs-material 9.7.7, 29 packages. Lift it when 2.x ships a permissive `LICENSE`, verified from the repository rather than from the notice.



## WP-115 — the check that could not see what it was capping

<a id="wp-115"></a>

`scripts/audit_licenses.py` read three surfaces and each of them matched a forbidden pattern against a declaration, which meant a distribution declaring nothing matched nothing and passed — in the same run that printed "license audit clean". [WP-114b](https://borda.github.io/lucid-YOLO/work_packages/wp-114b/) hit that wall directly: the exposure it wanted to prevent could only be bounded with a version cap, because the audit had no way to report it. The fourth surface is the absence of a licence.

**Reading a licence with no field to read it from.** One installed distribution declares nothing: `faster-coco-eval` 1.7.2, PEP 639 metadata with no `License`, no `License-Expression`, no `License ::` classifier, shipping `licenses/LICENSE` holding the Apache-2.0 text. So the naive rule — no field, no pass — would have failed on a genuinely Apache-2.0 base dependency on its first run, which is how a gate gets turned off. `bundled_license_indicators` was no help by design: it reads `License: ...` declaration lines rather than prose, because a bundled LGPL names GPL on dozens of its own lines, and the Apache text carries no such line at all. The recognizer added here identifies a document from a distinctive phrase in its own first 15 lines, and identification *is* the verdict.

**The copyleft pattern is deliberately not run over a recognized text as a second opinion.** MPL-2.0's own Secondary-License clause names the GNU General Public License, the GNU Lesser General Public License and the GNU Affero General Public License in consecutive lines. A belt-and-braces re-scan would therefore fail the audit on a licence this environment already carries through `certifi` and `pathspec`. The table lists permissive texts only, so the recognizer never has to decide that something is forbidden — an omission costs a false alarm, never a silent pass. The escalation runs the other way: a document that was *not* recognized and whose header names a GPL-family licence is a failure in any tier, because "no AGPL at any cost" does not care which group reached the package.

**Two tiers, and only the fourth check is graded.** A package in the `[project.dependencies]` closure is republished in this project's own wheel metadata and installed by everyone; an unreadable licence there fails. A package reachable only through `dev` or `docs` appears in no wheel metadata — PEP 735 groups are not published at all — is imported by `src/` never, and is vendored into no artifact; it is printed as a flag and the run passes. Both-tier packages take the stricter attribution, and so do packages the walk cannot attribute, so a hole in the resolver presents as a loud failure rather than as a quiet demotion.

**Attributing the tier was the larger half.** The audit walks a flat installed environment; the tiers are a property of `pyproject.toml`, so the closure has to be resolved the way an installer would. The first attempt dropped every requirement carrying an `extra` marker and left 19 of 105 distributions unattributed. Carrying the requested extras through the walk and evaluating each marker against them closes that: the residue is one distribution, `lucid-yolo` itself, which no dependency list mentions because it *is* the project. Final attribution: 56 base, 29 dev, 19 docs. `packaging` became a declared `dev` dependency in the process — it was already there transitively through both pytest and mkdocs, and a direct import is not a transitive dependency (R34).

The audit still prints one line on a clean run, and the flag lines only when there are any.



## WP-115b — the first thing the new check found was real

<a id="wp-115b"></a>

[WP-115](https://borda.github.io/lucid-YOLO/work_packages/wp-115/) shipped a check for the absence of a licence and it fired within a day, on a CUDA environment this laptop is not: `cuda-toolkit`, reported as a base-tier failure. The first question was whether the check was wrong. It is not. The wheel was downloaded and opened: `cuda_toolkit-13.3.1.dist-info/` contains `METADATA`, `WHEEL` and `RECORD`, and nothing else. No `License`, no `License-Expression`, no `License ::` classifier, no licence document declared or undeclared. Its every requirement is extras-gated, so a bare install pulls nothing at all. There is genuinely no statement of terms in that distribution to read.

**So the resolution is an allowlist, not a weaker check.** `UNREADABLE_ALLOWLIST` is the fourth allowlist in this file and is kept separate from the three beside it for the reason D16 already gave about the third: the exposures differ, and folding them together lets one package's silence excuse another package's declaration. `ALLOWLIST` excuses a package's own copyleft declaration, `BUNDLED_ALLOWLIST` and `BUNDLED_BINARY_ALLOWLIST` excuse a vendored copyleft library, and this one excuses metadata that says nothing. Entries require a DECISIONS.md row first, as all three others do; `cuda-toolkit` is the only one.

**CUDA is admitted as an environment, not as a licence** (D17). The NVIDIA CUDA EULA is proprietary, which the permissive-only policy would otherwise refuse outright. What makes it out of scope is that `pyproject.toml` declares no CUDA package, this project redistributes no part of the toolkit, and it publishes no trained weights (D14) -- so what the EULA covers is the machine an accelerator run happens on rather than anything this repository ships. That is written down rather than left as a silence, because an exemption nobody recorded is indistinguishable from an oversight.

**A second, smaller thing the finding exposed.** The report printed `(base)` for two different situations: a package the shipped closure actually reaches, and a package the walk never reached at all, which defaults to the strict tier by design. Both are failures, for opposite reasons, and spelling them identically sends a reader looking for a dependency declaration that does not exist -- which is exactly the wrong first move when the real answer is "nothing requires this; it was installed by hand". The unattributed case now says so in its own reason.

**And the suite stopped reading the installed environment.** Seven tests and nine doctests asserted over `metadata.distributions()` -- the whole venv is clean, `shapely` really does ship GEOS, `torch` attributes to base -- and none of those is an assertion about this code. A contributor who installs anything can fail them without touching a line of the audit, which is exactly what `cuda-toolkit` did. Had the allowlist landed on its own, `test_the_live_environment_has_no_unreadable_license` would have gone green and the next package installed on that box would have turned it red again. The live claim belongs to the pre-commit hook, which runs the audit against the real environment on every commit and is the actual gate; a test's job is that the audit *decides* correctly, and synthetic distributions answer that without depending on what anyone happens to have installed. The shapes those fixtures are written to were read off real wheels once and recorded in their docstrings -- the `cuda_toolkit` dist-info listing above is one of them.

Two things that went with it are worth naming. The GEOS true-positive test is gone; what stands in for it is the synthetic manylinux spelling (`libgeos_c-abcdd5fa.so.1.19.2` through `library_stem`) plus the parametrized normalizer cases, so the regression it guarded -- the stem normalizer going blind on Linux, where CI actually runs -- is still caught. And `audit = _load_audit()` at module level became a session-scoped fixture: loading a non-package script by path is fine, doing it as an import side effect at collection time is not.



## WP-116 — a formatter that reads `Eq.` as the end of a sentence

<a id="wp-116"></a>

docformatter joins the three formatters already on the commit hook, configured in `[tool.docformatter]` and ordered ahead of ruff-format so the docstring body is rewritten first and the quotes and indentation around it normalized second -- the reverse order converges too, but only on the commit after.

**Wrapping is off in both axes, and that is the load-bearing setting rather than a timid one.** `wrap-descriptions = 0` because this project's docstrings carry argued paragraphs, `**bold**` lead-ins, and Google `Args:`/`Returns:` blocks whose indentation is what Napoleon parses; re-flowing them to a column would run those blocks together and rewrite hand-chosen line breaks across the whole tree in a single commit nobody could review. `wrap-summaries = 0` for a smaller reason: a summary is one sentence by construction and has nothing to gain from being re-flowed.

**What it cost once was 32 summary lines.** docformatter decides where the summary ends by finding the first period it reads as a sentence end, and it reads every abbreviation that way. This repository writes the papers' own notation -- `R1 Eq. 15`, `blueprint sec. 5.9`, `Redmon et al.` -- in summary lines, so those summaries were cut mid-citation and everything after them re-indented as continuation text, which is what turns an `Args:` block into prose Napoleon cannot parse:

```
"""Initialize the two coarse-level projections required by Eq.

8.
        Args:
            in_channels: Per-level neck channel counts ``(N3, N4, N5)`` in
```

No option prevents it. `--wrap-summaries 0` disables wrapping, not the split; `--non-strict` governs reST list detection; `--docstring-length` would only choose which docstrings get mangled. So the source is what changed: `Eq.` and `sec.` are spelled out and `et al.` rephrased, **in summary lines only** -- the bodies keep the papers' abbreviations, since nothing reads them for sentence boundaries. One summary genuinely held two sentences and was split into a summary and a body paragraph, which is what PEP 257 asked for anyway.

**`non-cap` names 17 identifiers.** The formatter capitalizes a summary's first word, which is right for English and wrong for a name: `lucid-yolo:` became `Lucid-yolo:`, `mAP` became `MAP`, and `empty()`, `backward()`, `rboxes[i]`, `o2o` and the nine subpackage names were all renamed to something that does not exist. Each is listed rather than the check being disabled, so a future summary opening with an ordinary English word is still capitalized.

**The hook now reformats nothing, and that is the honest result.** Every line in this commit's diff is the one-time rephrase; docformatter's own second pass is empty, because these docstrings were already PEP 257-clean. Its value is prospective -- it is a gate on what gets written next, not a repair of what is there.



## WP-116b — a hook pinned to no interpreter in particular

<a id="wp-116b"></a>

[WP-116](https://borda.github.io/lucid-YOLO/work_packages/wp-116/)'s `docformatter` hook carried no `language_version`, so pre-commit built its isolated environment against whatever `python3` resolved first on `PATH` at hook-creation time. That happened to be a 3.11 install then; a later `python3.10` framework install on this machine moved ahead of it in `PATH`, and the next `pre-commit run` silently rebuilt the hook's environment against 3.10 -- a version below this project's own `requires-python = ">=3.11"` floor.

docformatter's own config reader needs `tomllib`, stdlib only since 3.11, to parse `[tool.docformatter]` out of `pyproject.toml`. Under 3.10 that import is simply absent, and the hook failed with `NameError: name 'tomllib' is not defined` rather than anything naming the real cause. Every other hook in the file that needs a specific interpreter says so explicitly -- `mypy` and `license-audit` both pin `entry: .venv/bin/python`. `docformatter` is a third-party hook rather than a local one, so it takes `language_version: python3.11` instead: the same guarantee, expressed the way pre-commit resolves environments for repos it clones rather than ones this project owns.

This is the same fault as `cross-venv` above -- an environment silently reconstructed between two commands -- reached through a different door: there it was `uv run` rebuilding `.venv`, here it is `pre-commit` rebuilding one hook's env off unpinned `PATH` resolution. Neither the tree nor the hook's own config changed; only what interpreter answered to `python3` did.



## WP-117 — documented and verified are not the same claim

<a id="wp-117"></a>

[WP-085](https://borda.github.io/lucid-YOLO/work_packages/wp-085/) wired `--doctest-modules` over `src`, `scripts` and `tests`, which turns a helper's `Examples:` block into an executable check the moment that block exists -- but nothing forced it to exist. A fresh `ast` count over `tests/**/test_*.py` found 261 non-fixture, non-`test_` helper functions, every one already carrying a docstring, and only 1 of the 261 carrying a `>>>` line `--doctest-modules` could actually run. Prose describing behaviour a reader had to trust, not a line pytest ever executed. 23 of 678 `test_` functions carried no docstring at all, all 23 in one file, `tests/data/test_download.py`.

Closing the gap by hand, file by file, surfaced real bugs the doctests would otherwise have hidden behind a passing suite: `_write_split`'s and `build.convert_split`'s return values auto-printed inside a `with` block under doctest's "single" exec mode, needing `_ = ...` to suppress; `AssignResult` has no `.labels` attribute, only `.target_labels`; `Trainer` has no `.deterministic` attribute; `_decode_boxes`'s hand-derived expected output was numerically wrong until computed from a real run rather than worked out on paper. Each was a doctest that failed on first execution -- exactly the outcome [WP-085](https://borda.github.io/lucid-YOLO/work_packages/wp-085/)'s infrastructure exists to produce, and exactly what a prose-only docstring would never have surfaced.

A second, unrelated inconsistency turned up mid-pass: some docstrings wrote a bare `>>>` block, others wrapped it in a proper `Examples:` header matching the Napoleon style `predict.py` and `export.py` already use. An `ast`-based script fixed 166 docstrings across 56 files in one pass -- locate each docstring's span, detect a `>>>` line with no preceding `Examples:`, insert the header and re-indent -- verified against a doctest re-run before and after rather than assumed safe.

The "8 groups across 6 files" class-regrouping half of the original scope did not survive contact with measurement: a first-shared-word `ast` scan returns ~90 candidate groups, because this suite's own descriptive-sentence naming house style is indistinguishable from genuinely flat enumeration by any heuristic tried. Split off into [WP-128](https://borda.github.io/lucid-YOLO/work_packages/wp-128/) rather than guessed at.



## WP-118 — one log grew two audiences

<a id="wp-118"></a>

`RESEARCH_LOG.md` had accumulated 85 entries answering two different questions -- what a reproduction claim cost to establish, and what the repo's own tooling cost to build -- under one charter that named only the first. The immediate trigger was smaller: rows 115, 115b and 116 had each grown to a paragraph of debugging narrative in the roadmap's own Scope column, because the log existed but the habit of writing to it first did not.

The split rule: does an entry record what the reproduction *claims* (model, loss, data protocol, eval numbers, paper ambiguity), or *how the repo builds, checks, and publishes itself* (CI, hooks, packaging, licence, formatting, CLI, docs tooling)? Most of the 85 sorted cleanly -- the whole of `Cross-cutting` and Phase 0 turned out to be tooling-kind, the whole of Phase 7 and most of Phase 8/9 stayed fidelity-kind. Five did not sort cleanly, because their finding genuinely was both: [WP-066](https://borda.github.io/lucid-YOLO/work_packages/wp-066/) (export tooling *and* an architecture-fidelity confirmation), [WP-079](https://borda.github.io/lucid-YOLO/work_packages/wp-079/) (a DataLoader bug *and* a measured cost to augmentation diversity), [WP-083](https://borda.github.io/lucid-YOLO/work_packages/wp-083/) (a golden-harness design *and* the frozen numbers it pins), [WP-091b](https://borda.github.io/lucid-YOLO/work_packages/wp-091b/) (a decoder implementation *and* an unsourced-threshold provenance investigation), [WP-097](https://borda.github.io/lucid-YOLO/work_packages/wp-097/) (a download-verify bug *and* what R18's published totals actually describe). Those five were not forced into one bucket; each is cut into two pieces, cross-linked both ways, sized to what each piece actually carries rather than split down the middle.

The mechanical risk in a retroactive split of this size is silent loss -- a paragraph dropped in the move, an anchor a roadmap row still cites that no longer resolves. The move itself was scripted rather than hand-edited per entry, asserting the parsed anchor set equalled the classification set before writing anything, so nothing could be silently skipped or double-counted; `scripts/_tests/test_audit_docs_present.py::test_every_log_link_resolves` (renamed from its single-file predecessor) now checks both files' anchors against the roadmap's citations of either.



## WP-119 — a symlink kept for a command nothing still runs literally

<a id="wp-119"></a>

[WP-038](https://borda.github.io/lucid-YOLO/work_packages/wp-038/) moved the experiment configs into `src/lucid_yolo/configs` as package data, so a wheel-only install could run `--config`, and kept a repo-root `configs -> src/lucid_yolo/configs` symlink so the commands already documented at the time -- ones that typed `configs/det_smoke.yaml` from repo root -- kept working. That was the right call then: a compatibility shim is cheaper than rewriting every example the moment a layout changes underneath them.

What made it safe to drop now is that nothing in the tree still needed it. Every path that resolves the configs directory in code does so package-relatively -- `Path(lucid_yolo.__file__).resolve().parent / "configs"` in `tests/ptl/test_cli.py` and `scripts/overfit_micro.py`, `Path(__file__).resolve().parents[1] / "configs"` in `packaged_config()` -- none of which touch the repo root at all. And `_resolve_config_args` (also [WP-038](https://borda.github.io/lucid-YOLO/work_packages/wp-038/)) already does the harder version of what the symlink did: a `--config` value that names a packaged config by bare filename, with or without `.yaml`, resolves onto the installed `lucid_yolo/configs` tree when no such file exists locally, so `lucid-yolo fit --config det_smoke.yaml` already works identically from a checkout or a wheel install, no path and no symlink required.

The one place the symlink was load-bearing rather than redundant was prose: `train.py`'s own module docstring showed `python -m lucid_yolo.cli.train fit --config configs/det_smoke.yaml`, a literal root-relative path that only resolved because the symlink put a real directory at `configs/`. That is now the bare-name form the resolver was actually built for. `AGENTS.md`'s `configs/data/*.yaml` and two comments in `tests/ptl/test_cli.py` and `scripts/overfit_micro.py` that named the bare `configs/` directory are reworded to `src/lucid_yolo/configs/...` or `lucid_yolo/configs/...` for the same reason -- accurate once nothing at repo root answers to that name. Historical entries elsewhere (`ENGINEERING_LOG.md`'s own [WP-099b](https://borda.github.io/lucid-YOLO/work_packages/wp-099b/) entry, completed roadmap rows for [WP-038](https://borda.github.io/lucid-YOLO/work_packages/wp-038/) and [WP-099c](https://borda.github.io/lucid-YOLO/work_packages/wp-099c/)) describe what was true when they were written and are left alone; a backfilled record does not get edited to match a later decision.



## WP-127 — a version number is not a run's identity

<a id="wp-127"></a>

Lightning's own default checkpoint path is `lightning_logs/version_N/checkpoints/epoch=X-step=Y.ckpt` -- `version_N` numbers the *run*, sequentially, across every task and scale a working tree has ever fit; nothing in the path names what the run actually trained. [WP-111](https://borda.github.io/lucid-YOLO/work_packages/wp-111/)'s re-scoring of [WP-064](https://borda.github.io/lucid-YOLO/work_packages/wp-064/)'s OBB-smoke checkpoint hit this directly: `version_10` said nothing until cross-referenced against its `hparams.yaml`'s `task: obb` field, a step that only works because the checkpoint happened to still carry its hyperparameters.

The fix stays inside `DetectionCLI.instantiate_trainer` (`cli/train.py`), the same injection seam [WP-038](https://borda.github.io/lucid-YOLO/work_packages/wp-038/) already uses for the progress bar and the default logger pair: unless a config places its own `ModelCheckpoint` in `trainer.callbacks`, one is now injected with `filename=_checkpoint_filename(self.model.task, variant)` -- a new pure helper prefixing `{task}_{variant}_` onto Lightning's own `{epoch}-{step}` template, left as a literal placeholder so `ModelCheckpoint`'s own metric-filling machinery still resolves it. `dirpath` is left at Lightning's default (`<version dir>/checkpoints`), so `lightning_logs/version_N` numbering is untouched and no already-written checkpoint moves or is renamed -- only new runs pick up the prefix. `task` comes from `self.model.task` (the instantiated module, already built by the time `instantiate_trainer` runs) rather than the raw config, since only the module's own default (`task="detect"`) is guaranteed correct once link-computed arguments are in play.

The injection duplicated the progress-bar callback's four-line "append without replacing the config's own list" pattern verbatim, so it moved into a small `_add_trainer_default_callback` method both call into, rather than being copy-pasted a second time.



## WP-128 — a shared prefix is not a shared caller

<a id="wp-128"></a>

[WP-117](https://borda.github.io/lucid-YOLO/work_packages/wp-117/)'s own count -- "8 groups across 6 files" -- was a one-off human read, not a reproducible measurement, and the row it left behind said so: a first-shared-word `ast` scan over `test_` names returns roughly 90 candidate groups, because this suite's house style already writes full descriptive sentences (`test_boundary_sits_at_midpoint_of_overlap_band`) that share a topic noun with unrelated tests -- indistinguishable from genuinely flat, mechanical enumeration (`test_<subject>_<case>`) by prefix-matching alone.

The discriminator that breaks the tie: a candidate prefix confirms only if it is a contiguous token subsequence -- prefix, suffix, or infix -- of some `ast.Call` function-name target actually present in the same file (`Attribute.attr` or `Name.id`, underscore-split, leading-underscore tokens dropped). A house-style sentence sharing a topic noun calls nothing that looks like it and never confirms; `test_check_data_a/b/c` calling `check_data(...)` does. Confirmed groups need 3+ members to enforce, matching the size floor `tests/eval/test_tile_merge.py`'s own precedent already implied by example. Verified both directions against real cases in this tree: correctly includes `check_data`, `plan_archives`, `verify_split`, `configure_optimizers`, `warmup_decay`, `on_after_batch_transfer`, `detections_to_predictions`, and `rotated_topk` (confirmed as an infix of `o2o_rotated_topk`); correctly excludes `deployed_view`-style near-misses and the `a_named_worker_count_*` / `an_auto_chosen_*` house-style family [WP-117](https://borda.github.io/lucid-YOLO/work_packages/wp-117/)'s row already flagged as the hard case.

New `scripts/lint/audit_flat_test_groups.py` owns this rather than extending `audit_test_doctests.py` (129) -- that script's own docstring disclaims scanning for anything but missing doctest Examples, and stapling an unrelated structural check onto it would have made neither job legible from the file alone. Same shape as every other `scripts/lint/` checker: `main(argv) -> int` CLI, a `flat-test-group-audit` local hook gated on `files: ^(tests/.*/test_[^/]+\.py|scripts/_tests/test_[^/]+\.py)$` (pure repo content, not installed-environment state, so no `always_run`), and `scripts/_tests/test_audit_flat_test_groups.py` as the functional core -- synthetic `tmp_path` fixtures per confirmed/rejected case, plus the two "live tree is currently clean" sanity tests the hook itself depends on staying true.

Nine files regrouped under `class Test<Subject>:` once the discriminator was run against the whole `tests/` and `scripts/_tests/` tree rather than continuing [WP-117](https://borda.github.io/lucid-YOLO/work_packages/wp-117/)'s by-eye survey: `test_verify.py`, `test_download.py`, `test_coco_eval.py`, `test_schedule.py`, `test_dota_parse.py`, `test_coco.py`, `test_obb_head.py`, `test_tiling.py`, `test_proto.py`, `test_module.py`, and `scripts/_tests/test_audit_docs_present.py` (all 12 of its check-function pairs, for whole-file consistency, though the discriminator's crude grouping key only flagged 6) and `test_audit_docs_site.py`. Regrouping moves node ids, not test bodies -- Python resolves a module-level helper name at call time, not at class-body-definition time, so relocating a scattered function into a contiguous class needed no change to whatever module-level fixtures it called. Every citation of a moved node id in `docs/ROADMAP.md` (rows 044, 051, 110) and `docs/ASSUMPTIONS.md` (A45) was checked against a grep for the old id and updated; row 044's citation had already drifted stale before this WP touched the file (`test_oracle` moved under a pre-existing `TestOracleRoundTrip` by earlier, unrelated work) and was corrected in the same pass since the row was already open.



## WP-129 — a check that scans the repo is not a unit test of it

<a id="wp-129"></a>

[WP-117](https://borda.github.io/lucid-YOLO/work_packages/wp-117/) landed its enforcement as `tests/meta/test_test_suite_quality.py`: a pytest test whose body is an `ast` walk over `tests/**/test_*.py`, asserting no helper lacks a doctest Example. Functionally that is a lint check -- it scans repo content for a house-style violation, the same shape as `audit_licenses.py` scanning the installed environment for a licence violation -- wearing a pytest test's clothes because that was the fastest place to put it at the time.

The fix is the split this project already draws for `audit_licenses.py` and `check_commit_trailers.py`: the check itself is a `scripts/` script with a `main(argv)` CLI entry point, invoked by a `.pre-commit-config.yaml` local hook; pytest keeps a meta-test exercising the script's own functions (`module_level_helpers`, `has_doctest_example`, `find_missing`) against synthetic `tmp_path` fixtures, the way `test_commit_trailers.py` exercises `validate_message` against synthetic commit messages rather than real repository history. Pytest stays scoped to the functional core -- does this function do what it claims, on cases chosen to exercise it -- and the hook owns the enforcement gate against the live tree.

Unlike `license-audit`, the new `test-doctest-audit` hook is not `always_run: true`: it scans pure repo content (`tests/**/test_*.py`), not installed-environment state that can drift without a tracked-file change, so it is gated on `files: ^tests/.*/test_.*\.py$` and only runs when a test file is actually part of the commit.

Moving the script surfaced one more inconsistency worth fixing in the same pass rather than leaving for a future WP to trip over: `scripts/` had grown to eleven files at its root with no separation between the two things a hook-invoked checker and a golden-regression producer actually are. `audit_licenses.py` and `check_commit_trailers.py` -- already hook-invoked -- move into a new `scripts/lint/` alongside the new script; `audit_licenses.py`'s own `PYPROJECT` path, resolved via `Path(__file__).resolve().parents[N]`, needed its `N` bumped for the extra directory level, caught immediately by the pre-commit hook itself failing on the next run rather than silently reading the wrong file. `check_goldens.py`, `golden_producers.py`, `shapes_regression.py` and the rest of the golden/training/release tooling stay at `scripts/`'s root: their dotted `scripts.<module>:<function>` paths are written into checked-in `goldens/**/*.json` `producer` fields as data, not just import statements, and rewriting those paths is a fixture migration across files that took minutes-each GPU runs to produce -- a materially different, materially riskier change than moving two files whose only callers are a Makefile line, a CI step and their own meta-tests.

______________________________________________________________________



## WP-130 — a test of a script belongs beside the script

<a id="wp-130"></a>

[WP-129](https://borda.github.io/lucid-YOLO/work_packages/wp-129/) split the doctest-audit check itself out of `tests/meta/` but left its test where the whole family had always lived, `tests/meta/test_audit_test_doctests.py` -- and left the other eight `tests/meta/` files, plus `tests/scripts/`, `tests/train/test_shapes_regression.py` and `tests/integration/test_overfit_micro.py`, exactly where they were: every one of them a test whose subject is a `scripts/` module, sitting in `tests/` anyway because that is where tests conventionally go. The convention was wrong for this shape: `src/lucid_yolo/` and `tests/unit/` mirror each other by design, and `scripts/` had grown its own testable surface without growing its own mirror. Fixed by moving all fifteen files into a new `scripts/_tests/`, each renamed `test_<script_name>.py` after the module it exercises -- `test_license_audit.py` becomes `test_audit_licenses.py`, `test_golden_harness.py` becomes `test_check_goldens.py` -- so the pairing reads directly off the two filenames rather than needing the docstring to say which script a test covers.

The other five `tests/meta/` checkers still inline (docs presence, docs site, figure captions, license headers, version single source) convert the same way [WP-129](https://borda.github.io/lucid-YOLO/work_packages/wp-129/) converted the doctest audit: a `scripts/lint/<name>.py` script plus a `.pre-commit-config.yaml` hook, `scripts/_tests/test_<name>.py` left as the functional-core test. `release_guard.py` and `check_goldens.py` were already scripts with no hook -- both gain one, `stages: [manual]` like nothing else in the file, since a release tag and a golden recompute are not per-commit checks. `release_guard.py` needed one real change to be hook-shaped at all: `--tag` was `required=True`, unrunnable from a hook that fires with no arguments on every commit. It now defaults to `git describe --tags --exact-match HEAD` and no-ops when `HEAD` isn't exactly a tag -- true for nearly every commit, which is the point.

Every hook's `name:` field gained a distinct leading emoji chosen for the action it performs (🧠 mypy, ⚖️ license audit, 📋 docs presence, 🗺️ docs site, 🖼️ figure captions, 📄 license headers, 🔢 version single source, 📝 doctest audit, 🏷️ commit trailers, 🥇 golden check, 🛡️ release guard) -- eleven hooks read a lot faster on `pre-commit run --all-files`'s output when each line starts with a different glyph than when all eleven start with the same bullet.

The move itself broke two things that had never been exercised before, both structural rather than content bugs. **mypy**: `[tool.mypy]` scans `files = ["src", "scripts"]` with no `mypy_path` entry for the repo root, so a file inside `scripts/` got two identities at once -- `absolutize_readme` from the direct directory walk, `scripts.absolutize_readme` from `scripts/_tests/test_absolutize_readme.py`'s own `from scripts.absolutize_readme import ...`, now itself swept into the `scripts` scan for the first time. `explicit_package_bases = true` plus `mypy_path = "src:."` gives both paths the same resolution; `scripts/_tests/` itself is excluded from strict checking afterward, matching `tests/`'s own long-standing exemption for exactly the same fixture-heavy, `monkeypatch`-typed, dynamically-imported character. **pytest**: `scripts/_tests/conftest.py` and `tests/conftest.py` are both non-package `conftest.py` files with the bare module name `conftest`; under the default prepend import mode, loading them from two different top-level collection roots (`scripts` and `tests` in the same `pytest ... src scripts tests` invocation) rather than nested under a shared one raised `import file mismatch`. `--import-mode=importlib` fixes it structurally, keying each module's identity on its resolved path rather than its stem, and is now the addopts default for every invocation, not just this one.

`scripts/README.md` is new: a table per subtree (core, `lint/`, `_tests/`) naming every script, what it does, what invokes it, and which test covers it -- the crossroad a developer new to `scripts/` reads first, and the thing that makes "no dedicated test, doctests only" for `golden_producers.py`, `plot_training.py`, and `dump_debug_grid.py` a documented decision instead of a silent gap.

______________________________________________________________________



## WP-130b — an emoji requirement is not the same claim as "every local hook"

<a id="wp-130b"></a>

[WP-130](https://borda.github.io/lucid-YOLO/work_packages/wp-130/)'s emoji pass read "every hook" as "every `repo: local` hook" and stopped at eleven -- the other eleven, all third-party (`pre-commit-hooks`, `mdformat`, `docformatter`, `ruff-pre-commit`), kept their plain default or custom names. Fixed by giving each of those eleven a `name:` override where none existed (`end-of-file-fixer`, `check-yaml` x1, `check-toml`, `check-added-large-files`, `ruff-check`, `ruff-format`) or an emoji prefix on the existing custom one (`check-yaml`'s unsafe-tags instance, `mdformat` x2, `docformatter`). Twenty-two hooks, twenty-two distinct emoji, chosen per action rather than per category so the same glyph never covers two conceptually different checks.

Separately: `pre-commit run --all-files` was only ever exercising nineteen of the twenty-two hooks, by design at the time -- `commit-trailers` runs at `commit-msg` stage, `golden-check`/`release-guard` at `stages: [manual]`, none of which `--all-files` (a `pre-commit`-stage run) reaches. Asked to close that gap too. `golden-check` and `release-guard` are cheap enough off a release tag (one `git describe` call, one already-fast golden recompute) that `stages: [pre-commit, manual]` costs nothing meaningful on every commit -- the accepted trade is `make gate` now calls `check_goldens.py` twice, once inside `precommit`'s `--all-files` and once via the `golden` target, rather than carrying a second exemption to remember. `commit-trailers` cannot take the same fix: a `pre-commit`-stage run has no commit message to validate, since no commit is in progress. Its `--all-files`-reachable half is a new sibling hook, `commit-trailers-history`, wired to the same script's pre-existing `--range` mode (`--range origin/main..HEAD`) rather than new code -- it re-validates every commit not yet pushed, which is a strictly more useful check than re-reading the newest message alone, and was already there waiting to be pointed at from a hook.

______________________________________________________________________

