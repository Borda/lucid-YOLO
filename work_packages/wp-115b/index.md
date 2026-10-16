# 📦 WP-115b — `fix(scripts): allowlist a licence that cannot be read`

> Engineering log · [Phase 11 — Rolling, toward v0.5.0](https://borda.github.io/lucid-YOLO/engineering/phase-11-rolling/)
>
> 🔗 Depends on [WP-115](https://borda.github.io/lucid-YOLO/work_packages/wp-115/) · Needed by none
>
> ⬅️ [WP-115](https://borda.github.io/lucid-YOLO/work_packages/wp-115/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-116](https://borda.github.io/lucid-YOLO/work_packages/wp-116/) ➡️


## WP-115b — the first thing the new check found was real

<a id="wp-115b"></a>

[WP-115](https://borda.github.io/lucid-YOLO/work_packages/wp-115/) shipped a check for the absence of a licence and it fired within a day, on a CUDA environment this laptop is not: `cuda-toolkit`, reported as a base-tier failure. The first question was whether the check was wrong. It is not. The wheel was downloaded and opened: `cuda_toolkit-13.3.1.dist-info/` contains `METADATA`, `WHEEL` and `RECORD`, and nothing else. No `License`, no `License-Expression`, no `License ::` classifier, no licence document declared or undeclared. Its every requirement is extras-gated, so a bare install pulls nothing at all. There is genuinely no statement of terms in that distribution to read.

**So the resolution is an allowlist, not a weaker check.** `UNREADABLE_ALLOWLIST` is the fourth allowlist in this file and is kept separate from the three beside it for the reason D16 already gave about the third: the exposures differ, and folding them together lets one package's silence excuse another package's declaration. `ALLOWLIST` excuses a package's own copyleft declaration, `BUNDLED_ALLOWLIST` and `BUNDLED_BINARY_ALLOWLIST` excuse a vendored copyleft library, and this one excuses metadata that says nothing. Entries require a DECISIONS.md row first, as all three others do; `cuda-toolkit` is the only one.

**CUDA is admitted as an environment, not as a licence** (D17). The NVIDIA CUDA EULA is proprietary, which the permissive-only policy would otherwise refuse outright. What makes it out of scope is that `pyproject.toml` declares no CUDA package, this project redistributes no part of the toolkit, and it publishes no trained weights (D14) -- so what the EULA covers is the machine an accelerator run happens on rather than anything this repository ships. That is written down rather than left as a silence, because an exemption nobody recorded is indistinguishable from an oversight.

**A second, smaller thing the finding exposed.** The report printed `(base)` for two different situations: a package the shipped closure actually reaches, and a package the walk never reached at all, which defaults to the strict tier by design. Both are failures, for opposite reasons, and spelling them identically sends a reader looking for a dependency declaration that does not exist -- which is exactly the wrong first move when the real answer is "nothing requires this; it was installed by hand". The unattributed case now says so in its own reason.

**And the suite stopped reading the installed environment.** Seven tests and nine doctests asserted over `metadata.distributions()` -- the whole venv is clean, `shapely` really does ship GEOS, `torch` attributes to base -- and none of those is an assertion about this code. A contributor who installs anything can fail them without touching a line of the audit, which is exactly what `cuda-toolkit` did. Had the allowlist landed on its own, `test_the_live_environment_has_no_unreadable_license` would have gone green and the next package installed on that box would have turned it red again. The live claim belongs to the pre-commit hook, which runs the audit against the real environment on every commit and is the actual gate; a test's job is that the audit *decides* correctly, and synthetic distributions answer that without depending on what anyone happens to have installed. The shapes those fixtures are written to were read off real wheels once and recorded in their docstrings -- the `cuda_toolkit` dist-info listing above is one of them.

Two things that went with it are worth naming. The GEOS true-positive test is gone; what stands in for it is the synthetic manylinux spelling (`libgeos_c-abcdd5fa.so.1.19.2` through `library_stem`) plus the parametrized normalizer cases, so the regression it guarded -- the stem normalizer going blind on Linux, where CI actually runs -- is still caught. And `audit = _load_audit()` at module level became a session-scoped fixture: loading a non-package script by path is fine, doing it as an import side effect at collection time is not.

