# 📦 WP-130b — `fix(ci): emoji on every hook, three now run on --all-files`

> Engineering log · [Phase 11 — Rolling, toward v0.5.0](https://borda.github.io/lucid-YOLO/engineering/phase-11-rolling/)
>
> 🔗 Depends on [WP-130](https://borda.github.io/lucid-YOLO/work_packages/wp-130/) · Needed by none
>
> ⬅️ [WP-130](https://borda.github.io/lucid-YOLO/work_packages/wp-130/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-118](https://borda.github.io/lucid-YOLO/work_packages/wp-118/) ➡️


## WP-130b — an emoji requirement is not the same claim as "every local hook"

<a id="wp-130b"></a>

[WP-130](https://borda.github.io/lucid-YOLO/work_packages/wp-130/)'s emoji pass read "every hook" as "every `repo: local` hook" and stopped at eleven -- the other eleven, all third-party (`pre-commit-hooks`, `mdformat`, `docformatter`, `ruff-pre-commit`), kept their plain default or custom names. Fixed by giving each of those eleven a `name:` override where none existed (`end-of-file-fixer`, `check-yaml` x1, `check-toml`, `check-added-large-files`, `ruff-check`, `ruff-format`) or an emoji prefix on the existing custom one (`check-yaml`'s unsafe-tags instance, `mdformat` x2, `docformatter`). Twenty-two hooks, twenty-two distinct emoji, chosen per action rather than per category so the same glyph never covers two conceptually different checks.

Separately: `pre-commit run --all-files` was only ever exercising nineteen of the twenty-two hooks, by design at the time -- `commit-trailers` runs at `commit-msg` stage, `golden-check`/`release-guard` at `stages: [manual]`, none of which `--all-files` (a `pre-commit`-stage run) reaches. Asked to close that gap too. `golden-check` and `release-guard` are cheap enough off a release tag (one `git describe` call, one already-fast golden recompute) that `stages: [pre-commit, manual]` costs nothing meaningful on every commit -- the accepted trade is `make gate` now calls `check_goldens.py` twice, once inside `precommit`'s `--all-files` and once via the `golden` target, rather than carrying a second exemption to remember. `commit-trailers` cannot take the same fix: a `pre-commit`-stage run has no commit message to validate, since no commit is in progress. Its `--all-files`-reachable half is a new sibling hook, `commit-trailers-history`, wired to the same script's pre-existing `--range` mode (`--range origin/main..HEAD`) rather than new code -- it re-validates every commit not yet pushed, which is a strictly more useful check than re-reading the newest message alone, and was already there waiting to be pointed at from a hook.

______________________________________________________________________

