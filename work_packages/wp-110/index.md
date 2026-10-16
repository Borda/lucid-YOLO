# 📦 WP-110 — `refine(cli): drop the deprecated download alias`

> Engineering log · [Phase 10 — Consolidation](https://borda.github.io/lucid-YOLO/engineering/phase-10-consolidation/)
>
> 🔗 Depends on [WP-089](https://borda.github.io/lucid-YOLO/work_packages/wp-089/), [WP-096](https://borda.github.io/lucid-YOLO/work_packages/wp-096/) · Needed by none
>
> ⬅️ [WP-109](https://borda.github.io/lucid-YOLO/work_packages/wp-109/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-067](https://borda.github.io/lucid-YOLO/work_packages/wp-067/) ➡️


## WP-110 — a grep for a dead command cannot find its dead grammar

<a id="wp-110"></a>

[WP-096](https://borda.github.io/lucid-YOLO/work_packages/wp-096/) deprecated `lucid-download` in 0.3.0 and wrote the removal date into three places: `[project.scripts]`, AGENTS.md sec. 2, and the 0.3.0 changelog. 0.4.0 is that date. The alias existed so published reproduction instructions kept running, which is a real cost to removing it — and the way that cost is paid is not by keeping the alias forever but by making sure every string this repository prints names a command the reader can still run.

**The sweep was driven by grepping the alias name, and that method has a hole the grep cannot see.** A hint that names `lucid-download` is findable; a hint that uses the alias's *flag grammar* without naming it is not. Parse-testing every CLI fragment in the module's own docstrings found one: the checksum policy spelled `--sha256 val2017.zip=<hex>`, the bare argparse form, which the shipped jsonargparse command rejects outright — `error: Parser key "sha256": Expected a <class 'collections.abc.Sequence'>`. It needed `'[val2017.zip=<hex>]'`. Nothing that greps for a name would ever have surfaced it, and it would have shipped as documentation that fails when followed.

The same shape of mistake was in the test that guarded the repair hint: it asserted the hint *string appeared* on stderr. That assertion would have passed unchanged while the command it named was being deprecated out from under it — it tests presence where the thing worth testing is runnability. It now shell-splits the printed hint and re-parses it through the shipped parser, so a hint that stops parsing fails here rather than in an operator's terminal.

Nothing turned out to be shared. `lucid-data download` reaches `download_dataset` through jsonargparse's `add_function_arguments`, never `add_arguments`, never `main`, never an `argparse.ArgumentParser` — the deleted pair was referenced from exactly two places, each other and the test file. The near-trap was that `add_arguments`'s own docstring claimed it registered "the `lucid-data download` flags", untrue since [WP-096](https://borda.github.io/lucid-YOLO/work_packages/wp-096/); anyone trusting the docstring over the call graph would have kept it as shared code.

Removing the alias also removed the one exemption to the project's underscore-flag rule, and AGENTS.md sec. 2 was stale on a second count anyway: `lucid-predict` shipped in [WP-089](https://borda.github.io/lucid-YOLO/work_packages/wp-089/) and the bullet still counted three commands. Both corrected in the same edit, since they are one sentence describing one surface.

Two references are left standing deliberately. The reproduction report's "commands as run" blocks keep their dashed spellings — they record what was executed at the version named, and editing them would falsify the record rather than update it — with the surrounding prose changed to say the alias *was* removed and that the block must be read through its replacements rather than run as written.

