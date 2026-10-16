# 📦 WP-154c — `fix(goldens): freeze-goldens stops re-copying what it cannot re-freeze`

> Engineering log · [Phase 14 — Augmentation upstreaming to `fuse`, v0.7.0](https://borda.github.io/lucid-YOLO/engineering/phase-14-augmentation-upstreaming/)
>
> 🔗 Depends on [WP-154](https://borda.github.io/lucid-YOLO/work_packages/wp-154/) · Needed by none
>
> ⬅️ [WP-154](https://borda.github.io/lucid-YOLO/work_packages/wp-154/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-154b](https://borda.github.io/lucid-YOLO/work_packages/wp-154b/) ➡️


## WP-154c — the mechanism stops reintroducing what it was already told to remove

<a id="wp-154c"></a>

[WP-154](https://borda.github.io/lucid-YOLO/work_packages/wp-154/) closed by naming why [WP-132](https://borda.github.io/lucid-YOLO/work_packages/wp-132/)'s wall came back: `make freeze-goldens` was a blind `cp goldens/*.json goldens/frozen/$(MINOR)/`, with no notion that `fixture_checksums.json` and `data_checksums.json` are generator-derived — their producers render synthetic images through `fuse-augmentations`, so once that package's generator moves, no future code change can satisfy "the frozen copy still holds". [WP-132](https://borda.github.io/lucid-YOLO/work_packages/wp-132/) removed the two files that wall produced; [WP-140](https://borda.github.io/lucid-YOLO/work_packages/wp-140/) and [WP-153](https://borda.github.io/lucid-YOLO/work_packages/wp-153/) each re-froze them anyway, because a copy of a green golden is green until the generator moves, and nothing in the copy path could tell the two categories apart. Removing the files a third time at [WP-159](https://borda.github.io/lucid-YOLO/work_packages/wp-159/) would only restart the same clock.

The fix is a schema field rather than a human checklist item. `fixture_checksums.json` and `data_checksums.json` now carry `"freezable": false`; every other live golden is freezable by omission, which keeps the six pure-code goldens' files untouched and states the rule as "false is the exception, not true". `scripts/freeze_goldens.py` replaces the Makefile's inline `cp`: it reads each live golden's `freezable` field and copies only the eligible ones, reporting what it skipped rather than skipping silently. `Makefile`'s `freeze-goldens` target calls it. `scripts/check_goldens.py`'s `_parse_golden` reads the same field, and `check_golden` rejects — with a named error, not a silent skip — any golden it finds under `goldens/frozen/` whose `freezable` is `false`; this should never happen once the freeze path is fixed, but a manual copy or a retroactively-flagged old snapshot must still be caught by the gate that reads the frozen tree, not assumed away.

Two things this row does not do. It does not touch either `goldens/frozen/` directory — [WP-154](https://borda.github.io/lucid-YOLO/work_packages/wp-154/) already removed what needed removing, and this row's job is only to stop the mechanism, not to re-litigate the removal. And it does not re-freeze anything: no live golden's values changed, so `goldens/fixture_checksums.json` and `goldens/data_checksums.json` gain one field and nothing else.

