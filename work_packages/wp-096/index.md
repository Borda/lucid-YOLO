# 📦 WP-096 — `refine(cli): one data CLI, one eval CLI`

> Engineering log · [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/engineering/phase-08-oriented-detection/)
>
> 🔗 Depends on [WP-094](https://borda.github.io/lucid-YOLO/work_packages/wp-094/), [WP-095](https://borda.github.io/lucid-YOLO/work_packages/wp-095/) · Needed by [WP-097](https://borda.github.io/lucid-YOLO/work_packages/wp-097/), [WP-110](https://borda.github.io/lucid-YOLO/work_packages/wp-110/), [WP-138](https://borda.github.io/lucid-YOLO/work_packages/wp-138/)
>
> ⬅️ [WP-095](https://borda.github.io/lucid-YOLO/work_packages/wp-095/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-097](https://borda.github.io/lucid-YOLO/work_packages/wp-097/) ➡️


## WP-096 — a rename with a published cost

<a id="wp-096"></a>

Parsing moved to jsonargparse so that flags and help text derive from the **operation signatures** rather than from a hand-maintained parser that is free to disagree with the function it calls. The cost is real and is stated rather than absorbed: jsonargparse spells a parameter with underscores, so every documented flag changed shape in one commit.

`lucid-download` therefore survives as a deprecated alias rather than being dropped. It is named in published reproduction instructions, and an instruction that no longer runs is a worse outcome than a duplicate entry point — the alias keeps accepting its **old** flags, which is the part a copy-pasted command depends on.

