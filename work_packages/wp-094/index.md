# 📦 WP-094 — `feat(data): DOTA tiles into a trainable layout`

> Engineering log · [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/engineering/phase-08-oriented-detection/)
>
> 🔗 Depends on [WP-057](https://borda.github.io/lucid-YOLO/work_packages/wp-057/), [WP-088](https://borda.github.io/lucid-YOLO/work_packages/wp-088/) · Needed by [WP-095](https://borda.github.io/lucid-YOLO/work_packages/wp-095/), [WP-096](https://borda.github.io/lucid-YOLO/work_packages/wp-096/), WP-098, [WP-100](https://borda.github.io/lucid-YOLO/work_packages/wp-100/)
>
> ⬅️ [WP-088](https://borda.github.io/lucid-YOLO/work_packages/wp-088/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-095](https://borda.github.io/lucid-YOLO/work_packages/wp-095/) ➡️


## WP-094 — the difficult flag died at the loader

<a id="wp-094"></a>

[WP-057](https://borda.github.io/lucid-YOLO/work_packages/wp-057/#wp-057) manufactures difficult instances at crop time and A51 defines the channel they are supposed to travel on, but nothing connected the two: the reader did not forward `difficult` out of the annotation record, so every instance the tiler had flagged arrived at the loss indistinguishable from a whole one. Nothing disagreed anywhere — the counts are right, the boxes are right, and the only thing wrong is that a protocol decision the register spent a row on was not in force.

A53's non-schema keys are what make the flag survivable through a COCO container at all, since COCO's schema has no field for it or for the tile's window provenance, and a standard reader ignores both and still sees an ordinary detection set.

