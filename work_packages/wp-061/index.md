# 📦 WP-061 — `feat(assign): rotated containment for TAL and STAL`

> Research log · [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/research/phase-08-oriented-detection/)
>
> 🔗 Depends on [WP-058](https://borda.github.io/lucid-YOLO/work_packages/wp-058/), [WP-060](https://borda.github.io/lucid-YOLO/work_packages/wp-060/) · Needed by [WP-062](https://borda.github.io/lucid-YOLO/work_packages/wp-062/)
>
> ⬅️ [WP-060](https://borda.github.io/lucid-YOLO/work_packages/wp-060/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-062](https://borda.github.io/lucid-YOLO/work_packages/wp-062/) ➡️


## WP-061 — rotated containment for TAL and STAL

<a id="wp-061"></a>

The axis-aligned path is an untouched early-return branch below the new one, and that was verified rather than assumed: assignments were snapshotted at the parent commit on a scene whose anchors sit **exactly on ground-truth boundaries**, then compared field by field with `torch.equal` — 20 tensors, 6,144 elements, bit-identical. "Its tests still pass" would not have distinguished a boundary case that moved by a rounding step.

