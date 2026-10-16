# 📦 WP-058 — `feat(data): rotated-aware augmentation`

> Research log · [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/research/phase-08-oriented-detection/)
>
> 🔗 Depends on [WP-057](https://borda.github.io/lucid-YOLO/work_packages/wp-057/) · Needed by [WP-061](https://borda.github.io/lucid-YOLO/work_packages/wp-061/)
>
> ⬅️ [WP-057](https://borda.github.io/lucid-YOLO/work_packages/wp-057/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-059](https://borda.github.io/lucid-YOLO/work_packages/wp-059/) ➡️


## WP-058 — rotated-aware augmentation

<a id="wp-058"></a>

The three policy choices are A40. What belongs here is the **live defect it fixed**: mirroring negated `theta` without re-wrapping, so any box above 45° left the long-edge range. Nothing had ever hit it because no rotated box could reach a flip through a real pipeline — the augmentations raised `NotImplementedError` on non-empty `rboxes` until this package removed the guard.

A defect that is unreachable is still a defect, and it becomes reachable in the same commit that removes the guard. The general shape: when lifting a `NotImplementedError`, the code behind it has never executed and has never been tested, whatever its coverage says.

