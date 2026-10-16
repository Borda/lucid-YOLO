# 📦 WP-085 — `test(repo): doctests join the offline gate`

> Engineering log · [Phase 6 — Evaluation, v0.1.0](https://borda.github.io/lucid-YOLO/engineering/phase-06-evaluation/)
>
> 🔗 Depends on [WP-004](https://borda.github.io/lucid-YOLO/work_packages/wp-004/) · Needed by [WP-117](https://borda.github.io/lucid-YOLO/work_packages/wp-117/)
>
> ⬅️ [WP-084](https://borda.github.io/lucid-YOLO/work_packages/wp-084/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-052](https://borda.github.io/lucid-YOLO/work_packages/wp-052/) ➡️


## WP-085 — doctests join the offline gate

<a id="wp-085"></a>

Every public function is required to carry an `Examples` section, and **no gate had ever run one**: `testpaths` pointed at `tests/` alone and `addopts` carried no `--doctest-modules`. Four examples had rotted unnoticed over roughly eighty work packages.

The worst was not stale — it was never true. `orthogonalize`'s example asserted near-exact orthogonality that the Newton–Schulz coefficients cannot reach, and which that function's own unit test documents as unattainable. It could not have passed at any seed, and was additionally unseeded. It now states the guarantee that actually holds (no singular value expanded). The other three pinned stale float reprs (`0.01` against `0.010000000000000009`; `0.01...` against `0.00999...`).

Wiring it up took the offline suite from **558 to 699 cases** and coverage from 96% to 97%. The lesson recorded for later gates: a requirement nothing executes is a documentation convention, not a gate, and the two are indistinguishable from a green run.

