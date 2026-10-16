# 📦 WP-087f — `refine(configs): name tiers by what they are, not by letters`

> Research log · [Phase 7 — Instance segmentation, v0.2.0](https://borda.github.io/lucid-YOLO/research/phase-07-instance-segmentation/)
>
> 🔗 Depends on [WP-087](https://borda.github.io/lucid-YOLO/work_packages/wp-087/) · Needed by none
>
> ⬅️ [WP-087e](https://borda.github.io/lucid-YOLO/work_packages/wp-087e/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-054](https://borda.github.io/lucid-YOLO/work_packages/wp-054/) ➡️


## WP-087f — tier names

<a id="wp-087f"></a>

The configs were named `<task>_tier_<tier letter>_<scale letter>`, putting two letters from unrelated alphabets side by side, with the tier letter reading backwards (A the cheap smoke tier, B the longer paired-run one).

The scale letter was not merely confusing but **false**: it sets `variant` inside the file and `--variant` overrides it, so `seg_tier_a_n.yaml --variant m` is an unremarkable invocation whose filename says `n` while the run is `m`. A filename that can contradict the run it names is a defect, not a style question.

Renames propagated to historical mentions too — a reader meeting "Det-A attempt 2" in the A13 row has no file left to connect it to — with git history keeping the old names. Each config header keeps its `blueprint sec. 10 tier A` citation, because the blueprint is not in this repository and still speaks in letters, so the mapping has to survive somewhere explicit.

______________________________________________________________________

