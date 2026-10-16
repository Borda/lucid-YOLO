# 📦 WP-105 — `fix(eval): a report that survives, and a pass you can watch`

> Engineering log · [Phase 8 — Oriented detection, v0.3.0](https://borda.github.io/lucid-YOLO/engineering/phase-08-oriented-detection/)
>
> 🔗 Depends on [WP-104](https://borda.github.io/lucid-YOLO/work_packages/wp-104/) · Needed by WP-106
>
> ⬅️ [WP-104](https://borda.github.io/lucid-YOLO/work_packages/wp-104/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-064](https://borda.github.io/lucid-YOLO/work_packages/wp-064/) ➡️


## WP-105 — a report that survives

<a id="wp-105"></a>

Two defects the first real oriented evaluation exposed, both in the CLI rather than in what it measures.

`--output` into a directory that did not exist yet raised **after** the scoring pass finished and printed its numbers — minutes of accelerator work producing a traceback and nothing else. The parent is now created rather than required, because the file is the only durable form of the expensive part. The general rule: a long computation's output path should be validated or created *before* the computation, never discovered after it.

Second, a 10,132-tile pass showed one line at the start and nothing until it ended. The run has a known length; it now draws a progress bar, wrapped at the CLI rather than inside the evaluator, which tests drive in-process.

______________________________________________________________________

