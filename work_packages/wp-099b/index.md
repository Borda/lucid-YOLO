# 📦 WP-099b — `feat(data): a datamodule that dispatches on layout`

> Engineering log · [Phase 9 — Inference and generalization](https://borda.github.io/lucid-YOLO/engineering/phase-09-inference/)
>
> 🔗 Depends on [WP-099](https://borda.github.io/lucid-YOLO/work_packages/wp-099/) · Needed by [WP-099d](https://borda.github.io/lucid-YOLO/work_packages/wp-099d/), [WP-099c](https://borda.github.io/lucid-YOLO/work_packages/wp-099c/)
>
> ⬅️ [WP-099](https://borda.github.io/lucid-YOLO/work_packages/wp-099/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-099d](https://borda.github.io/lucid-YOLO/work_packages/wp-099d/) ➡️


## WP-099b — the tie-break that must not exist

<a id="wp-099b"></a>

`resolve_split` already had a precedence rule: candidate spellings are tried in table order and the first match wins. Reusing that shape across the two *readers* is the obvious move and it is wrong, for a reason worth stating in one line — **within a table the loser is another spelling of one reader, and between the tables the loser is a different label space, a different image set and a different parser.** A first-match rule in the first case picks the wrong directory of the right dataset; in the second it trains on annotations nobody named and reports a plausible number while doing it. So the probe evaluates both tables in full and returns a pair of booleans: there is no order for the answer to depend on, and ambiguity is refused rather than resolved (A63).

The refusal only works if the operator has a way to answer it, which is what earned the explicit `layout` override its place. Two states are otherwise unreachable: an ambiguous root on a read-only dataset mount, where the only remedy would be moving files; and a YOLO root whose `data.yaml` points its splits outside the convention table, which `from_root` resolves — it is why `val: ../valid/images` works at all — and which no directory probe can see. A knob that exists because two legitimate datasets are otherwise unreadable is a different thing from a knob that exists because one was possible.

The construction-versus-`setup` split was forced by something outside this package. Refusing at construction is the better instinct and it collides with `tests/ptl/test_cli.py`, which instantiates every shipped config through `LightningCLI(run=False)` while `configs/det_smoke.yaml` carries a placeholder `data_root`. A filesystem verdict at construction fails that dry-parse for every run not yet pointed at data. The line drawn instead: everything decidable from the arguments alone raises at construction, and anything needing the disk raises at `setup()` — still before an image is decoded and before the trainer takes a step.

Two absences the YOLO path has to declare rather than discover. `mask_targets` is refused, because the format carries no per-instance rings and rasterising empty ones would produce a segmentation run supervised by nothing that still reports a detection number. Copy-paste is the quieter one: unlike the oriented path it does **not** raise on polygon-free targets — it decodes an entire extra source sample, finds no candidates and does nothing — so at `copy_paste = 0.1` every YOLO run would have paid for an augmentation that can never fire. The suppression keeps consuming the RNG draw, which is what leaves the COCO path byte-identical and the frozen data checksums unmoved. An augmentation that fails silently costs more than one that raises.

One more property of the override, which is what keeps it from costing anything: naming any COCO-shaped path — `train_images`, `val_annotations`, any of them — is itself read as stating the COCO layout. Every caller that predates the probe is therefore already off it, and the probe runs only for a root that has said nothing about its own shape.

