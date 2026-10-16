# 📦 WP-146 — `fix(tests): reconcile the R21 pin with the LetterShape family`

> Engineering log · [Phase 13 — Contributor admission, augmentation guard and keypoint parity, v0.6.0](https://borda.github.io/lucid-YOLO/engineering/phase-13-contributor-admission/)
>
> 🔗 Depends on none · Needed by [WP-147](https://borda.github.io/lucid-YOLO/work_packages/wp-147/)
>
> ⬅️ [WP-145](https://borda.github.io/lucid-YOLO/work_packages/wp-145/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-147](https://borda.github.io/lucid-YOLO/work_packages/wp-147/) ➡️


## WP-146 — the pin was right and the register was wrong

<a id="wp-146"></a>

This row was written expecting to advance a pin. It advanced nothing. `pyproject.toml` has read `fedde0c1a482ac1add313fd823f6bc18e7b47436` since [WP-132](https://borda.github.io/lucid-YOLO/work_packages/wp-132/), and [WP-132](https://borda.github.io/lucid-YOLO/work_packages/wp-132/)'s own commit trailer cites that same commit; what [WP-132](https://borda.github.io/lucid-YOLO/work_packages/wp-132/) wrote into `docs/PROVENANCE.md` was `0a0cc64072b854936f3a15c36c21b5ed37012891`, one upstream commit behind the thing it installed. The register and the manifest disagreed for eight work packages and one release, and nothing could have noticed: no gate reads the register, and both hashes name real commits of the right repository.

**A pin bump that spans two upstream commits needs both named.** [WP-132](https://borda.github.io/lucid-YOLO/work_packages/wp-132/) wanted `SymbolShape` for the pose overfit gate's kite slice, and `SymbolShape` is `0a0cc640` — so the register's prose is about the right *feature* while its hash is about the wrong *commit*, which is the failure mode that makes this kind of drift survive review. Reading either field alone gives a coherent story. Reading them together against `pyproject.toml` is what does not.

**What `LetterShape` did to the fixtures: nothing, measured.** The generator was installed at `0a0cc640`, both generator-derived producers were run, and every value came back identical to the frozen goldens — 90 detection annotations, 895 segmentation points, 55131.455 box-area sum, 61 oriented annotations, and the ten `data_pipeline_metrics` values including the image mean and std sums. The generator was then restored to `fedde0c1` and the producers re-run to confirm the restoration. So the whole of [WP-132](https://borda.github.io/lucid-YOLO/work_packages/wp-132/)'s re-freeze is attributable to `0a0cc640` — the obtuse-scalene `TRIANGLE` and the area-centroid re-centring, exactly as its register prose says — and none of it to the fourth shape family. No golden moves here, which is the correct outcome for a reconciliation row and would have been a defect in a bump row.

Neutrality was worth measuring rather than arguing. `fedde0c1` widens the `Shape` union to 49 members and changes how `class_id_of` resolves ids against a narrowed category list, both of which are exactly the kind of change that reaches a fixture's category block. The argument for neutrality — that this project draws only `DEFAULT_SHAPES`, whose ids are a stable prefix — is a good argument and is not evidence. Installing the other commit and reading the numbers costs two installs.

**The fixture module was describing behaviour it does not have.** `tests/fixtures/synthetic.py` claimed the emitted COCO `categories` list "still names every animal fuse-augmentations ships even though none is ever drawn here". It does not, and did not at any pin this project has used: the detection fixture emits four categories, `square`, `rectangle`, `triangle`, `circle`. The claim is corrected here rather than left for a later row, because this row's whole subject is a register describing R21 wrongly, and a second document describing R21 wrongly a directory away is the same defect. Its cause is not `LetterShape` — the golden frozen at `74af9855` already carried four categories — so this is a claim that was never checked rather than one that went stale, which is the more useful thing for a later reader to know.

The general lesson is narrow and worth keeping: a register row carrying a hash is a manifest with no gate behind it, and the hash is the field that drifts, because prose gets re-read and a forty-character hex string does not. Where the same fact lives in two files and only one is executable, the executable one is the one to trust and the other is the one to check.

