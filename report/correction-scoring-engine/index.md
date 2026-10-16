# ⚙️ Correcting note — which engine scored each tier

Appended, not merged (D10): this adds a fact the four sections above never recorded and revises no number any of them states. It is not a fifth tier — no run stands behind it.

## What the record left out

Every `lucid-eval` invocation printed above names a checkpoint, a data root and sometimes an output path, and none names a scoring engine. That was unambiguous when they were written and stopped being so when [WP-138](https://borda.github.io/lucid-YOLO/work_packages/wp-138/) landed in 0.5.0: since then `--eval_backend` defaults to `auto`, which takes `hotcoco` on any host where hotcoco is usable and falls back to `faster_coco_eval` only where it is not. A reader running the printed commands today therefore scores with a different engine than the one that produced the figures above, on most machines, with nothing in the command or the output to say so. The engine each run actually used is written to `info["eval_backend"]` in the report JSON (`detect_eval.py`, `pose_eval.py`) — but those JSONs are archived off-repo under `.experiments/`, so the repository alone could not answer the question.

This is a prospective reproducibility gap, not a retroactive falsification: no figure above is wrong, and none moves.

## Engine per tier

| tier | box/mask engine | how it is known | affected figures |
| -- | -- | -- | -- |
| 0.1.0 Det-smoke | `faster_coco_eval` | the run predates [WP-138](https://borda.github.io/lucid-YOLO/work_packages/wp-138/); no other engine existed in the package | every mAP/mAR figure in the Det-smoke table |
| 0.2.0 Seg-smoke | `faster_coco_eval` | same | box and `segm_` figures alike |
| 0.3.0 OBB-smoke | not applicable | the rotated protocol is [WP-063](https://borda.github.io/lucid-YOLO/work_packages/wp-063/)'s own mAP, not COCOeval; `lucid-eval` ignores `--eval_backend` for an `obb` checkpoint | none |
| 0.5.0 Pose-smoke | **not determinable from this repository** | the run was accepted the day [WP-138](https://borda.github.io/lucid-YOLO/work_packages/wp-138/) landed, so `auto` may have resolved either way on the machine that scored it; the answer is in that run's archived report JSON, not here | the box mAP only — `0.5030 e2e box mAP` in the acceptance table |

The ten OKS keypoint statistics are unaffected in every tier: keypoint scoring is hand-driven `faster_coco_eval` regardless of `--eval_backend` (A73), so `0.2738 e2e OKS AP`, `0.3357 train OKS AP` and the RLE-versus-Laplace-NLL comparison are engine-independent by construction.

## Reproducing a figure above

Add the engine to the printed command rather than taking today's default:

```bash
lucid-eval --checkpoint <checkpoint> --data_root <root> --eval_backend faster_coco_eval
```

That is required for 0.1.0 and 0.2.0, where the recorded figures are `faster_coco_eval` figures. For 0.5.0 it is the conservative choice rather than a known match, for the reason the table gives. The printed command blocks in the sections above are left unedited, as those sections' own notes say they are.

## How close the two engines are

Held equal by `tests/eval/test_coco_eval.py::TestHotcocoParity`, to `1e-6` absolute on CPU: the 12 box statistics, the 24 box-plus-mask statistics, targets carrying their own `iscrowd` and `area` through the box pass and the mask pass, the never-updated all-zero case, and one end-to-end `DualPathEvaluator` run over a fixture. Not held by anything: the per-class entries the report drops, and any device other than CPU. The `iscrowd`/`area` cases are what closed a real divergence — the hotcoco path once rebuilt both fields from the box, so its size buckets and crowd-ignore rule were not COCO's; that was fixed in 0.8.0 ([WP-166](https://borda.github.io/lucid-YOLO/work_packages/wp-166/)), which is after every tier above was accepted and therefore changes none of their numbers.
