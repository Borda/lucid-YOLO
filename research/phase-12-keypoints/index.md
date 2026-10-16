# 🕺 Phase 12 — Keypoint detection, v0.5.0


## WP-125 — the flow beats its own control, at nano scale too

<a id="wp-125"></a>

R14's mechanism claim is that RLE's learned residual density buys real accuracy over the reparameterization, the sigmoid-bounded per-point scale and the `log sigma_hat` Jacobian it ships alongside — not that those three alone do the work. That claim needed a paired run differing only in the flow, which is what [WP-135](https://borda.github.io/lucid-YOLO/work_packages/wp-135/) built (`LaplaceNLLLoss`, R14 Table 7's own published control) and this row reads.

Both arms: `pose_nano_smoke.yaml` / `pose_nano_smoke_laplace_nll.yaml`, byte-identical but for `keypoint_loss`, nano variant, 50 epochs, COCO `person_keypoints_{train,val}2017`, same seed. RLE is Lightning run `version_11`; Laplace-NLL is `version_14` (two false starts, `version_12`/`version_13`, 11 seconds apart — each wrote only the TensorBoard writer's own opening record and nothing else, no `config.yaml`, no step, before exiting; the cause is not in what synced to Drive). Both scored with the identical `lucid-eval` pass, EMA weights, full val2017 split:

|  | RLE (`version_11`) | Laplace-NLL (`version_14`) |
| -- | -- | -- |
| e2e OKS AP | 0.2738 | 0.2527 |
| e2e OKS AP50 | 0.6094 | 0.5692 |
| e2e OKS AP75 | 0.2138 | 0.1933 |
| nms OKS AP | 0.2688 | 0.2525 |
| e2e box mAP | 0.5030 | 0.4471 |

RLE beats the ablation on every OKS statistic, both decode paths — a 8.3% relative gap on `e2e OKS AP` (0.2738 vs 0.2527), same sign as R14 Table 7's own 70.5-vs-67.4 AP (4.5% relative) at full COCO scale and full training budget. The absolute figures are not R14's — nano width, 50 epochs and no pretraining put a floor under both arms nothing here controls for — but the acceptance this row was scoped to is the mechanism's direction of effect, and a paired run differing only in `keypoint_loss` shows the flow term earning a real, consistent margin rather than none or a reversed one. `keypoint_flip_pairs` ran live in both arms — A64 was already resolved by [WP-132](https://borda.github.io/lucid-YOLO/work_packages/wp-132/)'s own reading of the training path, and this row is the first to exercise that resolved pairing at tier scale rather than through overfit-100's augmentation-off gate. `keypoint_gain = 1.0` (A68) trained a real pose head in both arms to a checkpoint that beats its own control rather than one swamped by or swamping the detection objective — the first evidence either way past a memorization-scale run, though the gain's own tuned adequacy stays unmeasured and A68 stays open on that question.

