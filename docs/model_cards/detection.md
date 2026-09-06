# 🎯 Model card — lucid-yolo detector (n scale)

Covers the detection model accepted at the Det-smoke tier for release `0.1.0`. One card per task family; segmentation and oriented detection get their own at `0.2.0` and `0.3.0`.

## 📇 Model details

|  |  |
| -- | -- |
| Developed by | Independent reproduction, single operator; see `PROVENANCE.md` |
| Model type | Single-stage anchor-free object detector, DFL-free dual head |
| Scale | `n` — 2.4M parameters, 5.4 GFLOPs at 640×640 |
| Input | RGB, letterboxed to 640×640, float32 in `[0, 1]` |
| Output | Two decode paths from one forward: NMS-free end-to-end (`decode/topk_e2e.py`, 300 detections as `[x1, y1, x2, y2, score, class]`) and a one-to-many path with NMS (`decode/nms_path.py`) |
| Version | `0.0.1.dev12` (checkpoint from run v8, 2026-08-06) |
| License | Apache-2.0 (code and report). Weights derive from COCO 2017 — see Licensing below |
| Paper | Methods per the YOLO26 paper, arXiv:2606.03748. Independent implementation; not affiliated with, endorsed by, or derived from Ultralytics or its codebase |

The architecture is typed Python, not a config DSL (ADR-001): five scale rows (`n/s/m/l/x`) multiply depth, width and max-channels over one topology. Only the `n` scale has been trained.

## 🧭 Intended use

**Intended.** Reproduction research: verifying the paper's architectural and training claims, ablating its mechanisms, and serving as a readable from-scratch implementation of NMS-free detection. The `e2e` path is the interesting artifact — it exports without suppression ops.

**Not intended.** Production detection, safety-critical or rights-affecting decisions, surveillance, biometric identification, or any deployment where 25 mAP on 80 common object categories would be mistaken for a reliable perception system. This is a smoke-tier reproduction at the smallest scale, roughly half the accuracy of a well-trained detector of this family, and it has been evaluated on exactly one benchmark.

## 🗂️ Training data

COCO 2017 `train2017` — 118,287 images, 80 categories, annotations under CC-BY-4.0 with images under their original Flickr terms. No other data; no pretraining (D2: no Objects365 initialization).

COCO's documented composition limits apply to this model directly: category frequencies are long-tailed, scenes are web photography skewed toward particular geographies and contexts, and the `person` category carries all the demographic imbalance of that source. Nothing here corrects for it, and no fairness evaluation across subgroups has been run.

## 🏋️ Training procedure

|  |  |
| -- | -- |
| Recipe | batch 128, 50 epochs (46,250 steps), MuSGD (`w_muon 0.5 / w_sgd 0.5`), lr 0.02 with 3-epoch warmup then linear decay to `lr0 × 0.01`, weight decay 5e-4, momentum 0.95 |
| Loss | dual-branch; per branch CIoU 7.5 / classification 0.5 / L1 6.0 (stride units), Progressive branch weight `0.8 → 0.1` |
| Assignment | TAL (`alpha 1, beta 6`) with STAL small-target surrogates; one-to-many topk 10, one-to-one 7→1 |
| Augmentation | mosaic `p=1.0` (disabled for the final 10 epochs), fused affine+letterbox, size-aware mixup and copy-paste, HSV jitter, horizontal flip `p=0.5` |
| Precision / seed | 16-mixed, seed 0 |
| Hardware | one RTX PRO 6000, ~8m45s/epoch, ~7.3 hours total |
| EMA | decay 0.9999, tau 2000 |

**`w_muon` and `w_sgd` are independent gains (A7), so `w_muon = 0` is a supported SGD-only configuration** — `MuSGD._parameter_update` branches on it explicitly and skips the Newton–Schulz iteration rather than computing and discarding it, and `tests/optim/test_musgd.py` pins the resulting update. Supported is all it is: no run in this project has ever trained a tier under it, so there is **no measurement of what the Muon branch is worth on this architecture** and this row should not be read as one arm of an ablation. Every tier above trained at `0.5 / 0.5`.

## 📊 Evaluation

COCO val2017, all 5000 images, `torchmetrics` `MeanAveragePrecision` on the `faster_coco_eval` backend. Both decode paths from one forward per batch.

| weights | path | mAP50-95 | mAP50 | mAP75 | mAP_S | mAP_M | mAP_L |
| -- | -- | -- | -- | -- | -- | -- | -- |
| EMA | NMS | **25.30** | 38.12 | 26.94 | 11.86 | 27.09 | 33.88 |
| EMA | E2E | 23.84 | 36.00 | 25.15 | 11.64 | 25.86 | 31.38 |
| raw | NMS | 25.33 | 38.45 | 27.05 | — | — | — |
| raw | E2E | 24.23 | 36.72 | 25.82 | 11.24 | — | — |

Recall at 100 detections (E2E, EMA): 46.07. The NMS-free path costs 1.11 AP against the NMS path on raw weights, 1.46 on EMA.

No latency figures are published. Throughput is hardware-dependent and the project treats parameter/FLOP fidelity as the substitute claim (all five scales within ±2% params, ±5% FLOPs of [R1 Table 7]).

## ⚠️ Limitations

- **Accuracy.** 25.3 mAP is a smoke-tier result at the smallest scale, ~50 epochs against the paper's 500–600. Expect frequent misses and confusions on anything but large, unambiguous, well-lit instances.
- **Small objects are weakest** — 11.9 mAP_S against 33.9 mAP_L. STAL is implemented but its benefit is a Det-ablations trend claim, unmeasured here.
- **80 COCO categories only.** No open-vocabulary capability; anything outside those categories is either missed or misclassified as the nearest one.
- **640×640 letterboxed.** Untested at other resolutions or aspect extremes.
- **One benchmark, one seed, one scale, one device.** No variance estimate across seeds, and no cross-dataset generalization measurement at all: the RF100-VL tier that would have supplied one (WP-074/075) was **dropped on 2026-08-18** over an unresolved per-dataset licence question, so nothing replaces it and none is scheduled. Training has also never spanned more than a single accelerator. This task's metrics are torchmetrics metrics that do synchronise across ranks, so a multi-device run is not obstructed the way the oriented and keypoint tiers are — but unobstructed is an untested inheritance from Lightning, not a result this project has.
- **No robustness evaluation** — weather, blur, occlusion, adversarial or distribution-shift behavior is entirely uncharacterized.
- **Not bitwise reproducible across platforms** (A26): libm last-bit rounding differs across OS and architecture; the goldens assert structural metrics with tolerance, not byte hashes.
- **Determinism is weaker on MPS than the recipe reads.** `default_determinism` (`cli/train.py`) resolves `deterministic` to `"warn_only"` whenever MPS is the auto-picked accelerator, because MPS ships no deterministic kernel for some backwards this model hits (`index_put_with_accumulate` in the assignment/loss backward); strict `True` would abort the step. CPU and CUDA keep strict determinism. An Apple-silicon run is therefore reproducible only up to those non-deterministic kernels, and says so in warnings rather than by failing.
- **The seed reproduces a run only at the same worker count.** `--data.num_workers 0` replays the augmentation stream byte for byte; with workers each one is re-seeded per worker and per epoch from the loader's own generator (WP-079). Both are fully determined by `seed`, but they are *different* streams — a run at 32 workers does not reproduce the same seed at 0 workers, or at 16.

## 🤝 Ethical considerations

A detector trained on COCO inherits COCO's `person` category, and any deployment that detects people carries surveillance and privacy implications this project has not assessed. The intended-use section is a boundary, not a disclaimer: the model has no evaluation supporting use on people, and none is planned.

Failure modes are unbounded in the sense that matters — no calibration study exists, so the confidence scores should not be read as probabilities, and the model gives no signal when it is operating outside its training distribution.

## 📜 Licensing and provenance

Code and this report: Apache-2.0. Weights derive from COCO 2017; no trained weights are published (D14).

**Clean-room statement.** No file, configuration, weight, or code fragment from any Ultralytics repository, package, documentation site, or released checkpoint was opened, downloaded, imported, or consulted at any point in this project's history. Every design input traces to a paper, its cited primary literature, a registered assumption, or — for diagnosis only, never copying — a permissively licensed independent implementation registered under D13/ADR-004. The audit trail is `PROVENANCE.md`; the standing prohibitions are `AGENTS.md` sec. 7.

## 🔖 Citation

This reproduction has no publication. Cite the paper it reproduces:

```bibtex
@article{yolo26,
  title  = {YOLO26},
  note   = {arXiv:2606.03748},
  year   = {2026}
}
```
