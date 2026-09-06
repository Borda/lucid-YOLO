# 🖌️ Model card — lucid-yolo segmenter (n scale)

Covers the segmentation model produced by the Seg-smoke tier run, released as `0.2.0`. That tier was accepted at its principal gate (roadmap 054) on 2026-08-10, against the criterion recorded in `REPRODUCTION_REPORT.md`. One card per task family; detection has its own, oriented detection gets one at `0.3.0`.

## 📇 Model details

|  |  |
| -- | -- |
| Developed by | Independent reproduction, single operator; see `PROVENANCE.md` |
| Model type | Single-stage anchor-free instance segmenter — DFL-free dual detection head plus prototype–coefficient masks (R16) |
| Scale | `n` — 2.72M parameters, 9.00 GFLOPs at 640×640 (`goldens/params_flops_seg.json`) |
| Input | RGB, letterboxed to 640×640, float32 in `[0, 1]` |
| Output | Per detection: box, score, class, and 32 mask coefficients assembled against a 160×160 prototype bank (A15). Same two decode paths as the detector — NMS-free end-to-end and one-to-many with NMS — with masks paired to the kept anchors by index, not re-ranked (`decode/topk_e2e.py`) |
| Mask decode | Assemble, sigmoid, bilinear upsample to the letterboxed input, crop to the predicted box, threshold at 0.5, then nearest-neighbour inverse letterbox (A37) |
| Version | Run v9 checkpoint `epoch=49-step=46250.ckpt`, 2026-08-10. The package version installed for the run is not recorded in any run artifact |
| License | Apache-2.0 (code and report). Weights derive from COCO 2017 — see Licensing below |
| Paper | Methods per the YOLO26 paper, arXiv:2606.03748. Independent implementation; not affiliated with, endorsed by, or derived from Ultralytics or its codebase |

The segmentation head sits on the detector's own backbone, neck and dual head: the detection objective is unchanged and the two mask terms ride on top of it (A38). An auxiliary semantic branch trains alongside and is absent from the deployed model — `forward` returns `None` in eval mode and `deploy()` does not hold the branch at all (A17), which is why the parameter gate has exactly one convolution to account for.

Parameter and FLOP fidelity is gated rather than asserted: `test_param_flops.py` holds all five scales within ±3% params and ±5% FLOPs of [R1 Table S9]. The parameter tolerance is wider than detection's ±2% because the head's sizing rests on five registered assumptions (A14, A15, A18, A34, A35) rather than on published structure.

## 🧭 Intended use

**Intended.** Reproduction research: verifying the paper's segmentation claims, ablating the prototype–coefficient mechanism, and serving as a readable from-scratch implementation of NMS-free instance segmentation. As with the detector, the `e2e` path is the interesting artifact — masks come out of a forward that contains no suppression op.

**Not intended.** Production segmentation, safety-critical or rights-affecting decisions, medical or scientific measurement from mask areas, surveillance, biometric identification, or any deployment where 19 segm mAP on 80 common object categories would be mistaken for a reliable perception system. This is a smoke-tier reproduction at the smallest scale, evaluated on exactly one benchmark. Mask boundaries are quantized by a 160×160 prototype grid — 4 input pixels per prototype cell — so any use that depends on precise object extent, and especially on small objects, is outside what this model supports.

## 🗂️ Training data

COCO 2017 `train2017` — 118,287 images, 80 categories, polygon instance annotations under CC-BY-4.0 with images under their original Flickr terms. No other data; no pretraining (D2: no Objects365 initialization).

COCO's documented composition limits apply to this model directly: category frequencies are long-tailed, scenes are web photography skewed toward particular geographies and contexts, and the `person` category carries all the demographic imbalance of that source. Nothing here corrects for it, and no fairness evaluation across subgroups has been run. COCO's polygon annotations are also themselves coarse — the masks this model imitates are human-drawn polygons, not pixel-exact segmentations.

## 🏋️ Training procedure

|  |  |
| -- | -- |
| Recipe | batch 128, 50 epochs (46,250 steps), MuSGD (`w_muon 0.5 / w_sgd 0.5`), lr 0.02 with 3-epoch warmup then linear decay to `lr0 × 0.01`, weight decay 5e-4, momentum 0.95, gradient clip 10.0 |
| Loss | detection unchanged — per branch CIoU 7.5 / classification 0.5 / L1 6.0 (stride units), Progressive branch weight `0.8 → 0.1` — plus instance mask 2.5 and auxiliary semantic 0.5 (A38) |
| Mask supervision | Both branches' coefficients, each against its own assignment, combined under the same Progressive `alpha` split the box terms use; per-pixel BCE on box-cropped masks, box-area normalized (A16) |
| Assignment | TAL (`alpha 1, beta 6`) with STAL small-target surrogates; one-to-many topk 10, one-to-one 7→1 |
| Augmentation | mosaic `p=1.0` (disabled for the final 10 epochs), fused affine+letterbox, size-aware mixup and copy-paste, HSV jitter, horizontal flip `p=0.5` |
| Precision / seed | bf16-mixed, seed 0, `deterministic: true` |
| Hardware | one unrecorded Colab GPU (`default_root_dir` under `/content/`); the model was not logged, and no per-epoch timing was recorded |
| EMA | decay 0.9999, tau 2000 |

The wiring gate that precedes any COCO launch is `scripts/overfit_micro.py --task segment`: 100 images, 592 instances, 4 classes, 100 epochs at 320 px, decoded through the deployed one-to-one path. It measures train mask IoU **0.8146** against a 0.7 floor (`goldens/gpu/overfit_micro_seg.json`).

## 📊 Evaluation

COCO val2017, all 5000 images, at 640 px through `lucid-eval`, which runs one forward per batch and decodes both paths from it. Boxes and masks are scored by two `torchmetrics` `MeanAveragePrecision` instances on the `faster_coco_eval` backend, masks at original image resolution against ground truth decoded from the COCO polygons (R12). EMA weights; raw weights were not evaluated for this run.

| path | box mAP50-95 | box mAP50 | box mAP75 | segm mAP50-95 | segm mAP50 | segm mAP75 |
| -- | -- | -- | -- | -- | -- | -- |
| NMS | **26.12** | 39.12 | 27.93 | **19.01** | 34.40 | 18.63 |
| E2E | 24.85 | 37.25 | 26.59 | 18.20 | 32.72 | 18.03 |

| path | box S / M / L | segm S / M / L | box mAR_100 | segm mAR_100 |
| -- | -- | -- | -- | -- |
| NMS | 12.33 / 28.54 / 34.86 | 3.74 / 20.06 / 31.79 | 46.75 | 32.75 |
| E2E | 12.43 / 27.14 / 32.91 | 3.57 / 19.31 / 30.34 | 46.78 | 32.50 |

The NMS-free path costs 1.27 box AP and 0.81 segm AP against the NMS path. Masks reach 0.728 of the box figure on the NMS path and 0.732 on E2E; the `REPRODUCTION_REPORT.md` Seg-smoke section reads that ratio in detail.

No latency figures are published. Throughput is hardware-dependent and the project treats parameter/FLOP fidelity as the substitute claim.

## ⚠️ Limitations

- **Accuracy.** 19.0 segm mAP is a smoke-tier result at the smallest scale, 50 epochs against the paper's 500–600. Expect frequent misses and confusions on anything but large, unambiguous, well-lit instances.
- **Small objects are far weaker in masks than in boxes** — 3.7 segm mAP_S against 12.3 box mAP_S, a ratio of 0.30 where large objects reach 0.91. A 32-pixel object spans 8 prototype cells; below that, boundary quantization dominates whatever the coefficients predict.
- **High-IoU localization is where masks lose.** The mask-to-box ratio is 0.88 at IoU 0.50 and 0.67 at IoU 0.75: the model finds and labels instances about as well as its boxes do, and pays at the boundary.
- **Masks are cropped to the predicted box.** A wrong box truncates its mask by construction; mask errors and box errors are not independent.
- **80 COCO categories only.** No open-vocabulary capability; anything outside those categories is either missed or misclassified as the nearest one.
- **640×640 letterboxed.** Untested at other resolutions or aspect extremes.
- **One benchmark, one seed, one scale, one device.** No variance estimate across seeds, and no cross-dataset generalization measurement at all: the RF100-VL tier that would have supplied one (WP-074/075) was **dropped on 2026-08-18** over an unresolved per-dataset licence question, so nothing replaces it and none is scheduled. Training has also never spanned more than a single accelerator. This task's metrics are torchmetrics metrics that do synchronise across ranks, so a multi-device run is not obstructed the way the oriented and keypoint tiers are — but unobstructed is an untested inheritance from Lightning, not a result this project has.
- **No robustness evaluation** — weather, blur, occlusion, adversarial or distribution-shift behavior is entirely uncharacterized.
- **No raw-weight evaluation** for this run, so the EMA contribution is unmeasured here.
- **Not bitwise reproducible across platforms** (A26): libm last-bit rounding differs across OS and architecture; the goldens assert structural metrics with tolerance, not byte hashes.
- **`deterministic: true` above is what the run *requests*, not uniformly what it gets.** `default_determinism` (`cli/train.py`) resolves it to `"warn_only"` whenever MPS is the auto-picked accelerator, because MPS ships no deterministic kernel for some backwards this model hits; CPU and CUDA keep strict `True`. An Apple-silicon run is reproducible only up to those kernels, and says so in warnings rather than by failing.
- **The seed reproduces a run only at the same worker count.** `--data.num_workers 0` replays the augmentation stream byte for byte; with workers each is re-seeded per worker and per epoch from the loader's own generator (WP-079). Both are fully determined by `seed`, but they are *different* streams — a run at 32 workers does not reproduce the same seed at 0 workers, or at 16.

## 🤝 Ethical considerations

A segmenter trained on COCO inherits COCO's `person` category, and a mask is a more precise readout of a person than a box is — silhouette, pose and body extent come out of it. Any deployment that segments people therefore carries surveillance and privacy implications this project has not assessed, and carries them further than the detector does. The intended-use section is a boundary, not a disclaimer: the model has no evaluation supporting use on people, and none is planned.

Failure modes are unbounded in the sense that matters — no calibration study exists, so the confidence scores should not be read as probabilities, mask boundaries carry no uncertainty estimate at all, and the model gives no signal when it is operating outside its training distribution.

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
