# Provenance Log

Every design decision in this repository cites its public source. This file is
the clean-room evidence record: sources are logged here on first use, and every
commit carries a `Provenance:` trailer referencing the source ids below. No
Ultralytics code surface, config, weight, or web property (including
docs.ultralytics.com) has been consulted at any point — see the denylist in
AGENTS.md.

## Source allowlist

Admissible references. Anything not on this list requires a DECISIONS.md entry
before first use; denylisted surfaces are never admissible.

### Papers under reproduction

| ID | Reference | Role |
|---|---|---|
| R1 | Jocher, G. et al., *Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models*, arXiv:2606.03748 (2026) | Method source: Eq. 1–15, Figs. S1/S2, Tables 2–11, S1–S11 |
| R2 | Sapkota, R. et al., *YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection*, arXiv:2509.25164 (2026) | Independent analysis; benchmark corroboration (incl. seg/OBB tables) |
| R3 | Hidayatullah, P., Tubagus, R., *YOLO26: A Comprehensive Architecture Overview and Key Improvements*, arXiv:2602.14582 (2026) | Third-party architectural specification: variant multipliers, block diagram, SPPF shortcut, Top-K decoding, STAL minimum-anchor behavior |

### Cited primary literature

| ID | Reference | Role |
|---|---|---|
| R4 | Feng, C. et al., *TOOD: Task-Aligned One-Stage Object Detection*, arXiv:2108.07755 (ICCV 2021) | TAL formulation and alpha/beta defaults (A2) |
| R5 | Li, X. et al., *Generalized Focal Loss*, arXiv:2006.04388 (NeurIPS 2020) | DFL definition (the mechanism being removed) |
| R6 | Wang, A. et al., *YOLOv10: Real-Time End-to-End Object Detection*, arXiv:2405.14458 (NeurIPS 2024) | Consistent dual assignment |
| R7 | Liu, J. et al., *Muon is Scalable for LLM Training*, arXiv:2502.16982 (2025) | Muon update, Newton–Schulz orthogonalization |
| R8 | Jordan, K., *Muon: An optimizer for hidden layers in neural networks*, kellerjordan.github.io/posts/muon (2024) | Reference NS iteration count and step scaling (A5, A6) |
| R9 | Bochkovskiy, A. et al., *YOLOv4*, arXiv:2004.10934 (2020) | Mosaic augmentation; CIoU adoption in the lineage |
| R10 | Zheng, Z. et al., *Distance-IoU Loss*, arXiv:1911.08287 (AAAI 2020) | CIoU definition (A1) |
| R11 | Hidayatullah, P. et al., *YOLOv8 to YOLO11: A Comprehensive Architecture In-depth Comparative Review*, arXiv:2501.13400 (2025) | C3k2/C2PSA/SPPF internals in the YOLO11 lineage (A3) |
| R13 | Zhou, Y. et al., *MMRotate*, arXiv:2205.14672 (ACM MM 2022) | Long-edge angle convention; rotated-box tooling conventions (A21, A23) |
| R14 | Li, J. et al., *Human Pose Regression with Residual Log-Likelihood Estimation*, arXiv:2107.11291 (ICCV 2021) | RLE (future pose milestone only) |
| R16 | Bolya, D. et al., *YOLACT: Real-time Instance Segmentation*, arXiv:1904.02689 (ICCV 2019) | Prototype–coefficient mask formulation; protonet, K, coefficient, mask-loss conventions (A14–A18) |
| R17 | Llerena, J. M. et al., *Gaussian Bounding Boxes and Probabilistic IoU*, arXiv:2106.06072 (2021) | ProbIoU rotated loss (A19) |
| R20 | Redmon, J. et al., *You Only Look Once: Unified, Real-Time Object Detection*, arXiv:1506.02640 (CVPR 2016) | Origin of the YOLO family term; MIT-licensed Darknet lineage (naming attribution) |

### Dataset and tooling documentation (Ultralytics-unrelated)

| ID | Reference | Role |
|---|---|---|
| R12 | Lin, T.-Y. et al., *Microsoft COCO*, arXiv:1405.0312 (ECCV 2014); cocodataset.org | Dataset and eval protocol (bbox + segm) |
| R18 | Xia, G.-S. et al., *DOTA: A Large-scale Dataset for Object Detection in Aerial Images*, arXiv:1711.10398 (CVPR 2018) + official devkit | Dataset, crop convention, val evaluation protocol, license terms |
| R21 | fuse-augmentations, github.com/Borda/fuse-augmentations, commit 5834dc5ed5a245f9a7477ab326c307fc1a279c2b, Apache-2.0 | Synthetic scene generator for test fixtures and dataset stand-ins (A26). No YOLO-implementation lineage; verified Apache-2.0 |
| R22 | torchmetrics, Lightning AI, github.com/Lightning-AI/torchmetrics, Apache-2.0 | `MeanAveragePrecision` bbox mAP engine for the detection acceptance instrument (WP-069). Apache-2.0 verified from the installed 1.9.0 wheel dist-info `licenses/LICENSE` |
| R23 | faster-coco-eval, github.com/MiXaiLL76/faster_coco_eval, Apache-2.0 | COCOeval-faithful, pycocotools-free backend for `MeanAveragePrecision` (WP-069). Apache-2.0 verified from the installed 1.7.2 wheel dist-info `licenses/LICENSE` (PyPI metadata omits the license field) |
| R24 | Lin, T.-Y. et al., *Focal Loss for Dense Object Detection*, arXiv:1708.02002 (ICCV 2017) | Prior-probability classification bias initialization (sec. 5.1, pi = 0.01) for dense one-stage heads (A30) |

### Placeholders

| ID | Status |
|---|---|
| R15 | Withdrawn (blueprint v1.4). Formerly the prose pages of docs.ultralytics.com; removed because the site is generated from the AGPL repository. Retained as a numbered placeholder so R16–R21 ids stay stable. No page was ever consulted. |
| R19 | Legal-evidence citation only, never an implementation source: Ultralytics GitHub org discussion 14008 (public, non-code) stating licensing terms "apply to the specific implementation of the YOLO models, not the conceptual model itself." Collected at planning time by the planning role. |

## Naming evidence record

Package name `open-yolos` uses "YOLO" as a model-family/category term. Evidence:

1. **Origin**: "YOLO" coined by Redmon et al. (2016) [R20]; the original Darknet
   implementations are MIT-licensed.
2. **Multi-party generic usage**: YOLOX (Megvii), YOLOv6 (Meituan), YOLOv7/v9
   (Wang et al.), PP-YOLOE (Baidu), YOLO-NAS (Deci), YOLOv13 (Lei et al.) — a
   decade of unrelated parties using the family name.
3. **Ultralytics' own position** [R19]: licensing terms "apply to the specific
   implementation of the YOLO models, not the conceptual model itself."
4. **Word-mark status**: the bare "YOLO" US software word mark is registered to
   an unrelated entity (Next Computer Inc., Reg. 6927573), itself subject to a
   pending cancellation proceeding.

Usage discipline: "YOLO26", "YOLO11", "Ultralytics", and their logos never
appear in package names, module paths, class names, or model identifiers.
Nominative references to the paper ("the YOLO26 paper, arXiv:2606.03748") are
the only usage. The YOLO26 method implementation is exposed under the feature
name `e2e` (variants `open-yolos-e2e-{n,s,m,l,x}`), never a version number.

## Audit record

Periodic audits of the clean-room contract (AGENTS.md denylist) against the
full execution record. Method and detailed findings live in the project's
design document (sec. 12); this table is the versioned summary.

| Date | Scope | Evidence layers | Verdict |
|---|---|---|---|
| 2026-08-02 | WP-001–WP-069, 56 commits, audited at `4dd18ba` | codebase/dependency sweep · commit provenance trail (55/56 trailers, all allowlist-cited) · complete session fetch log (only R1, R7, R8, R12, R21 endpoints; zero denylisted domains) · delegated-work hand-over records · persistent agent memory | Upheld — no Ultralytics code, config, weights, or docs consulted at any point; every design input traces to R1–R24 or a registered assumption |

## Usage log

First-use log of sources consulted during implementation. Papers are cited by
arXiv id; access dates recorded per session.

| Date | Source | Used for |
|---|---|---|
| 2026-07-31 | R20 | NOTICE attribution, README disclaimer (WP-002) |
| 2026-08-01 | R1–R20 | Policy docs seeded from the blueprint (WP-003) |
| 2026-08-01 | R21 | Synthetic fixture/dataset generator decision (A26, D12) |
| 2026-08-02 | R22, R23 | torchmetrics `MeanAveragePrecision` + `faster_coco_eval` backend replace pycocotools in the bbox evaluator (WP-069) |
| 2026-08-02 | R24 | Prior-probability cls bias init adopted after the Det-A collapse diagnosis (A30) |
