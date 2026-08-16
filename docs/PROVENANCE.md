# 📚 Provenance Log

Every design decision in this repository cites its public source. This file is the clean-room evidence record: sources are logged here on first use, and every commit carries a `Provenance:` trailer referencing the source ids below. No Ultralytics code surface, config, weight, or web property (including docs.ultralytics.com) has been consulted at any point — see the denylist in AGENTS.md.

## ✅ Source allowlist

Admissible references. Anything not on this list requires a DECISIONS.md entry before first use; denylisted surfaces are never admissible.

Reference *implementations* (R25–R28) are admitted under D13/ADR-004 on two standing conditions. **Licence:** MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause or ISC only — copyleft (AGPL, GPL, LGPL), source-available, research-only, non-commercial, commercially licensed and unlicensed code are inadmissible at any cost, and an unestablished licence counts as inadmissible rather than permissive. **Use:** reading for diagnosis only; copying code from any external detection repository remains prohibited (AGENTS.md sec. 7). A per-use provenance check still applies to individual files — the entries below record licence verification, not a blanket clearance of every file in the repository.

### Papers under reproduction

| ID | Reference | Role |
| -- | -- | -- |
| R1 | Jocher, G. et al., *Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models*, arXiv:2606.03748 (2026) | Method source: Eq. 1–15, Figs. S1/S2, Tables 2–11, S1–S11 |
| R2 | Sapkota, R. et al., *YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection*, arXiv:2509.25164 (2026) | Independent analysis; benchmark corroboration (incl. seg/OBB tables) |
| R3 | Hidayatullah, P., Tubagus, R., *YOLO26: A Comprehensive Architecture Overview and Key Improvements*, arXiv:2602.14582 (2026) | Third-party architectural specification: variant multipliers, block diagram, SPPF shortcut, Top-K decoding, STAL minimum-anchor behavior |

### Cited primary literature

| ID | Reference | Role |
| -- | -- | -- |
| R4 | Feng, C. et al., *TOOD: Task-Aligned One-Stage Object Detection*, arXiv:2108.07755 (ICCV 2021) | TAL formulation and alpha/beta defaults (A2) |
| R5 | Li, X. et al., *Generalized Focal Loss*, arXiv:2006.04388 (NeurIPS 2020) | DFL definition (the mechanism being removed) |
| R6 | Wang, A. et al., *YOLOv10: Real-Time End-to-End Object Detection*, arXiv:2405.14458 (NeurIPS 2024) | Consistent dual assignment |
| R7 | Liu, J. et al., *Muon is Scalable for LLM Training*, arXiv:2502.16982 (2025) | Muon update, Newton–Schulz orthogonalization |
| R8 | Jordan, K., *Muon: An optimizer for hidden layers in neural networks*, kellerjordan.github.io/posts/muon (2024) | Reference NS iteration count and step scaling (A5, A6) |
| R9 | Bochkovskiy, A. et al., *YOLOv4*, arXiv:2004.10934 (2020) | Mosaic augmentation; CIoU adoption in the lineage |
| R10 | Zheng, Z. et al., *Distance-IoU Loss*, arXiv:1911.08287 (AAAI 2020) | CIoU definition (A1) |
| R11 | Hidayatullah, P. et al., *YOLOv8 to YOLO11: A Comprehensive Architecture In-depth Comparative Review*, arXiv:2501.13400 (2025) | C3k2/C2PSA/SPPF internals in the YOLO11 lineage (A3) |
| R13 | Zhou, Y. et al., *MMRotate*, arXiv:2204.13317 (ACM MM 2022) | Long-edge angle convention; rotated-box tooling conventions (A21, A23) |
| R14 | Li, J. et al., *Human Pose Regression with Residual Log-Likelihood Estimation*, arXiv:2107.11291 (ICCV 2021) | RLE (future pose milestone only) |
| R16 | Bolya, D. et al., *YOLACT: Real-time Instance Segmentation*, arXiv:1904.02689 (ICCV 2019) | Prototype–coefficient mask formulation; protonet, K, coefficient, mask-loss conventions (A14–A18) |
| R17 | Llerena, J. M. et al., *Gaussian Bounding Boxes and Probabilistic IoU*, arXiv:2106.06072 (2021) | ProbIoU rotated loss (A19) |
| R20 | Redmon, J. et al., *You Only Look Once: Unified, Real-Time Object Detection*, arXiv:1506.02640 (CVPR 2016) | Origin of the YOLO family term; MIT-licensed Darknet lineage (naming attribution) |

### Dataset and tooling documentation (Ultralytics-unrelated)

| ID | Reference | Role |
| -- | -- | -- |
| R12 | Lin, T.-Y. et al., *Microsoft COCO*, arXiv:1405.0312 (ECCV 2014); cocodataset.org | Dataset and eval protocol (bbox + segm) |
| R18 | Xia, G.-S. et al., *DOTA: A Large-scale Dataset for Object Detection in Aerial Images*, arXiv:1711.10398 (CVPR 2018) + official devkit | Dataset, crop convention, val evaluation protocol, license terms |
| R21 | fuse-augmentations, github.com/Borda/fuse-augmentations, commit 5834dc5ed5a245f9a7477ab326c307fc1a279c2b, Apache-2.0 | Synthetic scene generator for test fixtures and dataset stand-ins (A26). No YOLO-implementation lineage; verified Apache-2.0 |
| R22 | torchmetrics, Lightning AI, github.com/Lightning-AI/torchmetrics, Apache-2.0 | `MeanAveragePrecision` bbox mAP engine for the detection acceptance instrument (WP-069). Apache-2.0 verified from the installed 1.9.0 wheel dist-info `licenses/LICENSE` |
| R23 | faster-coco-eval, github.com/MiXaiLL76/faster_coco_eval, Apache-2.0 | COCOeval-faithful, pycocotools-free backend for `MeanAveragePrecision` (WP-069). Apache-2.0 verified from the installed 1.7.2 wheel dist-info `licenses/LICENSE` (PyPI metadata omits the license field) |
| R24 | Lin, T.-Y. et al., *Focal Loss for Dense Object Detection*, arXiv:1708.02002 (ICCV 2017) | Prior-probability classification bias initialization (sec. 5.1, pi = 0.01) for dense one-stage heads (A30) |
| R25 | torchvision, github.com/pytorch/vision, BSD-3-Clause | Permissive reference implementation admitted under D13/ADR-004 for diagnostic reading only (never copied). BSD-3-Clause verified 2026-08-05 from the repository `LICENSE` ("Copyright (c) Soumith Chintala 2016"). No Ultralytics lineage: PyTorch-project detection stack, independent of the YOLO family |
| R26 | YOLOX, Megvii-BaseDetection, github.com/Megvii-BaseDetection/YOLOX, Apache-2.0 | Permissive reference implementation admitted under D13/ADR-004 for diagnostic reading only (never copied). Apache-2.0 verified 2026-08-05 from the repository `LICENSE` ("Copyright (c) 2021-2022 Megvii Inc."). Independent anchor-free YOLO lineage (arXiv:2107.08430), no Ultralytics code |
| R27 | MMDetection, OpenMMLab, github.com/open-mmlab/mmdetection, Apache-2.0 | Permissive reference implementation admitted under D13/ADR-004 for diagnostic reading only (never copied). Apache-2.0 verified 2026-08-05 from the repository `LICENSE` ("Copyright 2018-2023 OpenMMLab"). **Core `mmdetection` only** — the sibling `mmyolo` repository reimplements Ultralytics-family models and is deliberately excluded pending a provenance check it has not been given |
| R28 | PaddleDetection (PP-YOLOE), PaddlePaddle, github.com/PaddlePaddle/PaddleDetection, Apache-2.0 | Permissive reference implementation admitted under D13/ADR-004 for diagnostic reading only (never copied). Apache-2.0 verified 2026-08-05 from the repository `LICENSE` (unmodified Apache template, no named holder in the file). Independent PaddlePaddle detection stack |
| R29 | matplotlib, github.com/matplotlib/matplotlib, PSF-based Matplotlib License | Vector figure rendering for the reproduction report (WP-080), a dev-extra tool only — not a runtime dependency. Permissive PSF-derived license verified 2026-08-06 from the installed 3.11.1 wheel dist-info `LICENSE` ("Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved") |
| R30 | mdformat, github.com/hukkin/mdformat, MIT, with its `mdformat-gfm` plugin, github.com/hukkin/mdformat-gfm, MIT | Markdown formatter run as a commit-time hook, a pre-commit tool only — installed in the hook's own environment, never a project dependency. MIT verified 2026-08-11 from each repository's `LICENSE` ("Copyright (c) 2021 Taneli Hukkinen" and "Copyright (c) 2020 Taneli Hukkinen"). The plugin is what makes the formatter table-aware: core mdformat is CommonMark-only and would read a GFM table as prose |
| R31 | shapely, github.com/shapely/shapely, BSD-3-Clause | Polygon-intersection oracle for the WP-063 rotated-IoU gate (A24). A `dev` dependency-group package only — `src/` never imports it, and A24 places the oracle in tests precisely so it stays an independent implementation rather than our own float64 code checking our own float32 code. BSD-3-Clause verified 2026-08-11 from the installed 2.1.2 wheel dist-info `licenses/LICENSE.txt` ("Copyright (c) 2007, Sean C. Gillies. 2019, Casper van der Wel. 2007-2022, Shapely Contributors"), corroborated by the `License: BSD 3-Clause` metadata field and the `License :: OSI Approved :: BSD License` classifier. **Disclosure:** the same wheel bundles the GEOS native library, whose `licenses/LICENSE_GEOS` declares **LGPLv2.1** — dynamically linked, present only in the development and test environment, never vendored into this repository, never linked into any published artifact, and absent from the runtime dependency set. Allowed under D15, which also made the license audit read bundled `License-File` documents rather than metadata fields alone; that change immediately surfaced the same class of exposure in `numpy`, a runtime dependency, which no audit here had ever seen |
| R32 | Roboflow Universe dataset *Cars*, workspace `gongx`, project `cars-jnnoy`, version 1, YOLO-format export, CC BY 4.0 (licence stated in the export's own `README.dataset.txt` and `data.yaml`) | The YOLO label format as a **dataset artifact**, read for WP-099: a `data.yaml` (`names` as an index-ordered list, `nc`, split entries written `../train/images` beside a real `<root>/train/images` tree) and a 70-image train split of normalized `cls cx cy w h` rows. Registered because the format has no specification on the paper allowlist while AGENTS.md forbids reading any implementation's reader — a published export is the format "as published by the datasets themselves". Data only: no code, no configuration written by a detector implementation, no weights. Verified 2026-08-14 by reading the export on disk |
| R33 | MkDocs, github.com/mkdocs/mkdocs, BSD-2-Clause; MkDocs Material, github.com/squidfunk/mkdocs-material, MIT; `mdformat-mkdocs`, github.com/KyleKing/mdformat-mkdocs, MIT | Documentation-site toolchain (WP-114), a publishing tool only — installed through the `docs` dependency group and the pre-commit hook's own environment, never imported by `src/` and never by a gate. Licences verified 2026-08-16 from each repository's `LICENSE` ("Copyright © 2014-present, Tom Christie", "Copyright (c) 2016-2025 Martin Donath", "Copyright (c) 2024 Kyle King"). The site renders the same markdown files GitHub does; nothing here generates a page from source |
| R34 | `packaging`, github.com/pypa/packaging, dual Apache-2.0 OR BSD-2-Clause | Requirement, marker and extras parsing for the dependency-tier walk in `scripts/audit_licenses.py` (WP-115). A `dev` dependency-group package only -- `src/` never imports it, and it arrived transitively (pytest and mkdocs both require it) before it was declared. Licence verified 2026-08-16 from the installed distribution's own `License-Expression` field, `Apache-2.0 OR BSD-2-Clause`, with both texts shipped as `LICENSE.APACHE` and `LICENSE.BSD`; GitHub's licence detector reports NOASSERTION for the dual grant, which is why the wheel rather than the API is what this row cites |

### Placeholders

| ID | Status |
| -- | -- |
| R15 | Withdrawn (blueprint v1.4). Formerly the prose pages of docs.ultralytics.com; removed because the site is generated from the AGPL repository. Retained as a numbered placeholder so R16–R21 ids stay stable. No page was ever consulted. |
| R19 | Legal-evidence citation only, never an implementation source: Ultralytics GitHub org discussion 14008 (public, non-code) stating licensing terms "apply to the specific implementation of the YOLO models, not the conceptual model itself." Collected at planning time by the planning role. |

## 🏷️ Naming evidence record

Package name `lucid-yolo` uses "YOLO" as a model-family/category term. Evidence:

1. **Origin**: "YOLO" coined by Redmon et al. (2016) [R20]; the original Darknet implementations are MIT-licensed.
2. **Multi-party generic usage**: YOLOX (Megvii), YOLOv6 (Meituan), YOLOv7/v9 (Wang et al.), PP-YOLOE (Baidu), YOLO-NAS (Deci), YOLOv13 (Lei et al.) — a decade of unrelated parties using the family name.
3. **Ultralytics' own position** \[R19\]: licensing terms "apply to the specific implementation of the YOLO models, not the conceptual model itself."
4. **Word-mark status**: the bare "YOLO" US software word mark is registered to an unrelated entity (Next Computer Inc., Reg. 6927573), itself subject to a pending cancellation proceeding.

Usage discipline: "YOLO26", "YOLO11", "Ultralytics", and their logos never appear in package names, module paths, class names, or model identifiers. Nominative references to the paper ("the YOLO26 paper, arXiv:2606.03748") are the only usage. The YOLO26 method implementation is exposed under the feature name `e2e` (variants `lucid-yolo-e2e-{n,s,m,l,x}`), never a version number.

## 🔍 Audit record

Periodic audits of the clean-room contract (AGENTS.md denylist) against the full execution record. Method and detailed findings live in the project's design document (sec. 12); this table is the versioned summary.

| Date | Scope | Evidence layers | Verdict |
| -- | -- | -- | -- |
| 2026-08-02 | WP-001–WP-069, 56 commits, audited at `975ff50` | codebase/dependency sweep · commit provenance trail (55/56 trailers, all allowlist-cited) · complete session fetch log (only R1, R7, R8, R12, R21 endpoints; zero denylisted domains) · delegated-work hand-over records · persistent agent memory | Upheld — no Ultralytics code, config, weights, or docs consulted at any point; every design input traces to R1–R24 or a registered assumption |

## 📝 Usage log

First-use log of sources consulted during implementation. Papers are cited by arXiv id; access dates recorded per session.

| Date | Source | Used for |
| -- | -- | -- |
| 2026-07-31 | R20 | NOTICE attribution, README disclaimer (WP-002) |
| 2026-08-01 | R1–R20 | Policy docs seeded from the blueprint (WP-003) |
| 2026-08-01 | R21 | Synthetic fixture/dataset generator decision (A26, D12) |
| 2026-08-02 | R22, R23 | torchmetrics `MeanAveragePrecision` + `faster_coco_eval` backend replace pycocotools in the bbox evaluator (WP-069) |
| 2026-08-02 | R24 | Prior-probability cls bias init adopted after the Det-smoke collapse diagnosis (A30) |
| 2026-08-11 | R30 | mdformat + mdformat-gfm adopted as the markdown commit hook (WP-001) |
