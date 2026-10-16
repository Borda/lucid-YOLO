# 📈 Reproduction report

A living document (D10): one section lands with each `0.MINOR`, recording what was reproduced, what was assumed, and what diverged. Sections are append-only — a later release corrects an earlier claim by adding to it, never by editing the record away.

Independent, from-scratch implementation of the methods described in the YOLO26 paper (arXiv:2606.03748, [R1]). No Ultralytics source code, configuration, or weights were consulted at any point; see `PROVENANCE.md` for the source allowlist and the audit trail.

## 🗂️ Pages

Pages, one per release section, in the order D10 appended them:

- [v0.1.0: Detection](report/detection.md)
- [v0.2.0: Instance segmentation](report/segmentation.md)
- [v0.3.0: Oriented detection](report/oriented-detection.md)
- [v0.4.0: Consolidated note — detection, segmentation, oriented detection](report/consolidated-note.md)
- [v0.5.0: Keypoint detection](report/keypoints.md)
- [Unreleased: Correcting note — which engine scored each tier](report/correction-scoring-engine.md)
- [Unreleased: Correcting note — the v0.5.0 section's source claims](report/correction-keypoint-sources.md)
