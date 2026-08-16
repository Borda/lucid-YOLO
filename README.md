# lucid-yolo

> lucid-yolo is an independent, from-scratch PyTorch Lightning implementation of the real-time detection, instance segmentation, and oriented detection methods described in the Ultralytics YOLO26 paper (arXiv:2606.03748). "YOLO" refers to the family of real-time detectors originated by Redmon et al. (2016). This project is not affiliated with, endorsed by, or derived from Ultralytics or its codebase. No Ultralytics source code, configurations, or model weights were consulted or used. See docs/PROVENANCE.md.

## What this is

A research reproduction for knowledge sharing: independent verification of the paper's published ablation claims (STAL small-object gains, Progressive Loss schedule ranking, MuSGD convergence, DFL-removal neutrality, segmentation prototype-fusion and auxiliary-loss gains, OBB long-edge angle formulation) in a codebase with no shared lineage to the reference implementation. Everything implemented here is a fact, equation, or procedure published in the papers cited in docs/PROVENANCE.md.

## Versioning

Perpetual 0.x release train — each 0.MINOR is a gated capability milestone (0.1 detection, 0.2 instance segmentation, 0.3 oriented detection, 0.4+ rolling). No 1.0 is planned: the project tracks a living specification (the paper plus this project's assumption register), and each release freezes its golden metrics; later releases must never regress them. This is a deliberate policy, not an abandonment signal — see docs/DECISIONS.md.

Current release: **0.4.0** — the three tiers of 0.3 plus the inference path that reads their checkpoints, a second dataset layout, whole-image tile merging and an export gate. It is the first release that trains no new tier and publishes no new accuracy figure: 0.1 through 0.3 each landed a capability and a run, and this one consolidates what those three produced. No `v0.1.0` was ever tagged; the detector's history ships inside the 0.2.0 changelog section, and `goldens/frozen/0.2/` is the first frozen set. Releases publish no trained weights (D14), which for the oriented model is a licence matter as well as a policy one: DOTA permits academic use only, so weights trained on it could not ship under this repository's Apache-2.0 terms (O4).

Every oriented figure this repository publishes is still **per tile**. `merge_whole_images` exists as of 0.4 (roadmap 107), but no oriented run has been re-scored through it, so the published numbers remain comparable to nothing outside this repository — see the reproduction report. Building the merge and reporting a number through it are two work packages, and only the first is done.

## Development

```bash
make setup   # venv + editable install + pre-commit hooks
make gate    # lint + pre-commit + tests + golden regression: the merge gate
```

Datasets are never committed and never fetched by the test suite — docs/DATASETS.md says where COCO 2017 and DOTA-v1.0 come from and what has to be on disk before a `[DATA]` run. docs/TRAINING.md carries the launch command for each of the three tiers.

The execution contract for contributors and agents lives in AGENTS.md; every design decision cites its public source (docs/PROVENANCE.md), and every point where the papers underdetermine the implementation is a recorded assumption (docs/ASSUMPTIONS.md).

## Examples

Releases publish no trained weights (D14), so every command below takes a checkpoint you trained yourself — docs/TRAINING.md carries the launch command for each tier.

One image to a prediction, as printed detections and as a JSON report:

```bash
lucid-predict --checkpoint runs/det.ckpt --image street.jpg
lucid-predict --checkpoint runs/seg.ckpt --image street.jpg --output masks.json
lucid-predict --checkpoint runs/obb.ckpt --image aerial.png --output rboxes.json
```

The same three predictions drawn over the picture they were made on:

```bash
python scripts/draw_predictions.py runs/det.ckpt street.jpg --output street_det.png
python scripts/draw_predictions.py runs/seg.ckpt street.jpg --output street_seg.png --conf-threshold 0.4
python scripts/draw_predictions.py runs/obb.ckpt aerial.png --output aerial_obb.png --img-size 1024
```

A detection checkpoint draws boxes, a segmentation one adds each instance's own mask overlay, and an oriented one draws rotated quadrilaterals — never their upright envelopes, which is a different rectangle from the one the model reported. The task is read from the checkpoint; `--task` can name it, and a checkpoint that contradicts the name is refused rather than drawn through the wrong path. Colour is a function of the class index alone, so two figures of the same scene are comparable by eye. `matplotlib` is a `dev` dependency and stays one: the drawing lives in `scripts/`, and a wheel a consumer installs pulls no plotting stack.

## License

Apache-2.0 (see LICENSE and NOTICE).
