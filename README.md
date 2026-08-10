# lucid-yolo

> lucid-yolo is an independent, from-scratch PyTorch Lightning implementation of the real-time detection, instance segmentation, and oriented detection methods described in the Ultralytics YOLO26 paper (arXiv:2606.03748). "YOLO" refers to the family of real-time detectors originated by Redmon et al. (2016). This project is not affiliated with, endorsed by, or derived from Ultralytics or its codebase. No Ultralytics source code, configurations, or model weights were consulted or used. See docs/PROVENANCE.md.

## What this is

A research reproduction for knowledge sharing: independent verification of the
paper's published ablation claims (STAL small-object gains, Progressive Loss
schedule ranking, MuSGD convergence, DFL-removal neutrality, segmentation
prototype-fusion and auxiliary-loss gains, OBB long-edge angle formulation) in
a codebase with no shared lineage to the reference implementation. Everything
implemented here is a fact, equation, or procedure published in the papers
cited in docs/PROVENANCE.md.

## Versioning

Perpetual 0.x release train — each 0.MINOR is a gated capability milestone
(0.1 detection, 0.2 instance segmentation, 0.3 oriented detection, 0.4+
rolling). No 1.0 is planned: the project tracks a living specification (the
paper plus this project's assumption register), and each release freezes its
golden metrics; later releases must never regress them. This is a deliberate
policy, not an abandonment signal — see docs/DECISIONS.md.

Current release: **0.2.0** — detection and instance segmentation, at the smoke
tier and the smallest scale. No `v0.1.0` was ever tagged; the detector's history
ships inside the 0.2.0 changelog section, and `goldens/frozen/0.2/` is the first
frozen set. Releases publish no trained weights (D14).

## Development

```bash
make setup   # venv + editable install + pre-commit hooks
make gate    # lint + pre-commit + tests + golden regression: the merge gate
```

The execution contract for contributors and agents lives in AGENTS.md; every
design decision cites its public source (docs/PROVENANCE.md), and every point
where the papers underdetermine the implementation is a recorded assumption
(docs/ASSUMPTIONS.md).

## License

Apache-2.0 (see LICENSE and NOTICE).
