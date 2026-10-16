# ⚡ Phase 5 — Lightning training loop


## WP-039 — a package whose finding is that no code was needed

<a id="wp-039"></a>

Deterministic checkpoint and resume was proved rather than implemented: a four-epoch reference run against a run stopped after two epochs and resumed produces the same per-step loss trajectory over the post-resume epochs (rel 1e-4), with `global_step`, `current_epoch` and every MuSGD momentum buffer restored. **No source changes were needed** — Lightning's checkpointing plus existing module and optimizer state already round-trips.

Two scoping choices made that provable: all runs share `max_epochs`, so the progressive-loss alpha ramp (a pure function of `current_epoch / max_epochs`, recomputed each epoch) is identical either side of the resume; and the data stream is a fixed unshuffled in-memory dataset, keeping sampler-state restoration out of scope. Without the first, a correct resume would still have produced divergent losses.



## WP-040 — overfit-100 on an accelerator

<a id="wp-040"></a>

Train-set recall **0.9747** against the 0.95 floor, MPS, 4:47 wall clock, decoded NMS-free through the deployed path.

The structural decision that outlived it: `goldens/gpu/` is excluded from default harness discovery and recomputed only under `--include-gpu`, which is what keeps the offline gate accelerator-free while still holding accelerator numbers under version control. The cost of that split is that a `gpu/` golden can go stale unnoticed — `shapes_regression_det.json` currently has an outstanding re-freeze.

*Superseded by [WP-167](https://borda.github.io/lucid-YOLO/work_packages/wp-167/#wp-167).* The outstanding re-freeze is closed: all five files under `goldens/gpu/`, `shapes_regression_det.json` among them, were regenerated on an L4. The sentence above is left as written because the staleness it predicted is exactly what happened — `golden-gpu` returned 46/48 on the pre-WP-169 values — and a log that edits its own correct predictions into the past tense stops being evidence that anything was predicted.

