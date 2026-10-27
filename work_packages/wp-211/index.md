# WP-211 — per-task strategies

> Engineering log · [Phase 19 — Codebase audit remediation](https://borda.github.io/lucid-YOLO/engineering/phase-19-codebase-audit/)
>
> 🔗 Depends on [WP-210](https://borda.github.io/lucid-YOLO/work_packages/wp-210/) · Needed by [WP-212](https://borda.github.io/lucid-YOLO/work_packages/wp-212/)
>
> ⬅️ [WP-210](https://borda.github.io/lucid-YOLO/work_packages/wp-210/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-212](https://borda.github.io/lucid-YOLO/work_packages/wp-212/) ➡️


<a id="wp-211"></a>

## WP-211 — per-task strategies with exhaustive CLI dispatch

**Finding.** Audit C-TASKS step 2 found that the task registry declared the supported tasks while the Lightning module and three commands still selected behavior through task-string comparisons. A fifth task could therefore inherit detection behavior without an explicit implementation.

**Strategies.** One stateless private object per task lives in `ptl/_task_strategies.py`, selected once after task validation. The single `DetectionLitModule` and `Trainer` contract stays intact. Strategies own padding extras, extra-loss composition and logging, replay routing, metric construction and task-specific validation; registered modules, replay holders and accumulators remain on the Lightning module.

| Task | Target extras / forward | Extra loss and logs | Replay route | Validation |
| -- | -- | -- | -- | -- |
| `detect` | Boxes only | Scalar zero | Complete network; eager or compiled detection objective | Box mAP, distributed metric |
| `segment` | Mask/prototype forward | Mask + semantic terms | Existing segmentation features and head routes | Box + mask mAP, distributed metrics |
| `obb` | Padded rotated boxes | Rotated box + stride-normalized L1 + angle terms | Certified complete oriented network | Rotated mAP, rank-local lists; box metric stays unused |
| `keypoints` | Padded coordinates and visibility | Keypoint + OKS terms | Shared features and pose stems; certified training loss capture | Box mAP + OKS mAP, rank-local point lists |

**Compatibility.** Tensor arithmetic, operation order, dtypes, log keys and initialization order are retained. The task remains a plain string in hyperparameters and checkpoints; strategies register no parameters or buffers. Module methods checked by replay certificates remain the call-through boundary; the oriented certificate also pins the new strategy instance and extra-loss implementation, so changing that consumer cannot silently retain full-network replay. Segmentation retains its collective mask-presence decision at epoch end.

**Fifth-task guard.** Explicit `Task`-keyed tables select evaluation protocols and applicable options, prediction handlers and the three training-loader target flags. Exhaustiveness tests compare every table and the strategy mapping with `set(Task)`; module-resolution cases pin each task to its own strategy class. Prediction handlers retain the existing report formats and evaluation resolves the existing protocol functions at call time.

**Survivors.** No `self._task` or `self.task` equality, inequality or membership comparison remains in `ptl/module.py`. Constructor-local task checks remain to preserve head/loss registration and RNG order; argument validation remains before strategy lookup. These are construction and validation boundaries rather than step-time dispatch.

**Verification.**

- **CPU.** `make gate` green on this commit's tree, the strategy and CLI table exhaustiveness tests, the copy and pickle singleton tests and the oriented-certificate tests among it.
- **Parity.** Before/after CPU comparison covers all four tasks with ragged positive and empty batches: initial/final state dictionaries, hyperparameters, losses, raw outputs, ordered logs and gradients are byte-identical in all eight cases. Capture flags are disabled in this experiment; the existing replay suites cover CPU routing, and CUDA parity remains pending.
- **Goldens.** CPU sweep: 50/50 passed; no golden file is edited.
- **CUDA.** `make gate-gpu` green on an A100 (py3.11, torch 2.9.1+cu129) at the stack's tip `72b5505`, the tree [WP-218](https://borda.github.io/lucid-YOLO/work_packages/wp-218/) left: 359 passed and 2 skipped (a `grad_dtype` case that needs a newer torch, and a replay test that needs two CUDA devices), every notebook in full, `demo_obb` on DOTA-v1.0 among them, and 55/55 accelerator goldens. `tests/ptl -m "gpu or data"` also passed on Colab's py3.13 / torch 2.11 (242 passed). The same selection on this package's own commit, `c50ed22`, passed in that session too (242 passed).
- **Release notes.** No changelog entry: this package changes no user-visible behavior.

