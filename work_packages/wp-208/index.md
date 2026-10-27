# WP-208 — the three CUDA replay holders on shared primitives, segmentation head replay under accumulation, and the red-main rule for CUDA tests

> Engineering log · [Phase 19 — Codebase audit remediation](https://borda.github.io/lucid-YOLO/engineering/phase-19-codebase-audit/)
>
> 🔗 Depends on [WP-207](https://borda.github.io/lucid-YOLO/work_packages/wp-207/), [WP-209](https://borda.github.io/lucid-YOLO/work_packages/wp-209/) · Needed by [WP-210](https://borda.github.io/lucid-YOLO/work_packages/wp-210/)
>
> ⬅️ [WP-207](https://borda.github.io/lucid-YOLO/work_packages/wp-207/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-209](https://borda.github.io/lucid-YOLO/work_packages/wp-209/) ➡️


<a id="wp-208"></a>

## WP-208 — one copy of the recording, gradient, release and lifecycle steps, and the segmentation head holder moved to the forward-time copy

**Why.** The network, segmentation-head and loss holders (`ptl/network_capture.py`, `ptl/segment_head_capture.py`, `ptl/loss_capture.py`) share one design, but each carried its own copy of the steps they have in common:

- the four global-hook registries, read three times;
- the autocast re-entry with the cache off, written three times;
- the precision signature, which the other two imported from the network holder;
- the BF16 runtime gate, which the loss holder duplicated rather than imported;
- the buffer-snapshot, `fork_rng`, synchronize and restore block around recording, in the network and head holders.

They also differed where they should not:

- **The head holder still copied live gradients from a backward pre-hook**, the pattern [WP-206](https://borda.github.io/lucid-YOLO/work_packages/wp-206/) found wrong on CUDA for the network holder. Because of that it refused gradient accumulation, at setup and again in its runner.
- **Only the head holder drained its streams** before dropping its graphs; the network and loss holders dropped theirs with replays possibly still queued.
- **The loss holder's callback cast `module._loss_capture` without a check**, so a trainer reused on a module without that attribute raised `AttributeError`. Its `retire` cleared the holder again on every later cause.
- **On a failed fit**, the network and loss callbacks let a failing drain replace the trainer's own exception; the head callback already turned it into a warning.

**Change.** A new private module, `ptl/_graph_replay.py`, holds functions rather than a base class, so the holder classes, their attribute names and every warning text stay as they were:

| Primitive | Replaces |
| -- | -- |
| `BF16_REPLAY_SUPPORTED` | the version check in the network and loss holders |
| `precision_signature`, `tensor_signature` | the network holder's precision read and the storage fingerprint in the network and head holders |
| `global_hook_breach`, `module_hook_breach` | the hook checks in all three holders |
| `replay_autocast` | the autocast re-entry in all three holders |
| `record_graphed` | the recording block in the network and head holders |
| `copy_out_live_gradients` | the gradient copy in the network holder and the head holder's pre-hook |
| `release_graphs`, `drain_device` | the head holder's per-stream drain, and a device-wide drain for the other two |
| `ReplayLifecycle` | the three Lightning callbacks, now one-line subclasses under their old names |

Each holder re-binds a moved name under the module-level name it always used (`_BF16_REPLAY_SUPPORTED`, `_BF16_SUPPORTED`, `_precision_signature`, `_tensor_signature`), so a patch of that module still reaches the value the holder reads. The holders shrink from 706, 561 and 438 lines to 619, 490 and 426, against 354 for the shared module, most of it docstrings and doctests.

**The head holder copies at forward time and accumulates.** `_RecordedGroup.run` calls `copy_out_live_gradients` on the group's parameters right before the replay, and the pre-hook is gone. The post-hook that releases the group's hold stays. That hold is what makes the forward-time copy sufficient, as in [WP-206](https://borda.github.io/lucid-YOLO/work_packages/wp-206/): while a replay's backward is pending, every later call of that group runs native, so nothing else can alias the static gradient buffers before that backward writes them. With the copy moved, the setup condition `accumulate_grad_batches == 1` and the runner's "gradient accumulation changed after setup" retirement are removed. The setup warning drops "with accumulate_grad_batches=1". The lifecycle callback no longer writes `entry.pending_backward` itself; the holder gained `start_batch()`, the same call the other two make.

**Release drains everywhere (D6).** `release_graphs` synchronizes each distinct stream once, in first-seen order, before a holder drops its graphs. The head holder keeps the stream each group recorded on. The network and loss holders do not record a stream. They now record the device, and `drain_device` synchronizes the whole device, so a replay launched from any stream has finished. It does nothing without a recording, for a non-CUDA device or before CUDA is initialized, so a CPU run and unpickling never touch the CUDA runtime. The drain runs only when a holder is cleared (fit start and end, failure, retirement, loss workspace growth), never per step. Every `clear()` resets its fields first and keeps the recording alive in a local reference until the drain returns, so a failing drain still leaves an empty holder. The network holder clears every pose stem holder even when an earlier one's drain fails, and then raises the first failure. The reason for draining everywhere rather than proving it unnecessary: nothing shows that dropping a graph pool with replay work still queued on a stream is safe, and the project cannot produce that proof offline. One synchronize at a rare point is the cheaper side of the trade.

**The lifecycle fixes (D5).** `ReplayLifecycle._holder` reads the attribute with `getattr` and checks its type for every holder, so a module without it makes each hook a no-op. `on_exception` catches a `RuntimeError` from the drain and warns `<flag>=True could not drain its recordings after the fit failed (...)`, the text the head holder already used. `_LossCapture.retire` returns early once the holder is disabled: one warning per fit, and no second `clear()`.

**Not consolidated, on purpose.**

- **Dispatch certificates.** The network holder and the module certify by method-name shadowing, the head holder by class-method identity plus a type allow-list, and the loss holder by callable identities. Merging them would change what each guard admits, and tests call each check directly.
- **Activation gates.** The holders read different trainer facts, and their warning texts are pinned by tests; only the accumulation half of the head holder's gate is removed.
- **The loss holder's recording.** It has no buffers, suspends the garbage collector and makes the first replay inside its own `try`, so it keeps its block. Its silent BF16 fallback is unchanged.

**A CPU seam for the whole cycle.** The network and head holders record only through `_graph_replay.record_graphed`, looked up on the module at call time. The `dry_recording` fixture in `tests/ptl/conftest.py` patches it to return the module itself, so a CPU test runs recording, the forward-time copy, the hold, its release and `clear` as a CUDA fit does, with the native arithmetic standing in for the graph. Graph numerics are what it cannot show; the `gpu` tests own them. No production flag was added for it.

**The replay flags' lifetime and the red-main rule (D32).** The principal answered P5 and the third decision of the 2026-10-20 [escalation](https://borda.github.io/lucid-YOLO/ESCALATION/) on 2026-10-23, and [D32](https://borda.github.io/lucid-YOLO/DECISIONS/) records both:

- **Red main.** A known failing CUDA test leaves `main` red until it is fixed, and a [GPU] work package does not close without a green `make gate-gpu` run. `AGENTS.md` carries it in two places: the `make gate-gpu` sentence of section 2, and "leave `main` red" in section 7. [WP-209](https://borda.github.io/lucid-YOLO/work_packages/wp-209/), which landed before this package to make that run green, was the first package closed under it.
- **Flag lifetime.** Each flag is kept or retired on its own record. A new section of [CUDA performance](https://borda.github.io/lucid-YOLO/advanced/cuda-performance/#lifetime-of-the-replay-flags) gives one row per flag, with its package, documented workload, measured gain and the two triggers. Every documented gain is at least 10%, so no flag is a retirement candidate; nothing is removed here.

**Tests.**

- `tests/ptl/test_graph_replay.py` (new):
    - stream drain order and de-duplication;
    - no CUDA call without an initialized CUDA recording, and a device-wide synchronize with one;
    - drain before release on `clear` for the network and loss holders;
    - every stem holder cleared, the device drained and the graphed forward dropped whichever of them fails, with the first failure raised;
    - a failing drain raised at fit end and warned on a failed fit;
    - the no-op callback for a missing or foreign holder, across all three holders.
- `test_network_capture_guards.py::TestDryRecordingCycle::test_accumulated_microbatches_record_once_and_match_eager`: two accumulated micro-batches through the dry seam record once and end bitwise equal to eager.
- `test_segment_head_capture_guards.py`:
    - `test_live_gradient_leaves_storage_at_forward_time`: right after `run`, before any backward, the parameter holds a separate tensor with the old values.
    - `test_dry_recording_accumulates_like_native_and_drains_on_clear`: two micro-batches through one group record once, hold until backward, match a native run bitwise and drain once on `clear`.
    - `test_fit_setup_activates_under_gradient_accumulation` and `test_gradient_accumulation_keeps_certificate`: no warning and no retirement at accumulation 2. The accumulation case of `test_contract_drift_retires_with_its_reason` is removed.
- `test_loss_capture.py::test_later_retire_cause_stays_silent`.
- `test_segment_network_capture.py`:
    - `heads-bf16-accumulated` of `test_segment_fit_replays_across_empty_partial_validation_and_ema` now expects both holders to replay with no warning.
    - New CUDA test `test_segment_head_replay_under_accumulation_matches_native`: head replay on and off, BF16, accumulation 3, EMA, clipping and validation. Each window runs a full batch, a partial batch that accumulates natively and a second full batch. The empty-target batch sits alone in the last window. It compares weights, optimizer, EMA and scaler bitwise. A spy on the forward-time copy asserts that every recorded prediction group replayed at least once inside a window, onto the gradients its earlier batches accumulated, so a run whose in-window replay silently went native cannot pass. A batch opening a window is not counted, since the previous step's gradients are alive there at every accumulation setting (see Limits).
    - The `heads-bf16-accumulated` cell asserts recording, not replay onto accumulated gradients; the dedicated test above carries that claim.
- Two `_RecordedGroup` tests built with `__new__` gain `group.parameters = ()`, since `run` now reads them.

A mutation check, which made `copy_out_live_gradients` a no-op, failed four tests: the network dry cycle, the head forward-time copy, the head dry cycle and the existing head backward-storage test.

**Verification.**

- **CPU.** `make gate` green on this commit's tree, the nine `tests/ptl/test_graph_replay.py` tests and the new guard and head-accumulation tests among it.
- **CUDA.** `colab_11` ran `make gate-gpu` on an A100 before [WP-209](https://borda.github.io/lucid-YOLO/work_packages/wp-209/) and was red only on the failures that package fixed, which the `colab_12` control reproduced on the tree without this package. `make gate-gpu` green on an A100 (py3.11, torch 2.9.1+cu129) at the stack's tip `72b5505`, the tree [WP-218](https://borda.github.io/lucid-YOLO/work_packages/wp-218/) left: 359 passed and 2 skipped (a `grad_dtype` case that needs a newer torch, and a replay test that needs two CUDA devices), every notebook in full, `demo_obb` on DOTA-v1.0 among them, and 55/55 accelerator goldens. `tests/ptl -m "gpu or data"` also passed on Colab's py3.13 / torch 2.11 (242 passed). The same selection on this package's own commit, `92e775b`, passed in that session too (242 passed). That green run closes the package, as D32 requires.

**Goldens.** No frozen golden is touched. Every replay flag is off by default, and the overfit goldens build their module with the defaults.

**Limits.**

- **Throughput** of head replay under accumulation is not measured. This package claims gradient and state equality with native training at accumulations 2 and 3, on the GPU named above.
- **A clone on every replayed step, not measured.** Lightning's automatic optimization runs the training step's forward, then `zero_grad`, then backward (pytorch_lightning 2.6.6, `loops/optimization/automatic.py`). So the forward-time copy meets the previous step's gradients on every replayed step, at accumulation 1 too: it clones the recorded parameters' gradients, and `zero_grad` then drops the clones. The network holder has paid this since [WP-206](https://borda.github.io/lucid-YOLO/work_packages/wp-206/), and the head holder pays it from this package on; its old pre-hook ran after `zero_grad` and cloned nothing at accumulation 1. The cost is one device copy of the recorded parameters' gradient bytes per step. No speed-up in the flag-lifetime table was re-measured with this clone in place, so the next measurement of each flag is what confirms it still holds. Avoiding the clone would mean tracking which gradients alias the static buffers. A pose network holder also synchronizes the device once per stem holder on `clear`, which is rare and left as is.
- **[WP-193c](https://borda.github.io/lucid-YOLO/work_packages/wp-193c/) and [WP-193d](https://borda.github.io/lucid-YOLO/work_packages/wp-193d/) stay `🔄`.** Their DoD names runs that are still outstanding, and this package does not run them.
