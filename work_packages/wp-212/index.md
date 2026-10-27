# WP-212 — named loss inputs, recorded alpha

> Engineering log · [Phase 19 — Codebase audit remediation](https://borda.github.io/lucid-YOLO/engineering/phase-19-codebase-audit/)
>
> 🔗 Depends on [WP-211](https://borda.github.io/lucid-YOLO/work_packages/wp-211/) · Needed by [WP-213](https://borda.github.io/lucid-YOLO/work_packages/wp-213/)
>
> ⬅️ [WP-211](https://borda.github.io/lucid-YOLO/work_packages/wp-211/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-213](https://borda.github.io/lucid-YOLO/work_packages/wp-213/) ➡️


<a id="wp-212"></a>

## WP-212 — named loss inputs, recorded alpha

**Findings.** CS-17/DP-18 identified positional tensor groups in dual-branch scoring and loss capture, with four copies of the native fallback. CS-21 found that an explicit tensor override drove the total while `DualLossOutput.alpha` recorded the attribute. C-SMALL also identified long capture routing and sequential replay guards.

**Inputs and routing.** Public `BranchPrediction` pairs each branch's logits and decoded boxes. Private `_LossInputs` names the loss recorder's ten inputs in their existing order; the CUDA boundary still receives a flat tuple of tensors. One native fallback handles ineligible loss calls. Replay output ownership is separate from workspace preparation, and segmentation runner checks stop at the first reason while retaining the stage-replacement warning's existing precedence.

**Recorded weight.** Without an override, `DualLossOutput.alpha` remains the attribute float. With an override, it is the supplied tensor, including on replay, without a host read inside compilation or capture. The detection blend keeps its tensor expressions, operation order and dtypes. Extra task losses cast the recorded tensor weight per branch after forming its complement in double precision, preserving the former float-weight rounding and scalar dtype. Log keys, hyperparameters and state dictionaries are unchanged.

**Verification.**

- **CPU.** `make gate` green on this commit's tree. Regression: `test_tensor_alpha_blends_bit_equal_to_attribute_and_records_override`; capture contracts: `test_recorder_preserves_flat_input_order` and `test_replay_records_live_alpha_override`.
- **Goldens.** CPU sweep: 50/50 passed; no golden file is edited.
- **CUDA.** `make gate-gpu` green on an A100 (py3.11, torch 2.9.1+cu129) at the stack's tip `72b5505`, the tree [WP-218](https://borda.github.io/lucid-YOLO/work_packages/wp-218/) left: 359 passed and 2 skipped (a `grad_dtype` case that needs a newer torch, and a replay test that needs two CUDA devices), every notebook in full, `demo_obb` on DOTA-v1.0 among them, and 55/55 accelerator goldens. `tests/ptl -m "gpu or data"` also passed on Colab's py3.13 / torch 2.11 (242 passed). The same selection on this package's own commit, `c7e2ad7`, passed in that session too (242 passed). Both include the loss replay and compiled-step parity tests that exercise the tensor `alpha` blend.

