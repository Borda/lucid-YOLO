# 📦 Phase 10 — Consolidation


## WP-065 — what three tiers say that none of them says alone

<a id="wp-065"></a>

The row said "merge det/seg/obb sections". It appends instead, and the reason is D10: the report is append-only, a later release correcting an earlier claim by adding to it rather than editing the record away. Merging three accepted sections would rewrite three records that three `[PRINCIPAL]` gates signed off on. So the consolidation is a fourth section that reads the other three against each other and revises none of them, and its own first line says it is not a fourth tier: no run stands behind it.

**The consolidation found an absence, and the absence is the result.** The NMS-free deploy path's cost against NMS is the one quantity all three tiers could have reported in the same units — Det-smoke measures 1.11 AP raw and 1.46 EMA, Seg-smoke 1.27 box and 0.81 segm. OBB-smoke reports EMA against raw weights and never evaluated an NMS baseline at all, so the project has no figure for what NMS-free deployment costs orientation, on a path built precisely to deploy without NMS. Three sections can each be complete and the set still have a hole in it; nothing but putting them in one table shows it.

The assumption tables turned out not to overlap the way the package expected. Twenty-three ids, and **no id appears in two tiers' tables** — the spawn brief assumed A11 did, and it does not: A11 is detection's anchor rule, and segmentation's table has its own row for it saying the mask crop reuses that rule rather than defining a second one. A13 crosses the same way, named inside the oriented A49 row as the gain the rotated term inherited. Two cross-references, zero duplicate rows, which is a tidier register than anyone had checked for.

What the union does show is a count that no single tier's table could: **eleven of the twenty-three are `held` or `not exercised at its recorded value`** — an assumption a passing run carried without isolating. That is not eleven weak results, and the note says so; the runs cleared their gates. It is eleven places where the evidence is "the run worked" rather than "this was measured", and A21 is the extreme case, where the tier ran at R18's 512 px overlap and the register's own 200 px value therefore remains unmeasured by the run that was supposed to exercise it.

The learning-rate line is the other thing only the side-by-side view produces. Det-smoke and Seg-smoke both scale lr from a batch-64 recipe; OBB-smoke, at half the batch and 2.56x the pixels per step, leaves it at the config value. Each section states its own choice and none argues it; three facts in three places become one open question in one.

Four hypotheses close the section, each with the evidence already in the file and the experiment that would settle it, and one of them is there specifically to *not* be upgraded: the segmentation run's box accuracy exceeds detection's, and the 0.2.0 section calls that confounded by precision, hardware and pipeline. A consolidation is exactly where a confounded observation gets quietly promoted to a finding by being restated one more time, so it is restated as the confound.

The gate is `test_report_sections`, reading headings only and never their content — which is what makes it compatible with append-only. A correction landing inside a section leaves every heading where it was; a section renamed out of the file does not, and that is the failure the test exists to catch.



## WP-066 — the exported graph confirms the one-to-one claim, not just its shape

<a id="wp-066"></a>

All three deploy paths export on the legacy TorchScript tracer at opset 20 and none of them contains a `NonMaxSuppression` node — a single `TopK` stands where suppression would be, and the two `Softmax` nodes belong to the attention block, not a distribution-focal-loss box head. The DFL-free, NMS-free one-to-one branch R1 claims shows up in the emitted graph itself, not only in a docstring. Verifying the graph's *arithmetic* — not just its shape — against the checkpoint it came from turned out to be almost entirely an export-tooling story rather than a fidelity one; it is recorded in full at [wp-066.md#wp-066-engineering](https://borda.github.io/lucid-YOLO/work_packages/wp-066/#wp-066-engineering).

