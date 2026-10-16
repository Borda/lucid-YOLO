# 📦 WP-090b — `fix(eval): masks for an empty instance axis`

> Engineering log · [Phase 9 — Inference and generalization](https://borda.github.io/lucid-YOLO/engineering/phase-09-inference/)
>
> 🔗 Depends on [WP-090](https://borda.github.io/lucid-YOLO/work_packages/wp-090/) · Needed by none
>
> ⬅️ [WP-090](https://borda.github.io/lucid-YOLO/work_packages/wp-090/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-091](https://borda.github.io/lucid-YOLO/work_packages/wp-091/) ➡️


## WP-090b — the degenerate input arrives with the second caller

<a id="wp-090b"></a>

`decode_instance_masks` has raised on a zero-length instance axis since WP-053a wrote it. `F.interpolate` treats that axis as channels and rejects an empty one outright. No test caught it and no reviewer would have: from the evaluator the case is unreachable, because both decoders always hand the function a fixed 300 rows with their padding included, so the axis is never empty. Predict decodes only what survives the confidence cut, and an image can have nothing above it.

The rule worth keeping: **a function whose only caller pads to a fixed length has never been exercised at its degenerate input, and that input arrives with the second caller rather than the first.** The shape of the bug is not "someone forgot the empty case" — it is that the empty case did not exist while there was one caller, so no amount of care at the time would have surfaced it.

Returning the empty stack rather than raising is the shape contract, not politeness: a caller assembling per-image results needs a tensor of the right rank and dtype to stack, and an exception pushes the count check out into every such caller, which is where the copies start.

