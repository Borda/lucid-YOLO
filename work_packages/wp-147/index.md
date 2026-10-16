# 📦 WP-147 — `refine(data): split sampling from application on stochastic transforms`

> Engineering log · [Phase 13 — Contributor admission, augmentation guard and keypoint parity, v0.6.0](https://borda.github.io/lucid-YOLO/engineering/phase-13-contributor-admission/)
>
> 🔗 Depends on [WP-146](https://borda.github.io/lucid-YOLO/work_packages/wp-146/) · Needed by [WP-148](https://borda.github.io/lucid-YOLO/work_packages/wp-148/)
>
> ⬅️ [WP-146](https://borda.github.io/lucid-YOLO/work_packages/wp-146/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-148](https://borda.github.io/lucid-YOLO/work_packages/wp-148/) ➡️


## WP-147 — a seam is a move, not a rewrite

<a id="wp-147"></a>

Every stochastic transform did two separable things inside one `__call__`: it drew parameters from a generator, then applied them. Only the second is the contract that has to survive the augmentation stack moving upstream, because the upstream sampler will not draw the same numbers from the same seed — different call order, different distributions, different consumption of the stream. Freezing an expectation against a seed would therefore break at the swap for reasons that have nothing to do with correctness, and the only available repair would be a re-freeze at exactly the moment the guard was supposed to hold. So the seam comes first, and everything frozen afterwards is keyed to stated parameters.

Seven transforms now carry it: `RandomAffine` (which had `AffineParams` and a private `_sample` already, and is the precedent the other six are modelled on), `FusedAffineLetterbox`, `HSVJitter` with a new `HSVParams`, `HorizontalFlip` with `FlipParams`, `MosaicAssembly` with `MosaicParams`, `Mixup` with `MixupParams` and `CopyPaste` with `CopyPasteParams`. Each gains a public `sample()` that is the only method touching `generator`, and a public `apply()` that draws nothing; `__call__` is now literally `self.apply(inputs, self.sample(...))`. `Letterbox` needed nothing — it is deterministic and its `forward_affine` / `warp_targets` / `inverse_map` have been public since WP-009.

**The draw code was moved, not rewritten.** That is the whole discipline of this package. Every `sample()` body is the original draw sequence lifted verbatim: same helper, same order, same count. The guard for it is a temporary same-codebase equivalence test in `tests/data/test_aug_seam.py` — record the generator state, run `__call__`, rewind, run `sample()` then `apply()`, assert the two agree bit for bit on the image and on every target modality. Rewinding rather than re-seeding is deliberate: a path that consumed a different number of draws fails immediately instead of drifting into some later transform's numbers. That file is old-against-old inside one repository with no upstream package involved, so it is not the old-versus-new comparison the plan rules out, and [WP-149](https://borda.github.io/lucid-YOLO/work_packages/wp-149/) deletes it once the frozen tiers make it redundant.

**Two params types are two-valued on purpose.** `MixupParams.lam` is `float | None` and `CopyPasteParams.selected` is a possibly-empty tuple, because for both transforms *not acting* is a sampled outcome rather than an absence of parameters. A `MixupParams` that could only express a blend would make the pass-through branch unreachable through the seam, and the pass-through is the branch that must consume the trigger uniform and nothing else — consuming a blend factor there would desynchronise every later draw in the pipeline. Both branches are pinned by their own test.

**`CopyPaste` was the one transform whose loop genuinely interleaved.** It drew per candidate and pasted inside the same loop, so splitting it meant establishing that the draw count depends on the candidate count and the `max_paste` cap alone, never on pixels. It does, so `sample(candidate_count)` reproduces the loop's stream exactly and `apply` rasterises the named rings afterwards. The cap is what makes this worth a test of its own: a seam that drew once per candidate regardless of the cap would consume more of the stream than the original and pass every assertion about the pasted image.

**One ordering did change, and only on an error path.** Validation that used to sit between the draw and the application now sits after both — `RandomAffine`'s rotated-pairing check, `Mixup`'s and `CopyPaste`'s same-size-images check. For any input that does not raise, the stream and the result are identical, which is what the equivalence test asserts. For an input that raises, the generator has advanced where it previously had not. That is a real difference and worth writing down rather than discovering later; it is accepted because a raising sample is an aborted sample, and no caller resumes a pipeline from the RNG state left behind by an exception.

The `last_params` / `last_matrix` / `last_gains` / `last_center` / `last_lam` / `last_flipped` attributes all still exist and are set by `apply`, not by `sample`. Putting them on the application half keeps them meaning what they always meant — what the most recent *call* did — including when a caller supplies parameters instead of drawing them.

