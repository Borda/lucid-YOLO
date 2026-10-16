# 🔗 Phase 14 — Augmentation upstreaming to `fuse`, v0.7.0

--8<-- "work_packages/wp-154.md:engineering"

--8<-- "work_packages/wp-154c.md:engineering"

--8<-- "work_packages/wp-154b.md:engineering"

--8<-- "work_packages/wp-155.md:engineering"

--8<-- "work_packages/wp-156.md:engineering"

--8<-- "work_packages/wp-157.md:engineering"

--8<-- "work_packages/wp-158.md:engineering"

--8<-- "work_packages/wp-159.md:engineering"

## Phase 14 — what each row does, and where its boundary is

<a id="phase-14-rows"></a>

Phase 14's roadmap cells were the longest in the register by a factor of three, because each row carries a boundary that is easy to cross by accident: the swaps are large, and the difference between "this much of the engine moved" and "and this other thing came with it" is exactly what a scope cell exists to state. That reasoning belongs here, where there is room for it, and the roadmap keeps the summary. Each row below is written before it lands, so it is a specification rather than a record; the `### WP-nnn` entry written when the row lands is what says what actually happened.

<a id="wp-154-scope"></a>

**[WP-154](../work_packages/wp-154.md) — the pin bump alone.** The dependency bump and the re-freeze it forces, with nothing delegated. R21 advances from `fedde0c1` to `a5a276a` — the commit `v0.12.0` is tagged from once this roll-out is clean. Two things break loudly and are repaired here. `tests/fixtures/synthetic.py` imports `animal_shapes`, which upstream's restructure removed; the replacement is `tuple(AnimalShape)[:KEYPOINTS_ANIMAL_COUNT]`, the same declaration-order selection the helper performed, and the other three imports from that package are unaffected. The synthetic fixtures themselves also moved: `PrimitiveShape.TRIANGLE` was redesigned from obtuse-scalene to equilateral, `polygon_to_obb` re-derived from an upright frame rather than minimum-area calipers, and the animal family rebuilt around a packaged SVG zoo, against goldens whose tolerance is zero. Every changed golden value is traced to a named upstream change before it is re-frozen; one that cannot be traced is a regression and stops the row rather than being frozen over.

<a id="wp-154c-scope"></a>

**[WP-154c](../work_packages/wp-154c.md) — the freeze mechanism, fixed rather than the files, a third time.** [WP-154](../work_packages/wp-154.md) hit [WP-132](../work_packages/wp-132.md)'s wall again: `goldens/frozen/0.5` and `0.6` both re-acquired the two generator-derived goldens, because `make freeze-goldens` is a blind copy with no notion that a golden can be pinned to an external package's output rather than to this project's own code. A `"freezable": false` field on the two live goldens, a `scripts/freeze_goldens.py` that skips them at copy time, and a `check_goldens.py` that rejects one found frozen anyway — so the row that removed the files this time is not the row that has to remove them again next time. No golden value changes and neither frozen directory is touched.

<a id="wp-154b-scope"></a>

**[WP-154b](../work_packages/wp-154b.md) — the conventions, before anything delegates.** Absorbs FA-1, FA-2 and FA-3, and gives up the two conventions §8.12 measured as genuinely different — deliberately before any delegation, so the frozen literals they move travel with a local edit rather than with a swap. The composition centre becomes `((W-1)/2, (H-1)/2)` and the mirror axis `(width - 1) / 2` (`data/augment.py`, `data/rotated_aug.py`), on the principal's reasoning that one convention across image and coordinate transport is worth a re-freeze and two conventions inside one pipeline are worth nothing; rotated angles are unaffected, since `pi - theta` and `-theta` differ by a half turn the long-edge convention absorbs.

What is asserted rather than changed: the datamodule's existing `torch.Generator` reaches `fuse`'s own draws rather than a second stream — the property [WP-079](../work_packages/wp-079.md) was opened by losing — and end to end under `num_workers > 0`, which upstream does not exercise; and constant-value padding is a `fill=` parameter requiring `padding_mode="zeros"` rather than the `PaddingMode.CONSTANT` member the plan expected, in the image's own value range so `114/255` transfers unchanged. The half-pixel convention has nothing left to pin: 0.12 declined to add `align_corners` and sampled with `align_corners=True` against the normalization sandwich derived for that same flag, so the pixel-space map is the one an `align_corners=False` implementation produces for the same matrix, measured to `atol=1e-5`. That measurement is recorded, not frozen — there is no flag whose flip it would catch.

<a id="wp-155-scope"></a>

**[WP-155](../work_packages/wp-155.md) — the letterbox and the instance filter.** Absorbs FA-4 and FA-5. `Letterbox` keeps `inverse_map` and `forward_affine` unchanged in signature and semantics, so its consumers — six evaluation and prediction call sites, four since [WP-152](../work_packages/wp-152.md) added `predict_keypoints` — are untouched, and delegates the resize-and-pad internals. `RandomAffine` and `MosaicAssembly` consume one upstream keep mask instead of two local `_keep_mask` copies that must agree; `Targets.filter` is unchanged, since which instances survive is a `fuse` question and what a surviving instance means is not.

The trap is in the defaults. Upstream's `min_size` and `min_visibility` are both `0.0`, deliberately — the defaults drop nothing rather than encode a plausible-looking minimum — against this project's `2.0` and `0.1`. Omit either argument at any call site and every instance survives the warp, including ones pushed entirely off canvas, with no exception and no shape change. Both are passed explicitly at every call site and a test asserts it. Deletes `_resize_pad`, `_LetterboxGeom` and both `_keep_mask`.

<a id="wp-155b-scope"></a>

**WP-155b — the resampling delta, measured on purpose.** The row decision 6 asked for, holding no swap. Delegating the letterbox keeps its geometry exactly — unrounded ratio, `max(1, round(size * r))` content size, `(out - new) // 2` floor-halved pads with the odd pixel right or bottom, verified in both directions on integer arithmetic — while changing its pixels: this project resized with an antialiased bilinear `F.interpolate` followed by `F.pad`, where upstream runs one `grid_sample` from source canvas to letterboxed canvas with `antialias=False` by default and, when enabled, a Gaussian mipmap prefilter rather than torch's area-weighted kernel. So every coordinate round-trip test passes and only image-derived metrics move, on the evaluation path, across the six evaluation and prediction call sites. A delta that hides behind a green coordinate suite is exactly the one that has to be measured on purpose. This row measures it against the 0.6 frozen goldens, records the number, and re-freezes what moved. Nothing moving is a legitimate outcome and is recorded as one.

<a id="wp-156-scope"></a>

**[WP-156](../work_packages/wp-156.md) — the affine reduces to ranges over a segment.** Absorbs FA-3 and FA-5 into the warp itself, and takes the local fusion with it. What remains locally is R1 Table S3's ranges and the sampling of them. `FusedAffineLetterbox` is deleted outright — [WP-155](../work_packages/wp-155.md) and this row composed inside `fuse` *are* that composition, so the local class is a second implementation of the package's whole subject. Polygon rings ride the composed matrix rather than a duplicate warp path: `return_matrix=True` hands the composed pixel matrix back and `apply_affine_to_points` applies it to rings here.

What decides whether that matrix is the whole chain is adjacency, not `clip_policy` — which governs when a fused *colour* chain clamps and shares nothing with this. Only consecutive geometric operations group into one segment, so a colour operation between two geometric ones splits the run and the returned matrix then describes the last segment alone. Ordering geometry before the letterbox before any colour is therefore a requirement of this row, and the single-geometric-segment condition is asserted as a tier-C invariant because silence is how it would fail. The returned matrix is also `None` for an exact-only pipeline and for a letterbox-only one, where `letterbox_matrix()` is the route instead, so a consumer checks *not `None`* **and** single segment before treating it as the whole chain — two conditions, not one. Deletes `_warp_image`, `_theta_from_pixel_matrix`, `_transform_box_corners` and the whole fused class.

<a id="wp-157-scope"></a>

**[WP-157](../work_packages/wp-157.md) — the two task conventions.** Absorbs FA-7 and FA-8, together because each passes a table or a callback upstream rather than moving a rule there. `rotated_aug.py` shrinks to what upstream declines to own rather than being deleted outright: `clip_rboxes_to_canvas` stays, because clipping a rotated box yields a polygon and not a rotated box, so upstream refuses the operation by design and bridges through `rbox_envelopes` to a plain-box clip and the keep mask for the survival decision instead; `check_rotated_pairing` stays too, validating a local `Targets` pairing invariant ([WP-056](../work_packages/wp-056.md)) rather than any geometry. `canonicalize` is passed as a callback rather than reimplemented upstream, since angle range and periodicity are a task convention (A22) and not a resampling concern, and `rotated_geom.py` is untouched — the assigner, rotated NMS and the OBB head all read it.

The left/right index-swap table is passed through rather than owned; `HorizontalFlip` delegates, the table staying dataset metadata read from the caller (A64), which is what keeps `K` generic. `_mirror_targets` shrinks rather than being deleted: upstream's `permute_keypoint_pairs` moves coordinates only and leaves the visibility flags where they were, so the local permutation of `keypoint_vis` stays. Drop it and every mirrored sample carries a visible point marked occluded and its partner marked visible, with identical shapes throughout and no failing coordinate assertion. The mirror axis is not this row's concern — [WP-154b](../work_packages/wp-154b.md) already moved it, which is why nothing frozen moves here.

<a id="wp-158-scope"></a>

**[WP-158](../work_packages/wp-158.md) — the closing sweep.** The last of the four rows that re-freeze; like the other three, it holds no swap. `data/__init__.py` re-exports 54 symbols — 55 when this paragraph was written, less the `FusedAffineLetterbox` [WP-156](../work_packages/wp-156.md) deleted — a count set when the package owned the whole augmentation stack and raised by [WP-147](../work_packages/wp-147.md)'s five parameter dataclasses; it is reduced to what survives, so the public surface states what this project still implements. `docs/ROADMAP.md`, `docs/ASSUMPTIONS.md` (A10, A11, A21, A26), `docs/PROVENANCE.md`, the model cards and `README.md` are rewritten to describe the boundary rather than a stack this project no longer implements, with A11's half-open rasterisation rule kept — that one stays local and still governs mask supervision.

Any structural fixture metric the swaps moved is re-frozen with the move recorded. Tier D is *not* re-frozen, because there is nothing there to re-freeze: it holds no literal and no golden, only same-seed-equal and different-seed-differ against whichever implementation is live, so it either still holds after the swaps or names a real loss of reproducibility. Tiers A, B and C are not re-frozen either: if any moved, the swap was wrong and the row that moved it is the defect.
