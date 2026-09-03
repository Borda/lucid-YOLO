# 🚨 Escalation Log

Blocked-WP entries per the anti-guessing rule (AGENTS.md sec. 4). An agent stops and writes an entry here — then waits for a principal decision — when any of these occur:

1. A gate fails after **two** documented assumption iterations (each iteration = a recorded ASSUMPTIONS.md revision plus a re-run).
2. Answering a question would require a denylisted source. Never resolve an ambiguity by looking at the reference implementation.
3. A WP's roadmap spec conflicts with the technical specification, or the specification conflicts with the papers.
4. A change would require altering a frozen golden.
5. A run would exceed 4 GPU-hours and is not already marked [PRINCIPAL].

Escalation is success, not failure: this log plus the assumption register is the research output a from-code port could never produce.

## 🧾 Entry template

```markdown
## <date> — WP-<NNN> <slug>

- **Symptom**:
- **Hypotheses tried** (with ASSUMPTIONS.md revision ids):
- **Sources consulted** (allowlisted only):
- **Decision requested**:
- **Resolution** (filled by the principal):
```

## 📌 Entries

## 2026-09-03 — WP-154b the composition centre moves a frozen golden

- **Symptom**: WP-154b changes `AffineParams.matrix`'s composition centre from `(W/2, H/2)` to `((W-1)/2, (H-1)/2)`, the convention `fuse-augmentations` composes about. `goldens/frozen/0.6/aug_invariants.json` — a pure-code frozen golden written by the v0.6.0 release commit `22739e1` — pins five metrics the change moves past their tolerance: `affine_image_mean` 0.5565 → 0.5542 (tol 0.002), `affine_bbox_coord_sum` 121.3428 → 121.9856 (tol 0.05), `affine_polygon_area` 276.7792 → 277.6373 (tol 0.5), `fused_image_mean` 0.5529 → 0.5506 (tol 0.002), `fused_bbox_coord_sum` 91.0071 → 91.4892 (tol 0.05). Every hook, every one of 2543 unit tests and the other 35 goldens are green; the three residual pytest failures (`scripts/_tests/test_check_goldens.py::test_real_goldens_pass`, `::test_main_exit_code_zero_on_real_goldens`, the `check_all` doctest) all assert "every real golden passes" and resolve with this one file. This is trigger 4: the row cannot be both gate-green and frozen-golden-clean.
- **Hypotheses tried** (with ASSUMPTIONS.md revision ids): none — this is not an ambiguity to iterate on. The delta is fully traced rather than guessed: re-running `aug_invariants` with the centre reverted (and nothing else) reproduces all fifteen stored values bit-exactly, so the whole movement is the centre and there is no second cause to look for. `mixup_image_mean` (-0.0009) and `rotated_rbox_area` (-0.0001) move within tolerance from the same cause; every mirror-independent metric is unchanged, the mirror axis not being exercised by this producer.
- **Sources consulted** (allowlisted only): `docs/ENGINEERING_LOG.md` WP-154b and WP-155b scope entries, `docs/ROADMAP.md` Phase 14 preamble, AGENTS.md sec. 4 and sec. 7, the WP-132 and WP-154 removals of generator-derived frozen goldens.
- **Decision requested**: which of three, noting that the ruling covers the phase and not only this row — WP-155b plans to measure the letterbox resampling delta "against the 0.6 frozen goldens" and re-freeze what moves, and the metrics it will move (`letterbox_image_mean`, `letterbox_pad_fraction`, the fused pair) live in this same file.
    1. **Remove `goldens/frozen/0.6/aug_invariants.json`**, on the WP-132/WP-154 precedent. Different in kind from that precedent: those two were generator-derived and could never again be satisfied by any code change, whereas this one is pure-code and would be satisfiable by any release that did not deliberately give up the convention. Recoverable in history; costs a real regression guard for one release.
    2. **Update its five values in place**, which sec. 7 forbids outright and which makes "current code still satisfies every value a past release pinned" mean something weaker than it does today.
    3. **Re-scope**: hold WP-154b until 0.7 is released and freeze the new convention there, leaving 0.6 untouched — which blocks every remaining Phase 14 row, all of which depend on 154b.
- **Resolution** (filled by the principal): Overriding AGENTS.md §7's standing prohibition on modifying a frozen golden, for this one file, on record — the values are updated in place rather than the file removed. `goldens/frozen/0.6/aug_invariants.json`'s seven traced metrics (the five past tolerance plus `mixup_image_mean` and `rotated_rbox_area`, both moved within tolerance from the same cause) now match the live `goldens/aug_invariants.json` exactly. Presented with the removal alternative (same mechanism as WP-132/WP-154, no rule override needed) and chose the override instead.

## 2026-09-03 — WP-155 the letterbox resample moves the same frozen golden again

- **Symptom**: WP-155 delegates `Letterbox`'s resize-and-pad to `fuse-augmentations`, replacing an antialiased bilinear `F.interpolate` plus `F.pad` with one `grid_sample` from the source canvas to the letterboxed one. Geometry is bit-unchanged — `letterbox_pad_fraction` holds at `0.4`, every coordinate round-trip case passes, `goldens/data_checksums.json` does not move — and exactly one image-derived value moves: `letterbox_image_mean` 0.5024 → 0.4994 against a 0.002 tolerance, in the live `goldens/aug_invariants.json` and in `goldens/frozen/0.6/aug_invariants.json`. 34 of 36 goldens pass; the two failures are that one metric in those two files. This is trigger 4 again, on the same frozen file WP-154b overrode eight commits ago, whose resolution was recorded as one dated exception and explicitly not a precedent.
- **Hypotheses tried** (with ASSUMPTIONS.md revision ids): the delta is measured rather than assumed, and one alternative was tested rather than argued. Upstream's opt-in `antialias=True` was enabled and the producer re-run: the metric stays at 0.4994, the Gaussian mipmap prefilter not engaging at this producer's downscale, so recovering the old pixels is not available short of keeping the local implementation. A32 is revised in this change (non-antialiased resampling widens from the train path's deviation to the whole pipeline's filter) rather than contradicted.
- **Sources consulted** (allowlisted only): the installed `fuse_augmentations` 0.12.0.dev0 source (`affine/matrix.py`, `affine/segment.py`, `targets.py`, `factories.py`, `pipeline.py`), `docs/ENGINEERING_LOG.md` WP-155 and WP-155b scope entries, `docs/ASSUMPTIONS.md` A32, AGENTS.md sec. 4, sec. 6.5 and sec. 7, and the 2026-09-03 WP-154b entry above.
- **Decision requested**: four, noting that WP-155 and WP-155b cannot both keep their commit boundary and a green gate — the swap moves the value, so WP-155 alone fails `make gate` and WP-155b alone has nothing left to measure.
    1. **Keep the row split**, accepting that WP-155's own commit fails the golden check and `main` is red between the two commits, which sec. 5 forbids.
    2. **One commit, override the frozen value again**: land the swap, the live re-freeze and `frozen/0.6/aug_invariants.json`'s `letterbox_image_mean` together. Gate green, but a second sec. 7 override on the same file, and it merges a delegation swap with a frozen move — the exact pairing sec. 6.5 exists to prevent.
    3. **Remove `goldens/frozen/0.6/aug_invariants.json`** on the WP-132/WP-154 precedent, no rule override — at the cost of all fifteen of that file's metrics, fourteen of which are still satisfiable.
    4. **Narrow WP-155** to the keep mask and the letterbox geometry, keeping `_resize_pad` local and antialiased so nothing moves — leaving two resample implementations in the tree and deferring the delegation to 0.7's freeze.
- **Resolution** (filled by the principal): Option 2. Overriding sec. 7 a second time, on record: `letterbox_image_mean` becomes 0.4994 in both the live and the frozen copy, and WP-155 and WP-155b land as one commit. Presented with option 3 (removal, no override needed) and option 4 (no pixel moves at all) and chose the override. Two dated exceptions now stand on this file; neither is a standing waiver, and the next frozen-golden case is a fresh decision rather than one this precedent settles.
