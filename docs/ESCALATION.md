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
