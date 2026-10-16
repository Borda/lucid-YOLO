# 🏗️ Phase 2 — Architecture


## WP-023 — the param/FLOP fidelity gate

<a id="wp-023"></a>

The gate passed all five scales within ±2% params and ±5% FLOPs of R1 Table 7 (worst case `l` FLOPs +4.2%) after **exactly two assumption iterations**, both registered rather than tuned quietly: A28 (the original full-conv box stem left the head **7–16× the reference budget**) and A3 (nested `C3k` inner-bottleneck count 1 instead of 2, because at depth 1.0 the `l`/`x` variants overshot with two).

The third change was explicitly **not** an iteration: A29's counting convention (GFLOPs = 2× fvcore MACs, on the deployed NMS-free model; params over the full dual-branch checkpoint) was *determined empirically* by trying the candidates. Trying three conventions to discover which one a table used is measurement; changing the model until it matches a number is fitting. The distinction is recorded because the two are indistinguishable from the outside once the gate is green.

