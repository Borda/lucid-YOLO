# 🎯 Phase 6 — Evaluation, v0.1.0


## WP-071 — packed uint8 batch transport

<a id="wp-071"></a>

The transport format and its error bound are A33. What the register does not carry is the shape of the win: flattening the ragged `list[Targets]` into eight dense tensors takes the DataLoader IPC hop from roughly **600 pickled segments per batch to 9**, with bytes down 4x from the uint8 quantization. Segment count, not byte count, is what makes a containerized shm ceiling bite.



## WP-073 — compact annotation store

<a id="wp-073"></a>

`CocoDetectionDataset` held the raw JSON annotation dicts after building its per-image `Targets`. Millions of small Python objects, each with a refcount touched on every access — and refcount traffic materializes copy-on-write pages in every persistent DataLoader worker, so a fork-shared dataset is only shared until Python reads it.

Observed on a Colab 176 GB host, killed at validation boundaries: **epoch 1 at 48 workers, epoch 24 at 32 workers**. The worker count sets how fast the leak accumulates, not whether it does.

Dropping the dicts fixed the growth; the val-loader worker cap (`min(num_workers, 4)`) addressed the second half, a persistent val pool doubling the worker population at exactly the moment the training pool is still resident. That cap is what [WP-102](https://borda.github.io/lucid-YOLO/work_packages/wp-102/#wp-102) later narrowed and [WP-103](https://borda.github.io/lucid-YOLO/work_packages/wp-103/#wp-103) restored for inherited counts.



## WP-076 — epoch-recycled loader workers

<a id="wp-076"></a>

[WP-073](https://borda.github.io/lucid-YOLO/work_packages/wp-073/)'s compact store did not end the OOMs: Colab still killed the run at **epoch 24 and epoch 38**. The residue is not one leak but a class of them — allocator high-water growth, arena fragmentation, and whatever copy-on-write pages survive — all of which are per-worker and none of which a dataset change reaches.

Respawning the pool costs seconds against a 9-minute epoch and hard-resets every one of them, so `persistent_workers` defaults **off**. This is a case where the cheap blunt fix dominates: no diagnosis of *which* allocator behaviour dominated was needed, because recycling ends all of them and the measurement said the cost was noise.



## WP-079 — per-worker, per-epoch augmentation RNG: what the collision cost

<a id="wp-079"></a>

Measured exposure: **1/3200 of the intended draws** at 32 workers over 100 epochs. Two DataLoader defects compose multiplicatively and neither is visible in a loss curve — the run trains, it simply trains on far less augmentation variety than the config describes. The mechanism is a `_TrainPipeline` bug fixed the same day; the full story is at [wp-079.md#wp-079-engineering](https://borda.github.io/lucid-YOLO/work_packages/wp-079/#wp-079-engineering).



## WP-082 — objective invariant gate

<a id="wp-082"></a>

Written after WP-078 and [WP-079](https://borda.github.io/lucid-YOLO/work_packages/wp-079/) survived 474 tests and three tier runs, which is the fact that justifies the package: the suite was large and did not contain the shape of test that would have caught either.

The useful negative result is the test **not** written. An obvious guard for WP-078's defect is "the L1 term must not exceed some share of the objective". On a converged fixture the broken pixel frame reads **10.7%** against the correct frame's **1.5%** — both comfortably under any threshold anyone would have picked. Such a test would look like protection and catch nothing. The gate pins the coordinate frame directly instead (`1/stride` against the pixel frame, per level) and its scale on a near-converged state (0.75 stride-units against 5.99 broken).



## WP-083 — synthetic-shapes generalization golden

<a id="wp-083"></a>

Frozen at NMS mAP50-95 **0.8823**, E2E **0.8583**, mAP50 **0.9842** — three MPS runs identical to four decimals, 127–169 s wall clock. The spread in wall clock against the exactness of the metric is the point: it is a cheap gate whose *number* is stable even though its *timing* is not. The harness this measures — 2000 generated scenes, held-out scoring, mosaic held on for five of six epochs specifically so a collapsed augmentation RNG ([WP-079](https://borda.github.io/lucid-YOLO/work_packages/wp-079/#wp-079)) would be detectable — is described at [wp-083.md#wp-083-engineering](https://borda.github.io/lucid-YOLO/work_packages/wp-083/#wp-083-engineering).

