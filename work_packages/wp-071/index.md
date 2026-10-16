# 📦 WP-071 — `perf(data): packed + uint8 batch transport`

> Research log · [Phase 6 — Evaluation, v0.1.0](https://borda.github.io/lucid-YOLO/research/phase-06-evaluation/)
>
> 🔗 Depends on [WP-014](https://borda.github.io/lucid-YOLO/work_packages/wp-014/) · Needed by none
>
> ⬅️ [WP-040](https://borda.github.io/lucid-YOLO/work_packages/wp-040/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-073](https://borda.github.io/lucid-YOLO/work_packages/wp-073/) ➡️


## WP-071 — packed uint8 batch transport

<a id="wp-071"></a>

The transport format and its error bound are A33. What the register does not carry is the shape of the win: flattening the ragged `list[Targets]` into eight dense tensors takes the DataLoader IPC hop from roughly **600 pickled segments per batch to 9**, with bytes down 4x from the uint8 quantization. Segment count, not byte count, is what makes a containerized shm ceiling bite.

