# 🔄 Phase 8 — Oriented detection, v0.3.0


## WP-105 — a report that survives

<a id="wp-105"></a>

Two defects the first real oriented evaluation exposed, both in the CLI rather than in what it measures.

`--output` into a directory that did not exist yet raised **after** the scoring pass finished and printed its numbers — minutes of accelerator work producing a traceback and nothing else. The parent is now created rather than required, because the file is the only durable form of the expensive part. The general rule: a long computation's output path should be validated or created *before* the computation, never discovered after it.

Second, a 10,132-tile pass showed one line at the start and nothing until it ended. The run has a known length; it now draws a progress bar, wrapped at the CLI rather than inside the evaluator, which tests drive in-process.

______________________________________________________________________



## WP-094 — the difficult flag died at the loader

<a id="wp-094"></a>

[WP-057](https://borda.github.io/lucid-YOLO/work_packages/wp-057/#wp-057) manufactures difficult instances at crop time and A51 defines the channel they are supposed to travel on, but nothing connected the two: the reader did not forward `difficult` out of the annotation record, so every instance the tiler had flagged arrived at the loss indistinguishable from a whole one. Nothing disagreed anywhere — the counts are right, the boxes are right, and the only thing wrong is that a protocol decision the register spent a row on was not in force.

A53's non-schema keys are what make the flag survivable through a COCO container at all, since COCO's schema has no field for it or for the tile's window provenance, and a standard reader ignores both and still sees an ordinary detection set.



## WP-095 — a run monitor is not an acceptance instrument

<a id="wp-095"></a>

The oriented tier reached its release row with no way to produce an acceptance number. What existed was the epoch metric `val/rotated_mAP` ([WP-102](https://borda.github.io/lucid-YOLO/work_packages/wp-102/#wp-102)), and that is a run monitor: it scores whatever the training loader hands it, under torchmetrics' own protocol rather than A46's recall grid, A47's cap and A48's difficult rule. Two figures computed under two protocols are not the same measurement, and the one a release quotes has to be the registered one.

One decision inside the instrument is worth recording. It scores in **letterbox coordinates, with the ground truth letterboxed alongside**, rather than inverting the predictions back to tile coordinates first. Both sides then pass through one geometry, which makes the comparison an equality rather than a proxy — the inverse is where a frame error would enter, and here it is not on the path at all. The same reasoning put checkpoint loading in `lucid_yolo.eval.checkpoint` instead of a second copy: one loader, one place for a task to be read wrong.



## WP-096 — a rename with a published cost

<a id="wp-096"></a>

Parsing moved to jsonargparse so that flags and help text derive from the **operation signatures** rather than from a hand-maintained parser that is free to disagree with the function it calls. The cost is real and is stated rather than absorbed: jsonargparse spells a parameter with underscores, so every documented flag changed shape in one commit.

`lucid-download` therefore survives as a deprecated alias rather than being dropped. It is named in published reproduction instructions, and an instruction that no longer runs is a worse outcome than a duplicate entry point — the alias keeps accepting its **old** flags, which is the part a copy-pasted command depends on.



## WP-097 — a published total no correct download can reach

<a id="wp-097-engineering"></a>

`check_dota_root` compared its summed train+val totals against R18's whole-dataset figures ([wp-097.md#wp-097](https://borda.github.io/lucid-YOLO/work_packages/wp-097/#wp-097)), so it failed on every correct download, at the first command an operator is told to run.

The fix is not a looser tolerance but a different question. A count the caller did not state is **reported**, not compared: on an unstated expectation the check's job is to say what it found, and a number with no stated referent cannot fail. What still fails is a stated expectation that misses — and, new here, a DOTA expectation aimed at a COCO root, which had previously been accepted and silently ignored.

Recorded because the failure was in the gate rather than in the data, and a gate that fires on every correct input is the one shape of gate that teaches its operator to disbelieve gates.

