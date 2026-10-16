# 🧵 Cross-cutting

## Float reproducibility has two axes, not one

<a id="cross-float"></a>

A26 already records that the synthetic fixture generator is byte-identical per seed **on a given platform** and not across platforms — libm last-bit rounding differs, macOS arm64 against ubuntu x86-64 CI. That entry is about image generation. The same fault line runs through **torch reductions**, and it was found the hard way on 2026-08-14.

`test_obb_training.py` carried a snapshot of a `task="detect"` training step asserted with `==`, captured on arm64 at one intra-op thread. Three values of that same total have now been observed from source trees computing identical arithmetic:

| total | where |
| -- | -- |
| 22.80110550 | arm64, 1 intra-op thread (the recorded snapshot) |
| 22.80111694 | arm64, 12 threads |
| 22.80112076 | x86-64, ubuntu CI |

The spread is 1.5e-5 at magnitude 22.8 — about eight float32 ulps — and comes from the summation order a build chooses for the dense classification term. The test passed for the whole of Phase 8 because until 0.3.0 was pushed it had only ever run on the machine its snapshot came from; its first x86 execution failed it. It now compares at a relative 1e-4 with an absolute floor.

The general rule this establishes, beyond that one test: **bit-identity is a within-architecture property.** A gate that asserts it across machines is asserting something no library promises. What survives across architectures is a tolerance argued from ulp count, and what such a gate can still catch is a change to the objective — those move by percent. What it can no longer catch is a change of one rounding step. No tolerance choice recovers that.

## A green gate is only as trustworthy as the interpreter it ran under

<a id="cross-venv"></a>

Mid-session on 2026-08-13, `uv run` silently rebuilt `.venv` — Python 3.14.2, torch 2.13 — replacing the 3.11.16 environment a passing `make gate` had just run under. Nothing announced it; the first sign was unrelated behaviour changing. Restored with `make setup`, and every later invocation named `.venv/bin/python` explicitly rather than going through `uv run`.

The general point: "the gate is green" is a claim about an environment as much as about a tree, and a tool that can reconstruct that environment between two commands can invalidate the claim without touching a tracked file.

## Deterministic SVG is deterministic per matplotlib, not per run

<a id="cross-svg"></a>

WP-080 fixes `svg.hashsalt` and suppresses date metadata so an unchanged run regenerates its figure byte-identically. That holds — within one matplotlib. On 2026-08-14, regenerating the committed detection and segmentation figures produced files differing from the committed ones, and the cause was isolated by regenerating with the *git-stashed original* script: the committed script also differs, so the difference is the renderer, not the edit.

The committed figures were left untouched rather than churned. The consequence to know: a figure diff is not evidence that a plotting change altered a curve. Prove which by re-rendering with the pre-change script before reading anything into it.

## Two process failures that each cost a run

<a id="cross-process"></a>

Both recurred often enough to be written into AGENTS.md rather than left in notes.

**The gate certifies a tree, not an intention.** The loop runs `make gate` at step 4 and flips a roadmap row at step 6, and the gap between them is where a docs edit rides into a code commit that was never gated with it. That is how commit `fcf3040` left `main` red on two meta tests, and the next agent spent a full run reporting a blocker it was forbidden to fix. The invariant is a set equality between what was gated and what is committed; the docs-only exemption is the thing it is constantly confused with.

**A completion notification is not a completion.** A delegated work package returns marked *completed* whether the agent finished or stopped mid-sentence on a partial edit, and the notification carries a result field that reads like a report — so an unfinished run looks like a finished one that summarized badly. It happened **four times in Phase 8**. The check that catches it is one `git status` against the spec, and resuming the same agent beats respawning it, because its transcript is the context.
