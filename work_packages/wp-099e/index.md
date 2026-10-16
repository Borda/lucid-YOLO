# 📦 WP-099e — `feat(data): the splits a check is asked for`

> Engineering log · [Phase 9 — Inference and generalization](https://borda.github.io/lucid-YOLO/engineering/phase-09-inference/)
>
> 🔗 Depends on [WP-099c](https://borda.github.io/lucid-YOLO/work_packages/wp-099c/) · Needed by none
>
> ⬅️ [WP-099c](https://borda.github.io/lucid-YOLO/work_packages/wp-099c/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-065](https://borda.github.io/lucid-YOLO/work_packages/wp-065/) ➡️


## WP-099e — the gap was one layer above where it looked

<a id="wp-099e"></a>

[WP-099c](https://borda.github.io/lucid-YOLO/work_packages/wp-099c/) closed by naming this itself: `splits` was still not a CLI flag, so a root shipping only `train` fails on a missing `val`. That was recorded so it would be a choice rather than an oversight, and this is the follow-up. A third-party export whose validation split was never cut is a correct export, and the first command an operator runs against a fresh provisioning refused it over a directory nobody had promised.

What is worth writing down is where the parameter was actually missing. `check_yolo_root` and `check_dota_root` had both taken `splits` from the day each was written — `tests/data/test_check.py` had been calling `check_yolo_root(..., splits=("train",))` all along, because the fixtures only ever build one split. The capability existed; the route from the command line to it did not. So the fix is not a feature the layout checkers lacked but one argument `check_dataset` and `_check_root` dropped on the floor. A gap that presents as missing functionality is worth pricing against the layer below before implementing: the two root checkers were the expensive place to fix this, and the wrong one.

The default belongs to the layout rather than to the call. `_check_root` names `YOLO_SPLITS` or `DOTA_SPLITS` at the branch that already knows which layout it is dispatching to, instead of defaulting `splits` in `check_dataset`'s signature. The two constants are equal today and are documented for different reasons — DOTA's testing third has no public labels, YOLO's `test:` entry is a split no run of this project opens — and an entry-point default would have quietly made them one fact, so that changing either would have to be discovered rather than read.

COCO refuses the flag rather than defaulting it, on the reasoning that already governs `--expected_images` there: `check_coco_root` checks `train2017` and `val2017` against 118,287 and 5,000, and the names and the numbers are a single published fact. There is no subset of that to ask for. Following `_check_flags_apply`'s existing message shape was the whole of the design decision — a flag that silently does nothing makes the report read as though it had been honoured.

The empty tuple is the sharper case, because it inverts the failure mode the rest of the module guards. Every other bad argument here fails loudly; `splits=()` fails *cleanly* — no split checked means no problem found means `PASS: dataset layout valid` printed over a root nothing looked at, a false green from the one command whose purpose is to be believed. It is refused in `_require_splits`, called from `check_dataset` before the layout is even inferred and from both root checkers, because those two are documented as the importable core: guarding only the entry point would leave the false green reachable by the shorter route, and a library caller reading `ok` off the result has no report to notice the emptiness in. The entry-point call is what makes `--splits '[]'` fail at the command line, which is where an empty list is easiest to type by accident.

The annotation is `tuple[str, ...] | None`, and the CLI surface it produces was verified rather than assumed: jsonargparse renders it as a list literal, `--splits '[train]'`, which is the spelling `lucid-data download --splits` has always required — a bare `--splits train` is refused by both. One flag's syntax is not a second thing to learn. The `--help` line for it is less self-describing than `download`'s, the `| None` union suppressing the `[ITEM,...]` metavar; cosmetic, and not worth changing the annotation for.

