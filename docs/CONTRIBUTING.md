<!-- SPDX-License-Identifier: Apache-2.0 -->

# 🤝 Contributing

This is a clean-room reproduction. That single fact decides almost everything below, so it is worth stating plainly before any process: **the value of this repository is not that the code works — it is that the code's lineage is known.** A patch that improves a number while making its own derivation unclear costs more than it adds, because it converts a traceable artifact into an ordinary one.

So the contribution process asks for two things that most projects do not: where your change came from, and your representation that you had the right to submit it. Everything else here is the usual: run the gate, write the commit, open the pull request.

## 🧭 Before you write anything

**Never read a reference implementation of YOLO.** Not the Ultralytics repository, not a mirror, not a vendored copy inside another package, not `docs.ultralytics.com`. This is the prime directive and it has no exception for "just to check". The admissible sources are the papers and the small set of permissively licensed implementations enumerated in [`PROVENANCE.md`](PROVENANCE.md), each admitted by name under D13 and ADR-004, and each admitted for diagnostics rather than for method.

The rule is broader than its most famous instance. What you must not copy from is **any** source that is copyleft (AGPL, GPL, LGPL, SSPL), source-available (BSL, Elastic, PolyForm), paid or proprietary, or whose licence cannot be read at all (D17). Ultralytics is enumerated because it is the one a YOLO contributor reaches for by reflex, not because it is the definition.

If you have already read one, that is not a disaster and it is not something to conceal — say so in the pull request. What cannot be repaired is a derivation nobody knew about.

## 🚪 The three admission layers

Each layer answers a different question, and none of them answers the other two. That is why there are three rather than one.

| Layer | Mechanism | What it establishes | What it does **not** |
| -- | -- | -- | -- |
| **DCO sign-off** | `Signed-off-by:` on every commit, checked in CI | Ownership and right to submit — your own representation, on the record, per commit | Nothing about what the work derives from |
| **Licence attestation** | The clean-room checklist in the pull-request template | That you did not copy from a non-permissive source | Nothing reviewable — it is a promise, not an artifact |
| **Provenance trailers** | `WP:`/`Provenance:`/`Assumptions:`/`Gate:`, validated in CI | What an algorithmic change is derived from, in a form a reviewer can check against the register | Nothing about your right to submit |

The third is the only one producing a reviewable artifact rather than a promise, which is why an unnamed derivation on a loss or an assigner becomes visible at review rather than after release. The first two are cheap and they are still worth having: a promise on the record is a different thing from silence.

### Signing off

Add `-s` to your commit, or write the trailer yourself:

```bash
git commit -s -m "fix(data): ..."
```

The line must name you and reach you: `Signed-off-by: Ada Lovelace <ada@example.com>`. It means you agree to the [Developer Certificate of Origin](https://developercertificate.org/) — that you wrote the patch, or have the right to pass it on. CI checks every commit in the pull request, not just the last one, and the check is a file in this repository rather than a third-party app, so it survives whatever happens to that service.

## 📐 When a change needs a work package

Most of this repository's history is one work package per commit: a numbered row in [`ROADMAP.md`](ROADMAP.md), a definition of done, a green gate, and a commit carrying provenance trailers. That contract is a **reproduction instrument, not a permanent process** (D18), and it has an end condition.

Once the reproduction report carries all four accepted tiers *and this repository is public*, a change that

- alters no shipped behaviour,
- adds or removes no public symbol,
- moves no golden, and
- changes no documented assumption

lands as an ordinary gated commit. A typo fix, a clearer docstring, a test that covers an existing branch: open a pull request, no roadmap row needed.

Everything else still opens one. And adding a new **task** to the model family — a fifth alongside detection, segmentation, oriented detection and keypoints — restarts the full procedure: its own phase, numbered work packages, a smoke tier, a principal gate and a `0.MINOR` release. A new task is a new reproduction claim, and a claim is what this contract exists to substantiate.

**The gate never relaxes**, under either regime. That is the part D18 does not touch.

## ✅ Running the gate

```bash
make setup        # create .venv and install the dev dependency group
make gate         # every linter, strict typing, the full offline suite, all goldens
```

`make gate` is the merge bar and it runs offline. It must be green **locally** before you open a pull request — CI re-runs it, but a red gate discovered in CI is a round trip that costs a reviewer's attention for something your own machine could have told you.

Two failures deserve a specific response rather than a retry:

- **A golden moved.** Do not re-freeze it. A frozen expectation may change only in a change whose subject *is* that expectation, with the reason recorded in [`ENGINEERING_LOG.md`](ENGINEERING_LOG.md): what moved, from which value to which, and why the new value is correct. A frozen value moving in the same commit as an implementation swap is indistinguishable from "the new code disagreed and we adjusted the test", which is why it is refused.
- **An assumption is in your way.** The register in [`ASSUMPTIONS.md`](ASSUMPTIONS.md) records every place where the papers underdetermine the implementation. If your change depends on reading one differently, revise the row and say why in the same change. Do not work around it silently — the register being complete is the point.

## ✍️ The commit message

```
<type>(<scope>): <detail>          # <= 72 characters

<body: what changed and why, in prose>

WP: 160                            # the roadmap row, when there is one
Provenance: R1 sec. 3.2.1          # at least one source id from PROVENANCE.md
Assumptions: A22, A40              # A-ids touched, or `none`
Gate: tests/data/test_x.py::test_y # what proves the definition of done
```

`type` is one of `feat`, `fix`, `test`, `ci`, `docs`, `chore`, `perf`, `refactor`, `refine`, `exp`, `release`. Trailers are validated against the register, so a `Provenance:` id that does not resolve fails the check rather than the review.

Under D18's relaxation a change with no roadmap row writes `WP: none` — the literal word, matching how `Assumptions:` states its own absence, so the message says no row applies rather than leaving a reader to decide whether one was forgotten. The other three trailers stay mandatory: the derivation question does not go away just because the tracking did.

## 🐛 Reporting a problem

Issue templates are in the repository. Two things make a report actionable here that are easy to leave out:

- **The commit or release you are on.** "Latest" is not a version; this project ships release commits ahead of its own tags, so name the commit.
- **Whether `make gate` is green on your machine before your change.** A gate that is already red locally usually means an environment problem rather than a defect, and it is the fastest thing to rule out.

For a numeric disagreement — a metric that does not reproduce — the useful report names the golden or the reported figure, the value you got, and the hardware. Cross-platform last-bit divergence is expected and documented (A26); a structural difference is not.

## 🔒 Security

Do not open a public issue for a vulnerability. This project ships no trained weights (D14) and has no network surface at inference, so the realistic classes are dependency-borne and deserialization-borne. Report privately through the repository's security advisory form.

## 📜 Licence

Contributions are Apache-2.0, matching the project. Signing off is your statement that you are entitled to submit them under those terms.
