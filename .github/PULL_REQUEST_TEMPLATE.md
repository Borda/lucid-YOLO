<!-- SPDX-License-Identifier: Apache-2.0 -->

The `squash-message` check reads this description together with the title as the commit message a squash merge will land, and runs it through `scripts/lint/check_commit_trailers.py`. It stays red until the title is `<type>(<scope>): <detail>` and the four trailer lines at the end are filled; it re-runs whenever the description is edited. Replace this paragraph with the change itself: what it does and what it derives from, in prose. No hash-numbered issue references and no at-sign mentions — GitHub auto-links them in whichever repository the commit lands in.

Clean-room acknowledged: nothing here was copied from a source outside the permissive allowlist — MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC (D13) — which excludes copyleft (AGPL, GPL, LGPL, SSPL), source-available (BSL, Elastic, PolyForm), paid or proprietary source, any source whose licence cannot be read (D17), and the Ultralytics repository, its mirrors, packaged copies and `docs.ultralytics.com`. Every commit is signed off (`git commit -s`), and `make gate` is green locally including every frozen golden.

The trailers, one per line, values as `AGENTS.md` sec. 5 defines them: `WP:` one roadmap row id from `docs/ROADMAP.md` or `none`; `Provenance:` source ids from `docs/PROVENANCE.md` (`R7`) or `none (reason)`; `Assumptions:` A-ids from `docs/ASSUMPTIONS.md` or `none`; `Gate:` the test id proving the definition of done.

WP: none
Provenance:
Assumptions: none
Gate:
