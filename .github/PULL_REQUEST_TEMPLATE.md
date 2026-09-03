<!-- SPDX-License-Identifier: Apache-2.0 -->

## Work package

- WP id:
- Commit subject follows `<type>(<scope>): <detail>` (\<= 72 chars):

## Clean-room checklist

- [ ] Provenance ids cited (from `docs/PROVENANCE.md`):
- [ ] Assumptions recorded (A-ids) or `none`:
- [ ] Gate test id proving the DoD:
- [ ] Clean-room acknowledged. Nothing here was copied from a source outside the permissive allowlist — MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC (D13) — which excludes:
  - **copyleft**: AGPL, GPL, LGPL, SSPL
  - **source-available**: BSL, Elastic, PolyForm
  - **paid or proprietary** source of any kind
  - **any source whose licence cannot be read** (D17) — unreadable is not permissive
  - and, as the instance a YOLO contributor reaches for by reflex rather than as the definition: the Ultralytics repository, any mirror, any vendored or packaged copy, and `docs.ultralytics.com`
- [ ] Sign-off present on every commit (`git commit -s`) — see `docs/CONTRIBUTING.md`
- [ ] `make gate` is green locally, including every frozen golden
