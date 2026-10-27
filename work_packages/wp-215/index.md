# WP-215 — shared lint helpers

> Engineering log · [Phase 19 — Codebase audit remediation](https://borda.github.io/lucid-YOLO/engineering/phase-19-codebase-audit/)
>
> 🔗 Depends on [WP-214](https://borda.github.io/lucid-YOLO/work_packages/wp-214/) · Needed by [WP-216](https://borda.github.io/lucid-YOLO/work_packages/wp-216/)
>
> ⬅️ [WP-214](https://borda.github.io/lucid-YOLO/work_packages/wp-214/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-216](https://borda.github.io/lucid-YOLO/work_packages/wp-216/) ➡️


<a id="wp-215"></a>

## WP-215 — shared lint helpers

**Finding.** K3’s remaining tooling duplication was fence state tracking in the doctest and WP-link audits, seven script-local repository-root definitions and the docs-site audit’s recursive Markdown walk. The existing common helpers already owned root resolution, sorted recursive walks and the standard counted failure report.

**Change.** A private shared fence tracker toggles state using each caller’s existing regular expression, retaining the doctest audit’s language and closing-blank checks and the WP-link audit’s tilde and blockquote handling. The seven scripts import their root from the appropriate common module, retaining a patchable module attribute. The docs-site audit uses the shared sorted walk. Standard counted failure reports already use the common reporter; specialized license, frozen-manifest and commit-trailer reports retain their distinct wording. Test-local root constants remain unchanged. No hook or output format changes.

**Verification.**

- **CPU.** `make gate` green on this commit's tree. `scripts/_tests` remains unchanged; its existing doctest-fence, WP-link, docs-site, frozen-manifest, commit-trailer, README, release-guard, overfit and shapes-regression assertions cover the affected callers. The `FenceTracker` doctest exercises language retention, ordinary lines and quoted mixed-delimiter toggles. No new test functions or hooks.
- **Ownership.** Only the two common owners define a script-level `REPO_ROOT`; the other matches are unchanged test-fixture constants. Only the lint common helper calls `rglob`. Specialized failure formats remain local because the counted reporter cannot reproduce them byte for byte.

