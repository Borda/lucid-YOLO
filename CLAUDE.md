# CLAUDE.md

@AGENTS.md

`AGENTS.md` is the execution contract and wins on every point; this file only imports it and restates the commit rule agents most often miss.

## Commit sign-off

Every commit carries a DCO `Signed-off-by:` trailer — normal, merge, cherry-pick, fixup and agent-made commits alike. Use `git commit -s`, `git merge --signoff` or `git cherry-pick -s`; a merge staged with `--no-commit` is finished with `git commit -s`. Sign at creation time: the DCO GitHub App checks every PR commit, and a pushed commit without the trailer can only be fixed by rewriting the branch. Never reword already-pushed commits to add it. Before any push, confirm every commit ahead of the remote shows a sign-off:

```bash
git log --format='%h %(trailers:key=Signed-off-by,valueonly)' @{upstream}..HEAD
```
