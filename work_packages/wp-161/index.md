# 📦 WP-161 — `docs: the README names the paper nominatively`

> Engineering log · [Phase 15 — Audit remediation, v0.8.0](https://borda.github.io/lucid-YOLO/engineering/phase-15-audit-remediation/)
>
> 🔗 Depends on [WP-158](https://borda.github.io/lucid-YOLO/work_packages/wp-158/) · Needed by none
>
> ⬅️ [WP-160](https://borda.github.io/lucid-YOLO/work_packages/wp-160/) · [Roadmap](https://borda.github.io/lucid-YOLO/ROADMAP/) · [WP-162](https://borda.github.io/lucid-YOLO/work_packages/wp-162/) ➡️


## WP-161 — the discipline was written down and the banner did not follow it

<a id="wp-161"></a>

`docs/PROVENANCE.md` sec. 3.5 states the rule in one sentence: "YOLO26", "YOLO11", "Ultralytics" and their logos never appear in package names, module paths, class names or model identifiers, and *nominative references to the paper ("the YOLO26 paper, arXiv:2606.03748") are the only usage*. The README's first line said "the Ultralytics YOLO26 paper". That is not a nominative reference to a paper; it is a vendor name bound to the method, which is the one binding this project's whole thesis argues against — the method is what the paper describes, and a codebase's choices are a separate thing that the shared name makes impossible to see. The register had the rule and the most-read file in the repository did not follow it.

**Three mentions, two categories, and only one of them moved.** The attributive one names the work: it is now "the YOLO26 paper", matching the register's own quoted form. The other two are the non-affiliation disclaimer — "not affiliated with, endorsed by, or derived from Ultralytics or its codebase" and "No Ultralytics source code, configurations, or model weights were consulted or used" — and they are a different instrument. A disclaimer that does not name the party it disclaims is not a disclaimer, it is a sentence about nobody. Both are also pinned as literal fragments in `scripts/lint/audit_license_headers.py` and required in `NOTICE`, so removing them from the README is a three-file change to a legal statement rather than a wording fix, and it is not what "stop naming the vendor as the method's owner" asks for.

**The id became the link it already was.** `arXiv:2606.03748` sat as bare text in a file where every other reference resolves. It is now `[arXiv:2606.03748](https://arxiv.org/abs/2606.03748)`, fetched and read before being written: the page returns the paper R1 cites, by the authors R1 names. The RLE id on the keypoints line is deliberately left alone — linking it is a separate verification pass, and this row's subject is the paper this project reproduces.

One thing this row does not fix, recorded so the next person does not have to rediscover it: `NOTICE` carries the identical drift, "the Ultralytics YOLO26 paper", and unlike the README's disclaimer sentences that phrase is *not* an audited fragment there. It is the same one-word correction. It is left for whoever touches `NOTICE` next rather than widened into this row.

