# 🚨 Escalation Log

Blocked-WP entries per the anti-guessing rule (AGENTS.md sec. 4). An agent stops and writes an entry here — then waits for a human decision — when any of these occur:

1. A gate fails after **two** documented assumption iterations (each iteration = a recorded ASSUMPTIONS.md revision plus a re-run).
2. Answering a question would require a denylisted source. Never resolve an ambiguity by looking at the reference implementation.
3. A WP's roadmap spec conflicts with the technical specification, or the specification conflicts with the papers.
4. A change would require altering a frozen golden.
5. A run would exceed 4 GPU-hours and is not already marked [HUMAN].

Escalation is success, not failure: this log plus the assumption register is the research output a from-code port could never produce.

## 🧾 Entry template

```markdown
## <date> — WP-<NNN> <slug>

- **Symptom**:
- **Hypotheses tried** (with ASSUMPTIONS.md revision ids):
- **Sources consulted** (allowlisted only):
- **Decision requested**:
- **Resolution** (filled by human):
```

## 📌 Entries

*(none yet)*
