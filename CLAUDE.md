# CLAUDE.md
## HPE-AFF Data Engineering Pipeline

This file is read automatically by Claude Code at the start of every session.

---

## Two documents govern this project — read both before doing anything

**1. `AGENTS.md`** (repo root) — the technical specification:
what to build, in what order, all schemas, all rules, all constraints.

**2. `.github/CLAUDE_WORKFLOW.md`** — the GitHub workflow:
how to branch, when to commit, commit message format,
CI configuration, when and how to open pull requests.

Read both completely. Then check `pipeline_state.json` to find where to resume.

---

## Absolute rules — these override everything else

- Never commit to `main` directly
- Never commit `.env`, `data/raw/`, `data/consolidated/`, `data/generated/`
- Never commit secrets or API keys
- Always run the relevant test before committing
- Never open a PR until the full test suite passes with zero failures
- `pipeline_state.json` is the source of truth for what is done — trust it

---

## Quick orientation

```
data_pipeline/     source code — this is what gets committed
data/              all data — gitignored except data/test_forms/
.github/           CI workflow + this workflow instructions
AGENTS.md          full technical specification
CLAUDE.md          this file
requirements.txt   pinned dependencies
pyproject.toml     pytest + ruff config
.env.example       environment variable reference (no secrets)
```

---

## Session start — run this first

```bash
git branch --show-current
cat pipeline_state.json 2>/dev/null || echo "not started"
pytest data_pipeline/tests/ --tb=line -q 2>/dev/null | tail -10
git status
```

Then proceed to the next incomplete stage per `pipeline_state.json`.