# Claude Code — GitHub workflow

How Claude Code interacts with this repository: branches, commits, CI,
pull requests. The absolute rules in `CLAUDE.md` (never commit to main,
never commit `data/` or secrets, run tests before commit) override
anything here.

---

## Branch strategy

**Never commit directly to `main`.** Always work on a feature branch.

Branch naming — `<scope>/<short-desc>`, kebab-case:

| Scope | When | Example |
| --- | --- | --- |
| `approach/<lane>` | Blank-form approach lanes (one per worktree under `~/Code/exadaps-aff-ds-synth-worktrees/`) | `approach/pymupdf-redact` |
| `synth/<topic>` | Synth-dataset module work | `synth/classifier` |
| `feature/<topic>` | General feature work | `feature/eval-harness` |
| `docs/<topic>` | Documentation-only changes | `docs/post-merge-refresh` |
| `fix/<short-desc>` | Bug fixes | `fix/widget-clear-regression` |

Create the branch at session start:

```bash
git checkout -b approach/page-rebuild
```

If a branch for the work already exists, check it out and continue.

---

## Commit rules

### When to commit

Commit **coherent units of work**, not file-by-file or batched. Each
commit should leave the tree in a passing-tests state for its scope.

Reasonable commits include:

- A new module file + its tests, when both pass.
- A schema change + every call-site updated.
- A new approach implementation + its docs entry.
- Documentation refresh as a single commit.

### Commit message format

```
<type>(<scope>): <one-line summary>

<optional body — explain the why if non-obvious>
```

Types: `feat`, `fix`, `test`, `refactor`, `docs`, `chore`, `ci`, `build`.

Examples (drawn from this repo's history):

```
feat(blank_forms): pymupdf-redact content-stream + widget redaction
test(golden-set): add 3 XFUND form examples, curate FUNSD/XFUND raw
ci: strip test steps, keep ruff + add pylint linting
refactor: archive prior pipeline under legacy/, scaffold src/aff
```

### What to always include

- Implementation + its tests in the same commit.
- Updated documentation if behaviour or interface changed.

### What to never commit

- `.env` or any file with secrets / API keys.
- `data/raw/`, `data/synth_dataset/`, `data/process_steps/`, or any other
  subdirectory of `data/` except `data/test_forms/` (committed fixtures).
- `__pycache__/`, `.pyc`, `.DS_Store`, `*.parquet`, `*.tiff`.
- Generated PNGs except those in `tests/fixtures/golden_set/`
  (whitelisted via `.gitignore`).

The `.gitignore` already covers these — check `git status` before staging.

---

## Test protocol

```bash
# Toolchain
nix develop
uv sync

# Run all tests
uv run pytest

# Run scoped to a module
uv run pytest tests/blank_forms/ -v
```

Run the relevant tests before each commit. Do not commit on a failing
test in the scope you're touching. For a feature branch, run the full
suite before opening the PR.

Lint:

```bash
uv run ruff check src/ tests/
uv run pylint --disable=all --enable=E src/aff/
```

Ruff is required; pylint errors-only is what CI checks.

---

## CI

`.github/workflows/ci.yml` runs **lint only** — ruff + pylint-errors —
on every push to a branch matching the configured triggers and on every
PR targeting `main`.

CI does **not** run pytest. The test suite depends on local fixture data
that is too large for CI; tests run locally before commit and before PR.

If you add a new branch-naming scope (e.g. `migrate/*`), update
`.github/workflows/ci.yml`'s `branches:` block so CI fires on it.

---

## Pull requests

### When to open

One PR per coherent change. Open it after:

1. The branch is committed and pushed.
2. The full local test suite passes.
3. Lint passes (`ruff check src/ tests/` + `pylint -E`).

Worktree lanes (`approach/*`) follow a different rule: **they do not
push and do not open PRs** — they commit locally and the human reviews,
then merges manually. See `~/.claude/skills/worktree-lanes/SKILL.md`.

### PR title / body

Keep the title short (<70 chars). Use the body for detail.

```
gh pr create --title "feat(blank_forms): pymupdf-redact approach" --body "$(cat <<'EOF'
## Summary
- What changed and why, 1–3 bullets.

## Test plan
- [ ] What to run to verify, as a markdown checklist.
EOF
)"
```

### Pre-PR checklist

- [ ] `uv run pytest` passes locally with zero failures.
- [ ] `uv run ruff check src/ tests/` is clean.
- [ ] `uv run pylint --disable=all --enable=E src/aff/` is clean.
- [ ] No data files, secrets, or generated artifacts in the diff.
- [ ] Commits are coherent; no `wip:` left behind.

Target `main`. Do not force-push to `main`.

---

## Worktree lanes

Approach lanes live in sibling worktrees:
`~/Code/exadaps-aff-ds-synth-worktrees/approach-<name>/`. Each carries:

- An `INSTRUCTIONS.md` (durable brief, git-ignored).
- A `STATE.md` (per-run resume brief, git-ignored).
- A branch `approach/<name>`.

Lanes commit but do **not** push and do **not** open PRs. The human
inspects via `git -C <worktree> log`, then merges manually into `main`.

See `~/.claude/skills/worktree-lanes/SKILL.md` for the full skill spec.

---

## Session start

```bash
git branch --show-current
git status --short
uv run pytest tests/ --tb=line -q 2>/dev/null | tail -10
```

Then continue the in-progress work, or branch off `main` for new work.
