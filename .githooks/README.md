# Repo-managed git hooks

## Enable (once per clone)

```bash
./.githooks/install.sh
# equivalently: git config core.hooksPath .githooks
```

`core.hooksPath` is local config and is **not** cloned, so every fresh checkout
must run this. It is also listed in the root README Quick Start.

## `pre-commit` -> `check-notebook-outputs.py`

Blocks a commit when a **staged** `.ipynb` carries cell **outputs** or a
non-null **`execution_count`**, so notebook diffs stay to source-only.

- Only staged content is inspected (`git show :<path>`), not the working tree.
- Notebooks already carrying outputs in history are flagged only when you stage
  a new change to them.
- Fix: `jupyter nbconvert --clear-output --inplace <notebook>` (or `nbstripout
  <notebook>`), then `git add` again.
- Exempt a notebook by adding a glob to `notebook-output-whitelist.txt`.
- One-off bypass: `git commit --no-verify`.

Stdlib-only Python 3 — the `gnt` conda env does not need to be active.

## `pre-commit` -> `check-confidential.py`

Blocks a commit when a staged file may carry confidential mine data (CLAUDE.md § "Confidential
mine data"):

- anything under a `confidential/` directory (also git-ignored);
- data files (`.csv`, `.parquet`, `.duckdb`, `.xls[x]`, `.gpkg`, ...);
- figures and documents whose path mentions mine, mining, snl or corridor.

Paths in `confidential-allowlist.txt` are exempt; an entry asserts the file holds only aggregate
estimates. The check reads paths, not content, so it backs up the rule rather than replacing it.
