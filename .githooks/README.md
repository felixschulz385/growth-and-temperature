# Repo-managed git hooks

## Enable (once per clone)

```bash
./.githooks/install.sh
# equivalently: git config core.hooksPath .githooks
```

`core.hooksPath` is local config and is **not** cloned, so every fresh checkout
must run this. It is also listed in the root README Quick Start.

## `pre-commit` -> `check-notebook-outputs.py`

Blocks a commit when a **staged** `.ipynb` that may show mine-level data carries cell
**outputs** or a non-null **`execution_count`**. Other notebooks keep their outputs.

- `notebook-output-whitelist.txt` allows every notebook (`*.ipynb`), then re-blocks, with
  `!` lines, the notebooks that read mine-level data. The last matching line wins.
- Independently of the list, a notebook whose code reads a mine-level source (`snl_mining`,
  `berman_mining`, `BCRT_`, `property_id`) is blocked. A new mine notebook is therefore
  caught without a list entry. Add a `!` line for one that reaches mine data indirectly,
  e.g. through `src.analysis.corridors`.

- Only staged content is inspected (`git show :<path>`), not the working tree.
- Notebooks already carrying outputs in history are flagged only when you stage
  a new change to them.
- Fix: `jupyter nbconvert --clear-output --inplace <notebook>` (or `nbstripout
  <notebook>`), then `git add` again.
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
