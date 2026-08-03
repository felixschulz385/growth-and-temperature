# 10. Fetch ledger and unified HPC transfer

Ground-up replacement for `UnifiedDataIndex` (Parquet-backed FETCH-completion
index, `src/data/common/index/unified_index.py`) and `TransferManifest`
(Parquet-backed PREPARE/GRID transfer manifest,
`src/data/common/hpc/transfer.py`) with one DuckDB-backed ledger per source
and one unified push-to-HPC primitive. "Nothing has to remain" — this is not
an incremental patch of the old system, and the old Parquet/JSON files are
never read by any of the code below (§5).

## 1. Problems with the old system

- **Whole-file Parquet rewrites on every mutation** (`_add_files_to_index`
  once per crawled entrypoint, `update_file_statuses_batch` once per ~50-file
  download batch) — not incremental, the core "not fast" problem.
- **Two duplicate HPC-push implementations**: FETCH's inline tar/rsync/
  extract/verify (`async_downloader.py`) and the separate `pipeline
  transfer` path (`hpc/transfer.py`) — two manifest formats for one
  operation.
- **`is_complete()` is 100% local-disk** (`src/data/sources/steps.py`) — no
  concept of "complete locally, not yet verified on HPC." `_check_requires`
  (`src/cli/pipeline/handlers.py`) inherits the same blind spot.
- **No local cleanup after PREPARE/GRID transfer** — only FETCH's own push
  cleans up; `pipeline transfer` never deletes the local artifact it just
  pushed.
- **No SLURM job dependency chaining** — sequencing fetch→prepare→grid and
  cross-source `REQUIRES` edges is pure human responsibility.

## 2. Architecture

Replaces `common/index/unified_index.py`, `common/fetch/async_downloader.py`,
`common/hpc/transfer.py` with:

- **`common/ledger/`** — one DuckDB file per source (`SourceLedger`),
  tracking two orthogonal concerns: the FETCH-only remote-file crawl catalog
  (`remote_files`/`entrypoints` tables — "what does the origin have"), and
  the local/remote transfer state of every artifact any step produces
  (`artifacts` table — "what state is each unit in, here and on HPC").
- **`common/hpc/push.py`** (`HPCPusher`) — one push-to-HPC primitive used by
  both FETCH (many-small-files, tar-batched) and `pipeline transfer`
  (few-large-artifacts, one tar or direct rsync per unit), built entirely on
  the existing, unchanged `HPCClient`.
- **`common/fetch/driver.py`** (`run_fetch`) — the new FETCH driver.

`HPCClient` (`common/hpc/client.py`) and the `RemoteFileCatalog` protocol
(`src/data/sources/base.py`) are unchanged — every source's crawler/
downloader logic (Selenium for EOG, Box headers for ACAG, fixed file lists
for GADM/OSM/country_classifications, ...) needs no changes, only the
machinery driving it.

## 3. `artifacts` table: local/remote state as orthogonal axes

```
step          fetch | prepare | grid
unit_id       FETCH: file_hash        PREPARE/GRID: StepTarget.key
local_state   missing | writing | complete | failed
remote_state  missing | pushed | verified | failed
```

`PRIMARY KEY (step, unit_id)`. A row can be locally complete and not yet
pushed, pushed-not-verified, or fully verified — the direct fix for "no
complete-but-not-transferred concept." `StepTarget.require_remote: bool`
(default `False`, preserving today's local-only behavior everywhere except
MODIS's PREPARE — the one source whose PREPARE output must cross machines)
makes `is_complete()` additionally check `remote_state == verified` only
where it matters.

## 4. Human/operational boundaries, stated not hidden

FETCH still cannot run as a SLURM job (Selenium/browser/internet-egress
constraint, unchanged) — confirmed by `orchestration/slurm/jobs.yaml`'s own
header comment. SLURM dependency chaining (`orchestration/slurm/
submit_chain.py`, `--dependency=afterok` derived from `REQUIRES` + step
order) therefore starts at PREPARE, not FETCH: an operator runs FETCH
manually/on an egress-capable host, confirms via `pipeline summary`, then
runs `submit_chain.py` to start the PREPARE→GRID chain on SLURM.

## 5. No migration of old data

The new ledger is bootstrapped by scanning real on-disk/HPC filesystem state
(`pipeline reconcile`, `common/ledger/bootstrap.py`) as ground truth, not by
converting old `parquet_*.parquet`/`transfer_*.parquet`/`entrypoints_*.json`
files — those are left for an operator to delete once bootstrap has run.

## 6. Rollout order

1. Ledger foundation (`common/ledger/{schema,store,catalog}.py`) — testable
   in isolation against a temp `.duckdb` + a fake `RemoteFileCatalog`.
2. Bootstrap (`bootstrap.py` + `pipeline reconcile`).
3. Unified push primitive (`hpc/push.py`) — standalone against `HPCClient`.
4. Rewire FETCH (`fetch/driver.py`; swap all `_execute_fetch`/`_plan_prepare`
   call sites); delete `common/index/unified_index.py` +
   `common/fetch/async_downloader.py`.
5. Rewire PREPARE/GRID transfer (`handle_transfer` uses `HPCPusher`; MODIS's
   PREPARE targets get `require_remote=True`); delete `common/hpc/
   transfer.py`.
6. Location-aware completion (`StepTarget.require_remote`, ledger-aware
   `is_complete()`/`_check_requires`).
7. SLURM dependency chaining (`submit_chain.py` + `jobs.yaml` additions).
