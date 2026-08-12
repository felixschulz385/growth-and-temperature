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
  extract/verify (`async_downloader.py`) and the separate `data
  transfer` path (`hpc/transfer.py`) — two manifest formats for one
  operation.
- **`is_complete()` is 100% local-disk** (`src/data/sources/steps.py`) — no
  concept of "complete locally, not yet verified on HPC." `_check_requires`
  (`src/cli/data/handlers.py`) inherits the same blind spot.
- **No local cleanup after PREPARE/GRID transfer** — only FETCH's own push
  cleans up; `data transfer` never deletes the local artifact it just
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
  both FETCH (many-small-files, tar-batched) and `data transfer`
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
manually/on an egress-capable host, confirms via `data summary`, then
runs `submit_chain.py` to start the PREPARE→GRID chain on SLURM.

## 5. No migration of old data

The new ledger is bootstrapped by scanning real on-disk/HPC filesystem state
(`data reconcile`, `common/ledger/bootstrap.py`) as ground truth, not by
converting old `parquet_*.parquet`/`transfer_*.parquet`/`entrypoints_*.json`
files — those are left for an operator to delete once bootstrap has run.

## 6. Rollout order

1. Ledger foundation (`common/ledger/{schema,store,catalog}.py`) — testable
   in isolation against a temp `.duckdb` + a fake `RemoteFileCatalog`.
2. Bootstrap (`bootstrap.py` + `data reconcile`).
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

## 7. Addendum: MODIS's PREPARE renamed to FETCH

§3/§6 above describe MODIS's STAC-streaming step as "PREPARE" -- accurate at
the time, but a mislabel: STAC search + `odc.stac.load` + QC-mask + annual
compositing genuinely *is* the download from Planetary Computer, it just also
transforms as it goes (there's no separate raw-asset-on-disk stage to insert
a real FETCH before). `ModisSource.STEPS` is now `(FETCH, GRID)`, not
`(PREPARE, GRID)` -- `StepTarget.require_remote=True` now applies to MODIS's
FETCH targets, same mechanism, renamed step.

This does **not** make MODIS satisfy `RemoteFileCatalog` like every other
FETCH source (GLASS/EOG/...): MODIS has no flat crawlable remote file list,
only per-(year, tile) STAC queries, so `list_remote_files`/`download_async`/
`get_all_entrypoints` are not implemented, and MODIS is excluded from
`tests/data/sources/test_fetch_protocol.py`'s parametrization. Instead
`ModisSource._execute_fetch()` tracks each (year, tile) unit's local state
directly in the ledger's generic `artifacts` table
(`ensure_artifact`/`set_local_state`) -- the same table PREPARE/GRID units
already use, just written to directly instead of seeded from a crawl catalog.
This gives partial/resumable FETCH runs: a tile-year that fails (e.g. a
transient STAC error) is left `failed` rather than silently retried forever,
and the next `data run --source modis --step fetch` call picks up
whatever isn't yet complete on disk, same as before -- now with per-unit
state visible via the ledger rather than only inferrable from the filesystem.

`handle_index`/`data reconcile`'s FETCH branch both now check
`isinstance(source, RemoteFileCatalog)` before assuming a crawl catalog
exists, rather than switching purely on the step name -- MODIS's FETCH
reconciles via the same `reconcile_step()` PREPARE/GRID use (enumerate
`plan()`, check `is_complete()`/local existence), not
`common.ledger.bootstrap.reconcile_fetch()`.

The physical output path is unchanged (`processed/stage_1` legacy /
`prepared/<data_path>` v2) -- `ModisSource.output_root()` special-cases
FETCH to preserve it, rather than adopting `layout.raw_root()`'s bare
`<data_path>/raw` convention every crawler-based FETCH source uses, which
would have silently orphaned already-fetched local/HPC GeoTIFFs on this
rename.

Orchestration: `orchestration/slurm/jobs.yaml`'s `modis-prepare` job (host:
egress, `step: prepare`) is renamed `modis-fetch` (`step: fetch`,
`transfer_after: fetch`); `orchestration/scripts/modis-prepare.sh` is
regenerated as `modis-fetch.sh`. `submit_chain.py`'s FETCH-skip logic is
unaffected (MODIS's SLURM chain still starts at GRID, unchanged).
