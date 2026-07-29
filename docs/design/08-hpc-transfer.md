# 08 — Generic HPC transfer capability

## 1. Why this exists, and why it's generic rather than MODIS-only

[`07-modis-ingest.md`](07-modis-ingest.md) §2 resolves the "do scicore compute nodes have outbound
internet access" question by making it not matter: stage "annual" (the streaming ingest, which needs
internet egress to Planetary Computer) runs on whatever host has that access, and its output is pushed
to scicore over SSH before stage "spatial" (compute-heavy, network-free reprojection) runs there as an
ordinary SLURM job. This document specifies that push mechanism.

**Decision, confirmed with the user: build this as a generic, source-agnostic capability, not a
MODIS-specific one.** Any preprocessor whose stage output is produced somewhere other than scicore
gets the same escape hatch, via one reusable interface, rather than duplicating ad hoc transfer code
per source.

**Not starting from scratch.** `src/data/common/hpc/client.py` (`HPCClient`) already implements
hardened, non-interactive SSH (`BatchMode=yes`, `StrictHostKeyChecking=no`,
`PreferredAuthentications=publickey`, etc. — `client.py:80-89`), rsync transfer with an scp/PowerShell
fallback (`rsync_transfer`, `client.py:543-750`), remote command execution
(`execute_command`, `client.py:423-443`), remote directory/tar handling (`ensure_directory`,
`extract_tar`, `client.py:302-343`, `920-965`), and file-existence/info checks
(`check_file_exists`, `get_file_info`, `client.py:345-421`). `src/data/download/async_downloader.py`
already has a working tar → rsync → remote-extract → verify loop for raw downloaded files
(`_create_tar_archive`, `_transfer_and_extract`, `_verify_extracted_files`). None of this currently
touches processed/zarr output: every preprocessor's `hpc_target`/`hpc_root` resolves to a **local**
filesystem path by the time it's used (`src/config/runtime.py:get_legacy_hpc_compat_config`,
`runtime.py:80-97` — `"target"` is set to the local `paths.data_root`, not the SSH target), and no
preprocessor source file imports `HPCClient`. This document closes that wiring gap by reusing
`HPCClient` for a new purpose, not by building a second SSH client.

**Target, confirmed with the user: reuse the existing scicore convention.** `remote.ssh_target`/
`remote.key_file` in `orchestration/configs/data.yaml`/`data.local.yaml`
(`schulz0022@transfer12.scicore.unibas.ch:/scicore/.../data_nobackup`,
`~/.ssh/id_ed25519_scicore`) — the same dedicated transfer node already used for raw-file and index
sync. No new credentials or config surface for authentication; the only new config is *which* stage(s)
a source wants transferred (§4).

**Transfer point, confirmed with the user: after stage "annual" only.** Stage "spatial" keeps running
as an ordinary local SLURM job on scicore, reading the now-locally-present annual composites —
unchanged in kind from how GLASS/ACAG's stage "spatial" works today. This keeps the blast radius of
the new capability small: only one stage's output ever needs to leave the ingest host.

## 2. Interface

One new **optional** hook on the preprocessor contract
(`src/data/preprocess/sources/base.py:AbstractPreprocessor`) — optional because sources that never run
off-cluster (the overwhelming majority today) don't implement it and are unaffected:

```python
def get_transfer_units(self, stage: str) -> List[Dict[str, Any]]:
    """Local paths produced by `stage` that should be pushed to the HPC target.
    Each dict: {local_path, remote_path, unit_id}."""
```

A default implementation on `AbstractPreprocessor` derives this from the existing
`get_hpc_output_path(stage)` for the common case — one local root maps to the same relative path under
the remote target's base path. MODIS overrides it because stage "annual"'s per-tile-year layout
(`.../processed/stage_1/<year>/<tile>.zarr`, matching `GlassPreprocessor`'s existing per-grid-cell
path shape at `glass.py:303`) benefits from per-unit transfer entries — finer-grained resumability
than syncing one large directory tree, and it matches the per-tile-year processing granularity
[`07-modis-ingest.md`](07-modis-ingest.md) already uses for stage "annual".

## 3. New module: `src/data/common/hpc/transfer.py`

A thin orchestration layer over `HPCClient` — no new SSH/rsync logic, only sequencing and
manifest-tracking around the existing client. For each transfer unit:

1. **Tar the local zarr store's chunk tree**, mirroring `async_downloader.py`'s
   `_create_tar_archive`/`_transfer_and_extract` pattern. This is a deliberate choice, not an
   arbitrary one: a Zarr store is a directory of thousands of small chunk files, and plain
   `rsync -a` on that tree pays per-file SSH/rsync protocol overhead for every chunk — tar-then-
   transfer-then-untar amortizes that overhead into one file transfer, exactly the problem the
   existing raw-download pipeline already solved for tar-of-raw-files. Do not reinvent a second
   solution to the same problem.
2. **rsync the tarball** via `HPCClient.rsync_transfer` (already handles SSH key/BatchMode options and
   the scp/PowerShell fallback — reuse as-is).
3. **Extract remotely** via `HPCClient.extract_tar` (already does `mkdir -p` + `tar -xzf` over SSH).
4. **Verify**: sample a handful of the transferred zarr's chunk/metadata files via
   `HPCClient.check_file_exists`, mirroring `async_downloader.py`'s `_verify_extracted_files` sampling
   approach — full checksum verification of every chunk is unnecessary given rsync's own transfer
   integrity guarantees; sampling catches a failed/partial extraction, which is the actual failure mode
   worth guarding against here.

**Idempotency/resumability: reuse `UnifiedDataIndex` (`src/data/common/index/unified_index.py`) as the
transfer manifest, rather than inventing a second index format.** It already provides the
parquet/sqlite-backed, HPC-syncable status tracking the download subsystem uses for completed-file
state; a transfer manifest is the same shape of problem (per-unit status: pending/in-progress/
completed/failed, with retry) applied to transfer units instead of downloaded files. Whether this
needs a schema addition (e.g. a `unit_type` column distinguishing "download" from "transfer" rows, or
reusing the existing schema with a different `data_path` namespace) is an implementation-time call —
recommend starting with a namespace convention (e.g. `data_path = "<source>/transfer/<stage>"`) before
adding new columns, since the existing schema's `status_category`/completion semantics already fit.

**Config**: read `remote.ssh_target`/`key_file` via the existing `get_remote_config()`
(`src/config/runtime.py:41-57`) — no new credential keys. The only new config surface is which
stage(s) a source wants transferred, e.g.:

```yaml
sources:
  modis:
    transfer:
      stages: [annual]
```

**Flag for implementation, don't fix silently as a side effect of this design doc:**
`HPCClient.execute_command` and `HPCClient.rsync_transfer` are each **defined twice** in the same class
body (`client.py:94` and `423`; `client.py:129` and `543` — the second definition of each silently
shadows the first in Python, so only the second is ever actually reachable). Since this is the first
time new code calls into `HPCClient` for a purpose beyond what `async_downloader.py`/
`unified_index.py` already exercise, delete the dead first definitions as part of implementing this
module — small, low-risk (the second definitions are already what's currently invoked, confirmed via
`async_downloader.py`'s call sites), and leaves the class in a state where a future reader isn't misled
by dead code sitting next to what actually runs.

## 4. CLI verb

Mirror the existing `preprocess run` registration exactly
(`src/cli/preprocess/commands.py`, `src/cli/preprocess/handlers.py`) — same `--config`/`--source`/
`--stage` flags `run` already defines, reused rather than redeclared:

```
run.py preprocess transfer --config data.yaml --source modis --stage annual [--direction push|pull]
```

New `add_parser("transfer", ...)` in `commands.py`; new `handle_transfer` in `handlers.py` that loads
the preprocessor via the existing `factory.py` dispatch (`get_preprocessor_class`/
`create_preprocessor`), calls `get_transfer_units(stage)`, and drives
`src/data/common/hpc/transfer.py`. `--direction` defaults to `push` (local → HPC, the only direction
this design currently needs); `pull` is included in the interface for symmetry with
`HPCIndexSynchronizer`'s existing push/pull shape, not because any current source needs it.

## 5. Execution model

- **Not a SLURM script by default.** Transfer runs from wherever stage "annual" ran, which — per
  [`07-modis-ingest.md`](07-modis-ingest.md) §7 — is `orchestration/scripts/modis-ingest-annual.sh`
  (deliberately outside `orchestration/slurm/`, since it isn't a SLURM compute-node job). If the ingest
  host is itself scicore's login/transfer node, no SLURM script is appropriate there either — SLURM is
  for compute-node jobs, and the transfer step here is neither compute-heavy nor something a scheduler
  needs to arbitrate.
- **Stage "spatial" SLURM scripts are unaffected.** `modis-preprocess-spatial.sh`
  ([`07-modis-ingest.md`](07-modis-ingest.md) §7) assumes local data presence exactly like every
  existing two-stage source's spatial stage; it gains only a soft dependency — "the transfer manifest
  shows this tile-year's transfer as `completed`" — which is an operational precondition check, not a
  code dependency on this new module.

## 6. Open items

Logged in [`06-open-questions.md`](06-open-questions.md):
- Real transfer throughput/time for ~6,700 tile-year annual composites (the backbone's full-panel
  scale, [`00-backbone-overview.md`](00-backbone-overview.md)) over the scicore transfer node — not
  measurable from the repo; needs a small real test with representative zarr chunk counts/sizes.
- Whether `UnifiedDataIndex`'s existing schema needs an addition for transfer-manifest use, or whether
  a namespace convention on existing columns suffices (§3) — a design call to make while implementing,
  not before.
- Whether scicore compute-node outbound internet access is ever actually confirmed — now non-blocking
  for this design (§1), but still worth eventually resolving to decide whether the transfer step is
  needed at all for future sources, versus ingest running directly on compute nodes.
