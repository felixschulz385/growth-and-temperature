"""The one place every pipeline path lives: a stage-name-first physical tree,
`raw/<data_path>`, `prepared/<data_path>/<agg>/...`.

docs/design/09-integrated-pipeline.md §3/§14: this was originally the "layout:
v2" opt-in rename, deferred behind a `legacy` default that reproduced the old
preprocess-era per-source `processed/stage_1`/`stage_2[_ease6933]` paths
byte-for-byte. No source or orchestration config ever set `layout=v2` in
production, so that duality was pure dead weight; this module now builds only
the one (former "v2") shape. `scripts/migrate_legacy_layout.py` physically
renames already-computed on-disk artefacts from the old layout to this one.

PREPARE's tree is further split by "aggregation level" (`agg`, one of
CRS_AGG/ADM_AGG/MISC_AGG below): an earlier sketch put every pixel-grid
zarr store under a separate top-level `grid/<grid_id>/` tree, disjoint from
`prepared/<data_path>/`, which meant a source's PREPARE-stage intermediates
(vector files, per-GID tables) and its GRID-stage pixel store lived under two
unrelated roots even though they're the same source's one PREPARE step
output. Folding GRID under `prepared/<data_path>/crs/<grid_id>/` fixes that
-- `crs` is still its own bucket (grid_id-namespaced, dual-grid-capable)
rather than merged flat into `<data_path>/`, so a source with both a
pixel-grid store and a non-pixel-grid table (e.g. gadm's `country_id.zarr` +
`GID_0_code_mapping.json`) doesn't have the two collide or get arbitrarily
ordered in one directory.
"""

from __future__ import annotations

import os

from src.data.sources.steps import PipelineStep

#: docs/design/05-migration.md §1's recommended config switch, finally given a
#: real implementation via `grid_id`. Identifies grid *CRS*, not directory
#: layout -- a source can target either grid_id under this one physical
#: layout.
LEGACY_GRID_ID = "legacy_4326"
EASE_GRID_ID = "ease6933"

#: PREPARE's three physical-layout sub-buckets under
#: `<data_root>/prepared/<data_path>/<agg>/...` -- unrelated to any source's
#: own `data_path` string (e.g. gadm/osm/country_classifications each read
#: `data_path="misc"` historically -- that "misc" is a *data_path*, distinct
#: from MISC_AGG below, a *physical layout bucket* every data_path can have
#: one of).
#:
#: - CRS_AGG: pixel-grid zarr stores -- what GRID always wrote, now filed
#:   under PREPARE's own tree instead of a disjoint top-level `grid/` one
#:   (module docstring). Still grid_id-namespaced (`crs/<grid_id>/<family>.zarr`)
#:   for the same reason the legacy layout's `stage_2` vs `stage_2_ease6933`
#:   split existed: dual-grid support (legacy_4326 vs ease6933), even though
#:   in practice this only ever runs against one grid at a time.
#: - ADM_AGG: admin-unit-keyed tables (GID_N-keyed parquet, e.g. gadm's own
#:   `mine_count_adm1.parquet`-style sidecars) AND admin-shaped vector/JSON
#:   intermediates that feed them (simplified GADM `.gpkg` boundary files,
#:   `GID_N_code_mapping.json` sidecars) -- kept together since they're read
#:   as one logical "admin data" unit by cross-source consumers
#:   (`gadm.gid_mapping_path()`).
#: - MISC_AGG: everything else non-spatial (e.g. commodity-price lookup
#:   tables) -- the physical-layout catch-all, not to be confused with the
#:   `data_path="misc"` string above.
CRS_AGG = "crs"
ADM_AGG = "adm"
MISC_AGG = "misc"


def raw_root(data_root: str, data_path: str, *, namespace: str | None = None) -> str:
    """Where FETCH lands raw bytes: `<data_root>/raw/<data_path>[/<namespace>]`."""
    base = os.path.join(data_root, "raw", data_path)
    if namespace:
        base = os.path.join(base, namespace)
    return base


def output_root(
    data_root: str,
    data_path: str,
    step: PipelineStep,
    *,
    namespace: str | None = None,
    grid_id: str = LEGACY_GRID_ID,
    agg: str | None = None,
) -> str:
    """The output directory for one (data_path, step):
    - FETCH   -> raw_root() -> <data_root>/raw/<data_path>[/<namespace>]
    - PREPARE -> <data_root>/prepared/<data_path>/<agg>[/<namespace>] -- `agg`
                 is required here (no default): every PREPARE call site must
                 say which physical bucket its output belongs in, rather than
                 silently defaulting to one.
    - GRID    -> <data_root>/prepared/<data_path>/crs/<grid_id> -- flat,
                 `namespace` not applied; the caller (grid_store_path())
                 appends the family filename. Always the `crs` bucket (a
                 pixel-grid store, by definition), so GRID doesn't take an
                 `agg` argument the way PREPARE does -- there is only one
                 physically meaningful choice.
    """
    if step is PipelineStep.FETCH:
        return raw_root(data_root, data_path, namespace=namespace)

    if step is PipelineStep.PREPARE:
        if agg is None:
            raise ValueError(
                "PREPARE output_root() calls must pass agg= (CRS_AGG/ADM_AGG/"
                "MISC_AGG) -- see this module's docstring for the crs/adm/misc "
                "physical-layout split."
            )
        base = os.path.join(data_root, "prepared", data_path, agg)
    elif step is PipelineStep.GRID:
        return os.path.join(data_root, "prepared", data_path, CRS_AGG, grid_id)
    else:  # pragma: no cover -- PipelineStep is a closed enum
        raise ValueError(f"Unknown step: {step}")

    if namespace:
        base = os.path.join(base, namespace)
    return base


def grid_store_path(
    data_root: str,
    data_path: str,
    *,
    grid_id: str = LEGACY_GRID_ID,
    family: str,
) -> str:
    """The full GRID-stage store path for one source's output:
    `<data_root>/prepared/<data_path>/crs/<grid_id>/<family>.zarr`, per
    docs/design/02-storage.md §2's "one store per variable family" decision
    -- a shared directory of per-family stores. `grid_id` is folded into the
    path so the directory is self-documenting about which CRS it holds.

    Purely for pixel-grid `<family>.zarr` stores -- non-pixel-grid PREPARE
    outputs (GID-keyed parquet tables, GADM's boundary files, etc.) should
    call `output_root(..., PipelineStep.PREPARE, agg=ADM_AGG/MISC_AGG)`
    directly plus their own filename instead of going through this function.
    """
    root = output_root(data_root, data_path, PipelineStep.GRID, grid_id=grid_id)
    return os.path.join(root, f"{family}.zarr")


def index_path(local_index_dir: str | None, data_path: str) -> str | None:
    """The completion-index parquet path for a given `data_path`:
    <local_index_dir>/parquet_<safe(data_path)>.parquet.

    Mirrors `UnifiedDataIndex`'s own filename derivation exactly
    (`safe_data_path = self.data_path.replace("/", "_")`,
    `parquet_{safe_data_path}.parquet`) -- this module does not (and, without
    touching the untouched `UnifiedDataIndex`, cannot) change how that
    filename is derived; it exists so callers outside a `DataSource` instance
    (tests, CLI diagnostics) can predict the index path from config alone.

    For most sources this keeps today's index file byte-identical (`acag`'s
    `data_path="acag/pm25"` still resolves to `parquet_acag_pm25.parquet`).
    The misc split (docs/design/09-integrated-pipeline.md §7) is the one place
    this changes on purpose: `osm`/`gadm`/`country_classifications` each get a
    distinct `data_path` (unlike today's single `data_path="misc"` shared by
    all four origins), so each gets its own index file -- adopted from the old
    shared index via `data index --adopt-local` rather than a rename.

    Directory: also unifies on the download side's convention
    (`paths.local_index_dir`, config-driven) rather than the preprocess side's
    hardcoded `<hpc_root>/hpc_data_index` -- the two only coincided today
    because local config happens to point `local_index_dir` at
    `~/hpc_data_index`.

    Returns `None` when `local_index_dir` isn't configured (`paths.local_index_dir`
    left unset in `data.yaml`) rather than raising -- every `_plan_prepare()`
    caller already treats "index file not found" as a normal, warn-and-return-[]
    outcome (the completion index just hasn't been built yet); "index directory
    not configured" is the same outcome from the caller's perspective; it should
    not be a different, unhandled `TypeError` from `os.path.join(None, ...)`.
    """
    if not local_index_dir:
        return None
    safe = data_path.replace("/", "_").replace("\\", "_")
    return os.path.join(local_index_dir, f"parquet_{safe}.parquet")
