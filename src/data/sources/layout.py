"""The one place every pipeline path lives: a stage-name-first physical tree,
`raw/<data_path>`, `prepared/<data_path>`, `grid/<grid_id>/<family>.zarr`.

docs/design/09-integrated-pipeline.md §3/§14: this was originally the "layout:
v2" opt-in rename, deferred behind a `legacy` default that reproduced the old
preprocess-era per-source `processed/stage_1`/`stage_2[_ease6933]` paths
byte-for-byte. No source or orchestration config ever set `layout=v2` in
production, so that duality was pure dead weight; this module now builds only
the one (former "v2") shape. `scripts/migrate_legacy_layout.py` physically
renames already-computed on-disk artefacts from the old layout to this one.
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
) -> str:
    """The output directory for one (data_path, step):
    - FETCH   -> raw_root() -> <data_root>/raw/<data_path>[/<namespace>]
    - PREPARE -> <data_root>/prepared/<data_path>[/<namespace>]
    - GRID    -> <data_root>/grid/<grid_id> -- flat, `namespace` not applied;
                 the caller (grid_store_path()) appends the family filename.
    """
    if step is PipelineStep.FETCH:
        return raw_root(data_root, data_path, namespace=namespace)

    if step is PipelineStep.PREPARE:
        base = os.path.join(data_root, "prepared", data_path)
    elif step is PipelineStep.GRID:
        return os.path.join(data_root, "grid", grid_id)
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
    `<data_root>/grid/<grid_id>/<family>.zarr`, per docs/design/02-storage.md
    §2's "one store per variable family" decision -- a shared directory of
    per-family stores. `grid_id` is folded into the path so the directory is
    self-documenting about which CRS it holds."""
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
