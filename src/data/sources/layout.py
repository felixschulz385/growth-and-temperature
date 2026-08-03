"""The one place every pipeline path lives: the legacy per-source
stage_1/stage_2/stage_2_ease6933 numbering, and the opt-in `layout: v2`
physical rename (raw/<data_path>, prepared/<data_path>, grid/<grid_id>/<family>.zarr).

docs/design/09-integrated-pipeline.md §3: this module originally existed to
reproduce the legacy physical paths byte-for-byte during the preprocess ->
pipeline migration, so behaviour-preservation was checkable by diffing
artefacts rather than trusted from reading a diff -- `layout=legacy`
(the default) still guarantees that. §14's "layout: v2" physical rename,
deferred at the time, is now implemented here too: every registered source
opts into it via `layout="v2"`, additive and off by default.
"""

from __future__ import annotations

import os

from src.data.sources.steps import PipelineStep

#: docs/design/05-migration.md §1's recommended config switch, finally given a
#: real implementation via `grid_id` instead of being MODIS-only ad hoc.
LEGACY_GRID_ID = "legacy_4326"
EASE_GRID_ID = "ease6933"

#: docs/design/09-integrated-pipeline.md §3/§14's deferred "layout: v2" task:
#: a physical rename to raw/<data_path>, prepared/<data_path>,
#: grid/<grid_id>/<family>.zarr (docs/design/02-storage.md §2's "one Zarr
#: store per variable family" decision, for the GRID step specifically).
#: Every registered source now opts into it. Additive: LEGACY_LAYOUT is the
#: default everywhere, so today's physical paths are unaffected unless a
#: caller explicitly selects `layout="v2"`.
LEGACY_LAYOUT = "legacy"
V2_LAYOUT = "v2"


def raw_root(
    data_root: str, data_path: str, *, namespace: str | None = None, layout: str = LEGACY_LAYOUT
) -> str:
    """Where FETCH lands raw bytes.

    `layout=legacy` (default): `<data_root>/<data_path>/raw[/<namespace>]`,
    matching e.g. ACAGPreprocessor._resolve_raw_path
    (src/data/preprocess/sources/acag.py:163-167): `<hpc_root>/<data_path>/raw/<relative_path>`.

    `layout=v2`: `<data_root>/raw/<data_path>[/<namespace>]` -- a top-level
    `raw/` tree instead of nesting under each source, matching the
    stage-name-first convention `prepared/` and `grid/<grid_id>/` already
    use under v2 (docs/design/09-integrated-pipeline.md §3's "raw/,
    prepared/, grid/<grid_id>/" sketch).
    """
    if layout == V2_LAYOUT:
        base = os.path.join(data_root, "raw", data_path)
    else:
        base = os.path.join(data_root, data_path, "raw")
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
    layout: str = LEGACY_LAYOUT,
) -> str:
    """The output directory for one (data_path, step).

    `layout=legacy` (default), matching today's per-source
    `get_hpc_output_path(stage)` conventions exactly:
    - FETCH   -> raw_root() (no "processed/" prefix -- raw files live directly
                 under the source's data_path, as every current source's
                 download side already does)
    - PREPARE -> <data_root>/<data_path>/processed/stage_1[/<namespace>]
    - GRID    -> <data_root>/<data_path>/processed/stage_2[/<namespace>],
                 or .../stage_2_ease6933 when grid_id == "ease6933"
                 (docs/design/05-migration.md §1's additive dual-grid path)

    `layout=v2`, a physical rename to a stage-name-first top-level tree
    (docs/design/09-integrated-pipeline.md §3):
    - FETCH   -> raw_root(layout="v2") -> <data_root>/raw/<data_path>[/<namespace>]
    - PREPARE -> <data_root>/prepared/<data_path>[/<namespace>]
    - GRID    -> <data_root>/grid/<grid_id> -- flat, `namespace` not applied;
                 the caller (grid_store_path()) appends the family filename.
    """
    if step is PipelineStep.FETCH:
        return raw_root(data_root, data_path, namespace=namespace, layout=layout)

    if step is PipelineStep.PREPARE:
        if layout == V2_LAYOUT:
            base = os.path.join(data_root, "prepared", data_path)
        else:
            base = os.path.join(data_root, data_path, "processed", "stage_1")
    elif step is PipelineStep.GRID:
        if layout == V2_LAYOUT:
            return os.path.join(data_root, "grid", grid_id)
        stage2_dir = "stage_2_ease6933" if grid_id == EASE_GRID_ID else "stage_2"
        base = os.path.join(data_root, data_path, "processed", stage2_dir)
    else:  # pragma: no cover -- PipelineStep is a closed enum
        raise ValueError(f"Unknown step: {step}")

    if namespace:
        base = os.path.join(base, namespace)
    return base


def grid_store_path(
    data_root: str,
    data_path: str,
    legacy_filename: str,
    *,
    namespace: str | None = None,
    grid_id: str = LEGACY_GRID_ID,
    layout: str = LEGACY_LAYOUT,
    v2_family: str | None = None,
) -> str:
    """The full GRID-stage store path for one source's output.

    `layout=legacy` (default): `<output_root(GRID)>/<legacy_filename>` --
    byte-identical to what every source already builds by hand today.

    `layout=v2` with `v2_family` given: `<data_root>/grid/<grid_id>/<v2_family>.zarr`,
    per docs/design/02-storage.md §2's "one store per variable family"
    decision -- a shared directory of per-family stores, replacing the
    per-source `processed/stage_2[/<namespace>]` convention. `grid_id` is
    folded into the path so the directory is self-documenting about which
    CRS it holds, the same way the legacy layout's `stage_2` vs
    `stage_2_ease6933` already is -- even though in practice `layout=v2` is
    only ever run against one grid at a time, so this is about readability,
    not supporting two live grids under v2 simultaneously. Every registered
    GRID-capable source sets `v2_family` today; `v2_family=None` (the
    default parameter value, kept for callers that haven't opted a source
    in) falls back fully to the legacy path -- computed with `layout`
    forced to legacy regardless of what the caller passed, not a v2 path
    with the legacy filename.
    """
    if layout == V2_LAYOUT and v2_family is not None:
        root = output_root(data_root, data_path, PipelineStep.GRID, grid_id=grid_id, layout=V2_LAYOUT)
        return os.path.join(root, f"{v2_family}.zarr")
    root = output_root(data_root, data_path, PipelineStep.GRID, namespace=namespace, grid_id=grid_id, layout=LEGACY_LAYOUT)
    return os.path.join(root, legacy_filename)


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
    shared index via `pipeline index --adopt-local` rather than a rename.

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
