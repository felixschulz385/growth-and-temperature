"""The one place the existing stage_1/stage_2/stage_2_ease6933 path numbering lives.

docs/design/09-integrated-pipeline.md §3: this module reproduces today's
physical paths byte-for-byte -- old code (src/data/preprocess/sources/*) and
new code (src/data/sources/*) must write to the same place during the
migration, so behaviour-preservation is checkable by diffing artefacts rather
than trusted from reading a diff. Do not change the strings this module
produces without updating docs/design/09-integrated-pipeline.md §3's
"do not rename physical directories" decision and its consumer checklist.
"""

from __future__ import annotations

import os

from src.data.sources.steps import PipelineStep

#: docs/design/05-migration.md §1's recommended config switch, finally given a
#: real implementation via `grid_id` instead of being MODIS-only ad hoc.
LEGACY_GRID_ID = "legacy_4326"
EASE_GRID_ID = "ease6933"

#: docs/design/09-integrated-pipeline.md §3/§14's deferred "layout: v2" task
#: (docs/design/02-storage.md §2's "one Zarr store per variable family"
#: decision), finally given a real implementation for the single-source
#: families only -- see grid_store_path(). Additive: LEGACY_LAYOUT is the
#: default everywhere, so today's physical paths are unaffected.
LEGACY_LAYOUT = "legacy"
V2_LAYOUT = "v2"

_STAGE_DIR = {
    PipelineStep.PREPARE: "stage_1",
    # GRID's stage_2 directory name depends on grid_id -- see output_root().
}


def raw_root(data_root: str, data_path: str, *, namespace: str | None = None) -> str:
    """Where FETCH lands raw bytes: <data_root>/<data_path>/raw[/<namespace>].

    Matches e.g. ACAGPreprocessor._resolve_raw_path
    (src/data/preprocess/sources/acag.py:163-167): `<hpc_root>/<data_path>/raw/<relative_path>`.
    """
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
) -> str:
    """The output directory for one (data_path, step), matching today's
    per-source `get_hpc_output_path(stage)` conventions exactly.

    - FETCH   -> raw_root() (no "processed/" prefix -- raw files live directly
                 under the source's data_path, as every current source's
                 download side already does)
    - PREPARE -> <data_root>/<data_path>/processed/stage_1[/<namespace>]
    - GRID    -> <data_root>/<data_path>/processed/stage_2[/<namespace>],
                 or .../stage_2_ease6933 when grid_id == "ease6933"
                 (docs/design/05-migration.md §1's additive dual-grid path)
    """
    if step is PipelineStep.FETCH:
        return raw_root(data_root, data_path, namespace=namespace)

    if step is PipelineStep.PREPARE:
        base = os.path.join(data_root, data_path, "processed", "stage_1")
    elif step is PipelineStep.GRID:
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

    `layout=v2` with `v2_family` given: `<data_root>/grid_v2/<v2_family>.zarr`,
    per docs/design/02-storage.md §2's "one store per variable family"
    decision -- a shared flat directory of per-family stores, replacing the
    per-source `processed/stage_2[/<namespace>]` convention. Only meaningful
    for the "single contributing source" families (docs/design/09-integrated-
    pipeline.md §14's deferred layout:v2 task, scoped narrowly here); pass
    `v2_family=None` (the default) for sources not yet part of that scope --
    they fall back to the legacy path even when `layout=v2` is selected.
    """
    if layout == V2_LAYOUT and v2_family is not None:
        return os.path.join(data_root, "grid_v2", f"{v2_family}.zarr")
    root = output_root(data_root, data_path, PipelineStep.GRID, namespace=namespace, grid_id=grid_id)
    return os.path.join(root, legacy_filename)


def index_path(local_index_dir: str, data_path: str) -> str:
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
    """
    safe = data_path.replace("/", "_").replace("\\", "_")
    return os.path.join(local_index_dir, f"parquet_{safe}.parquet")
