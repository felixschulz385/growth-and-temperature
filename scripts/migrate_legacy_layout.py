"""Physically move already-computed pipeline output from either of two older
directory layouts into the current stage-name-first tree (`raw/<data_path>`,
`prepared/<data_path>/<agg>/...`, pixel-grid stores under
`prepared/<data_path>/crs/<grid_id>/<family>.zarr` -- see
`src/data/sources/layout.py`).

There are two prior layouts this script knows how to migrate *from*, because
both were live in production at different points and each may have left real
data on disk:

- "legacy" (the original preprocess-era layout):
    <data_root>/<data_path>/raw[/<namespace>]
    <data_root>/<data_path>/processed/stage_1[/<namespace>]
    <data_root>/<data_path>/processed/stage_2[/<namespace>]/<legacy_filename>
    <data_root>/<data_path>/processed/stage_2_ease6933[/<namespace>]/<legacy_filename>

- "interim" (commit d43db13's short-lived flat "v2" layout -- legacy support
  had just been removed from `layout.py`, but the crs/adm/misc bucket split
  hadn't landed yet; a real pipeline run on HPC executed against this code
  before the bucket-split merge, so its output needs migrating too):
    <data_root>/raw/<data_path>[/<namespace>]                    (same as FETCH today -- no move needed)
    <data_root>/prepared/<data_path>[/<namespace>]
    <data_root>/grid/<grid_id>/<family>.zarr                     (one flat shared dir across all sources)

`src/data/sources/layout.py` no longer knows how to build either of those
paths -- this script rebuilds both formulas locally, and reads each source's
*current* code (registration, `output_root()`/`grid_store_path()` overrides)
to build the new-side paths, so the mapping tracks the real implementation
instead of a hand-copied list.

For each source/step, both old-layout candidates are checked; whichever one
actually exists on disk is what gets moved. If data exists under *both* old
layouts for the same target, that's a conflict needing human review (skipped,
never guessed at).

Dry-run by default -- pass `--execute` to actually touch the filesystem.
Safe to re-run: a source/step already migrated (new destination exists) is
skipped; a source/step with nothing to migrate (no old data under either
layout) is skipped; a conflicting destination (an old path and the new path
both exist, or both old layouts have data) is skipped with a warning rather
than silently overwritten. Any expected source path that doesn't exist on
disk is skipped/warned, never an error.

Usage:
    python scripts/migrate_legacy_layout.py --config orchestration/configs/data.yaml
    python scripts/migrate_legacy_layout.py --config orchestration/configs/data.yaml --execute
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cli.config import load_config_with_env_vars  # noqa: E402
from src.config.runtime import get_paths_config  # noqa: E402
from src.data.pipeline.config import get_source_config  # noqa: E402
from src.data.pipeline.context import PipelineContext  # noqa: E402
from src.data.sources import layout, registry  # noqa: E402
from src.data.sources.base import DataSource  # noqa: E402
from src.data.sources.layout import EASE_GRID_ID, LEGACY_GRID_ID  # noqa: E402
from src.data.sources.steps import PipelineStep  # noqa: E402

logger = logging.getLogger(__name__)

#: (source id) -> (legacy GRID filename, new store family name). Mirrors the
#: literal each source's own PREPARE/GRID target embeds -- kept as a static
#: table (not derived by calling .plan(), which needs real PREPARE input
#: already in place) so this script has no ordering dependency on the data
#: it's migrating. plad and snl_mining have per-instance-config-dependent
#: filenames/families and are handled separately below.
#: `legacy_filename` is `None` for sources that never existed under the
#: original preprocess-era "legacy" layout (only introduced during the
#: interim/current rework) -- their legacy candidate is skipped entirely
#: rather than built from a made-up filename.
GRID_FILENAME_AND_FAMILY = {
    "acag": ("acag_pm25_timeseries_reprojected.zarr", "pm25"),
    "esacci": ("esacci_lc_timeseries_reprojected.zarr", "land_cover"),
    "ntl_harm": ("ntl_harm_timeseries_reprojected.zarr", "ntl_harm"),
    "eog_dmsp": ("dmsp_timeseries_reprojected.zarr", "eog_dmsp"),
    "eog_viirs": ("viirs_annual_timeseries_reprojected.zarr", "eog_viirs_annual"),
    "eog_dvnl": ("viirs_dvnl_timeseries_reprojected.zarr", "eog_viirs_dvnl"),
    "modis": ("modis_21A2_timeseries_reprojected.zarr", "modis_lst_21a2"),
    "modis_robustness_11a1": ("modis_11A1_timeseries_reprojected.zarr", "modis_lst_11a1"),
    "glass_modis": ("modis_timeseries_reprojected.zarr", "glass_modis_lst"),
    "glass_ta_modis": (None, "glass_modis_ta"),
    "glass_avhrr": ("avhrr_timeseries_reprojected.zarr", "glass_avhrr_lst"),
    "osm": ("land_mask.zarr", "land_mask"),
    "gadm": ("countries_grid.zarr", "country_id"),
    "country_classifications": ("classifications_grid.zarr", "classifications"),
    "berman_mining": ("berman_mining_timeseries_reprojected.zarr", "berman_mining"),
    "ecoregions": ("ecoregions_grid.zarr", "ecoregions"),
}

#: GADM's GRID output has sidecar files living alongside the zarr store, not
#: produced via a StepTarget -- move them together with the store. Under the
#: true preprocess-era "legacy" layout these were a fixed pair with old-style
#: names (commit 7ea6e60 renamed them to the dynamic `{gid_col}_code_mapping.
#: json` scheme below, before the interim layout ever existed -- so "interim"
#: data never used these old names, see the glob-based interim handling in
#: migrate_grid()). (legacy filename, gid_col it corresponds to).
GADM_LEGACY_SIDECARS = (("country_code_mapping.json", "GID_0"), ("subdivision_code_mapping.json", "GID_1"))

#: Sources whose PREPARE step is a documented exception: it resolves its own
#: output path from config (`prepared_db_path`/`duckdb_path` overrides), not
#: necessarily the standard output_root() shape -- see
#: src/data/sources/snl_mining/source.py.
PREPARE_EXCEPTIONS = {"snl_mining"}

#: Sources whose PREPARE step is migrated as a single per-instance-filename
#: move rather than a generic whole-directory move: MODIS/MODIS-robustness
#: write directly into the *shared* GRID root (grid/<grid_id>/, a
#: `<family>.zarr` store already covered by GRID_FILENAME_AND_FAMILY); plad
#: writes one per-instance-config filename (`plad_adm{admin_level}_reg_fav.
#: parquet`) into the ADM_AGG bucket. A generic whole-directory move would
#: either clobber other sources' grid stores (MODIS) or risk assuming the
#: wrong legacy source path (plad's OUTPUT_PREFIX vs cfg.data_path). See
#: migrate_grid()/migrate_plad() below.
PREPARE_IS_GRID_SHAPED = {"modis", "modis_robustness_11a1", "plad"}

#: (source id) -> agg bucket its whole PREPARE-stage directory now lives
#: under (src/data/sources/layout.py's CRS_AGG/ADM_AGG/MISC_AGG split) --
#: every source not listed here defaults to CRS_AGG (the common case: a
#: pixel-grid raster intermediate feeding the source's own GRID/family.zarr
#: output). `plad` isn't listed: its PREPARE output is a per-instance
#: filename under ADM_AGG, handled by migrate_plad() below, not this
#: whole-directory move.
PREPARE_AGG_OVERRIDES = {
    "gadm": "adm",
    "country_classifications": "adm",
    "osm": "misc",
    # ecoregions' PREPARE directory only ever held its simplified vector
    # (`ecoregions_simplified.gpkg`) -- dominant_biome_by_gid3.parquet used
    # to fall back to the GRID root instead, so it never shared this
    # directory; migrated separately via RELOCATED_PREPARE_TABLES below.
    "ecoregions": "misc",
    "commodity_prices": "misc",
}

#: MODIS forces grid_id=ease6933 unconditionally, independent of the
#: pipeline's own grid_id choice (src/data/sources/modis/source.py). The
#: GRID mapping table above must be resolved against EASE_GRID_ID for it
#: regardless of --grid-id.
MODIS_FORCED_GRID_IDS = {"modis", "modis_robustness_11a1"}

#: Point 1 of the layout rework: two GID-keyed parquet tables used to fall
#: back to the GRID root (grid_store_path(..., family=None)) and now live
#: under prepared/ instead -- (source id, legacy GRID filename, new PREPARE
#: filename). Neither has any reader anywhere in src/ today; kept for future
#: use, per the design decision.
RELOCATED_PREPARE_TABLES = [
    ("ecoregions", "dominant_biome_by_gid3.parquet", "dominant_biome_by_gid3.parquet", None),
    ("country_classifications", "classifications_by_gid0.parquet", "classifications_by_gid0.parquet", "country_classifications"),
]


def legacy_raw_root(data_root: str, data_path: str, *, namespace: Optional[str] = None) -> str:
    base = os.path.join(data_root, data_path, "raw")
    if namespace:
        base = os.path.join(base, namespace)
    return base


def legacy_output_root(
    data_root: str,
    data_path: str,
    step: PipelineStep,
    *,
    namespace: Optional[str] = None,
    grid_id: str = LEGACY_GRID_ID,
) -> str:
    if step is PipelineStep.FETCH:
        return legacy_raw_root(data_root, data_path, namespace=namespace)
    if step is PipelineStep.PREPARE:
        base = os.path.join(data_root, data_path, "processed", "stage_1")
    elif step is PipelineStep.GRID:
        stage2_dir = "stage_2_ease6933" if grid_id == EASE_GRID_ID else "stage_2"
        base = os.path.join(data_root, data_path, "processed", stage2_dir)
    else:  # pragma: no cover
        raise ValueError(f"Unknown step: {step}")
    if namespace:
        base = os.path.join(base, namespace)
    return base


def interim_output_root(
    data_root: str,
    data_path: str,
    step: PipelineStep,
    *,
    namespace: Optional[str] = None,
    grid_id: str = LEGACY_GRID_ID,
) -> str:
    """Commit d43db13's short-lived flat layout: `prepared/<data_path>[/<ns>]`
    for PREPARE, and a single flat `grid/<grid_id>/` shared across every
    source for GRID (no per-source nesting, no agg buckets -- both added by
    the later crs/adm/misc-bucket merge). FETCH is identical to today's
    `raw/<data_path>[/<ns>]`, so callers never need an interim FETCH root."""
    if step is PipelineStep.PREPARE:
        base = os.path.join(data_root, "prepared", data_path)
    elif step is PipelineStep.GRID:
        return os.path.join(data_root, "grid", grid_id)
    else:  # pragma: no cover
        raise ValueError(f"Unknown step: {step}")
    if namespace:
        base = os.path.join(base, namespace)
    return base


@dataclass
class MigrationTally:
    moved: list = field(default_factory=list)
    skipped_already_migrated: list = field(default_factory=list)
    skipped_nothing_to_migrate: list = field(default_factory=list)
    skipped_conflict: list = field(default_factory=list)
    failed: list = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"moved={len(self.moved)} "
            f"skipped_already_migrated={len(self.skipped_already_migrated)} "
            f"skipped_nothing_to_migrate={len(self.skipped_nothing_to_migrate)} "
            f"skipped_conflict={len(self.skipped_conflict)} "
            f"failed={len(self.failed)}"
        )


def plan_move(label: str, old_path: str, new_path: str, tally: MigrationTally) -> Optional[tuple]:
    """Single-candidate form, kept for callers with only one possible old
    location (e.g. FETCH, which is identical under "interim" and "current")."""
    return plan_move_multi(label, [("legacy", old_path)], new_path, tally)


def plan_move_multi(
    label: str, old_candidates: list[tuple[str, str]], new_path: str, tally: MigrationTally
) -> Optional[tuple]:
    """Like `plan_move()`, but checks multiple named old-layout candidates
    (e.g. "legacy" and "interim") for the same eventual `new_path`, since a
    real HPC run may have produced data under either one depending on when it
    ran. Exactly one candidate existing is the normal case; zero is "nothing
    to migrate"; the new path already existing is "already migrated"; more
    than one old candidate existing, or an old candidate coexisting with the
    new path, is a conflict -- never guessed at, always left for a human."""
    existing = [(layout_name, p) for layout_name, p in old_candidates if os.path.exists(p)]
    new_exists = os.path.exists(new_path)

    if not existing and not new_exists:
        logger.debug("[%s] nothing to migrate (no old data): %s", label, [p for _, p in old_candidates])
        tally.skipped_nothing_to_migrate.append(label)
        return None
    if not existing and new_exists:
        logger.info("[%s] already migrated: %s", label, new_path)
        tally.skipped_already_migrated.append(label)
        return None
    if len(existing) > 1:
        logger.warning(
            "[%s] CONFLICT: old data exists under multiple layouts (%s) -- skipping, needs human review",
            label, existing,
        )
        tally.skipped_conflict.append(label)
        return None
    (layout_name, old_path) = existing[0]
    if new_exists:
        logger.warning(
            "[%s] CONFLICT: both %s (%s) and new (%s) paths exist -- skipping, needs human review",
            label, layout_name, old_path, new_path,
        )
        tally.skipped_conflict.append(label)
        return None
    return old_path, new_path


def _is_strict_subpath(candidate: str, of: str) -> bool:
    candidate, of = os.path.normpath(candidate), os.path.normpath(of)
    return candidate != of and candidate.startswith(of + os.sep)


def do_move(label: str, old_path: str, new_path: str, *, execute: bool, tally: MigrationTally) -> None:
    if not execute:
        logger.info("[DRY RUN] [%s] would move %s -> %s", label, old_path, new_path)
        tally.moved.append(label)
        return
    try:
        logger.info("[%s] moving %s -> %s", label, old_path, new_path)
        if _is_strict_subpath(new_path, old_path):
            # A whole-directory PREPARE move where the interim old directory
            # (e.g. prepared/<data_path>) becomes the parent of the new
            # agg-bucketed one (prepared/<data_path>/<agg>) -- new_path lives
            # *inside* old_path, so a direct rename would try to move the
            # directory into its own subdirectory (`shutil.move` raises
            # "Cannot move a directory into itself"). Stage it out of the way
            # first: rename old_path to a sibling temp dir, recreate old_path
            # as an empty parent, then move the temp dir's contents into the
            # final new_path.
            staging = old_path + ".migrate_staging"
            if os.path.exists(staging):
                raise FileExistsError(f"staging path already exists, refusing to overwrite: {staging}")
            os.rename(old_path, staging)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(staging, new_path)
        else:
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            shutil.move(old_path, new_path)
        logger.info("[%s] done", label)
        tally.moved.append(label)
    except Exception:
        logger.exception("[%s] failed to move %s -> %s", label, old_path, new_path)
        tally.failed.append(label)


def migrate_fetch(source_key: str, source: DataSource, *, execute: bool, tally: MigrationTally) -> None:
    if PipelineStep.FETCH not in type(source).STEPS:
        return
    label = f"{source_key}/FETCH"
    old_root = legacy_raw_root(source.ctx.data_root, source.cfg.data_path, namespace=source.cfg.namespace)
    new_root = source.output_root(PipelineStep.FETCH)
    planned = plan_move(label, old_root, new_root, tally)
    if planned is not None:
        do_move(label, *planned, execute=execute, tally=tally)


def migrate_prepare(source_key: str, source: DataSource, *, execute: bool, tally: MigrationTally) -> None:
    if PipelineStep.PREPARE not in type(source).STEPS:
        return
    if source_key in PREPARE_EXCEPTIONS:
        logger.debug("[%s/PREPARE] documented exception -- not migrated by this script", source_key)
        return
    if source_key in PREPARE_IS_GRID_SHAPED:
        return  # handled by migrate_grid()/migrate_plad() instead

    label = f"{source_key}/PREPARE"
    legacy_root = legacy_output_root(source.ctx.data_root, source.cfg.data_path, PipelineStep.PREPARE, namespace=source.cfg.namespace)
    interim_root = interim_output_root(source.ctx.data_root, source.cfg.data_path, PipelineStep.PREPARE, namespace=source.cfg.namespace)
    agg = PREPARE_AGG_OVERRIDES.get(source_key, layout.CRS_AGG)
    new_root = source.output_root(PipelineStep.PREPARE, agg=agg)
    planned = plan_move_multi(label, [("legacy", legacy_root), ("interim", interim_root)], new_root, tally)
    if planned is not None:
        do_move(label, *planned, execute=execute, tally=tally)


def migrate_grid(source_key: str, source: DataSource, *, grid_id: str, execute: bool, tally: MigrationTally) -> None:
    mapping = GRID_FILENAME_AND_FAMILY.get(source_key)
    if mapping is None:
        return
    legacy_filename, family = mapping
    effective_grid_id = EASE_GRID_ID if source_key in MODIS_FORCED_GRID_IDS else grid_id

    legacy_dir = legacy_output_root(source.ctx.data_root, source.cfg.data_path, PipelineStep.GRID, grid_id=effective_grid_id)
    # interim's GRID root is flat and shared across every source (keyed only
    # by grid_id, not data_path) -- see interim_output_root()'s docstring.
    interim_dir = interim_output_root(source.ctx.data_root, source.cfg.data_path, PipelineStep.GRID, grid_id=effective_grid_id)
    new_path = layout.grid_store_path(source.ctx.data_root, source.cfg.data_path, grid_id=effective_grid_id, family=family)

    candidates = [("interim", os.path.join(interim_dir, f"{family}.zarr"))]
    if legacy_filename is not None:
        candidates.insert(0, ("legacy", os.path.join(legacy_dir, legacy_filename)))

    label = f"{source_key}/GRID"
    planned = plan_move_multi(label, candidates, new_path, tally)
    if planned is not None:
        do_move(label, *planned, execute=execute, tally=tally)

    if source_key == "gadm":
        from src.data.sources.misc.gadm import gid_mapping_path

        # Both old layouts wrote these sidecars alongside the GRID zarr store
        # (legacy: <stage_2 dir>/<name>; interim: the shared flat
        # grid/<grid_id>/<name>). The *current* location moved to the
        # ADM_AGG bucket (gid_mapping_path()) alongside gadm's simplified
        # .gpkg boundary files, not alongside the CRS zarr -- see
        # gid_mapping_path()'s own docstring. Legacy only ever wrote the
        # fixed country/subdivision pair; interim already used the dynamic
        # `{gid_col}_code_mapping.json` naming (see GADM_LEGACY_SIDECARS'
        # comment) for however many ADM levels a given run actually
        # rasterized, so that side is discovered by globbing rather than
        # assumed to be exactly GID_0/GID_1.
        gid_cols_seen = set()
        for legacy_name, gid_col in GADM_LEGACY_SIDECARS:
            gid_cols_seen.add(gid_col)
            sidecar_label = f"{source_key}/GRID/{legacy_name}"
            sidecar_candidates = [("legacy", os.path.join(legacy_dir, legacy_name))]
            interim_current_name = os.path.join(interim_dir, f"{gid_col}_code_mapping.json")
            if os.path.exists(interim_current_name):
                sidecar_candidates.append(("interim", interim_current_name))
            sidecar_new = gid_mapping_path(source.ctx.data_root, effective_grid_id, gid_col)
            planned = plan_move_multi(sidecar_label, sidecar_candidates, sidecar_new, tally)
            if planned is not None:
                do_move(sidecar_label, *planned, execute=execute, tally=tally)

        if os.path.isdir(interim_dir):
            import glob

            for interim_sidecar in glob.glob(os.path.join(interim_dir, "*_code_mapping.json")):
                gid_col = os.path.basename(interim_sidecar).removesuffix("_code_mapping.json")
                if gid_col in gid_cols_seen:
                    continue  # already handled above alongside its legacy pair
                sidecar_label = f"{source_key}/GRID/{gid_col}_code_mapping.json"
                sidecar_new = gid_mapping_path(source.ctx.data_root, effective_grid_id, gid_col)
                planned = plan_move_multi(sidecar_label, [("interim", interim_sidecar)], sidecar_new, tally)
                if planned is not None:
                    do_move(sidecar_label, *planned, execute=execute, tally=tally)


def migrate_plad(source_key: str, source: DataSource, *, grid_id: str, execute: bool, tally: MigrationTally) -> None:
    """PLAD's PREPARE output is a single GID_N-keyed parquet file --
    ADM_AGG, via `output_root(PREPARE, agg=ADM_AGG)` (see
    `PlaDSource.output_root()`/`_discover_prepare()`,
    src/data/sources/plad.py). The legacy layout wrote it into the (shared)
    GRID root instead. `admin_level` is per-instance config, so the filename
    is read off the live source object, mirroring how `_discover_prepare()`
    builds it."""
    if source_key != "plad":
        return
    admin_level = getattr(source, "admin_level", None)
    if admin_level is None:
        logger.warning("[plad/PREPARE] source has no admin_level -- skipping")
        return
    filename = f"plad_adm{admin_level}_reg_fav.parquet"
    legacy_dir = legacy_output_root(source.ctx.data_root, source.OUTPUT_PREFIX, PipelineStep.GRID, grid_id=grid_id)
    # interim (d43db13) wrote straight into the flat PREPARE root (no agg
    # buckets existed yet), unlike legacy which shared the GRID root.
    interim_dir = interim_output_root(source.ctx.data_root, source.OUTPUT_PREFIX, PipelineStep.PREPARE)
    new_dir = layout.output_root(source.ctx.data_root, source.OUTPUT_PREFIX, PipelineStep.PREPARE, agg=layout.ADM_AGG)
    label = f"plad/PREPARE/{filename}"
    candidates = [
        ("legacy", os.path.join(legacy_dir, filename)),
        ("interim", os.path.join(interim_dir, filename)),
    ]
    planned = plan_move_multi(label, candidates, os.path.join(new_dir, filename), tally)
    if planned is not None:
        do_move(label, *planned, execute=execute, tally=tally)


def migrate_relocated_prepare_tables(source_key: str, source: DataSource, *, grid_id: str, execute: bool, tally: MigrationTally) -> None:
    """Point 1 of the layout rework: small per-GID parquet tables that used
    to fall back to the GRID root, now under prepared/ -- see module
    docstring's RELOCATED_PREPARE_TABLES."""
    for table_source_key, legacy_filename, new_filename, namespace in RELOCATED_PREPARE_TABLES:
        if table_source_key != source_key:
            continue
        label = f"{source_key}/PREPARE/{new_filename}"
        effective_namespace = namespace or source.cfg.namespace
        legacy_dir = legacy_output_root(source.ctx.data_root, source.cfg.data_path, PipelineStep.GRID, namespace=source.cfg.namespace, grid_id=grid_id)
        # interim (d43db13) already relocated these off the GRID root into
        # the flat PREPARE root (no agg buckets existed yet), using the same
        # filename as today (new_filename == legacy_filename for both
        # RELOCATED_PREPARE_TABLES entries).
        interim_dir = interim_output_root(source.ctx.data_root, source.cfg.data_path, PipelineStep.PREPARE, namespace=effective_namespace)
        # ADM_AGG: both of RELOCATED_PREPARE_TABLES' entries are GID_N-keyed
        # tables (src/data/sources/layout.py's crs/adm/misc split).
        new_dir = layout.output_root(
            source.ctx.data_root, source.cfg.data_path, PipelineStep.PREPARE,
            namespace=effective_namespace, agg=layout.ADM_AGG,
        )
        candidates = [
            ("legacy", os.path.join(legacy_dir, legacy_filename)),
            ("interim", os.path.join(interim_dir, new_filename)),
        ]
        planned = plan_move_multi(label, candidates, os.path.join(new_dir, new_filename), tally)
        if planned is not None:
            do_move(label, *planned, execute=execute, tally=tally)


def migrate_source(source_key: str, source: DataSource, *, grid_id: str, execute: bool, tally: MigrationTally) -> None:
    # GRID-shaped moves first: they use a static filename/family mapping
    # rather than a whole-directory move, so there's no ordering hazard
    # either way -- but doing them first keeps the larger, riskier
    # whole-directory PREPARE/FETCH moves from running before we know this
    # source's GRID-shaped output (if any) has a known mapping.
    migrate_grid(source_key, source, grid_id=grid_id, execute=execute, tally=tally)
    migrate_plad(source_key, source, grid_id=grid_id, execute=execute, tally=tally)
    migrate_relocated_prepare_tables(source_key, source, grid_id=grid_id, execute=execute, tally=tally)
    migrate_prepare(source_key, source, execute=execute, tally=tally)
    migrate_fetch(source_key, source, execute=execute, tally=tally)


def build_source(source_key: str, config: dict, data_root: str, grid_id: str) -> DataSource:
    cls = registry.load(source_key)
    cfg = get_source_config(config, source_key)
    ctx = PipelineContext(data_root=data_root, grid_id=grid_id)
    return cls(ctx, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to the unified pipeline config (data.yaml)")
    parser.add_argument("--data-root", default=None, help="Override paths.data_root from the config")
    parser.add_argument(
        "--grid-id", default=LEGACY_GRID_ID, choices=[LEGACY_GRID_ID, EASE_GRID_ID],
        help="Which grid's stores to migrate (default: legacy_4326). Ignored for modis/modis_robustness_11a1, which always use ease6933.",
    )
    parser.add_argument("--source", default=None, help="Migrate only this source (default: every source in the config)")
    parser.add_argument("--execute", action="store_true", help="Actually move files (default: dry run, logs only)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    config = load_config_with_env_vars(args.config)
    data_root = args.data_root or get_paths_config(config).get("data_root")
    if not data_root:
        raise SystemExit("No data_root found -- pass --data-root or set paths.data_root in the config")

    source_keys = [args.source] if args.source else sorted((config.get("sources") or {}).keys())

    logger.info("Migrating layout: legacy|interim -> current, data_root=%s, grid_id=%s, execute=%s", data_root, args.grid_id, args.execute)
    logger.info("Sources: %s", ", ".join(source_keys))

    tally = MigrationTally()
    for source_key in source_keys:
        try:
            source = build_source(source_key, config, data_root, args.grid_id)
        except Exception:
            logger.exception("[%s] failed to construct source -- skipping entirely", source_key)
            tally.failed.append(f"{source_key}/construct")
            continue
        migrate_source(source_key, source, grid_id=args.grid_id, execute=args.execute, tally=tally)

    logger.info("Migration complete: %s", tally.summary())
    if not args.execute:
        logger.info("This was a DRY RUN -- pass --execute to actually move files.")
    if tally.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
