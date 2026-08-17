"""Physically move already-computed pipeline output from the old per-source
directory layout into the current stage-name-first tree (`raw/<data_path>`,
`prepared/<data_path>`, `grid/<grid_id>/<family>.zarr` -- see
`src/data/sources/layout.py`).

The old ("legacy") layout was:
    <data_root>/<data_path>/raw[/<namespace>]
    <data_root>/<data_path>/processed/stage_1[/<namespace>]
    <data_root>/<data_path>/processed/stage_2[/<namespace>]/<legacy_filename>
    <data_root>/<data_path>/processed/stage_2_ease6933[/<namespace>]/<legacy_filename>

`src/data/sources/layout.py` no longer knows how to build those paths (the
legacy layout was removed there entirely) -- this script rebuilds the old
formula locally, and reads each source's *current* code (registration,
`output_root()`/`grid_store_path()` overrides) to build the new-side paths,
so the mapping tracks the real implementation instead of a hand-copied list.

Dry-run by default -- pass `--execute` to actually touch the filesystem.
Safe to re-run: a source/step already migrated (new destination exists) is
skipped; a source/step with nothing to migrate (no legacy data) is skipped;
a conflicting destination (both old and new paths exist) is skipped with a
warning rather than silently overwritten. Any expected source path that
doesn't exist on disk is skipped/warned, never an error.

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
    "glass_avhrr": ("avhrr_timeseries_reprojected.zarr", "glass_avhrr_lst"),
    "osm": ("land_mask.zarr", "land_mask"),
    "gadm": ("countries_grid.zarr", "country_id"),
    "country_classifications": ("classifications_grid.zarr", "classifications"),
    "berman_mining": ("berman_mining_timeseries_reprojected.zarr", "berman_mining"),
    "ecoregions": ("ecoregions_grid.zarr", "ecoregions"),
}

#: GADM's GRID output has sidecar files living alongside the zarr store, not
#: produced via a StepTarget -- move them together with the store.
GADM_SIDECAR_FILENAMES = ("country_code_mapping.json", "subdivision_code_mapping.json")

#: Sources whose PREPARE step is a documented exception: it resolves its own
#: output path from config (`prepared_db_path`/`duckdb_path` overrides), not
#: necessarily the standard output_root() shape -- see
#: src/data/sources/snl_mining/source.py.
PREPARE_EXCEPTIONS = {"snl_mining"}

#: Sources whose PREPARE step writes directly into the *shared* GRID root
#: (grid/<grid_id>/) rather than its own prepared/<data_path>/ directory, so
#: a generic whole-directory PREPARE move would clobber other sources' grid
#: stores. Migrated as a per-file move instead (see migrate_plad below).
#: MODIS is also GRID-shaped, but its output is a `<family>.zarr` store
#: already covered by GRID_FILENAME_AND_FAMILY.
PREPARE_IS_GRID_SHAPED = {"modis", "modis_robustness_11a1", "plad"}

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
    old_exists = os.path.exists(old_path)
    new_exists = os.path.exists(new_path)

    if not old_exists and not new_exists:
        logger.debug("[%s] nothing to migrate (no legacy data): %s", label, old_path)
        tally.skipped_nothing_to_migrate.append(label)
        return None
    if not old_exists and new_exists:
        logger.info("[%s] already migrated: %s", label, new_path)
        tally.skipped_already_migrated.append(label)
        return None
    if old_exists and new_exists:
        logger.warning(
            "[%s] CONFLICT: both old (%s) and new (%s) paths exist -- skipping, needs human review",
            label, old_path, new_path,
        )
        tally.skipped_conflict.append(label)
        return None
    return old_path, new_path


def do_move(label: str, old_path: str, new_path: str, *, execute: bool, tally: MigrationTally) -> None:
    if not execute:
        logger.info("[DRY RUN] [%s] would move %s -> %s", label, old_path, new_path)
        tally.moved.append(label)
        return
    try:
        logger.info("[%s] moving %s -> %s", label, old_path, new_path)
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
    old_root = legacy_output_root(source.ctx.data_root, source.cfg.data_path, PipelineStep.PREPARE, namespace=source.cfg.namespace)
    new_root = source.output_root(PipelineStep.PREPARE)
    planned = plan_move(label, old_root, new_root, tally)
    if planned is not None:
        do_move(label, *planned, execute=execute, tally=tally)


def migrate_grid(source_key: str, source: DataSource, *, grid_id: str, execute: bool, tally: MigrationTally) -> None:
    mapping = GRID_FILENAME_AND_FAMILY.get(source_key)
    if mapping is None:
        return
    legacy_filename, family = mapping
    effective_grid_id = EASE_GRID_ID if source_key in MODIS_FORCED_GRID_IDS else grid_id

    old_dir = legacy_output_root(source.ctx.data_root, source.cfg.data_path, PipelineStep.GRID, grid_id=effective_grid_id)
    new_path = layout.grid_store_path(source.ctx.data_root, source.cfg.data_path, grid_id=effective_grid_id, family=family)
    old_path = os.path.join(old_dir, legacy_filename)

    label = f"{source_key}/GRID"
    planned = plan_move(label, old_path, new_path, tally)
    if planned is not None:
        do_move(label, *planned, execute=execute, tally=tally)

    if source_key == "gadm":
        new_dir = os.path.dirname(new_path)
        for sidecar in GADM_SIDECAR_FILENAMES:
            sidecar_label = f"{source_key}/GRID/{sidecar}"
            sidecar_old = os.path.join(old_dir, sidecar)
            sidecar_new = os.path.join(new_dir, sidecar)
            planned = plan_move(sidecar_label, sidecar_old, sidecar_new, tally)
            if planned is not None:
                do_move(sidecar_label, *planned, execute=execute, tally=tally)


def migrate_plad(source_key: str, source: DataSource, *, grid_id: str, execute: bool, tally: MigrationTally) -> None:
    """PLAD's PREPARE output is a single parquet file written directly into
    the (shared) GRID root, not a `<family>.zarr` store -- see
    `PlaDSource.output_root()`/`_discover_prepare()`
    (src/data/sources/plad.py). `admin_level` is per-instance config, so the
    filename is read off the live source object, mirroring how
    `_discover_prepare()` builds it."""
    if source_key != "plad":
        return
    admin_level = getattr(source, "admin_level", None)
    if admin_level is None:
        logger.warning("[plad/PREPARE] source has no admin_level -- skipping")
        return
    filename = f"plad_adm{admin_level}_reg_fav.parquet"
    old_dir = legacy_output_root(source.ctx.data_root, source.OUTPUT_PREFIX, PipelineStep.GRID, grid_id=grid_id)
    new_dir = layout.output_root(source.ctx.data_root, source.OUTPUT_PREFIX, PipelineStep.GRID, grid_id=grid_id)
    label = f"plad/PREPARE/{filename}"
    planned = plan_move(label, os.path.join(old_dir, filename), os.path.join(new_dir, filename), tally)
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
        old_dir = legacy_output_root(source.ctx.data_root, source.cfg.data_path, PipelineStep.GRID, namespace=source.cfg.namespace, grid_id=grid_id)
        new_dir = layout.output_root(source.ctx.data_root, source.cfg.data_path, PipelineStep.PREPARE, namespace=namespace or source.cfg.namespace)
        planned = plan_move(label, os.path.join(old_dir, legacy_filename), os.path.join(new_dir, new_filename), tally)
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

    logger.info("Migrating layout: legacy -> current, data_root=%s, grid_id=%s, execute=%s", data_root, args.grid_id, args.execute)
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
