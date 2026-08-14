"""Physically move already-computed pipeline output from the legacy
per-source directory layout into `layout: v2`'s stage-name-first tree
(`raw/<data_path>`, `prepared/<data_path>`, `grid/<grid_id>/<family>.zarr`
-- see `src/data/sources/layout.py`).

Dry-run by default -- pass `--execute` to actually touch the filesystem.
Safe to re-run: a source/step already migrated (v2 destination exists) is
skipped; a source/step with nothing to migrate (no legacy data) is skipped;
a conflicting destination (both legacy and v2 paths exist) is skipped with a
warning rather than silently overwritten.

Usage:
    python scripts/migrate_layout_v2.py --config orchestration/configs/data.yaml
    python scripts/migrate_layout_v2.py --config orchestration/configs/data.yaml --execute
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli.config import load_config_with_env_vars  # noqa: E402
from src.config.runtime import get_paths_config  # noqa: E402
from src.data.pipeline.config import get_source_config  # noqa: E402
from src.data.pipeline.context import PipelineContext  # noqa: E402
from src.data.sources import registry  # noqa: E402
from src.data.sources.base import DataSource  # noqa: E402
from src.data.sources.layout import EASE_GRID_ID, LEGACY_GRID_ID, LEGACY_LAYOUT, V2_LAYOUT  # noqa: E402
from src.data.sources.steps import PipelineStep  # noqa: E402

logger = logging.getLogger(__name__)

#: (source config key in data.yaml) -> GRID legacy filename, v2 family.
#: Mirrors the literals each source's own _plan()/_plan_grid() embeds --
#: kept here rather than derived, since deriving it would mean calling
#: plan() (which needs real PREPARE output already in place, an ordering
#: dependency this migration deliberately avoids). plad and snl_mining are
#: handled separately below since their filename/family depend on
#: per-instance config (admin_level, output_filename).
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
}

#: GADM's GRID output has sidecar files living alongside the zarr store,
#: not produced via a StepTarget -- move them together with the store.
GADM_SIDECAR_FILENAMES = ("country_code_mapping.json", "subdivision_code_mapping.json")

#: Sources whose PREPARE step is a documented layout:v2 exception: it
#: resolves its own output path from config, entirely independent of
#: layout.py, and must not be migrated (src/data/sources/snl_mining/source.py).
PREPARE_EXCEPTIONS = {"snl_mining"}

#: MODIS forces grid_id=ease6933 unconditionally, independent of the
#: pipeline's own grid_id choice (src/data/sources/modis/source.py). The
#: GRID mapping table above must be resolved against EASE_GRID_ID for it
#: regardless of --grid-id.
MODIS_FORCED_GRID_IDS = {"modis", "modis_robustness_11a1"}


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
    """Decide whether (old_path -> new_path) should be migrated.

    Returns the (old_path, new_path) pair to move, or None if there's
    nothing to do (already migrated, nothing to migrate, or a conflict that
    needs human review).
    """
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
            "[%s] CONFLICT: both legacy (%s) and v2 (%s) paths exist -- skipping, needs human review",
            label,
            old_path,
            new_path,
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


def migrate_fetch_or_prepare(
    source_key: str,
    step: PipelineStep,
    source_legacy: DataSource,
    source_v2: DataSource,
    *,
    execute: bool,
    tally: MigrationTally,
) -> None:
    """Whole-directory move for FETCH/PREPARE -- each source's own
    directory shape is unchanged under v2, only the parent nesting moves."""
    if step is PipelineStep.PREPARE and source_key in PREPARE_EXCEPTIONS:
        logger.debug("[%s/%s] documented layout:v2 exception -- not migrated", source_key, step.name)
        return
    if step not in type(source_legacy).STEPS:
        return

    label = f"{source_key}/{step.name}"
    old_root = source_legacy.output_root(step)
    new_root = source_v2.output_root(step)

    planned = plan_move(label, old_root, new_root, tally)
    if planned is not None:
        do_move(label, *planned, execute=execute, tally=tally)


def _grid_filename_and_family(source_key: str, source_legacy: DataSource) -> Optional[tuple]:
    if source_key in GRID_FILENAME_AND_FAMILY:
        return GRID_FILENAME_AND_FAMILY[source_key]
    if source_key == "plad":
        admin_level = source_legacy.admin_level
        return (
            f"plad_adm{admin_level}_timeseries_reprojected.zarr",
            f"admin_panel_adm{admin_level}",
        )
    if source_key == "snl_mining":
        return (source_legacy.output_filename, "snl_mining")
    return None


def migrate_grid(
    source_key: str,
    source_legacy: DataSource,
    source_v2: DataSource,
    *,
    execute: bool,
    tally: MigrationTally,
) -> None:
    """Per-file move for GRID -- legacy stores live one-per-source-directory
    under a filename that must be renamed to the v2 family filename, not
    just relocated. Doesn't call .plan() (which can depend on PREPARE
    output already being wherever ctx.layout expects, not guaranteed true
    mid-migration) -- uses the static (filename, family) mapping instead.

    Deliberately does *not* gate on `PipelineStep.GRID in type(source_legacy).STEPS`:
    several sources that used to declare a separate GRID step (gadm, acag,
    osm, ...) have since had PREPARE+GRID merged into one PREPARE step
    (Plan 2, docs/design successor to the ledger) and no longer declare GRID
    at all -- but pre-merge runs may still have real legacy-layout GRID
    output on disk that needs migrating. `GRID_FILENAME_AND_FAMILY`/
    `_grid_filename_and_family()` (a static, historical mapping, not derived
    from the current class) is the authority on whether a source ever had a
    GRID output worth migrating."""
    mapping = _grid_filename_and_family(source_key, source_legacy)
    if mapping is None:
        logger.warning("[%s/GRID] no known legacy_filename/v2_family mapping -- skipping", source_key)
        return
    legacy_filename, _v2_family = mapping

    old_dir = source_legacy.output_root(PipelineStep.GRID)
    new_dir = source_v2.output_root(PipelineStep.GRID)
    old_path = os.path.join(old_dir, legacy_filename)
    new_path = os.path.join(new_dir, f"{_v2_family}.zarr")

    label = f"{source_key}/GRID"
    planned = plan_move(label, old_path, new_path, tally)
    if planned is not None:
        do_move(label, *planned, execute=execute, tally=tally)

    if source_key == "gadm":
        for sidecar in GADM_SIDECAR_FILENAMES:
            sidecar_label = f"{source_key}/GRID/{sidecar}"
            sidecar_old = os.path.join(old_dir, sidecar)
            sidecar_new = os.path.join(new_dir, sidecar)
            planned = plan_move(sidecar_label, sidecar_old, sidecar_new, tally)
            if planned is not None:
                do_move(sidecar_label, *planned, execute=execute, tally=tally)


def migrate_source(
    source_key: str,
    source_legacy: DataSource,
    source_v2: DataSource,
    *,
    execute: bool,
    tally: MigrationTally,
) -> None:
    # GRID first: it doesn't depend on PREPARE's current location (uses a
    # static filename/family mapping, not .plan()), so there's no ordering
    # hazard either way -- but doing it first keeps the (larger, riskier)
    # whole-directory PREPARE/FETCH moves from ever running before we know
    # the source's GRID output has a known mapping.
    migrate_grid(source_key, source_legacy, source_v2, execute=execute, tally=tally)
    migrate_fetch_or_prepare(source_key, PipelineStep.PREPARE, source_legacy, source_v2, execute=execute, tally=tally)
    migrate_fetch_or_prepare(source_key, PipelineStep.FETCH, source_legacy, source_v2, execute=execute, tally=tally)


def build_source_pair(source_key: str, config: dict, data_root: str, grid_id: str):
    cls = registry.load(source_key)
    cfg = get_source_config(config, source_key)
    effective_grid_id = EASE_GRID_ID if source_key in MODIS_FORCED_GRID_IDS else grid_id

    ctx_legacy = PipelineContext(data_root=data_root, grid_id=effective_grid_id, layout=LEGACY_LAYOUT)
    ctx_v2 = PipelineContext(data_root=data_root, grid_id=effective_grid_id, layout=V2_LAYOUT)
    return cls(ctx_legacy, cfg), cls(ctx_v2, cfg)


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

    logger.info("Migrating layout: legacy -> v2, data_root=%s, grid_id=%s, execute=%s", data_root, args.grid_id, args.execute)
    logger.info("Sources: %s", ", ".join(source_keys))

    tally = MigrationTally()
    for source_key in source_keys:
        try:
            source_legacy, source_v2 = build_source_pair(source_key, config, data_root, args.grid_id)
        except Exception:
            logger.exception("[%s] failed to construct source -- skipping entirely", source_key)
            tally.failed.append(f"{source_key}/construct")
            continue
        migrate_source(source_key, source_legacy, source_v2, execute=args.execute, tally=tally)

    logger.info("Migration complete: %s", tally.summary())
    if not args.execute:
        logger.info("This was a DRY RUN -- pass --execute to actually move files.")
    if tally.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
