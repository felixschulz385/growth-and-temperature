"""Find (and optionally remove) MODIS FETCH `.tif` tiles corrupted by the
antimeridian/oversized-bbox bug fixed in `ModisSource._tile_bbox_4326`/
`_load_tile_year` (docs/design/13-prepare-memory-parallelism.md,
2026-08-26): before that fix, `_tile_bbox_4326` could silently wrap a
poleward corner's longitude for high-latitude tiles (v rows near a pole),
producing a STAC-search bbox spanning nearly the whole globe -- and
`_load_tile_year` had no `geobox=` constraint, so `odc.stac.load` wrote
whatever oversized union footprint the (over-broad) search returned instead
of the tile's real ~1200x1200 km footprint. Real production symptom: PREPARE
crashed on `xr.combine_by_coords` "duplicate values"/"not monotonic global
indexes" trying to mosaic dozens of these bloated files together.

A genuine single MODIS sinusoidal tile is a fixed
`tiles.TILE_SIZE_M` (~1,111,950m) square. This scans every fetched
`<year>/h##v##.tif` under each MODIS-family source's FETCH output root,
compares its actual on-disk bounds (a cheap `rasterio.open()` header read,
no pixel data touched) against the tile's true bounds
(`tiles.tile_bounds_m(h, v)`), and flags any file whose width or height
exceeds `--oversize-factor` times the true tile size (default 1.5x -- a
genuine tile can be a little larger than nominal from resampling/edge
padding, but never anywhere near double).

Removing a flagged file is enough on its own to make it eligible for
re-fetch: MODIS FETCH completion is plain `Completion.PATH_EXISTS`
(`src/data/sources/modis/source.py`'s module docstring) with no separate
ledger entry to also clear, so the next
`data run --source <id> --step fetch --key <year>/<tile>` (or a plain
`--step fetch` full rerun) naturally redoes exactly the removed units.

Dry run and SLURM submission are both the default -- this can scan
thousands of files across ~20 years x ~300 tiles per source, so it defaults
to a background SLURM job rather than tying up a login-node shell, and
defaults to reporting what it *would* remove rather than deleting anything.
Pass `--no-dry-run` to actually delete, and/or `--no-slurm` to run directly
in the current shell instead of submitting.

Usage:
    # Dry run, submitted to SLURM (both defaults) -- review the job's log:
    python scripts/find_and_remove_oversized_modis_tiles.py --config orchestration/configs/data.yaml

    # Same, but run directly in this shell instead of via sbatch:
    python scripts/find_and_remove_oversized_modis_tiles.py --config orchestration/configs/data.yaml --no-slurm

    # Actually delete the flagged files, still via SLURM:
    python scripts/find_and_remove_oversized_modis_tiles.py --config orchestration/configs/data.yaml --no-dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shlex
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.cli.config import load_config_with_env_vars  # noqa: E402
from src.cli.data.slurm import load_slurm_config  # noqa: E402
from src.config.runtime import get_paths_config  # noqa: E402
from src.data.pipeline.config import get_source_config  # noqa: E402
from src.data.pipeline.context import PipelineContext  # noqa: E402
from src.data.sources import registry  # noqa: E402
from src.data.sources.modis import tiles as modis_util  # noqa: E402
from src.data.sources.steps import PipelineStep  # noqa: E402

logger = logging.getLogger(__name__)

TILE_RE = re.compile(r"^h(\d{2})v(\d{2})\.tif$")


def build_source(source_key: str, config: dict, data_root: str):
    cls = registry.load(source_key)
    cfg = get_source_config(config, source_key)
    ctx = PipelineContext(data_root=data_root)
    return cls(ctx, cfg)


def modis_source_keys(config: dict) -> list[str]:
    return [
        key for key, raw in (config.get("sources") or {}).items()
        if (raw or {}).get("type") == "modis"
    ]


def scan_source(source, *, oversize_factor: float) -> list[tuple[str, str, str, float, float, float, float]]:
    """Returns a list of (year, tile, path, actual_width_km,
    actual_height_km, expected_width_km, expected_height_km) for every
    oversized file found."""
    import rasterio

    stage1_root = source.output_root(PipelineStep.FETCH)
    flagged = []
    if not os.path.isdir(stage1_root):
        logger.warning("No FETCH output root at %s -- skipping", stage1_root)
        return flagged

    for year_name in sorted(os.listdir(stage1_root)):
        year_dir = os.path.join(stage1_root, year_name)
        if not os.path.isdir(year_dir) or not year_name.isdigit():
            continue
        for fname in sorted(os.listdir(year_dir)):
            m = TILE_RE.match(fname)
            if not m:
                continue
            h, v = int(m.group(1)), int(m.group(2))
            x0, y0, x1, y1 = modis_util.tile_bounds_m(h, v)
            expected_w, expected_h = x1 - x0, y1 - y0

            path = os.path.join(year_dir, fname)
            try:
                with rasterio.open(path) as src:
                    bounds = src.bounds
            except Exception:
                logger.exception("Failed to open %s -- skipping", path)
                continue
            actual_w = bounds.right - bounds.left
            actual_h = bounds.top - bounds.bottom

            if actual_w > oversize_factor * expected_w or actual_h > oversize_factor * expected_h:
                flagged.append(
                    (year_name, f"h{h:02d}v{v:02d}", path,
                     actual_w / 1000, actual_h / 1000, expected_w / 1000, expected_h / 1000)
                )
    return flagged


def resubmit_via_slurm(args: argparse.Namespace) -> None:
    cluster = load_slurm_config()["cluster"]

    forward = [f'{cluster["python_bin"]} {os.path.abspath(__file__)}', f'--config "{args.config}"']
    if args.data_root:
        forward.append(f'--data-root "{args.data_root}"')
    if args.source:
        forward.append("--source " + " ".join(shlex.quote(s) for s in args.source))
    forward.append(f"--oversize-factor {args.oversize_factor}")
    forward.append("--dry-run" if args.dry_run else "--no-dry-run")
    forward.append("--no-slurm")
    forward.append(f"--log-level {args.log_level}")

    wrap_lines = [
        f'cd {cluster["project_root"]}',
        f'eval "$({cluster["conda_hook"]} shell.bash hook)"',
        "conda activate gnt",
        " ".join(forward),
    ]
    log_dir = f'{cluster["project_root"]}/log/maintenance/find-and-remove-oversized-modis-tiles'
    argv = [
        "sbatch", "--parsable",
        "--job-name=find-and-remove-oversized-modis-tiles",
        f"--output={log_dir}/%x-%j.out",
        f"--error={log_dir}/%x-%j.err",
        f"--time={args.slurm_time}",
        f"--qos={args.slurm_qos}",
        f"--cpus-per-task={args.slurm_cpus}",
        f"--mem={args.slurm_mem}",
        f"--wrap={' && '.join(wrap_lines)}",
    ]
    print(" ".join(shlex.quote(a) for a in argv))
    os.makedirs(log_dir, exist_ok=True)
    result = subprocess.run(argv, capture_output=True, text=True, cwd=PROJECT_ROOT, check=True)
    job_id = result.stdout.strip()
    print(f"Submitted find-and-remove-oversized-modis-tiles -> job {job_id}")
    print(f"Log: {log_dir}/find-and-remove-oversized-modis-tiles-{job_id}.out")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to the unified pipeline config (data.yaml)")
    parser.add_argument("--data-root", default=None, help="Override paths.data_root from the config")
    parser.add_argument(
        "--source", action="append", default=None,
        help="Restrict to this MODIS-family source id (repeatable). Default: every source with type: modis in the config.",
    )
    parser.add_argument(
        "--oversize-factor", type=float, default=1.5,
        help="Flag a tile whose width or height exceeds this multiple of the true MODIS tile size (default: 1.5).",
    )
    parser.add_argument(
        "--dry-run", action=argparse.BooleanOptionalAction, default=True,
        help="Only report what would be removed (default: on). Pass --no-dry-run to actually delete.",
    )
    parser.add_argument(
        "--slurm", action=argparse.BooleanOptionalAction, default=True,
        help="Submit as a SLURM job instead of running in this shell (default: on). Pass --no-slurm to run directly.",
    )
    parser.add_argument("--slurm-time", default="00:30:00")
    parser.add_argument("--slurm-qos", default="30min")
    parser.add_argument("--slurm-cpus", type=int, default=2)
    parser.add_argument("--slurm-mem", default="4G")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    if args.slurm and "SLURM_JOB_ID" not in os.environ:
        resubmit_via_slurm(args)
        return

    config = load_config_with_env_vars(args.config)
    data_root = args.data_root or get_paths_config(config).get("data_root")
    if not data_root:
        raise SystemExit("No data_root found -- pass --data-root or set paths.data_root in the config")

    source_keys = args.source or modis_source_keys(config)
    if not source_keys:
        raise SystemExit("No MODIS-family (type: modis) sources found in the config")

    logger.info(
        "Scanning for oversized MODIS FETCH tiles, data_root=%s, sources=%s, oversize_factor=%.2f, dry_run=%s",
        data_root, source_keys, args.oversize_factor, args.dry_run,
    )

    total_flagged = 0
    total_failed = 0
    for source_key in source_keys:
        try:
            source = build_source(source_key, config, data_root)
        except Exception:
            logger.exception("[%s] failed to construct source -- skipping entirely", source_key)
            total_failed += 1
            continue

        flagged = scan_source(source, oversize_factor=args.oversize_factor)
        if not flagged:
            logger.info("[%s] no oversized tiles found", source_key)
            continue

        by_year: dict[str, list] = {}
        for entry in flagged:
            by_year.setdefault(entry[0], []).append(entry)

        for year, entries in sorted(by_year.items()):
            for year_, tile, path, aw, ah, ew, eh in entries:
                logger.info(
                    "[%s] %s/%s: %.0fx%.0f km (expected ~%.0fx%.0f km) -- %s",
                    source_key, year_, tile, aw, ah, ew, eh, path,
                )
            tiles_str = ", ".join(e[1] for e in entries)
            logger.info(
                "[%s] year %s: %d oversized tile(s): %s",
                source_key, year, len(entries), tiles_str,
            )

        total_flagged += len(flagged)
        if args.dry_run:
            logger.info("[%s] [DRY RUN] would remove %d file(s)", source_key, len(flagged))
        else:
            for _year, _tile, path, *_rest in flagged:
                try:
                    os.remove(path)
                except Exception:
                    logger.exception("[%s] failed to remove %s", source_key, path)
                    total_failed += 1
            logger.info("[%s] removed %d file(s)", source_key, len(flagged))

    logger.info("Total: %d oversized file(s) found (failed=%d)", total_flagged, total_failed)
    if args.dry_run:
        logger.info("This was a DRY RUN -- pass --no-dry-run to actually delete.")
    else:
        logger.info(
            "Removed files are now missing on disk, so the next "
            "`data run --step fetch` (optionally --key <year>/<tile>) will re-fetch them "
            "with the fixed _tile_bbox_4326/_load_tile_year."
        )
    if total_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
