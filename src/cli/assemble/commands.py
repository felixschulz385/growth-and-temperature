"""
Argument registration for the ``assemble`` domain.

Sub-commands
------------
create   Build the assembled panel for one grid resolution (+ optional grid-shake
         variants). Always merges every source in ``assembly.sources``.
update   Add or refresh one source in an already-built assembled table.

There are no named assemblies -- the panel always contains every configured
source; ``--grid`` picks the output resolution and ``--shake`` the grid-origin
robustness variants.
"""

from __future__ import annotations

import argparse

from src.cli.common import add_logging_args
from src.data.common.dask.client import DEFAULT_DASHBOARD_PORT
from src.data.assemble.constants import DEFAULT_GRID_LABEL, GRID_RESOLUTIONS_M


def register(top_subparsers: argparse._SubParsersAction) -> None:
    """Register ``assemble`` and its sub-commands on *top_subparsers*."""
    from .handlers import handle_create, handle_update

    assemble_parser = top_subparsers.add_parser(
        "assemble",
        help="Assemble or update the panel dataset",
        description="Build or maintain the assembled panel dataset.",
    )
    add_logging_args(assemble_parser)
    sub = assemble_parser.add_subparsers(
        dest="assemble_cmd",
        metavar="COMMAND",
    )
    sub.required = True

    # ── create ─────────────────────────────────────────────────────────────
    create_p = sub.add_parser(
        "create",
        help="Build the assembled panel for one grid (+ optional shake variants)",
        description=(
            "Merge every source in assembly.sources onto the requested grid and "
            "write tile-partitioned parquet under "
            "<output_root>/grid=<label>/shake=<label>/."
        ),
    )
    _add_assemble_common(create_p)
    create_p.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="Overwrite existing tiles (default: True)",
    )
    create_p.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Skip existing tiles instead of overwriting",
    )
    create_p.set_defaults(func=handle_create)

    # ── update ─────────────────────────────────────────────────────────────
    update_p = sub.add_parser(
        "update",
        help="Add or refresh one source in an existing assembled table",
        description="Recompute one source's columns in an already-built table.",
    )
    _add_assemble_common(update_p)
    update_p.add_argument(
        "--datasource",
        required=True,
        help="Source name (key in assembly.sources) to refresh",
    )
    update_p.set_defaults(func=handle_update)


# ---------------------------------------------------------------------------
# Shared argument helpers
# ---------------------------------------------------------------------------

def _add_assemble_common(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by create and update sub-commands."""
    add_logging_args(parser)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to unified configuration file (YAML or JSON)",
    )
    parser.add_argument(
        "--grid",
        default=DEFAULT_GRID_LABEL,
        choices=sorted(GRID_RESOLUTIONS_M),
        help=(
            f"Output grid resolution label (default: {DEFAULT_GRID_LABEL} -- the "
            "native canonical resolution). Coarser labels require pipeline.grid=ease6933."
        ),
    )
    parser.add_argument(
        "--shake",
        default="none",
        help=(
            "Grid-shake robustness variants: 'none' (default, writes shake=base "
            "only), a preset ('quad' -> base + s0/s1/s2), or a single 's<N>' "
            "offset label (writes only that partition). Only meaningful when --grid "
            "coarsens below the native resolution."
        ),
    )
    # DuckDB engine resource knobs (one process per grid=/shake= variant).
    parser.add_argument("--threads", type=int, help="DuckDB threads (default: all cores)")
    parser.add_argument(
        "--memory-limit",
        help='DuckDB memory limit for the whole process (e.g. "200GB")',
    )
    parser.add_argument("--temp-dir", help="DuckDB spill directory (large aggregations/joins)")
    # Deprecated Dask aliases -- accepted so existing scripts keep working.
    parser.add_argument("--dask-threads", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--dask-memory-limit", help=argparse.SUPPRESS)
    parser.add_argument("--dashboard-port", type=int, default=DEFAULT_DASHBOARD_PORT,
                        help=argparse.SUPPRESS)
    parser.add_argument("--local-directory", help=argparse.SUPPRESS)
    parser.add_argument("--tile-size", type=int, help="Tile size override")
    parser.add_argument("--compression", help="Parquet compression format override")

    # SLURM submission (src/cli/assemble/slurm.py) -- resource defaults come from
    # orchestration/configs/slurm_jobs.yaml's `assembly_jobs:` block, overridable
    # per invocation below.
    parser.add_argument(
        "--slurm", action="store_true",
        help="Submit as a SLURM job (sbatch) instead of running locally",
    )
    parser.add_argument("--slurm-time", help="Override this job's SLURM --time")
    parser.add_argument("--slurm-mem", help="Override this job's SLURM --mem")
    parser.add_argument("--slurm-cpus", type=int, help="Override this job's SLURM --cpus-per-task")
    parser.add_argument("--slurm-qos", help="Override this job's SLURM --qos")
    parser.add_argument("--slurm-partition", help="Override this job's SLURM --partition")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="With --slurm, print the sbatch command instead of submitting",
    )
