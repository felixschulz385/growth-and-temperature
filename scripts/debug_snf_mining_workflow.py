#!/usr/bin/env python3
"""Debug wrapper around ``gnt download snf-mining``.

Usage examples:
  python scripts/debug_snf_mining_workflow.py full
  python scripts/debug_snf_mining_workflow.py ids
  python scripts/debug_snf_mining_workflow.py detail-exports
  python scripts/debug_snf_mining_workflow.py detail-parse
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gnt.cli.main import main as cli_main
from gnt.data.download.sources.snf_mining.config import DEFAULT_DB_PATH, DEFAULT_WAIT_SECONDS

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_credentials() -> Path:
    return _repo_root() / "orchestration" / "secrets" / "spglobal.credentials.json"


def _build_cli_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "download",
        "snf-mining",
        "--credentials",
        str(args.credentials),
        "--db",
        str(args.db),
        "--wait",
        str(args.wait),
        "--download-wait",
        str(args.download_wait),
        "--log-level",
        args.log_level,
    ]

    if args.debug:
        argv.append("--debug")
    if args.headless:
        argv.append("--headless")
    if args.mine_ids:
        argv.extend(["--mine-ids", *[str(mine_id) for mine_id in args.mine_ids]])
    if args.subsections:
        argv.extend(["--subsections", *args.subsections])
    if args.fail_fast:
        argv.append("--fail-fast")
    if args.max_attempts != 3:
        argv.extend(["--max-attempts", str(args.max_attempts)])
    if args.sidebar_reload_attempts != 2:
        argv.extend(["--sidebar-reload-attempts", str(args.sidebar_reload_attempts)])
    if args.step_sleep_seconds != 0.35:
        argv.extend(["--step-sleep-seconds", str(args.step_sleep_seconds)])

    stages = _resolve_requested_stages(args.step)
    if stages is not None:
        argv.extend(["--stages", *stages])

    force_stages = _resolve_force_stages(args)
    if force_stages:
        argv.extend(["--force-stages", *force_stages])

    return argv


def _resolve_requested_stages(step: str) -> list[str] | None:
    mapping = {
        "ids": ["ids"],
        "collection": ["ids"],
        "detail-exports": ["detail_exports"],
        "scrape-exports": ["detail_exports"],
        "detail-parse": ["detail_parse"],
        "parse-exports": ["detail_parse"],
        "full": None,
    }
    return mapping[step]


def _resolve_force_stages(args: argparse.Namespace) -> list[str]:
    forced = list(args.force_stages or [])
    if args.redo_current_stage:
        current = _resolve_requested_stages(args.step)
        if current is not None:
            forced.extend(current)
    deduped: list[str] = []
    for stage in forced:
        normalized = {
            "detail-exports": "detail_exports",
            "scrape-exports": "detail_exports",
            "detail-parse": "detail_parse",
            "parse-exports": "detail_parse",
        }.get(stage, stage)
        if normalized == "ids":
            continue
        if normalized not in deduped:
            deduped.append(normalized)
    return deduped


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug wrapper for SNF mining CLI runs")
    parser.add_argument(
        "step",
        choices=[
            "ids",
            "collection",
            "detail-exports",
            "scrape-exports",
            "detail-parse",
            "parse-exports",
            "full",
        ],
        help="Debug step alias to run",
    )
    parser.add_argument("--credentials", type=Path, default=_default_credentials())
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--wait", type=int, default=DEFAULT_WAIT_SECONDS)
    parser.add_argument("--download-wait", type=int, default=90)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mine-ids", nargs="*", default=None)
    parser.add_argument(
        "--subsections",
        nargs="+",
        default=None,
        help="Restrict detail export/parse stages to specific subsection labels",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--sidebar-reload-attempts", type=int, default=2)
    parser.add_argument("--step-sleep-seconds", type=float, default=0.35)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--force-stages",
        nargs="*",
        default=None,
        choices=["detail_exports", "detail_parse", "detail-exports", "detail-parse"],
        help="Stage(s) to clear and rerun completely",
    )
    parser.add_argument(
        "--redo-current-stage",
        action="store_true",
        help="Force a clean rerun of the selected debug step",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    parser.add_argument("--debug", action="store_true", help="Enable CLI debug logging")

    args = parser.parse_args()
    argv = _build_cli_argv(args)

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
