#!/usr/bin/env python3
"""Debug wrapper for the SNL mining Selenium scraper (src/data/sources/snl_mining/scraper).

Standalone tool, not pipeline-wired -- see the module docstring in
src/data/sources/snl_mining/source.py for why FETCH is declared absent for
this source. Invokes run_full_workflow() directly since the scraper has no
CLI of its own.

Usage examples:
  python scripts/debug_snl_mining_scraper.py full
  python scripts/debug_snl_mining_scraper.py ids
  python scripts/debug_snl_mining_scraper.py detail-exports
  python scripts/debug_snl_mining_scraper.py detail-parse
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.cli.common import setup_logging
from src.data.sources.snl_mining.scraper.config import DEFAULT_DB_PATH, DEFAULT_WAIT_SECONDS
from src.data.sources.snl_mining.scraper.workflow import Stage, run_full_workflow

logger = logging.getLogger(__name__)

_STEP_TO_STAGES = {
    "ids": [Stage.IDS],
    "collection": [Stage.IDS],
    "detail-exports": [Stage.DETAIL_EXPORTS],
    "scrape-exports": [Stage.DETAIL_EXPORTS],
    "detail-parse": [Stage.DETAIL_PARSE],
    "parse-exports": [Stage.DETAIL_PARSE],
    "full": None,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_credentials() -> Path:
    return _repo_root() / "orchestration" / "secrets" / "spglobal.credentials.json"


def _resolve_force_stages(args: argparse.Namespace) -> list[Stage] | None:
    forced = {Stage(stage.replace("-", "_")) for stage in (args.force_stages or [])}
    if args.redo_current_stage:
        current = _STEP_TO_STAGES[args.step]
        if current is not None:
            forced.update(current)
    forced.discard(Stage.IDS)
    return list(forced) or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug wrapper for the SNL mining scraper")
    parser.add_argument(
        "step",
        choices=list(_STEP_TO_STAGES),
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
        choices=["detail-exports", "detail-parse"],
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
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logging(args.log_level, debug=args.debug)

    mine_ids = [str(mine_id) for mine_id in args.mine_ids] if args.mine_ids else None

    logger.info("Starting SNL mining scraper | step=%s | headless=%s | db=%s", args.step, args.headless, args.db)

    results = run_full_workflow(
        credentials_path=args.credentials,
        db_path=args.db,
        stages=_STEP_TO_STAGES[args.step],
        headless=args.headless,
        wait=args.wait,
        download_wait=args.download_wait,
        mine_ids=mine_ids,
        subsections=args.subsections,
        max_attempts=args.max_attempts,
        sidebar_reload_attempts=args.sidebar_reload_attempts,
        continue_on_error=not args.fail_fast,
        force_stages=_resolve_force_stages(args),
        step_sleep_seconds=args.step_sleep_seconds,
    )
    logger.info("SNL mining scraper finished: %s", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
