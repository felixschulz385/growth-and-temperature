#!/usr/bin/env python3
"""CLI wrapper for the SNL mining LLM year-imputation batch job
(src/data/sources/snl_mining/imputation.py).

Standalone tool, not pipeline-wired -- see the module docstring in
src/data/sources/snl_mining/source.py for why FETCH is declared absent for
this source, and imputation.py's module docstring for why this is a script
rather than a `data run --step` candidate (a genuinely async, hours-long
external OpenAI Batch API call; PipelineStep only has FETCH/PREPARE/GRID).
Mirrors scripts/debug_snl_mining_scraper.py's shape.

Usage examples:
  python scripts/run_snl_mining_imputation.py probe
  python scripts/run_snl_mining_imputation.py run
  python scripts/run_snl_mining_imputation.py run --watch --max-cycles 0
  python scripts/run_snl_mining_imputation.py run --overwrite
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli.common import setup_logging  # noqa: E402
from src.data.sources.snl_mining.imputation import (  # noqa: E402
    DEFAULT_CSV_DIR,
    DEFAULT_OUTPUT_DIR,
    MineYearBatchEngine,
    load_fused_property_texts,
    load_properties,
    run_imputation,
)
from src.data.sources.snl_mining.scraper.config import DEFAULT_DB_PATH  # noqa: E402

logger = logging.getLogger(__name__)


def _default_credentials() -> Path:
    return PROJECT_ROOT / "orchestration" / "secrets" / "openai_mine_opening.txt"


def _run_probe(args: argparse.Namespace) -> int:
    import duckdb
    from openai import OpenAI

    api_key = Path(args.credentials).read_text().strip()
    client = OpenAI(api_key=api_key, timeout=900.0)
    engine = MineYearBatchEngine(client=client, model=args.model, project_root=Path(args.db).parent, live_service_tier=args.service_tier)

    with duckdb.connect(str(args.db), read_only=True) as con:
        property_texts = load_fused_property_texts(con)
        properties = load_properties(con)
    candidates, summary = engine.prepare_candidates(property_texts=property_texts, properties=properties)
    logger.info("Candidate summary: %s", summary.to_dict(orient="records")[0])

    probe = engine.select_probe(candidates, preferred_property_id=args.property_id)
    if probe.empty:
        logger.error("No candidate rows available for a probe.")
        return 1
    row = probe.iloc[0]
    result, usage, _ = engine.query_one_live(property_id=str(row["property_id"]), work_history_text=row["raw_text"])
    logger.info("Probe result for property_id=%s: %s (usage=%s)", row["property_id"], result, usage)
    return 0


def _run_full(args: argparse.Namespace) -> int:
    # --watch or --max-cycles 0 both mean "block and poll until the queue
    # drains" (run_periodic_batch_monitor's max_cycles=None); otherwise use
    # the given bounded cycle count (default 1: a single non-blocking pass).
    max_cycles = None if (args.watch or args.max_cycles == 0) else args.max_cycles
    manifest = run_imputation(
        db_path=Path(args.db),
        credentials_path=Path(args.credentials),
        output_dir=Path(args.output_dir),
        csv_dir=Path(args.csv_dir),
        model=args.model,
        service_tier=args.service_tier,
        poll_interval_seconds=args.poll_interval,
        max_cycles=max_cycles,
        overwrite_manifest=args.overwrite,
    )
    logger.info(
        "snl_mining imputation manifest: %d chunk(s), statuses=%s",
        len(manifest), manifest["manifest_status"].value_counts().to_dict(),
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="CLI wrapper for the SNL mining LLM year-imputation batch job")
    parser.add_argument("step", choices=["probe", "run"], help="probe: one live sanity-check request; run: manifest + submit/ingest pass")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--credentials", type=Path, default=_default_credentials())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--service-tier", default="flex")
    parser.add_argument("--property-id", default=None, help="probe: preferred property_id to test (default: first candidate)")
    parser.add_argument("--poll-interval", type=int, default=300, help="run --watch: seconds between polling cycles")
    parser.add_argument(
        "--max-cycles", type=int, default=1,
        help="run: polling cycles before returning (default 1, a single non-blocking pass -- 0 or --watch means unbounded)",
    )
    parser.add_argument("--watch", action="store_true", help="run: block and poll until the whole queue drains (equivalent to --max-cycles 0)")
    parser.add_argument("--overwrite", action="store_true", help="run: rebuild the manifest/request files from scratch instead of loading the existing one")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()
    setup_logging(args.log_level, debug=args.debug)

    if args.step == "probe":
        return _run_probe(args)
    return _run_full(args)


if __name__ == "__main__":
    raise SystemExit(main())
