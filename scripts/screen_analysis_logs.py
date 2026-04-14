#!/usr/bin/env python3
"""Screen DuckReg analysis logs for dataset, model, and runtime summaries."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_ROOT = PROJECT_ROOT / "log" / "analysis"
DEFAULT_VERSION = "0.4.2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "analysis" / "log_screen"

ISO_EVENT_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2})?)\]\s+"
    r"(?P<event>Running model \d+/\d+:|Completed:)\s+(?P<model>\S+)"
)
LOG_TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")
DATA_SOURCE_RE = re.compile(r"^Data source:\s+(?P<data_source>.+?)\s*$")
DESCRIPTION_RE = re.compile(r"^Description:\s+(?P<description>.+?)\s*$")
FORMULA_RE = re.compile(r"^Formula:\s+(?P<formula>.+?)\s*$")
SUCCESS_RE = re.compile(r"(Analysis complete|Operation completed successfully)")
FAILED_RE = re.compile(
    r"(Traceback|\s-\sERROR\s-|\bFAILED\b|\bCANCELLED\b|TIME LIMIT|^Error:)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract dataset/model/runtime records from log/analysis/**/<version>/*.log."
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help=f"Analysis log root (default: {DEFAULT_LOG_ROOT})",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"DuckReg/log version directory to screen (default: {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / DEFAULT_VERSION,
        help="Directory for extracted CSV/JSON files.",
    )
    return parser.parse_args()


def parse_iso_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def parse_log_timestamp(line: str) -> datetime | None:
    match = LOG_TS_RE.match(line)
    if not match:
        return None
    return datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S")


def seconds_between(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return max((end - start).total_seconds(), 0.0)


def duration_hms(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def dataset_key(data_source: str) -> str:
    if not data_source:
        return ""
    path = Path(data_source.strip())
    return path.stem if path.suffix == ".parquet" else data_source.strip()


def infer_method(description: str, formula: str, model: str) -> str:
    joined = " ".join([description, formula, model]).upper()
    if " IV " in f" {joined} " or "REGFAV" in joined or " | (" in formula:
        return "IV"
    return "OLS"


def slurm_id_from_path(path: Path) -> str:
    match = re.search(r"slurm-(\d+)\.log$", path.name)
    return match.group(1) if match else ""


def read_lines(path: Path) -> list[str]:
    return path.read_text(errors="replace").splitlines()


def collect_table_runtimes(log_paths: Iterable[Path]) -> dict[tuple[str, str], dict[str, object]]:
    runtimes: dict[tuple[str, str], dict[str, object]] = {}

    for path in log_paths:
        if not path.parent.parent.name.startswith("table-"):
            continue

        slurm_id = slurm_id_from_path(path)
        active: dict[str, datetime] = {}

        for line in read_lines(path):
            match = ISO_EVENT_RE.match(line)
            if not match:
                continue

            ts = parse_iso_timestamp(match.group("ts"))
            event = match.group("event")
            model = match.group("model")

            if event.startswith("Running model"):
                active[model] = ts
            elif event.startswith("Completed"):
                start = active.pop(model, None)
                runtime = seconds_between(start, ts)
                if runtime is not None:
                    runtimes[(slurm_id, model)] = {
                        "runtime_seconds": runtime,
                        "started_at": start.isoformat() if start else "",
                        "completed_at": ts.isoformat(),
                        "runtime_source": "table_log",
                        "table_log_path": str(path),
                    }

    return runtimes


def extract_model_record(
    path: Path,
    table_runtimes: dict[tuple[str, str], dict[str, object]],
) -> dict[str, object] | None:
    model = path.parent.parent.name
    if model.startswith("table-"):
        return None

    slurm_id = slurm_id_from_path(path)
    lines = read_lines(path)

    first_ts: datetime | None = None
    last_ts: datetime | None = None
    data_source = ""
    description = ""
    formula = ""
    success = False
    failed = False

    for line in lines:
        ts = parse_log_timestamp(line)
        if ts is not None:
            first_ts = first_ts or ts
            last_ts = ts

        if not data_source:
            match = DATA_SOURCE_RE.match(line)
            if match:
                data_source = match.group("data_source").strip()

        if not description:
            match = DESCRIPTION_RE.match(line)
            if match:
                description = match.group("description").strip()

        if not formula:
            match = FORMULA_RE.match(line)
            if match:
                formula = match.group("formula").strip()

        success = success or bool(SUCCESS_RE.search(line))
        failed = failed or bool(FAILED_RE.search(line))

    table_runtime = table_runtimes.get((slurm_id, model))
    if table_runtime:
        runtime_seconds = table_runtime["runtime_seconds"]
        started_at = table_runtime["started_at"]
        completed_at = table_runtime["completed_at"]
        runtime_source = table_runtime["runtime_source"]
    else:
        runtime_seconds = seconds_between(first_ts, last_ts)
        started_at = first_ts.isoformat() if first_ts else ""
        completed_at = last_ts.isoformat() if last_ts else ""
        runtime_source = "model_log_timestamps" if runtime_seconds is not None else ""

    return {
        "model": model,
        "dataset": dataset_key(data_source),
        "data_source": data_source,
        "method": infer_method(description, formula, model),
        "runtime_seconds": runtime_seconds,
        "runtime_hms": duration_hms(runtime_seconds),
        "runtime_source": runtime_source,
        "started_at": started_at,
        "completed_at": completed_at,
        "success": success,
        "failed": failed,
        "slurm_id": slurm_id,
        "log_path": str(path),
        "description": description,
        "formula": formula,
    }


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = fraction * (len(ordered) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    weight = idx - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def summarize_group(rows: list[dict[str, object]], group_keys: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)

    summaries = []
    for key_values, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        values = [
            float(row["runtime_seconds"])
            for row in group_rows
            if row.get("runtime_seconds") not in (None, "")
        ]
        summary = {key: value for key, value in zip(group_keys, key_values)}
        summary.update(summary_stats(values))
        summary["success_count"] = sum(1 for row in group_rows if row.get("success"))
        summary["failed_count"] = sum(1 for row in group_rows if row.get("failed"))
        summaries.append(summary)

    return summaries


def summary_stats(values: list[float]) -> dict[str, object]:
    if not values:
        return {
            "n": 0,
            "total_seconds": "",
            "total_hms": "",
            "mean_seconds": "",
            "mean_hms": "",
            "median_seconds": "",
            "median_hms": "",
            "p25_seconds": "",
            "p75_seconds": "",
            "min_seconds": "",
            "max_seconds": "",
            "min_hms": "",
            "max_hms": "",
        }

    total = sum(values)
    mean = statistics.mean(values)
    median = statistics.median(values)
    p25 = quantile(values, 0.25)
    p75 = quantile(values, 0.75)
    minimum = min(values)
    maximum = max(values)
    return {
        "n": len(values),
        "total_seconds": round(total, 3),
        "total_hms": duration_hms(total),
        "mean_seconds": round(mean, 3),
        "mean_hms": duration_hms(mean),
        "median_seconds": round(median, 3),
        "median_hms": duration_hms(median),
        "p25_seconds": round(p25, 3),
        "p75_seconds": round(p75, 3),
        "min_seconds": round(minimum, 3),
        "max_seconds": round(maximum, 3),
        "min_hms": duration_hms(minimum),
        "max_hms": duration_hms(maximum),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    log_paths = sorted(args.log_root.glob(f"**/{args.version}/*.log"))
    table_runtimes = collect_table_runtimes(log_paths)
    records = [
        record
        for path in log_paths
        if (record := extract_model_record(path, table_runtimes)) is not None
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(args.output_dir / "analysis_log_records.csv", records)
    write_csv(args.output_dir / "summary_by_dataset.csv", summarize_group(records, ("dataset",)))
    write_csv(args.output_dir / "summary_by_method.csv", summarize_group(records, ("method",)))
    write_csv(
        args.output_dir / "summary_by_dataset_method.csv",
        summarize_group(records, ("dataset", "method")),
    )
    write_csv(args.output_dir / "summary_by_model.csv", summarize_group(records, ("model",)))

    runtime_values = [
        float(row["runtime_seconds"])
        for row in records
        if row.get("runtime_seconds") not in (None, "")
    ]
    overall = {
        "log_root": str(args.log_root),
        "version": args.version,
        "log_files_found": len(log_paths),
        "model_records": len(records),
        "records_with_runtime": len(runtime_values),
        "records_with_data_source": sum(1 for row in records if row.get("data_source")),
        "success_count": sum(1 for row in records if row.get("success")),
        "failed_count": sum(1 for row in records if row.get("failed")),
        **summary_stats(runtime_values),
    }
    (args.output_dir / "summary_overall.json").write_text(json.dumps(overall, indent=2))

    print(f"Wrote {len(records)} model-log records to {args.output_dir}")
    print(f"Logs screened: {len(log_paths)}")
    print(f"Records with runtime: {overall['records_with_runtime']}")
    print(f"Records with data source: {overall['records_with_data_source']}")
    print(f"Overall mean runtime: {overall['mean_hms']}")
    print(f"Overall median runtime: {overall['median_hms']}")
    print(f"Overall max runtime: {overall['max_hms']}")
    print("Files:")
    for filename in (
        "analysis_log_records.csv",
        "summary_by_dataset.csv",
        "summary_by_method.csv",
        "summary_by_dataset_method.csv",
        "summary_by_model.csv",
        "summary_overall.json",
    ):
        print(f"  {args.output_dir / filename}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
