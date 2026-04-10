#!/usr/bin/env python3
"""
Archive MODIS daily HDF directories (YYYY/DOY) into one .tar.zst per DOY.

Example:
    python archive_doy_to_coldstore.py \
        --input-root /scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/glass/LST/MODIS/Daily/1KM/raw \
        --output-root /scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/glass/LST/MODIS/Daily/1KM/coldstore \
        --jobs 8 \
        --level 19 \
        --checksum

Requirements on the system PATH:
    - tar
    - zstd

Notes:
    - One archive is created per YYYY/DOY directory.
    - Only directories containing at least one .hdf file are archived.
    - Archives are written as: OUTPUT_ROOT/YYYY/DOY.tar.zst
    - A SHA-256 sidecar file can optionally be written for each archive.
    - Originals are NOT deleted unless --delete-source is passed.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class Task:
    year: str
    doy: str
    source_dir: Path
    archive_path: Path


@dataclass
class Result:
    task: Task
    success: bool
    skipped: bool
    message: str
    archive_size_bytes: int = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Archive YYYY/DOY directories containing HDF files into tar.zst cold-storage archives."
    )
    p.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing YYYY/DOY directories.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination root for tar.zst archives.",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 4) // 2),
        help="Number of parallel DOY archives to build.",
    )
    p.add_argument(
        "--level",
        type=int,
        default=19,
        help="zstd compression level (recommended for cold storage: 15-19; max is 22).",
    )
    p.add_argument(
        "--long-window-log",
        type=int,
        default=27,
        help="zstd long-range mode window log; set 0 to disable. Larger can improve ratio on big archives.",
    )
    p.add_argument(
        "--checksum",
        action="store_true",
        help="Write SHA-256 sidecar file next to each archive.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing archive if present.",
    )
    p.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete the source YYYY/DOY directory after successful archival.",
    )
    p.add_argument(
        "--min-hdf-count",
        type=int,
        default=1,
        help="Only archive a DOY directory if it contains at least this many .hdf files.",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to a log file.",
    )
    return p.parse_args()


def setup_logging(log_file: Path | None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def check_dependencies() -> None:
    for exe in ("tar", "zstd"):
        if shutil.which(exe) is None:
            raise RuntimeError(f"Required executable not found on PATH: {exe}")


def find_tasks(input_root: Path, output_root: Path, min_hdf_count: int) -> List[Task]:
    tasks: List[Task] = []

    if not input_root.is_dir():
        raise ValueError(f"Input root does not exist or is not a directory: {input_root}")

    for year_dir in sorted(p for p in input_root.iterdir() if p.is_dir()):
        year = year_dir.name
        if not year.isdigit() or len(year) != 4:
            continue

        for doy_dir in sorted(p for p in year_dir.iterdir() if p.is_dir()):
            doy = doy_dir.name
            if not doy.isdigit():
                continue

            hdf_count = sum(1 for _ in doy_dir.glob("*.hdf"))
            if hdf_count < min_hdf_count:
                continue

            archive_path = output_root / year / f"{doy}.tar.zst"
            tasks.append(Task(year=year, doy=doy, source_dir=doy_dir, archive_path=archive_path))

    return tasks


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_tar_command(source_dir: Path) -> List[str]:
    # Archive contents as relative paths under "."
    # We use POSIX/PAX format for broad metadata support.
    return [
        "tar",
        "--format=pax",
        "-C",
        str(source_dir),
        "-cf",
        "-",
        ".",
    ]


def build_zstd_command(archive_path: Path, level: int, long_window_log: int) -> List[str]:
    cmd = [
        "zstd",
        f"-{level}",
        "-T0",          # use all cores within each archive
        "--check",      # frame checksum
        "-o",
        str(archive_path),
    ]
    if long_window_log and long_window_log > 0:
        cmd.append(f"--long={long_window_log}")
    return cmd


def write_sha256_sidecar(path: Path) -> None:
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8")


def archive_one(task: Task, overwrite: bool, level: int, long_window_log: int,
                write_checksum: bool, delete_source: bool) -> Result:
    task.archive_path.parent.mkdir(parents=True, exist_ok=True)

    if task.archive_path.exists() and not overwrite:
        return Result(task=task, success=True, skipped=True, message="archive exists")

    tmp_archive = task.archive_path.with_name(task.archive_path.name + ".partial")

    if tmp_archive.exists():
        tmp_archive.unlink()

    tar_cmd = build_tar_command(task.source_dir)
    zstd_cmd = build_zstd_command(tmp_archive, level, long_window_log)

    logging.info("Archiving %s/%s -> %s", task.year, task.doy, task.archive_path)

    tar_proc = None
    zstd_proc = None

    try:
        tar_proc = subprocess.Popen(
            tar_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert tar_proc.stdout is not None

        zstd_proc = subprocess.Popen(
            zstd_cmd,
            stdin=tar_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Allow tar to receive SIGPIPE if zstd fails early.
        tar_proc.stdout.close()

        zstd_stdout, zstd_stderr = zstd_proc.communicate()
        tar_stderr = tar_proc.stderr.read().decode("utf-8", errors="replace")
        tar_return = tar_proc.wait()

        if tar_return != 0:
            if tmp_archive.exists():
                tmp_archive.unlink(missing_ok=True)
            return Result(
                task=task,
                success=False,
                skipped=False,
                message=f"tar failed with code {tar_return}: {tar_stderr.strip()}",
            )

        if zstd_proc.returncode != 0:
            if tmp_archive.exists():
                tmp_archive.unlink(missing_ok=True)
            zerr = zstd_stderr.decode("utf-8", errors="replace").strip()
            return Result(
                task=task,
                success=False,
                skipped=False,
                message=f"zstd failed with code {zstd_proc.returncode}: {zerr}",
            )

        tmp_archive.replace(task.archive_path)

        if write_checksum:
            write_sha256_sidecar(task.archive_path)

        archive_size = task.archive_path.stat().st_size

        if delete_source:
            shutil.rmtree(task.source_dir)

        return Result(
            task=task,
            success=True,
            skipped=False,
            message="ok",
            archive_size_bytes=archive_size,
        )

    except Exception as exc:
        if tmp_archive.exists():
            tmp_archive.unlink(missing_ok=True)
        return Result(
            task=task,
            success=False,
            skipped=False,
            message=f"exception: {exc}",
        )
    finally:
        # Defensive cleanup
        try:
            if tar_proc and tar_proc.poll() is None:
                tar_proc.kill()
        except Exception:
            pass
        try:
            if zstd_proc and zstd_proc.poll() is None:
                zstd_proc.kill()
        except Exception:
            pass


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def main() -> int:
    args = parse_args()
    setup_logging(args.log_file)

    try:
        check_dependencies()
    except Exception as exc:
        logging.error(str(exc))
        return 2

    tasks = find_tasks(
        input_root=args.input_root,
        output_root=args.output_root,
        min_hdf_count=args.min_hdf_count,
    )

    if not tasks:
        logging.warning("No matching YYYY/DOY directories with .hdf files found.")
        return 0

    logging.info("Found %d DOY directories to process.", len(tasks))
    logging.info("Compression level=%d, jobs=%d, long-window-log=%d",
                 args.level, args.jobs, args.long_window_log)

    results: List[Result] = []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(
                archive_one,
                task,
                args.overwrite,
                args.level,
                args.long_window_log,
                args.checksum,
                args.delete_source,
            )
            for task in tasks
        ]
        for fut in cf.as_completed(futures):
            res = fut.result()
            results.append(res)
            if res.success and res.skipped:
                logging.info("SKIP %s/%s: %s", res.task.year, res.task.doy, res.message)
            elif res.success:
                logging.info(
                    "DONE %s/%s: %s (%s)",
                    res.task.year,
                    res.task.doy,
                    res.message,
                    human_size(res.archive_size_bytes),
                )
            else:
                logging.error("FAIL %s/%s: %s", res.task.year, res.task.doy, res.message)

    ok = sum(1 for r in results if r.success and not r.skipped)
    skipped = sum(1 for r in results if r.success and r.skipped)
    failed = sum(1 for r in results if not r.success)

    logging.info("Summary: %d archived, %d skipped, %d failed", ok, skipped, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())