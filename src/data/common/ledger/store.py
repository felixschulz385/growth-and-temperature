"""`SourceLedger`: the per-source DuckDB completion/transfer ledger.

docs/design/10-fetch-ledger.md. Replaces `UnifiedDataIndex` (Parquet, full
read+rewrite on every mutation -- src/data/common/index/unified_index.py)
and `TransferManifest` (a second, parallel Parquet format for the same
"is this thing already on HPC" question -- src/data/common/hpc/transfer.py)
with one incrementally-writable file per source.

Concurrency: DuckDB allows exactly one read-write connection to a given
`.duckdb` file at a time, across processes. This is safe here because a
given source+step is already only ever driven by one process at a time (a
human running `pipeline run`, or one SLURM job) -- but any future caller
adding a second concurrent writer against the same file (e.g. a background
`pipeline summary` refresher) would need its own connection strategy. Callers
that only ever read (`pipeline summary`, `pipeline plan`, `_check_requires`)
should open with `read_only=True`.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import duckdb

from src.data.common.ledger import schema
from src.data.common.ledger.paths import remote_ledger_path

logger = logging.getLogger(__name__)

_GLASS_DATE_PATTERN = re.compile(r"A(\d{4})(\d{3})")
_YEAR_PATTERN = re.compile(r"(\d{4})")


def _extract_year_day(relative_path: str) -> tuple[Optional[int], Optional[int]]:
    """Best-effort year/day-of-year extraction from a filename, feeding
    `remote_files.year`/`day_of_year` (used by year-scoped PREPARE planning
    queries). Ported from `UnifiedDataIndex._compute_derived_fields`'s same
    two-pattern fallback (GLASS's `A2000055`-style compound date first, then
    a bare 4-digit year) -- unlike that version, "no match" is `(None,
    None)`, not the old code's `(0, 0)` magic-number sentinel, since nothing
    in this new ledger depends on that sentinel.
    """
    filename = Path(relative_path).name
    match = _GLASS_DATE_PATTERN.search(filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = _YEAR_PATTERN.search(filename)
    if match:
        return int(match.group(1)), None
    return None, None


def entrypoint_key(entrypoint: dict[str, Any]) -> str:
    """Stable dedup key for an entrypoint dict. Matches the old
    `UnifiedDataIndex._find_missing_entrypoints`'s `f"{year}_{day}"` /
    `str(year)` key shape, so entrypoint identity is unchanged."""
    if "day" in entrypoint:
        return f"{entrypoint['year']}_{entrypoint['day']}"
    return str(entrypoint["year"])


@dataclass(frozen=True)
class FetchUnit:
    """One pending FETCH file, as returned by `pending_fetch()`."""

    file_hash: str
    relative_path: str
    source_url: str
    bytes: Optional[int]


@dataclass(frozen=True)
class DownloadResult:
    """One file's download outcome, fed to `record_download_batch()`."""

    file_hash: str
    ok: bool
    local_path: Optional[str] = None
    bytes: Optional[int] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class PushResult:
    """One unit's HPC-push outcome, fed to `record_push_batch()`. Used for
    both FETCH file-units and PREPARE/GRID step-units (`common/hpc/push.py`
    is the shared producer)."""

    step: str
    unit_id: str
    ok: bool
    bytes: Optional[int] = None
    error: Optional[str] = None


class SourceLedger:
    """One DuckDB file per source. Open via `SourceLedger.open(...)`; also
    usable as a context manager (`with SourceLedger.open(...) as ledger:`)."""

    def __init__(
        self, connection: "duckdb.DuckDBPyConnection", *, local_path: str, data_path: str, read_only: bool = False
    ):
        self._con = connection
        self.local_path = local_path
        self.data_path = data_path
        self.read_only = read_only

    @classmethod
    def open(cls, path: str, *, data_path: str, read_only: bool = False) -> "SourceLedger":
        """*path*: local `.duckdb` file path (see `paths.ledger_path()`).
        *data_path*: the source's `cfg.data_path`, needed to compute this
        ledger's remote counterpart path for `push_to_remote()`/
        `merge_from_remote()`."""
        if not read_only:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        con = duckdb.connect(path, read_only=read_only)
        ledger = cls(con, local_path=path, data_path=data_path, read_only=read_only)
        if not read_only:
            ledger._ensure_schema()
        return ledger

    def _ensure_schema(self) -> None:
        for statement in schema.ALL_DDL:
            self._con.execute(statement)

    def _execute_readonly_safe(self, query: str, params: list[Any]):
        """Every query a read-only-opened ledger runs goes through here.

        `open()`'s schema creation only runs `if not read_only` -- so a
        `.duckdb` file that exists on disk (passing every call site's own
        `os.path.exists()` guard) but was never opened read-write yet (or
        whose schema-creating open crashed before finishing) has no tables
        at all, and DuckDB can't run `CREATE TABLE` on a read-only
        connection to self-heal. Without this, that surfaces as a raw
        `duckdb.CatalogException` deep in unrelated source code (confirmed:
        crashed `plad`'s GRID step, and `esacci`'s PREPARE planning inside
        `pipeline summary`) instead of behaving like "no ledger yet," which
        every caller already handles gracefully via `os.path.exists()`.
        Returns `None` on a missing-schema catalog error so callers can
        return the same empty/absent result they'd give for a missing file;
        re-raises any other exception unchanged.
        """
        try:
            return self._con.execute(query, params)
        except duckdb.CatalogException:
            logger.warning(
                "Ledger %s has no schema yet (never opened read-write) -- treating as empty", self.local_path
            )
            return None

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> "SourceLedger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # catalog: entrypoints (FETCH-only, entrypoint-based sources)
    # ------------------------------------------------------------------

    def upsert_entrypoints(self, entrypoints: Iterable[dict[str, Any]]) -> int:
        """Register any entrypoints not already known. Returns the count
        newly inserted (existing ones are left untouched -- `crawled` state
        is only ever advanced by `mark_entrypoint_crawled`)."""
        entrypoints = list(entrypoints)
        if not entrypoints:
            return 0
        keys = [entrypoint_key(ep) for ep in entrypoints]
        existing = self._existing_keys("entrypoints", "entrypoint_key", keys)
        self._con.executemany(
            "INSERT INTO entrypoints (entrypoint_key, payload) VALUES (?, ?) "
            "ON CONFLICT (entrypoint_key) DO NOTHING",
            [(key, json.dumps(ep)) for key, ep in zip(keys, entrypoints)],
        )
        return len(entrypoints) - len(existing)

    def missing_entrypoints(self) -> list[dict[str, Any]]:
        rows = self._con.execute("SELECT payload FROM entrypoints WHERE crawled = false").fetchall()
        return [json.loads(r[0]) for r in rows]

    def mark_entrypoint_crawled(self, entrypoint: dict[str, Any]) -> None:
        self._con.execute(
            "UPDATE entrypoints SET crawled = true, crawled_at = now() WHERE entrypoint_key = ?",
            [entrypoint_key(entrypoint)],
        )

    def reset_crawl_state(self) -> None:
        """`pipeline index --rebuild`'s new meaning: force every entrypoint
        to be re-crawled on the next `catalog.refresh()` call. Deliberately
        does *not* clear `remote_files`/`artifacts` -- unlike the old
        `UnifiedDataIndex(rebuild=True)`, which wiped the whole Parquet file
        including tracked download/completion state, forcing a full
        re-download of everything even if already fetched. Here, a re-crawl
        just re-verifies against the origin; `add_remote_files()`'s
        `ON CONFLICT DO UPDATE last_seen_at` leaves already-tracked files'
        local/remote state untouched."""
        self._con.execute("UPDATE entrypoints SET crawled = false, crawled_at = NULL")

    # ------------------------------------------------------------------
    # catalog: remote files (FETCH-only crawl catalog)
    # ------------------------------------------------------------------

    def add_remote_files(
        self,
        files: Iterable[tuple[str, str]],
        *,
        get_file_hash: Callable[[str], str],
        entrypoint_key: Optional[str] = None,
    ) -> int:
        """*files*: iterable of `(relative_path, source_url)`, as returned by
        `RemoteFileCatalog.list_remote_files()`. Registers any not already
        known (by `file_hash`) into `remote_files`, and seeds a matching
        `artifacts(step='fetch')` row for each new file so `pending_fetch()`
        can find it. Existing files just get `last_seen_at` refreshed.
        Returns the count of genuinely new files.
        """
        files = list(files)
        if not files:
            return 0

        rows = []
        for relative_path, source_url in files:
            file_hash = get_file_hash(source_url)
            year, day_of_year = _extract_year_day(relative_path)
            rows.append((file_hash, relative_path, source_url, entrypoint_key, year, day_of_year))

        hashes = [r[0] for r in rows]
        existing = self._existing_keys("remote_files", "file_hash", hashes)

        self._con.executemany(
            """
            INSERT INTO remote_files
                (file_hash, relative_path, source_url, entrypoint_key, year, day_of_year)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (file_hash) DO UPDATE SET last_seen_at = now()
            """,
            rows,
        )
        self._con.executemany(
            "INSERT INTO artifacts (step, unit_id, source_url) VALUES ('fetch', ?, ?) "
            "ON CONFLICT (step, unit_id) DO NOTHING",
            [(r[0], r[2]) for r in rows],
        )
        return len(hashes) - len(existing)

    def _existing_keys(self, table: str, column: str, keys: list[str]) -> set[str]:
        if not keys:
            return set()
        placeholders = ",".join("?" * len(keys))
        rows = self._con.execute(f"SELECT {column} FROM {table} WHERE {column} IN ({placeholders})", keys).fetchall()
        return {r[0] for r in rows}

    def iter_remote_files(self) -> list[tuple[str, str]]:
        """All `(file_hash, relative_path)` pairs in the crawl catalog --
        used by bootstrap reconciliation (`common/ledger/bootstrap.py`) to
        match the catalog against a real HPC filesystem listing."""
        return self._con.execute("SELECT file_hash, relative_path FROM remote_files").fetchall()

    # ------------------------------------------------------------------
    # fetch driver
    # ------------------------------------------------------------------

    def pending_fetch(self, limit: int, *, max_attempts: int = 5) -> list[FetchUnit]:
        """FETCH units not yet HPC-verified, smallest first (matches the old
        `query_pending_files`' "smaller files first for faster initial
        progress" ordering). `remote_state != 'verified'` retries `missing`,
        `pushed` (never sample-verified), and `failed` alike -- but only up
        to *max_attempts* times each. The old system's equivalent `FAILED:`
        rows retried indefinitely within one run, bounded only by an outer
        "3 consecutive empty polls" loop -- which never fires for a
        permanently-broken file, since a `FAILED:` row is never `empty`. A
        real fetch driver call must still terminate on a permanently broken
        URL, so `pending_fetch` itself bounds retries; `run_fetch`'s loop
        then naturally ends once nothing is left both pending and under the
        attempt cap.
        """
        rows = self._con.execute(
            """
            SELECT rf.file_hash, rf.relative_path, rf.source_url, a.bytes
            FROM artifacts a
            JOIN remote_files rf ON rf.file_hash = a.unit_id
            WHERE a.step = 'fetch' AND a.remote_state != ? AND a.attempts < ?
            ORDER BY a.bytes ASC NULLS LAST
            LIMIT ?
            """,
            [schema.RemoteState.VERIFIED, max_attempts, limit],
        ).fetchall()
        return [FetchUnit(file_hash=r[0], relative_path=r[1], source_url=r[2], bytes=r[3]) for r in rows]

    def record_download_batch(self, results: Iterable[DownloadResult]) -> None:
        """One DuckDB transaction for the whole batch -- the direct fix for
        the old system's whole-file-parquet-rewrite-per-batch cost."""
        results = list(results)
        if not results:
            return
        self._con.executemany(
            """
            UPDATE artifacts
            SET local_path = ?, bytes = COALESCE(?, bytes), local_state = ?,
                last_error = ?, attempts = attempts + 1, updated_at = now()
            WHERE step = 'fetch' AND unit_id = ?
            """,
            [
                (
                    r.local_path,
                    r.bytes,
                    schema.LocalState.COMPLETE if r.ok else schema.LocalState.FAILED,
                    r.error,
                    r.file_hash,
                )
                for r in results
            ],
        )

    def record_push_batch(self, results: Iterable[PushResult]) -> None:
        """One DuckDB transaction for the whole batch. Shared by the FETCH
        driver and `pipeline transfer` (both push through `HPCPusher`)."""
        results = list(results)
        if not results:
            return
        self._con.executemany(
            """
            UPDATE artifacts
            SET remote_state = ?, bytes = COALESCE(?, bytes), last_error = ?,
                attempts = attempts + 1, updated_at = now()
            WHERE step = ? AND unit_id = ?
            """,
            [
                (
                    schema.RemoteState.VERIFIED if r.ok else schema.RemoteState.FAILED,
                    r.bytes,
                    r.error,
                    r.step,
                    r.unit_id,
                )
                for r in results
            ],
        )

    def completed_fetch_files(self, year: Optional[int] = None) -> list[str]:
        """`relative_path`s of FETCH files verified present on HPC -- the
        direct swap-in for the 6 sources' old `pd.read_parquet(index_file);
        filter status_category == 'completed'` `_plan_prepare()` block.
        `remote_state == 'verified'` alone (not local_state) matches the old
        system's semantics exactly: `status_category` there was only ever
        set to `'completed'` after HPC-side verification (see
        `AsyncHPCDownloader._update_file_statuses`), never on local download
        alone -- and FETCH's local staging copy is deleted right after a
        successful push (`common/hpc/push.py`), so by the time PREPARE runs
        (on a different machine/SLURM job), only the remote copy still
        exists.

        Ordered by `discovered_at` (insertion order) -- `ntl_harm`'s
        `_plan_prepare` deliberately preserves "plain-dict insertion order,
        not sorted-by-year" as a ported quirk from the old Parquet-row-order
        behavior, so this can't be an unordered `SELECT`.
        """
        query = """
            SELECT rf.relative_path
            FROM artifacts a
            JOIN remote_files rf ON rf.file_hash = a.unit_id
            WHERE a.step = 'fetch' AND a.remote_state = ?
        """
        params: list[Any] = [schema.RemoteState.VERIFIED]
        if year is not None:
            query += " AND rf.year = ?"
            params.append(year)
        query += " ORDER BY rf.discovered_at"
        result = self._execute_readonly_safe(query, params)
        if result is None:
            return []
        return [r[0] for r in result.fetchall()]

    # ------------------------------------------------------------------
    # universal artifact tracking (FETCH file-units and PREPARE/GRID
    # step-units alike)
    # ------------------------------------------------------------------

    def ensure_artifact(
        self, step: str, unit_id: str, *, local_path: Optional[str] = None, remote_path: Optional[str] = None
    ) -> None:
        """Idempotently ensure a tracking row exists for `(step, unit_id)`.
        FETCH units are seeded via `add_remote_files()`; PREPARE/GRID units
        (and bootstrap, and `pipeline transfer`) use this directly."""
        self._con.execute(
            """
            INSERT INTO artifacts (step, unit_id, local_path, remote_path)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (step, unit_id) DO UPDATE SET
                local_path = COALESCE(excluded.local_path, artifacts.local_path),
                remote_path = COALESCE(excluded.remote_path, artifacts.remote_path)
            """,
            [step, unit_id, local_path, remote_path],
        )

    def set_local_state(self, step: str, unit_id: str, state: str, *, size_bytes: Optional[int] = None) -> None:
        self._con.execute(
            "UPDATE artifacts SET local_state = ?, bytes = COALESCE(?, bytes), updated_at = now() "
            "WHERE step = ? AND unit_id = ?",
            [state, size_bytes, step, unit_id],
        )

    def set_remote_state(self, step: str, unit_id: str, state: str) -> None:
        self._con.execute(
            "UPDATE artifacts SET remote_state = ?, updated_at = now() WHERE step = ? AND unit_id = ?",
            [state, step, unit_id],
        )

    def local_state(self, step: str, unit_id: str) -> Optional[str]:
        row = self._con.execute(
            "SELECT local_state FROM artifacts WHERE step = ? AND unit_id = ?", [step, unit_id]
        ).fetchone()
        return row[0] if row else None

    def remote_state(self, step: str, unit_id: str) -> Optional[str]:
        result = self._execute_readonly_safe(
            "SELECT remote_state FROM artifacts WHERE step = ? AND unit_id = ?", [step, unit_id]
        )
        if result is None:
            return None
        row = result.fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # summary / requires
    # ------------------------------------------------------------------

    def stats(self, step: str) -> dict[str, int]:
        """One aggregate query -- replaces `get_stats()`/`get_download_stats()`
        (each of which read a whole Parquet column) and the `os.walk()`
        file-counting `pipeline summary` previously did for FETCH's
        `Completion.NEVER` pseudo-target."""
        row = self._con.execute(
            """
            SELECT
                count(*),
                sum(CASE WHEN local_state = ? THEN 1 ELSE 0 END),
                sum(CASE WHEN remote_state = ? THEN 1 ELSE 0 END),
                sum(CASE WHEN local_state = ? OR remote_state = ? THEN 1 ELSE 0 END),
                sum(coalesce(bytes, 0))
            FROM artifacts WHERE step = ?
            """,
            [
                schema.LocalState.COMPLETE,
                schema.RemoteState.VERIFIED,
                schema.LocalState.FAILED,
                schema.RemoteState.FAILED,
                step,
            ],
        ).fetchone()
        total, local_complete, remote_verified, failed, total_bytes = row
        return {
            "total": total or 0,
            "local_complete": local_complete or 0,
            "remote_verified": remote_verified or 0,
            "failed": failed or 0,
            "total_bytes": total_bytes or 0,
        }

    def step_complete(self, step: str) -> bool:
        """Used by `_check_requires()`: does *any* unit of *step* count as
        done? Deliberately coarse (any row locally complete or remote-
        verified) -- precise enough for a prerequisite gate, which only
        needs "has this step produced anything," not per-target detail."""
        result = self._execute_readonly_safe(
            "SELECT count(*) FROM artifacts WHERE step = ? AND (local_state = ? OR remote_state = ?)",
            [step, schema.LocalState.COMPLETE, schema.RemoteState.VERIFIED],
        )
        if result is None:
            return False
        row = result.fetchone()
        return bool(row[0])

    # ------------------------------------------------------------------
    # HPC sync -- the ledger file itself, not its contents' targets
    # ------------------------------------------------------------------

    def push_to_remote(self, client: "Any") -> bool:
        """rsync this ledger's local `.duckdb` file to the HPC target.
        Single-file rsync, same mechanism `UnifiedDataIndex.sync_index_with_hpc`
        used -- just called at a caller-controlled cadence (the FETCH driver
        pushes every N batches + once at shutdown, not after every batch)."""
        remote_path = remote_ledger_path(self.data_path)
        remote_dir = os.path.dirname(remote_path)
        if remote_dir:
            client.ensure_directory(remote_dir)
        success, summary = client.rsync_transfer(
            self.local_path,
            remote_path,
            source_is_local=True,
            options={"compress": True, "archive": True, "partial": True, "checksum": True, "verbose": False},
            show_progress=False,
        )
        if not success:
            logger.warning("Failed to push ledger to HPC: %s", summary)
        return success

    def merge_from_remote(self, client: "Any", tmp_dir: str) -> bool:
        """Pull the remote copy of this ledger (if one exists yet) and merge
        its rows into the local copy, newest `updated_at` wins per
        `(step, unit_id)`. Best-effort and non-fatal: a source's very first
        fetch has no remote copy yet, which is a no-op here, not an error.
        """
        remote_path = remote_ledger_path(self.data_path)
        full_remote_path = f"{client.base_path}/{remote_path}" if getattr(client, "base_path", None) else remote_path
        if not client.check_file_exists(full_remote_path):
            logger.debug("No remote ledger yet at %s -- nothing to merge", full_remote_path)
            return True

        os.makedirs(tmp_dir, exist_ok=True)
        local_tmp = os.path.join(tmp_dir, "remote_ledger.duckdb")
        success, summary = client.rsync_transfer(
            remote_path,
            local_tmp,
            source_is_local=False,
            options={"compress": True, "archive": True, "partial": True, "checksum": True, "verbose": False},
            show_progress=False,
        )
        if not success:
            logger.warning("Failed to pull remote ledger for merge: %s", summary)
            return False

        try:
            escaped_path = local_tmp.replace("'", "''")
            self._con.execute(f"ATTACH '{escaped_path}' AS remote_ledger (READ_ONLY)")
            try:
                self._con.execute(
                    "INSERT INTO remote_files SELECT * FROM remote_ledger.remote_files "
                    "ON CONFLICT (file_hash) DO UPDATE SET "
                    "last_seen_at = greatest(excluded.last_seen_at, remote_files.last_seen_at)"
                )
                self._con.execute(
                    "INSERT INTO entrypoints SELECT * FROM remote_ledger.entrypoints "
                    "ON CONFLICT (entrypoint_key) DO UPDATE SET "
                    "crawled = entrypoints.crawled OR excluded.crawled, "
                    "crawled_at = greatest(coalesce(excluded.crawled_at, entrypoints.crawled_at), "
                    "coalesce(entrypoints.crawled_at, excluded.crawled_at))"
                )
                self._con.execute(
                    "INSERT INTO artifacts SELECT * FROM remote_ledger.artifacts "
                    "ON CONFLICT (step, unit_id) DO UPDATE SET "
                    "local_path = CASE WHEN excluded.updated_at > artifacts.updated_at "
                    "THEN excluded.local_path ELSE artifacts.local_path END, "
                    "remote_path = CASE WHEN excluded.updated_at > artifacts.updated_at "
                    "THEN excluded.remote_path ELSE artifacts.remote_path END, "
                    "local_state = CASE WHEN excluded.updated_at > artifacts.updated_at "
                    "THEN excluded.local_state ELSE artifacts.local_state END, "
                    "remote_state = CASE WHEN excluded.updated_at > artifacts.updated_at "
                    "THEN excluded.remote_state ELSE artifacts.remote_state END, "
                    "bytes = CASE WHEN excluded.updated_at > artifacts.updated_at "
                    "THEN excluded.bytes ELSE artifacts.bytes END, "
                    "attempts = greatest(excluded.attempts, artifacts.attempts), "
                    "updated_at = greatest(excluded.updated_at, artifacts.updated_at)"
                )
            finally:
                self._con.execute("DETACH remote_ledger")
        except Exception:
            logger.exception("Error merging remote ledger")
            return False
        finally:
            try:
                os.remove(local_tmp)
            except OSError:
                pass
        return True
