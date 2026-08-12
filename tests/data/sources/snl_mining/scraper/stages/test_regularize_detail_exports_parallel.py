"""Tests for the parallelized `detail_regularize` stage: the parse+classify
step (`_regularize_worker.process_one_row`) runs across a
`ProcessPoolExecutor` when `max_workers > 1` (CPU-bound pure-Python XML
parsing -- threads gave no speedup, see the module docstring in
`regularize_detail_exports.py`), with DuckDB writes kept serial on the main
process.

Most tests here use `max_workers=1` (the in-process, no-pool path) with
`_regularize_worker.parse_subsection_xls`/`classify_and_regularize`
monkeypatched -- a spawned worker *process* re-imports that module fresh, so
a monkeypatch applied in the test process is invisible to it; `max_workers=1`
is what lets these tests exercise the orchestration/aggregation logic
without real .xlsx fixture files (per `_builders.py`'s own convention: the
parsing layer itself is exercised elsewhere, by `parsing/xls.py`'s own
tests). `test_process_pool_path_...` below separately exercises the real
`ProcessPoolExecutor` path end-to-end (no monkeypatching, since it can't
apply across the process boundary) using a case that doesn't need real
xlsx content: a missing file.

Each fake subsection type below writes to its own table name (mirroring how
every *real* subsection_label maps to its own dedicated `detail_*` table) --
sharing one table name across two subsections of the same mine would trip
`upsert_regularized_rows`' delete-then-insert-by-mine_id idiom and clobber
one row with the other, which is a test-fixture concern, not a product bug.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.data.sources.snl_mining.scraper.stages import _regularize_worker as worker
from src.data.sources.snl_mining.scraper.stages import regularize_detail_exports as stage
from src.data.sources.snl_mining.scraper.storage.database import get_connection
from src.data.sources.snl_mining.scraper.storage.subsections import ensure_detail_tables

# (mine_id, subsection_label) -> (status, tables). "Financings"/"Ownership"
# are just fake subsection labels here, each with its own fake table name.
_PLAN = {
    ("m1", "Financings"): ("completed", {"detail_fake_financings": [{"value": "m1-financings"}]}),
    ("m1", "Ownership"): ("reclassified", {"detail_fake_ownership": [{"value": "m1-ownership"}]}),
    ("m2", "Financings"): ("content_mismatch", {}),
    ("m2", "Ownership"): ("RAISE", {}),  # simulates a parse/classify exception
    ("m3", "Financings"): ("unverified", {"detail_fake_financings": [{"value": "m3-financings"}]}),
}


def _make_conn(tmp_path):
    conn = get_connection(tmp_path / "scraper.duckdb")
    ensure_detail_tables(conn)
    return conn


def _seed_export_rows(conn, tmp_path):
    rows = []
    for (mine_id, subsection_label) in _PLAN:
        xls_path = tmp_path / f"{mine_id}_{subsection_label}.xlsx"
        xls_path.write_bytes(b"")  # only existence is checked before parsing
        rows.append((mine_id, "Section", subsection_label, "", str(xls_path), "sha-placeholder"))
    conn.executemany(
        """
        INSERT INTO mine_subsection_exports
            (mine_id, section_label, subsection_label, subsection_href, xls_path, xls_sha256, exported_at)
        VALUES (?, ?, ?, ?, ?, ?, now())
        """,
        rows,
    )


def _fake_parse_subsection_xls(path, subsection_label):
    return SimpleNamespace(xls_sha256="fake-sha")


def _fake_classify_and_regularize(parsed, mine_id, subsection_label):
    status, tables = _PLAN[(mine_id, subsection_label)]
    if status == "RAISE":
        raise ValueError("simulated classify failure")
    return status, tables


def test_row_processing_matches_expected_aggregate_outcome(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "parse_subsection_xls", _fake_parse_subsection_xls)
    monkeypatch.setattr(worker, "classify_and_regularize", _fake_classify_and_regularize)

    conn = _make_conn(tmp_path)
    _seed_export_rows(conn, tmp_path)

    counts = stage.regularize_detail_exports(
        conn,
        mine_ids=["m1", "m2", "m3"],
        continue_on_error=True,
        max_workers=1,
    )

    assert counts["export_file_count"] == 5
    assert counts["completed_count"] == 1
    assert counts["reclassified_count"] == 1
    assert counts["content_mismatch_count"] == 1
    assert counts["unverified_count"] == 1
    assert counts["failed_count"] == 1
    assert counts["skipped_count"] == 0
    # completed + reclassified + unverified each contribute one row.
    assert counts["regularized_row_count"] == 3

    financings = conn.execute("SELECT value FROM detail_fake_financings ORDER BY value").fetchall()
    assert [row[0] for row in financings] == ["m1-financings", "m3-financings"]

    ownership = conn.execute("SELECT value FROM detail_fake_ownership ORDER BY value").fetchall()
    assert [row[0] for row in ownership] == ["m1-ownership"]

    statuses = dict(
        conn.execute(
            """
            SELECT mine_id || ':' || subsection_label, status FROM mine_subsection_stage_status
            WHERE stage_name = 'detail_regularize'
            """
        ).fetchall()
    )
    assert statuses == {
        "m1:Financings": "completed",
        "m1:Ownership": "reclassified",
        "m2:Financings": "content_mismatch",
        "m2:Ownership": "failed",
        "m3:Financings": "unverified",
    }

    conn.close()


def test_second_run_reprocesses_non_completed_statuses_but_skips_completed_ones(tmp_path, monkeypatch):
    # get_completed_stage_keys only treats status='completed' (with
    # completed_at set) as "done" -- reclassified/unverified/content_mismatch
    # rows are re-attempted on every subsequent run, same pre-existing
    # behavior STATUS_UNVERIFIED already had before reclassification was
    # added. Only m1's Financings row (STATUS_COMPLETED) should be skipped.
    monkeypatch.setattr(worker, "parse_subsection_xls", _fake_parse_subsection_xls)
    monkeypatch.setattr(worker, "classify_and_regularize", _fake_classify_and_regularize)

    conn = _make_conn(tmp_path)
    _seed_export_rows(conn, tmp_path)

    stage.regularize_detail_exports(conn, mine_ids=["m1", "m3"], max_workers=1)
    second_counts = stage.regularize_detail_exports(conn, mine_ids=["m1", "m3"], max_workers=1)

    assert second_counts["export_file_count"] == 3
    assert second_counts["skipped_count"] == 1
    assert second_counts["completed_count"] == 0
    assert second_counts["reclassified_count"] == 1
    assert second_counts["unverified_count"] == 1

    conn.close()


def test_process_pool_path_with_missing_files(tmp_path):
    # Exercises the real ProcessPoolExecutor path end-to-end (no
    # monkeypatching -- can't cross the process boundary): rows whose
    # xls_path was never created deterministically fail at the
    # `Path.exists()` check inside `_regularize_worker.process_one_row`,
    # without needing real xlsx content. Confirms `ExportRow`/`RowResult`/
    # `SubsectionRecord` round-trip pickling across worker processes, that
    # the worker module's stdlib-only import graph actually works when
    # freshly re-imported by a spawned process, and that aggregation still
    # lands correctly when results come back from real subprocesses.
    conn = _make_conn(tmp_path)
    rows = [
        ("m1", "Section", "Financings", "", str(tmp_path / "missing_1.xlsx"), None),
        ("m1", "Section", "Ownership", "", str(tmp_path / "missing_2.xlsx"), None),
        ("m2", "Section", "Financings", "", str(tmp_path / "missing_3.xlsx"), None),
    ]
    conn.executemany(
        """
        INSERT INTO mine_subsection_exports
            (mine_id, section_label, subsection_label, subsection_href, xls_path, xls_sha256, exported_at)
        VALUES (?, ?, ?, ?, ?, ?, now())
        """,
        rows,
    )

    counts = stage.regularize_detail_exports(conn, mine_ids=["m1", "m2"], max_workers=2)

    assert counts["export_file_count"] == 3
    assert counts["failed_count"] == 3
    assert counts["completed_count"] == 0

    errors = conn.execute(
        "SELECT error_msg FROM mine_subsection_stage_status WHERE stage_name = 'detail_regularize'"
    ).fetchall()
    assert all("Missing XLS file" in row[0] for row in errors)

    conn.close()
