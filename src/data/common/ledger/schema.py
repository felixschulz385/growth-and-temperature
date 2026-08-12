"""DDL and state-string constants for the per-source DuckDB ledger.

docs/design/10-fetch-ledger.md §3. One `.duckdb` file per source holds three
tables: `remote_files`/`entrypoints` (the FETCH-only crawl catalog -- "what
does the origin have") and `artifacts` (the universal local/remote transfer
ledger -- "what state is each unit in, here and on HPC", used by FETCH's
per-file units and PREPARE/GRID's per-target units alike).
"""

from __future__ import annotations


class LocalState:
    """`artifacts.local_state`: has this unit's local copy been produced?"""

    MISSING = "missing"
    WRITING = "writing"
    COMPLETE = "complete"
    FAILED = "failed"


class RemoteState:
    """`artifacts.remote_state`: has this unit reached the HPC target?

    Orthogonal to `LocalState` -- a row can be `local=complete,
    remote=missing` (produced, not yet pushed), `remote=pushed`
    (rsync/extract succeeded, not yet sample-verified), or `remote=verified`
    (sample-verified present on HPC). This is the direct fix for the old
    system's "is_complete() has no concept of complete-but-not-transferred."
    """

    MISSING = "missing"
    PUSHED = "pushed"
    VERIFIED = "verified"
    FAILED = "failed"


_CREATE_REMOTE_FILES = """
CREATE TABLE IF NOT EXISTS remote_files (
    file_hash      VARCHAR PRIMARY KEY,
    relative_path  VARCHAR NOT NULL,
    source_url     VARCHAR NOT NULL,
    entrypoint_key VARCHAR,
    year           INTEGER,
    day_of_year    INTEGER,
    discovered_at  TIMESTAMP NOT NULL DEFAULT now(),
    last_seen_at   TIMESTAMP NOT NULL DEFAULT now()
)
"""

_CREATE_ENTRYPOINTS = """
CREATE TABLE IF NOT EXISTS entrypoints (
    entrypoint_key VARCHAR PRIMARY KEY,
    payload        VARCHAR NOT NULL,
    crawled        BOOLEAN NOT NULL DEFAULT false,
    discovered_at  TIMESTAMP NOT NULL DEFAULT now(),
    crawled_at     TIMESTAMP
)
"""

_CREATE_ARTIFACTS = """
CREATE TABLE IF NOT EXISTS artifacts (
    step          VARCHAR NOT NULL,
    unit_id       VARCHAR NOT NULL,
    local_path    VARCHAR,
    remote_path   VARCHAR,
    local_state   VARCHAR NOT NULL DEFAULT 'missing',
    remote_state  VARCHAR NOT NULL DEFAULT 'missing',
    bytes         BIGINT,
    attempts      INTEGER NOT NULL DEFAULT 0,
    last_error    VARCHAR,
    source_url    VARCHAR,
    created_at    TIMESTAMP NOT NULL DEFAULT now(),
    updated_at    TIMESTAMP NOT NULL DEFAULT now(),
    PRIMARY KEY (step, unit_id)
)
"""

#: `CREATE TABLE IF NOT EXISTS` above is a no-op against a ledger created
#: before this column existed -- `ADD COLUMN IF NOT EXISTS` is the migration
#: for every such pre-existing `.duckdb` file. Holds `StepTarget.meta` as
#: JSON so a ledger-backed `plan()` (docs/design/10-fetch-ledger.md's
#: successor) can reconstruct a target without re-running discovery.
_ALTER_ARTIFACTS_ADD_META = "ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS meta VARCHAR"

_CREATE_ARTIFACTS_LOCAL_STATE_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_artifacts_local_state ON artifacts(step, local_state)"
)
_CREATE_ARTIFACTS_REMOTE_STATE_IDX = (
    "CREATE INDEX IF NOT EXISTS idx_artifacts_remote_state ON artifacts(step, remote_state)"
)

#: Executed in order against a fresh connection by `SourceLedger._ensure_schema()`.
ALL_DDL: tuple[str, ...] = (
    _CREATE_REMOTE_FILES,
    _CREATE_ENTRYPOINTS,
    _CREATE_ARTIFACTS,
    _ALTER_ARTIFACTS_ADD_META,
    _CREATE_ARTIFACTS_LOCAL_STATE_IDX,
    _CREATE_ARTIFACTS_REMOTE_STATE_IDX,
)
