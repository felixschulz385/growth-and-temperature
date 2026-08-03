"""The per-source DuckDB ledger: replaces `UnifiedDataIndex` (Parquet,
whole-file rewrite per mutation) and `TransferManifest` (a second, parallel
Parquet format) with one incrementally-writable file per source, tracking
both the FETCH-only remote-file catalog and the local/remote transfer state
of every artifact any step (fetch/prepare/grid) produces.

See docs/design/10-fetch-ledger.md for the full rationale. `common/ledger/`
is infrastructure `src/data/sources/*` builds on -- it must never import
from `src.data.sources`, only the reverse.
"""

from .store import SourceLedger

__all__ = ["SourceLedger"]
