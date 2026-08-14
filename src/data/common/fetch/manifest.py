"""FETCH bookkeeping without a ledger: a source declares the fixed list of
files it needs (`RequiredFile`), that list is diffed against one cached
directory listing (`snapshot_local_listing()`/`snapshot_remote_listing()`,
never re-stat per file), and per-unit failure history lives in the small
JSON sidecars `src.data.common.statusfile` provides.

Three buckets come out of `plan_fetch()`: `complete` (present, plausible
size), `outstanding` (missing/undersized, will be retried), `unavailable`
(a source has given up on this unit -- an operator needs to fix the URL,
not wait for another automatic attempt).
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from typing import Any, Optional

from src.data.common import statusfile

STATUS_RETRYING = "retrying"
STATUS_UNAVAILABLE = "unavailable"

#: Generous default -- most sources retry across many `data fetch` runs, not
#: within one, so this bounds total lifetime attempts, not a single run's.
DEFAULT_MAX_ATTEMPTS = 5

#: Bare presence isn't enough to trust a download -- a dropped connection can
#: leave a small, truncated file that still "exists". A file under this
#: floor with no `expected_bytes` to check against is treated as incomplete
#: rather than trusted blind.
_MIN_PLAUSIBLE_BYTES = 1024


@dataclass(frozen=True)
class RequiredFile:
    """One file a source needs FETCH to produce. `unit_id` is the stable key
    used for status-file bookkeeping and progress reporting (e.g.
    `"2020/h12v09"` for a MODIS tile-year, a bare filename for a
    `ConfiguredFile`); `relative_path` is where it lands under the source's
    raw root, POSIX-style regardless of host OS."""

    unit_id: str
    relative_path: str
    url: str
    expected_bytes: Optional[int] = None


@dataclass(frozen=True)
class ListingEntry:
    size: int
    mtime: float


@dataclass(frozen=True)
class FetchPlan:
    complete: list[RequiredFile]
    outstanding: list[RequiredFile]
    unavailable: list[RequiredFile]

    def counts(self) -> dict[str, int]:
        return {
            "complete": len(self.complete),
            "outstanding": len(self.outstanding),
            "unavailable": len(self.unavailable),
        }


def snapshot_local_listing(root: str, *, max_depth: Optional[int] = None) -> dict[str, ListingEntry]:
    """One walk of *root* -- callers cache this for the whole run, never
    re-stat per file. A missing *root* is an empty listing, not an error (a
    source's raw dir may not exist yet on a fresh checkout).

    *max_depth* (a source's `RAW_LISTING_DEPTH`, see `src.data.sources.base
    .RemoteFileCatalog`) prunes `os.walk()` once a subdirectory is *max_depth*
    levels below *root* -- e.g. depth 2 for a `<year>/<file>` layout stops
    descending past the year directories. `None` (default) walks unbounded,
    the only safe choice for a source whose real nesting isn't known/fixed.
    Always skips `statusfile.STATUS_SUBDIR` regardless of depth -- those are
    FETCH's own bookkeeping sidecars, never a downloaded file."""
    listing: dict[str, ListingEntry] = {}
    if not os.path.isdir(root):
        return listing
    root = os.path.normpath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        if statusfile.STATUS_SUBDIR in dirnames:
            dirnames.remove(statusfile.STATUS_SUBDIR)
        if max_depth is not None:
            depth = 0 if dirpath == root else dirpath[len(root) :].count(os.sep)
            if depth >= max_depth - 1:
                dirnames[:] = []
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            try:
                st = os.stat(full)
            except OSError:
                continue
            listing[rel] = ListingEntry(size=st.st_size, mtime=st.st_mtime)
    return listing


def snapshot_remote_listing(client: Any, remote_root: str) -> dict[str, ListingEntry]:
    """One `find` round trip over *remote_root* via *client* (an
    `HPCClient`), same contract as `snapshot_local_listing()`. A missing
    remote directory yields an empty listing, not an error."""
    quoted = shlex.quote(remote_root)
    command = f"find {quoted} -type f -printf '%s %T@ %P\\n' 2>/dev/null || true"
    ok, stdout, _stderr = client.execute_command(command)
    listing: dict[str, ListingEntry] = {}
    if not ok:
        return listing
    for line in stdout.splitlines():
        parts = line.split(" ", 2)
        if len(parts) != 3:
            continue
        size_s, mtime_s, rel = parts
        try:
            listing[rel] = ListingEntry(size=int(size_s), mtime=float(mtime_s))
        except ValueError:
            continue
    return listing


def remote_listing_for_local_root(ctx: Any, local_root: str) -> "dict[str, ListingEntry] | None":
    """Remote counterpart of `snapshot_local_listing(local_root)`: one `find`
    over the HPC path that mirrors *local_root*. `None` if no HPC target is
    configured -- callers should fall back to the local listing in that
    case, not treat it as "nothing there".

    The remote tree always mirrors the local one relative to their
    respective roots (`ctx.data_root` locally, `HPCClient.base_path`
    remotely -- see `DataSource.transfer_units()`'s docstring), so the only
    thing that needs computing is *local_root*'s relative offset from
    `ctx.data_root`; the listing keys that come back are then directly
    comparable to `snapshot_local_listing(local_root)`'s keys."""
    if not ctx.ssh_target:
        return None
    from src.data.common.hpc.client import HPCClient
    from src.data.common.hpc.push import _full_remote_path

    client = HPCClient(target=ctx.ssh_target, key_file=ctx.key_file)
    rel_root = os.path.relpath(local_root, ctx.data_root).replace(os.sep, "/")
    remote_root = _full_remote_path(client, rel_root)
    listing = snapshot_remote_listing(client, remote_root)
    status_prefix = f"{statusfile.STATUS_SUBDIR}/"
    return {rel: entry for rel, entry in listing.items() if not rel.startswith(status_prefix)}


def resolve_fetch_listing(
    source: Any, local_root: str, *, allow_remote: bool = True
) -> "tuple[dict[str, ListingEntry], bool]":
    """The directory listing a FETCH `_plan_fetch()`/`run_fetch()` should
    treat as ground truth for *local_root*, and whether it came from the HPC
    target (`True`) instead of local disk (`False`).

    Remote is used only when *allow_remote* (the caller's own local-only
    override, e.g. `data summary`'s deliberately network-free targets) and
    this source's `transfer_mode` (`src.data.common.fetch.transfer_mode`)
    resolves to `"auto"` -- a source pushing every fetched file to HPC right
    after FETCH, whose local copy is disposable. Falls back to the local
    listing whenever remote isn't applicable or isn't reachable (no
    `remote.ssh_target` configured), so a source with no HPC target set up
    yet still gets its normal local-disk behavior instead of silently
    treating everything as missing."""
    from src.data.common.fetch.transfer_mode import resolve_transfer_mode

    if allow_remote and resolve_transfer_mode(source) == "auto":
        remote = remote_listing_for_local_root(source.ctx, local_root)
        if remote is not None:
            return remote, True
    local = snapshot_local_listing(local_root, max_depth=getattr(source, "RAW_LISTING_DEPTH", None))
    return local, False


def _is_present(entry: Optional[ListingEntry], required: RequiredFile) -> bool:
    if entry is None:
        return False
    if required.expected_bytes is not None:
        return entry.size == required.expected_bytes
    return entry.size >= _MIN_PLAUSIBLE_BYTES


def plan_fetch(
    required: list[RequiredFile], listing: dict[str, ListingEntry], status_dir: str
) -> FetchPlan:
    """Diff *required* against *listing*. A required file with a status
    sidecar marked `STATUS_UNAVAILABLE` (see `record_failure()`) is bucketed
    separately from a plain `outstanding` retry.

    One `os.listdir()` of the status directory up front (`statusfile
    .list_status_filenames()`), not one `open()` attempt per missing file --
    the common case (a mostly-incomplete tree) is mostly units that were
    never attempted at all and have no status file, so this turns N mostly-
    failed opens into 1 listdir + a handful of real reads."""
    complete: list[RequiredFile] = []
    outstanding: list[RequiredFile] = []
    unavailable: list[RequiredFile] = []
    status_filenames = statusfile.list_status_filenames(status_dir)
    for req in required:
        if _is_present(listing.get(req.relative_path), req):
            complete.append(req)
            continue
        status = None
        if f"{statusfile.sanitize_unit_id(req.unit_id)}.json" in status_filenames:
            status = statusfile.read(statusfile.status_path(status_dir, req.unit_id))
        if status and status.get("status") == STATUS_UNAVAILABLE:
            unavailable.append(req)
        else:
            outstanding.append(req)
    return FetchPlan(complete=complete, outstanding=outstanding, unavailable=unavailable)


def record_failure(
    status_dir: str,
    unit_id: str,
    error: str,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    permanent: bool = False,
) -> str:
    """Bump *unit_id*'s attempt count and last error; flips to
    `STATUS_UNAVAILABLE` once *permanent* (e.g. a 404/410 from the origin)
    or `attempts >= max_attempts`. Returns the resulting status string."""
    path = statusfile.status_path(status_dir, unit_id)
    existing = statusfile.read(path) or {}
    attempts = int(existing.get("attempts", 0)) + 1
    status = STATUS_UNAVAILABLE if (permanent or attempts >= max_attempts) else STATUS_RETRYING
    statusfile.write(path, {"status": status, "attempts": attempts, "last_error": error})
    return status


def clear_failure(status_dir: str, unit_id: str) -> None:
    """A previously-failing unit that just succeeded -- drop its status file
    so it doesn't linger in a stale bucket forever."""
    statusfile.remove(statusfile.status_path(status_dir, unit_id))
