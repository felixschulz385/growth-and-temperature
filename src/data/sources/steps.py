"""PipelineStep and the target/transfer/completion primitives shared by every source.

docs/design/09-integrated-pipeline.md §2-3: replaces the old
stage="annual"/"spatial"/"vector" string vocabulary with a fixed, artefact-named
enum, and generalizes the atomic-write/resumability lesson from the MODIS
GeoTIFF migration (commit 28d7132) into a per-target completion policy every
source shares instead of reimplementing.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from src.data.common.ledger.store import SourceLedger


class PipelineStep(enum.StrEnum):
    """The three artefact-named steps every source implements a subset of.

    Named for what each step *produces*, not the mechanism that produces it.
    """

    FETCH = "fetch"
    PREPARE = "prepare"
    GRID = "grid"


#: Canonical ordering -- used for `--all-steps` and for sanity-checking that a
#: REQUIRES edge doesn't create an ordering contradiction within one source.
STEP_ORDER: tuple[PipelineStep, ...] = (PipelineStep.FETCH, PipelineStep.PREPARE, PipelineStep.GRID)


class Completion(enum.StrEnum):
    """How the runner decides whether a StepTarget's output already exists.

    Generalizes the MODIS lesson (commit 28d7132): a killed-mid-write
    directory (e.g. a partially-written Zarr store) looks "complete" to a bare
    os.path.exists check, but a temp-file + os.replace single-file write does
    not have that failure mode.
    """

    #: Directory outputs (Zarr stores): complete only once a sibling
    #: ``<output_path>.complete`` marker file has been written.
    MARKER = "marker"
    #: Single-file outputs written via temp-file + os.replace: a bare
    #: os.path.exists is safe because the rename is atomic.
    PATH_EXISTS = "path_exists"
    #: Always re-run (side-effecting or intentionally non-idempotent targets).
    NEVER = "never"


def is_complete(target: "StepTarget", ledger: "SourceLedger | None" = None) -> bool:
    """Local-disk completion, same as always -- unless *target* declares
    `require_remote=True` (docs/design/10-fetch-ledger.md §6: today, only
    MODIS's FETCH targets do, since that's the one step producing output on
    a machine other than the one GRID later reads it from). In that case,
    also requires *ledger* to confirm HPC-side verification, so a target
    isn't reported "complete" just because it happens to still be sitting on
    the machine that produced it. Passing no *ledger* falls back to the
    local-only check (preserves every existing caller's behavior unchanged)."""
    if target.completion is Completion.NEVER:
        return False
    if target.completion is Completion.PATH_EXISTS:
        local_ok = os.path.exists(target.output_path)
    elif target.completion is Completion.MARKER:
        local_ok = os.path.exists(marker_path(target.output_path))
    else:
        raise ValueError(f"Unknown completion policy: {target.completion}")

    if not target.require_remote or ledger is None:
        return local_ok
    if not local_ok:
        return False
    from src.data.common.ledger.schema import RemoteState

    return ledger.remote_state(target.step.value, target.key) == RemoteState.VERIFIED


def local_completion_state(target: "StepTarget") -> str:
    """The local-disk truth for *target*, as an `artifacts.local_state`
    value -- the same stat `is_complete()` performs locally, just returned
    as the ledger's own vocabulary instead of a bool. Not folded into
    `is_complete()` (which stays local-only-unless-`require_remote`,
    signature/behavior unchanged) because `local_drift()`'s self-heal caller
    (`src/cli/data/handlers.py`) needs this exact value to write back
    into the ledger when it disagrees, not just a yes/no."""
    from src.data.common.ledger.schema import LocalState

    if target.completion is Completion.PATH_EXISTS:
        disk_complete = os.path.exists(target.output_path)
    elif target.completion is Completion.MARKER:
        disk_complete = os.path.exists(marker_path(target.output_path))
    else:
        disk_complete = False
    return LocalState.COMPLETE if disk_complete else LocalState.MISSING


def local_drift(target: "StepTarget", ledger: "SourceLedger") -> bool:
    """True if the ledger's belief about *target* disagrees with a cheap,
    no-network on-disk check -- the trigger for automatic ledger self-heal
    (as opposed to a manual `data reconcile`). Reuses the exact stat
    `is_complete()` already performs locally (no new I/O), just compares it
    against `ledger.local_state(...)` instead of returning it directly.

    Deliberately one-directional in what it can catch: a file present on
    disk with *no* ledger row at all looks identical to "ledger says
    missing, disk agrees" from here, since there is no row to disagree
    with -- that unknown-unknown is only caught by an explicit reconcile,
    which crawls/discovers rather than reading existing rows. Not a bug;
    stated in docs/design/10-fetch-ledger.md's successor as an accepted gap.
    """
    if target.completion is Completion.NEVER:
        return False

    from src.data.common.ledger.schema import LocalState

    disk_state = local_completion_state(target)
    ledger_state = ledger.local_state(target.step.value, target.key)
    return disk_state != (ledger_state or LocalState.MISSING)


def marker_path(output_path: str) -> str:
    """Sibling marker file path for a MARKER-completion directory output."""
    return output_path.rstrip(os.sep) + ".complete"


def mark_complete(output_path: str) -> None:
    with open(marker_path(output_path), "w", encoding="utf-8") as fh:
        fh.write("")


@dataclass(frozen=True)
class StepTarget:
    """One unit of work for one (source, step)."""

    source_id: str
    step: PipelineStep
    key: str
    output_path: str
    inputs: tuple[str, ...] = ()
    completion: Completion = Completion.MARKER
    #: True only for targets a *different* machine/job than the one that
    #: produces them will read (today: MODIS's FETCH, which streams from
    #: Planetary Computer off-cluster and must be pushed to HPC before GRID's
    #: SLURM job can read it). Default False preserves every other source's
    #: existing local-only completion semantics unchanged.
    require_remote: bool = False
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransferUnit:
    """One (local_path, remote_path) pair to push/pull over HPC transfer.

    Mirrors the dict shape docs/design/08-hpc-transfer.md §2 specified
    (`{local_path, remote_path, unit_id}`) as a typed value instead of a bare
    dict, without changing the wire contract src/data/common/hpc/transfer.py
    consumes.
    """

    unit_id: str
    local_path: str
    remote_path: str


@dataclass(frozen=True)
class TargetSelection:
    """What subset of a source's targets to plan/run. Empty = everything."""

    years: tuple[int, ...] | None = None
    year_range: tuple[int, int] | None = None
    keys: tuple[str, ...] | None = None

    def matches_year(self, year: int | None) -> bool:
        if year is None:
            return True
        if self.years is not None and year not in self.years:
            return False
        if self.year_range is not None and not (self.year_range[0] <= year <= self.year_range[1]):
            return False
        return True

    def matches_key(self, key: str) -> bool:
        return self.keys is None or key in self.keys


class UnsupportedStepError(ValueError):
    """Raised when a source is asked to plan/execute a step it doesn't declare in STEPS."""

    def __init__(self, source_id: str, step: PipelineStep, supported: Sequence[PipelineStep]):
        self.source_id = source_id
        self.step = step
        self.supported = tuple(supported)
        supported_str = ", ".join(s.value for s in self.supported) or "(none)"
        super().__init__(
            f"Source '{source_id}' does not implement step '{step.value}'. "
            f"Supported steps: {supported_str}."
        )


class MissingPrerequisiteError(RuntimeError):
    """Raised when a REQUIRES edge's expected output is not present yet."""

    def __init__(
        self,
        source_id: str,
        requires_id: str,
        requires_step: PipelineStep,
        expected_path: str,
    ):
        self.source_id = source_id
        self.requires_id = requires_id
        self.requires_step = requires_step
        self.expected_path = expected_path
        super().__init__(
            f"Source '{source_id}' requires source '{requires_id}' step "
            f"'{requires_step.value}' to be complete first. Expected output not "
            f"found at: {expected_path}. Run: "
            f"run.py data run --source {requires_id} --step {requires_step.value}"
        )
