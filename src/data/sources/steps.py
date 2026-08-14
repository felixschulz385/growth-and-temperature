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
from typing import Any, Mapping, Sequence


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
    #: Decided once at plan time and carried on the target rather than
    #: re-derived from a live check (`target.meta["complete"]`) -- for a
    #: FETCH target whose completeness was judged against a remote HPC
    #: listing instead of local disk (`src.data.common.fetch.manifest
    #: .resolve_fetch_listing`), where there's no single local path left for
    #: a bare `os.path.exists()` to re-check against.
    PRECOMPUTED = "precomputed"


def is_complete(target: "StepTarget") -> bool:
    """Whether *target*'s output already exists, per its completion policy."""
    if target.completion is Completion.NEVER:
        return False
    if target.completion is Completion.PATH_EXISTS:
        return os.path.exists(target.output_path)
    if target.completion is Completion.MARKER:
        return os.path.exists(marker_path(target.output_path))
    if target.completion is Completion.PRECOMPUTED:
        return bool(target.meta.get("complete", False))
    raise ValueError(f"Unknown completion policy: {target.completion}")


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
    #: Whether a FETCH `_plan_fetch()` must judge completeness from local
    #: disk only, even for a `transfer_mode=auto` source that would
    #: otherwise check the HPC target instead
    #: (`src.data.common.fetch.manifest.resolve_fetch_listing`). Defaults to
    #: the safe, pre-existing local-only behavior; `data plan`/`data run`
    #: explicitly opt out of it (`src/cli/data/handlers.py`) since those are
    #: the commands that actually decide what to (re-)fetch. `data summary`
    #: leaves this at its default -- it's documented to never make a live
    #: remote call.
    local_only: bool = True

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
