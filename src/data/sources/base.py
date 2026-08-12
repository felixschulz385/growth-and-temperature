"""The DataSource contract: replaces BaseDataSource (download) and
AbstractPreprocessor (preprocess) with one ABC covering a source's whole
lifecycle.

docs/design/09-integrated-pipeline.md §4.
"""

from __future__ import annotations

import abc
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional, Protocol, runtime_checkable

from src.data.sources import layout
from src.data.sources.steps import PipelineStep, StepTarget, TargetSelection, TransferUnit, UnsupportedStepError

if TYPE_CHECKING:
    from src.data.common.ledger.store import ArtifactRow
    from src.data.pipeline.config import SourceConfig
    from src.data.pipeline.context import PipelineContext
    from src.data.sources.verify import VerificationResult

logger = logging.getLogger(__name__)


@runtime_checkable
class RemoteFileCatalog(Protocol):
    """The exact attribute set `UnifiedDataIndex` and `AsyncHPCDownloader`
    duck-type on today (verified by grep across both modules, not asserted):
    `data_source.{data_path, DATA_SOURCE_NAME, has_entrypoints,
    list_remote_files, get_file_hash, get_all_entrypoints,
    filename_to_entrypoint, download_async}`.

    Deliberately narrower than the old `BaseDataSource` ABC: `local_path`,
    `download` (sync), `get_authenticated_session`, `schema_dtypes`, and
    `gcs_upload_path` are NOT accessed by the shared index/downloader
    machinery (confirmed by grep) -- they are per-source implementation
    details, not part of this cross-module contract, and `gcs_upload_path`
    specifically is dead code (docs/design/09-integrated-pipeline.md §1) not
    carried forward at all.

    `FETCH`-capable sources implement this so `UnifiedDataIndex`/
    `AsyncHPCDownloader` (deliberately untouched by this refactor -- both are
    large and have zero test coverage) keep working unmodified.
    """

    data_path: str
    DATA_SOURCE_NAME: str
    has_entrypoints: bool

    def list_remote_files(self, entrypoint: dict | None = None) -> list[tuple[str, str]]: ...

    def get_file_hash(self, file_url: str) -> str: ...

    def get_all_entrypoints(self) -> list[dict[str, Any]]: ...

    def filename_to_entrypoint(self, relative_path: str) -> dict[str, Any] | None: ...

    async def download_async(self, source_url: str, output_path: str, session: Any = None) -> None: ...


class DataSource(abc.ABC):
    """Base class for every registered source.

    Subclasses declare `ID`, `STEPS`, and (optionally) `ALIASES`/`REQUIRES` as
    class attributes, then implement `plan()`/`execute()`. `plan()`/`execute()`
    on an undeclared step raise `UnsupportedStepError` -- enforced here, not
    left to each subclass to remember (§2's "no default step" decision).
    """

    ID: ClassVar[str]
    ALIASES: ClassVar[tuple[str, ...]] = ()
    STEPS: ClassVar[tuple[PipelineStep, ...]]
    #: Each entry is `(my_step, prereq_source_id, prereq_step)`: running
    #: *my_step* of this source requires *prereq_step* of *prereq_source_id*
    #: to be complete first. Scoped per-step (not source-wide) so e.g.
    #: ecoregions' FETCH isn't gated on gadm just because ecoregions' GRID
    #: needs it (docs/design/09-integrated-pipeline.md §2).
    REQUIRES: ClassVar[tuple[tuple[PipelineStep, str, PipelineStep], ...]] = ()

    def __init__(self, ctx: "PipelineContext", cfg: "SourceConfig"):
        self.ctx = ctx
        self.cfg = cfg

    @property
    def data_path(self) -> str:
        """Satisfies `RemoteFileCatalog.data_path` for every FETCH-capable
        subclass automatically, so no subclass has to remember to expose it
        itself. Split sources sharing one `cfg.data_path` across several
        registered ids (e.g. the misc split's osm/gadm/country_classifications,
        all `cfg.data_path="misc"`) must override this to a distinct string --
        `UnifiedDataIndex` derives its index filename from this attribute, and
        without an override they would collide on one shared index file,
        recreating the exact problem the split exists to fix
        (docs/design/09-integrated-pipeline.md §7)."""
        return self.cfg.data_path

    @classmethod
    def from_config(cls, ctx: "PipelineContext", cfg: "SourceConfig") -> "DataSource":
        return cls(ctx, cfg)

    def plan(self, step: PipelineStep, selection: TargetSelection) -> list[StepTarget]:
        self._require_step(step)
        return self._plan(step, selection)

    def execute(self, target: StepTarget) -> bool:
        self._require_step(target.step)
        return self._execute(target)

    @abc.abstractmethod
    def _plan(self, step: PipelineStep, selection: TargetSelection) -> list[StepTarget]: ...

    @abc.abstractmethod
    def _execute(self, target: StepTarget) -> bool: ...

    def discover(self, step: PipelineStep, selection: TargetSelection) -> list[StepTarget]:
        """Ground-truth target enumeration for *step* -- what SHOULD exist,
        derived by live disk/HPC crawl rather than read from the ledger.
        `reconcile_step` (src/data/sources/reconcile.py) is the only normal
        caller: it treats this as authoritative and writes the result into
        the ledger's `artifacts` table, which a ledger-backed `plan()` (via
        `_plan_from_ledger()` below) then reads back cheaply.

        Default: identical to `plan()`. A source that hasn't split its
        `_plan_prepare`/`_plan_grid` into a ledger-backed fast path plus a
        `_discover_prepare`/`_discover_grid` live-crawl counterpart yet still
        has `plan()` doing that live discovery itself -- so this default
        keeps returning correct (if uncached) ground truth for it, and nothing
        breaks for a not-yet-migrated source. A migrated source overrides
        `_discover()` to call its own `_discover_fetch`/`_discover_prepare`/
        `_discover_grid` methods instead, mirroring `plan()`/`_plan()`'s
        existing dispatch shape.
        """
        self._require_step(step)
        return self._discover(step, selection)

    def _discover(self, step: PipelineStep, selection: TargetSelection) -> list[StepTarget]:
        return self._plan(step, selection)

    def _plan_from_ledger(
        self,
        step: PipelineStep,
        selection: TargetSelection,
        build_target: "Callable[[ArtifactRow, Any], Optional[StepTarget]]",
    ) -> "Optional[list[StepTarget]]":
        """Ledger-backed target enumeration: reads persisted `artifacts` rows
        for `(self.data_path, step)` and reconstructs each `StepTarget` via
        *build_target* instead of re-running a live disk crawl every call.

        Returns `None` (not `[]`) when no ledger is configured, or none has
        been populated yet for this (source, step) -- the signal a source's
        `_plan_prepare`/`_plan_grid` uses to fall back to
        `self._discover(step, selection)` (today's exact live-crawl
        behaviour), so a zero-config setup -- or one that simply hasn't run
        `data reconcile` for this step yet -- is unaffected by this
        change rather than silently seeing zero targets.

        `TargetSelection.matches_key` is applied here, generically, since
        `StepTarget.key` and `artifacts.unit_id` are the same value for
        every source; `matches_year`/anything needing source-specific key
        parsing is *build_target*'s responsibility (it has the row's `meta`
        to work with) -- consistent with every source's existing
        enumerate-then-filter `_plan_*` pattern.

        *build_target* also receives the open (read-only) `SourceLedger`
        connection, so a GRID target's `inputs` can be re-derived from the
        upstream PREPARE step's `local_complete_units()` within the same
        connection, instead of persisting -- and going stale against --
        a snapshot of `inputs` at discovery time.
        """
        from src.data.common.ledger.paths import ledger_path
        from src.data.common.ledger.store import SourceLedger

        local_ledger_path = ledger_path(self.ctx.local_index_dir, self.data_path)
        if not local_ledger_path or not os.path.exists(local_ledger_path):
            return None

        with SourceLedger.open_for_read(local_ledger_path, data_path=self.data_path) as ledger:
            rows = ledger.artifacts_for_step(step.value)
            if not rows:
                return None

            targets: list[StepTarget] = []
            for row in rows:
                if not selection.matches_key(row.unit_id):
                    continue
                target = build_target(row, ledger)
                if target is not None:
                    targets.append(target)
        return targets

    def _require_step(self, step: PipelineStep) -> None:
        if step not in self.STEPS:
            raise UnsupportedStepError(self.ID, step, self.STEPS)

    def output_root(self, step: PipelineStep, *, namespace: str | None = None) -> str:
        return layout.output_root(
            self.ctx.data_root,
            self.cfg.data_path,
            step,
            namespace=namespace if namespace is not None else self.cfg.namespace,
            grid_id=self.ctx.grid_id,
            layout=self.ctx.layout,
        )

    def transfer_units(self, step: PipelineStep) -> list[TransferUnit]:
        """Local paths produced by *step* that should be pushed to the HPC target.

        Optional hook (docs/design/08-hpc-transfer.md §2, renamed per
        docs/design/09-integrated-pipeline.md §8) -- default: derive one
        transfer unit from `output_root(step)`, mapping the local step root
        onto the same relative path under the remote target's base path.
        Sources with a finer per-unit output layout (e.g. MODIS's per-tile-year
        GeoTIFFs) should override this for finer-grained transfer resumability.
        """
        self._require_step(step)
        local_path = self.output_root(step)
        if self.ctx.remote_data_root:
            # Relative to the LOCAL data root, not `remote_data_root` -- the
            # two are different absolute paths (often on different
            # machines/filesystems, e.g. a Windows local path vs. a scicore
            # POSIX path), so `os.path.relpath(local_path, remote_data_root)`
            # doesn't compute "this unit's place under the remote tree", it
            # computes the (meaningless, often `../../../..`-laden) path from
            # one unrelated absolute path to another. The remote tree mirrors
            # the local one *relative to their respective roots* -- the
            # relative suffix to preserve comes from `local_path` vs.
            # `ctx.data_root`, then gets joined under the remote base
            # separately (`HPCClient.base_path`/`_full_remote_path`).
            # `remote_path` must be POSIX regardless of the local OS -- the
            # HPC target is always a remote Linux host. `os.path.relpath` on
            # Windows (`ntpath`) emits backslash-separated output even given
            # forward-slash input, which breaks remote `mkdir -p`/tar
            # arcnames (a literal `foo\bar` entry, not nested dirs).
            remote_path = os.path.relpath(local_path, self.ctx.data_root).replace(os.sep, "/")
        else:
            remote_path = os.path.basename(os.path.normpath(local_path))
        return [TransferUnit(unit_id=step.value, local_path=local_path, remote_path=remote_path)]

    def close(self) -> None:
        """Release sessions/dask clients/etc. Default: nothing to release."""
        return None

    def verify_grid(self, target: StepTarget) -> "VerificationResult":
        """Cheap sanity check for a completed GRID target's output.

        Default: delegates to `src.data.sources.verify.verify_grid_output`
        using whatever `expected_vars`/`value_range` the source declared on
        `target.meta` when it planned this target -- sources don't need to
        override this, just populate `meta` in their `_plan_grid`."""
        from src.data.sources.verify import verify_grid_output

        return verify_grid_output(
            target.output_path,
            expected_vars=target.meta.get("expected_vars"),
            value_range=target.meta.get("value_range"),
            range_vars=target.meta.get("range_vars"),
        )

    @staticmethod
    def _extract_year(filename: str) -> int | None:
        """Generic delimiter-preferring 4-digit-year filename parser.

        Was duplicated byte-for-byte in acag/esacci before being factored out
        here. ntl_harm's `_extract_year_from_filename` and eog's
        `_extract_year_from_path` are deliberately NOT merged into this: they
        use different regex sets and year bounds (real behavioral
        differences, not copy-paste), so unifying them would risk silently
        changing what they match rather than removing duplication.
        """
        import re

        for pattern in (r"[._\-](\d{4})[._\-]", r"(\d{4})"):
            for match in re.finditer(pattern, filename):
                year = int(match.group(1))
                if 1990 <= year <= 2040:
                    return year
        return None

    @staticmethod
    def get_file_hash(file_url: str) -> str:
        """`RemoteFileCatalog`'s stable per-file identifier -- an md5 of the
        remote URL, used as `artifacts.unit_id`/`remote_files.file_hash` in
        the ledger.

        Was duplicated byte-for-byte across acag/plad/berman_mining/
        ntl_harm/esacci/misc/_fetch.py/glass/eog before being factored out
        here -- unlike `_extract_year`, every one of those copies had zero
        behavioral variation (same hash algorithm, same encoding), so there
        was no real difference to preserve by keeping them separate.
        """
        import hashlib

        return hashlib.md5(file_url.encode("utf-8")).hexdigest()

    def _dask_client(self):
        """Shared Dask client factory for raster/tile-processing sources.

        Was duplicated byte-for-byte across acag/esacci/ntl_harm/gadm/modis/
        eog before being factored out here. Requires `self.temp_dir` (set by
        every subclass's own `__init__`, via the shared
        `cfg.temp_dir or tempfile.mkdtemp(...)` pattern) and reads
        `self.ctx.dask_threads`/`dask_memory_limit`/`dashboard_port`.
        Sources needing different client construction (e.g. GLASS, which
        supports a per-source config override of the dashboard port) should
        override this method rather than special-case it here.
        """
        from src.data.common.dask.client import DaskClientContextManager

        return DaskClientContextManager(
            threads=self.ctx.dask_threads,
            memory_limit=self.ctx.dask_memory_limit,
            dashboard_port=self.ctx.dashboard_port,
            temp_dir=os.path.join(self.temp_dir, "dask_workspace"),
        )
