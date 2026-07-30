"""The DataSource contract: replaces BaseDataSource (download) and
AbstractPreprocessor (preprocess) with one ABC covering a source's whole
lifecycle.

docs/design/09-integrated-pipeline.md §4.
"""

from __future__ import annotations

import abc
import os
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from src.data.sources import layout
from src.data.sources.steps import PipelineStep, StepTarget, TargetSelection, TransferUnit, UnsupportedStepError

if TYPE_CHECKING:
    from src.data.pipeline.config import SourceConfig
    from src.data.pipeline.context import PipelineContext


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
    REQUIRES: ClassVar[tuple[tuple[str, PipelineStep], ...]] = ()

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
        remote_base = self.ctx.remote_data_root
        if remote_base:
            remote_path = os.path.relpath(local_path, remote_base)
        else:
            remote_path = os.path.basename(os.path.normpath(local_path))
        return [TransferUnit(unit_id=step.value, local_path=local_path, remote_path=remote_path)]

    def close(self) -> None:
        """Release sessions/dask clients/etc. Default: nothing to release."""
        return None
