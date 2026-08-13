"""World Bank commodity prices: fetch + normalize, no GRID.

A tiny (commodity, year) -> price lookup table, not spatial -- `STEPS =
(FETCH, PREPARE)` only, mirroring `docs/design/09-integrated-pipeline.md`'s
recorded escape hatch for splitting `country_classifications`'s HDI/World
Bank origins into independent `(FETCH, PREPARE)` sources (§7, end). Consumed
by `snl_mining`'s own PREPARE step via a `REQUIRES` entry `(PipelineStep.
PREPARE, "commodity_prices", PipelineStep.PREPARE)` and a second DuckDB
`ATTACH` (see
`src/data/sources/snl_mining/source.py`), resolving this source's output path
directly through `layout.output_root(...)` rather than any framework-injected
path -- `REQUIRES` is pure ordering/scheduling metadata in this codebase
(docs/design/09-integrated-pipeline.md §2: cross-source coupling is on
artefact paths, never a class import).

FETCH uses `ConfiguredFilesFetchMixin` (`src/data/sources/misc/_fetch.py`),
same pattern as `gadm`/`osm`/`country_classifications`. The World Bank Pink
Sheet's `thedocs.worldbank.org` download URL embeds a content hash + release
date and rotates roughly monthly -- `prices_url` in `data.yaml` is expected
to need an occasional manual bump (get the current link from
https://www.worldbank.org/en/research/commodity-markets); a `prices_path`
config override lets PREPARE read an already-staged local copy directly,
bypassing FETCH entirely (used for the copy already present at
`data/raw/commodity_prices/auxiliary/CMO-Historical-Data-Annual.xlsx`).
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import TYPE_CHECKING, Any, List, Optional

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.base import DataSource
from src.data.sources.commodity_prices.prices import read_and_normalize_prices
from src.data.sources.misc._fetch import ConfiguredFile, ConfiguredFilesFetchMixin
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection

if TYPE_CHECKING:
    from src.data.common.ledger.store import ArtifactRow

logger = logging.getLogger(__name__)

DEFAULT_PRICES_URL = (
    "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/"
    "CMO-Historical-Data-Annual.xlsx"
)
DEFAULT_PRICES_NAME = "CMO-Historical-Data-Annual.xlsx"


class CommodityPricesSource(ConfiguredFilesFetchMixin, DataSource):
    """World Bank Pink Sheet commodity prices -- normalized (commodity, year)
    real-price lookup table, no GRID step."""

    ID = "commodity_prices"
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)
    REQUIRES: tuple[tuple[str, PipelineStep], ...] = ()

    DATA_SOURCE_NAME = "commodity_prices"

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="commodity_prices")
        super().__init__(ctx, cfg)

        url = cfg.raw.get("prices_url", DEFAULT_PRICES_URL)
        name = cfg.raw.get("prices_name", DEFAULT_PRICES_NAME)
        self.CONFIGURED_FILES: List[ConfiguredFile] = [ConfiguredFile(key="prices", url=url, name=name)]

        # Escape hatch for an already-staged local copy (e.g. downloaded by
        # hand ahead of a fresh FETCH run) -- PREPARE reads from here instead
        # of `output_root(FETCH)` when set, bypassing the FETCH step's own
        # output entirely.
        prices_path_override = cfg.raw.get("prices_path")
        self._raw_prices_path_override = self._resolve_path(prices_path_override) if prices_path_override else None

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.ctx.data_root, path)

    # ------------------------------------------------------------------
    # plan()/execute() dispatch
    # ------------------------------------------------------------------

    def _plan_fetch(self) -> List[StepTarget]:
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.FETCH, key="all",
                output_path=self.output_root(PipelineStep.FETCH), completion=Completion.NEVER,
            )
        ]

    def _plan(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        if step is PipelineStep.FETCH:
            return self._plan_fetch()
        if step is PipelineStep.PREPARE:
            return self._plan_prepare()
        raise AssertionError(f"unreachable: {step}")

    def _discover(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        """Ground truth for `data reconcile` -- see gadm.py's identical
        `_discover()` for the full rationale. This source's PREPARE target is
        a singleton with no year/key selection to apply; `selection` is
        accepted for interface symmetry only."""
        if step is PipelineStep.FETCH:
            return self._plan_fetch()
        if step is PipelineStep.PREPARE:
            return self._discover_prepare()
        raise AssertionError(f"unreachable: {step}")

    def _execute(self, target: StepTarget) -> bool:
        if target.step is PipelineStep.FETCH:
            return self._execute_fetch(target)
        if target.step is PipelineStep.PREPARE:
            return self._execute_prepare(target)
        raise AssertionError(f"unreachable: {target.step}")

    def _execute_fetch(self, target: StepTarget) -> bool:
        if not self.ctx.ssh_target:
            logger.warning("Fetch requires an HPC/remote target to be configured.")
            return False

        from src.data.common.fetch.driver import run_fetch

        return run_fetch(self, **self.cfg.raw.get("download", {}))

    # -- PREPARE ----------------------------------------------------------

    def _raw_prices_file(self) -> str:
        if self._raw_prices_path_override:
            return self._raw_prices_path_override
        return os.path.join(self.output_root(PipelineStep.FETCH), self.CONFIGURED_FILES[0].name)

    def _plan_prepare(self) -> List[StepTarget]:
        """Ledger-backed fast path. Falls back to `_discover_prepare()` --
        today's exact live logic -- if no ledger is configured yet, or
        `data reconcile --step prepare` hasn't populated one yet."""

        def build_target(row: "ArtifactRow", _ledger: Any) -> Optional[StepTarget]:
            raw_file = row.meta.get("raw_file")
            # `raw_file`/`row.local_path` are absolute paths as recorded by
            # whichever host wrote this row -- `merge_from_remote()` (pulled
            # by the FETCH driver/`data reconcile`/`data transfer`) merges
            # ledger rows across hosts by design (so e.g. a push another
            # machine already verified is visible here), but a PREPARE row
            # written on a *different* host carries that host's absolute
            # paths, which don't resolve on this one. `os.path.exists` here
            # is the same cheap guard `_discover_prepare()` already performs
            # on `raw_file` -- a foreign path simply won't exist on this
            # filesystem, so this row gets skipped in favor of the live-
            # discovery fallback below instead of `_execute_prepare()` later
            # handing a foreign `local_path` straight to `os.makedirs()`
            # (confirmed happening in practice: `PermissionError: '/Users'`
            # on scicore from a Mac-written row).
            if raw_file is None or row.local_path is None or not os.path.exists(raw_file):
                return None
            return StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key=row.unit_id,
                output_path=row.local_path, inputs=(raw_file,),
                completion=Completion.PATH_EXISTS, meta=row.meta,
            )

        targets = self._plan_from_ledger(PipelineStep.PREPARE, TargetSelection(), build_target)
        if targets:
            return targets
        logger.warning(
            "No usable ledger row for source='%s' step='prepare' -- falling back to live discovery; "
            "run `data reconcile --source %s --step prepare` on this host for faster planning.",
            self.ID, self.ID,
        )
        return self._discover_prepare()

    def _discover_prepare(self) -> List[StepTarget]:
        raw_file = self._raw_prices_file()
        if not os.path.exists(raw_file):
            return []
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key="commodity_prices",
                output_path=os.path.join(self.output_root(PipelineStep.PREPARE), "commodity_prices.parquet"),
                inputs=(raw_file,), completion=Completion.PATH_EXISTS,
                meta={"raw_file": raw_file},
            )
        ]

    def _execute_prepare(self, target: StepTarget) -> bool:
        if not self.cfg.override and os.path.exists(target.output_path):
            logger.info("Skipping commodity prices processing, output already exists: %s", target.output_path)
            return True

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)
        (raw_file,) = target.inputs
        result_df = read_and_normalize_prices(raw_file)
        result_df.to_parquet(target.output_path, index=False)
        logger.info(
            "Commodity prices processing complete: %d (commodity, year) rows, %d commodities",
            len(result_df), result_df["commodity"].nunique(),
        )
        return True


registry.register(
    CommodityPricesSource.ID, __name__, CommodityPricesSource.__name__,
    CommodityPricesSource.STEPS, requires=CommodityPricesSource.REQUIRES,
)
