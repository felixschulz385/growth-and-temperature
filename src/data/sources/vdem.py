"""V-Dem (Varieties of Democracy) country-year indices: fetch + prepare, no grid.

The successor project on pollution along the growth path by institutions
(docs/successors/pollution-institutions-prespec.md) uses V-Dem as its measure
of institutions. V-Dem's website puts the download behind a form, so FETCH
takes the full country-year dataset from the V-Dem Institute's own R data
package on GitHub (`vdeminstitute/vdemdata`, `data/vdem.RData`), pinned to the
commit of a release. The file is ~34 MB, small enough for the proxy; like
`ports`, `download_async()` runs a `requests` download in an executor.

PREPARE reads the `.RData` with `pyreadr` and keeps the configured
`variables` (indices on 0-1 scales, plus `v2x_regime`, the Regimes of the World
category) with the country keys, as
`prepared/vdem/misc/vdem_country_year.parquet`: `country_text_id`
(V-Dem's three-letter code, ISO 3166 alpha-3 for almost every country),
`country_name`, `year` and one column per variable.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, List

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.misc._fetch import ConfiguredFile, ConfiguredFilesFetchMixin
from src.data.sources.steps import Completion, PipelineStep, StepTarget

logger = logging.getLogger(__name__)

#: commit "v16 release" (2026-03-17) of vdeminstitute/vdemdata
DEFAULT_COMMIT = "f4dd26922e658442524dfd954bf14f7ebe622d5d"
DEFAULT_NAME = "vdem_v16.RData"
DEFAULT_VARIABLES = (
    "v2x_polyarchy",   # electoral democracy
    "v2x_libdem",      # liberal democracy
    "v2x_rule",        # rule of law
    "v2x_corr",        # political corruption (higher = more corrupt)
    "v2xcs_ccsi",      # core civil society
    "v2x_regime",      # Regimes of the World: 0 closed autocracy ... 3 liberal democracy
)
_KEYS = ("country_text_id", "country_name", "year")


class VdemSource(ConfiguredFilesFetchMixin, DataSource):
    """V-Dem country-year indices."""

    ID = "vdem"
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)

    DATA_SOURCE_NAME = "vdem"

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="vdem")
        super().__init__(ctx, cfg)

        commit = cfg.raw.get("commit", DEFAULT_COMMIT)
        url = cfg.raw.get(
            "url", f"https://github.com/vdeminstitute/vdemdata/raw/{commit}/data/vdem.RData"
        )
        name = cfg.raw.get("name", DEFAULT_NAME)
        self.CONFIGURED_FILES: List[ConfiguredFile] = [ConfiguredFile(key="vdem", url=url, name=name)]
        self.variables = tuple(cfg.raw.get("variables", DEFAULT_VARIABLES))

    async def download_async(self, file_url: str, output_path: str, session: Any = None) -> None:
        """Synchronous `requests` download in an executor (module docstring)."""
        import asyncio

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.download, file_url, output_path, None)

    # -- PREPARE ---------------------------------------------------------------

    def _raw_file_path(self) -> str:
        return os.path.join(self.output_root(PipelineStep.FETCH), self.CONFIGURED_FILES[0].name)

    def _plan_prepare(self) -> List[StepTarget]:
        raw_file = self._raw_file_path()
        if not os.path.exists(raw_file):
            return []
        root = self.output_root(PipelineStep.PREPARE, agg=layout.MISC_AGG)
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key="vdem",
                output_path=os.path.join(root, "vdem_country_year.parquet"),
                inputs=(raw_file,), completion=Completion.PATH_EXISTS,
            )
        ]

    def _execute_prepare(self, target: StepTarget) -> bool:
        import pyreadr

        if not self.cfg.override and os.path.exists(target.output_path):
            logger.info("Skipping V-Dem PREPARE, output already exists: %s", target.output_path)
            return True

        frames = pyreadr.read_r(target.inputs[0])
        vdem = frames["vdem"] if "vdem" in frames else next(iter(frames.values()))
        missing = [c for c in (*_KEYS, *self.variables) if c not in vdem.columns]
        if missing:
            raise ValueError(f"V-Dem file lacks columns {missing}")
        out = vdem[[*_KEYS, *self.variables]].copy()
        out["year"] = out.year.astype("int32")

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)
        tmp = f"{target.output_path}.tmp"
        out.to_parquet(tmp, index=False)
        os.replace(tmp, target.output_path)
        logger.info(
            "V-Dem PREPARE complete: %d country-years, %d countries, %d-%d -> %s",
            len(out), out.country_text_id.nunique(), out.year.min(), out.year.max(), target.output_path,
        )
        return True


registry.register(VdemSource.ID, __name__, VdemSource.__name__, VdemSource.STEPS)
