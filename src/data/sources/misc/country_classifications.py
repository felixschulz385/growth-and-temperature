"""UNDP HDI + World Bank income-group country classifications: fetch + prepare.

docs/design/09-integrated-pipeline.md §7 (the misc split): the third of the
three sources `misc.py` used to bundle. Two independently-fetched origins
(HDI csv, World Bank xlsx) sharing one joined prepare step -- kept together
deliberately (not a 4-way split) because they're joined into one
`iso3`-keyed table; see the design doc for the full reasoning and the
recorded escape hatch to split further later.

There is no separate GRID step -- `_execute_prepare` builds the joined
`classifications.parquet` (written as a real, standalone artifact:
`src/analysis/subsets/resolve.py` reads it directly by its own iso3 key, not
just as this source's internal intermediate) and then maps it onto gadm's
`GID_0` ids in the same call. `REQUIRES` on gadm's **PREPARE** output --
gadm's PREPARE produces `GID_0_code_mapping.json` directly (gadm's own
module docstring) -- is scoped to this source's own PREPARE step, so FETCH
runs unblocked before gadm exists, but PREPARE itself must wait for it.

**Not rasterized.** Every classification value (HDI tier, World Bank
income group) is constant across all of a country's pixels -- it varies only
by `GID_0`, never by pixel location -- so this writes a tiny `GID_0`-keyed
parquet table instead of a full pixel-grid zarr. Assembly merges it directly
onto rows via `src.data.assemble.processors.TileProcessor`'s `join_on`
mechanism (matching an existing `GID_0` column contributed by a gadm dataset
entry in the same assemble config), rather than every pixel in a country
carrying an identical rasterized boolean.

Ports `src/data/download/sources/misc.py::MiscDataSource` (hdi + worldbank_
income_classes configured files) and `src/data/preprocess/sources/misc.py::
MiscPreprocessor`'s `_process_country_classifications_target`. Output paths:
`prepared/misc/country_classifications/classifications.parquet` (the real,
externally-read intermediate), `prepared/misc/country_classifications/
classifications_by_gid0.parquet` (this PREPARE target's own output, no
readers anywhere in src/ today -- kept for future use, not a pixel-grid
zarr).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.misc._fetch import ConfiguredFile, ConfiguredFilesFetchMixin
from src.data.sources.misc.hdi import read_hdi
from src.data.sources.misc.worldbank import read_worldbank
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection
from src.data.sources import verify

logger = logging.getLogger(__name__)

DEFAULT_HDI_URL = "https://hdr.undp.org/sites/default/files/2025_HDR/HDR25_Composite_indices_complete_time_series.csv"
DEFAULT_HDI_NAME = "HDR25.csv"
DEFAULT_WB_URL = "https://ddh-openapi.worldbank.org/resources/DR0095334/download"
DEFAULT_WB_NAME = "DR0095334.xlsx"


class CountryClassificationsSource(ConfiguredFilesFetchMixin, DataSource):
    """UNDP HDI + World Bank income-group classifications, joined on `iso3`
    and keyed by GADM's `GID_0` id for a direct merge during assembly."""

    ID = "country_classifications"
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)
    REQUIRES = ((PipelineStep.PREPARE, "gadm", PipelineStep.PREPARE),)

    DATA_SOURCE_NAME = "country_classifications"

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="misc")
        if cfg.namespace is None:
            cfg = dataclasses.replace(cfg, namespace="country_classifications")
        super().__init__(ctx, cfg)

        hdi_url = cfg.raw.get("hdi_url", DEFAULT_HDI_URL)
        hdi_name = cfg.raw.get("hdi_name", DEFAULT_HDI_NAME)
        wb_url = cfg.raw.get("worldbank_url", DEFAULT_WB_URL)
        wb_name = cfg.raw.get("worldbank_name", DEFAULT_WB_NAME)
        self.CONFIGURED_FILES: List[ConfiguredFile] = [
            ConfiguredFile(key="hdi", url=hdi_url, name=hdi_name),
            ConfiguredFile(key="worldbank", url=wb_url, name=wb_name),
        ]

        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="country_classifications_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

    @property
    def data_path(self) -> str:
        return f"{self.cfg.data_path}/{self.cfg.namespace}"

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

    def _execute(self, target: StepTarget) -> bool:
        if target.step is PipelineStep.FETCH:
            return self._execute_fetch(target)
        if target.step is PipelineStep.PREPARE:
            return self._execute_prepare(target)
        raise AssertionError(f"unreachable: {target.step}")

    def _execute_fetch(self, target: StepTarget) -> bool:
        # FETCH is local-disk only now -- no HPC target required. `data
        # transfer` (separate, manual or auto per source config) is the only
        # thing that pushes to HPC.
        from src.data.common.fetch.driver import run_fetch

        return run_fetch(self, **self.cfg.raw.get("download", {}))

    # -- PREPARE (raw HDI/World Bank files -> joined table -> GID_0 table) --

    def _raw_file(self, key: str) -> str:
        name = next(f.name for f in self.CONFIGURED_FILES if f.key == key)
        return os.path.join(self.output_root(PipelineStep.FETCH), name)

    def _classifications_path(self) -> str:
        """The joined-but-not-yet-GID_0-mapped intermediate -- a real,
        externally-read artefact (module docstring), so it keeps its own
        PREPARE-stage path (`prepared/<data_path>/<namespace>/`)."""
        return os.path.join(self.output_root(PipelineStep.PREPARE), "classifications.parquet")

    def _output_path(self) -> str:
        # A small per-GID parquet table, not a `<family>.zarr` pixel-grid
        # store, so it lives under prepared/, not grid/<grid_id>/ (no
        # readers anywhere in src/ today; kept for future use).
        return os.path.join(
            self.output_root(PipelineStep.PREPARE), "classifications_by_gid0.parquet"
        )

    def _plan_prepare(self) -> List[StepTarget]:
        hdi_file, wb_file = self._raw_file("hdi"), self._raw_file("worldbank")
        has_hdi, has_wb = os.path.exists(hdi_file), os.path.exists(wb_file)
        if not has_hdi and not has_wb:
            return []
        inputs = tuple(f for f, present in ((hdi_file, has_hdi), (wb_file, has_wb)) if present)
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key="country_classifications",
                output_path=self._output_path(),
                inputs=inputs, completion=Completion.PATH_EXISTS,
                # value_cols vary by which of hdi/worldbank were available,
                # so only the always-present join key is checked.
                meta={
                    "has_hdi": has_hdi, "has_wb": has_wb,
                    **verify.verification_meta(self.cfg.raw, expected_vars=("GID_0",)),
                },
            )
        ]

    def _build_classifications_table(self, target: StepTarget):
        """Phase 1: join HDI + World Bank into `classifications.parquet`,
        the real externally-read intermediate (module docstring) -- reused
        as-is if it already exists (resumable, same idea as every other
        merged source's phase-1 skip)."""
        classifications_path = self._classifications_path()
        if not self.cfg.override and os.path.exists(classifications_path):
            return classifications_path

        os.makedirs(os.path.dirname(classifications_path), exist_ok=True)
        hdi_wide = read_hdi(self._raw_file("hdi")) if target.meta.get("has_hdi") else None
        wb_wide = read_worldbank(self._raw_file("worldbank")) if target.meta.get("has_wb") else None

        if hdi_wide is not None and wb_wide is not None:
            result_df = hdi_wide.merge(wb_wide, on="iso3", how="left")
            bool_cols = [c for c in wb_wide.columns if c != "iso3"]
            result_df[bool_cols] = result_df[bool_cols].fillna(False).astype(bool)
        elif hdi_wide is not None:
            result_df = hdi_wide
        elif wb_wide is not None:
            result_df = wb_wide
        else:
            logger.error("No data to process")
            return None

        result_df.to_parquet(classifications_path, index=False)
        logger.info("Country classifications processing complete: %d countries", len(result_df))
        return classifications_path

    def _execute_prepare(self, target: StepTarget) -> bool:
        import pandas as pd

        if not self.cfg.override and os.path.exists(target.output_path):
            logger.info("Skipping country classifications GID_0 table -- already exists: %s", target.output_path)
            return True

        classifications_path = self._build_classifications_table(target)
        if classifications_path is None:
            return False

        from src.data.sources.misc.gadm import gid_mapping_path

        country_mapping_file = gid_mapping_path(self.ctx.data_root, self.ctx.grid_id, "GID_0")
        if not os.path.exists(country_mapping_file):
            logger.error("Country mapping file not found: %s", country_mapping_file)
            return False

        from src.analysis.subsets.registry import load_country_registry

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)
        classifications_df = pd.read_parquet(classifications_path)
        country_code_to_id = load_country_registry(Path(country_mapping_file)).country_to_id
        classifications_df["GID_0"] = classifications_df["iso3"].map(lambda x: country_code_to_id.get(x, 0))
        classifications_df = classifications_df[classifications_df["GID_0"] != 0]

        value_cols = [c for c in classifications_df.columns if c not in ("iso3", "GID_0")]
        out_df = classifications_df[["GID_0"] + value_cols]
        out_df.to_parquet(target.output_path, index=False)
        logger.info(
            "Country classifications GID_0 table complete: %d countries, columns %s",
            len(out_df), value_cols,
        )
        return True


registry.register(
    CountryClassificationsSource.ID, __name__, CountryClassificationsSource.__name__,
    CountryClassificationsSource.STEPS, requires=CountryClassificationsSource.REQUIRES,
)
