"""UNDP HDI + World Bank income-group country classifications: fetch +
prepare + grid.

docs/design/09-integrated-pipeline.md §7 (the misc split): the third of the
three sources `misc.py` used to bundle. Two independently-fetched origins
(HDI csv, World Bank xlsx) sharing one joined prepare+grid step -- kept
together deliberately (not a 4-way split) because they're joined into one
`iso3`-keyed table; see the design doc for the full reasoning and the
recorded escape hatch to split further later. `REQUIRES` on gadm's GRID
output is the first real use of that mechanism in this codebase -- it needs
`GID_0_code_mapping.json` to translate `iso3` into gadm's integer `GID_0` ids.

**No longer rasterized.** Every classification value (HDI tier, World Bank
income group) is constant across all of a country's pixels -- it varies only
by `GID_0`, never by pixel location -- so GRID now writes a tiny
`GID_0`-keyed parquet table instead of a full pixel-grid zarr. Assembly
merges it directly onto rows via `src.data.assemble.processors.TileProcessor`'s
`join_on` mechanism (matching an existing `GID_0` column contributed by a
gadm dataset entry in the same assemble config), rather than every pixel in
a country carrying an identical rasterized boolean.

Ports `src/data/download/sources/misc.py::MiscDataSource` (hdi + worldbank_
income_classes configured files) and `src/data/preprocess/sources/misc.py::
MiscPreprocessor`'s `_process_country_classifications_target` (`stage="vector"`
-> PREPARE). Output paths: `misc/processed/stage_1/country_classifications/
classifications.parquet` (unchanged), `misc/processed/stage_2/
country_classifications/classifications_by_gid0.parquet` (was
`classifications_grid.zarr`).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
from pathlib import Path
from typing import List

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
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE, PipelineStep.GRID)
    REQUIRES = (("gadm", PipelineStep.GRID),)

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

    def _plan(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        if step is PipelineStep.FETCH:
            return [
                StepTarget(
                    source_id=self.ID, step=PipelineStep.FETCH, key="all",
                    output_path=self.output_root(PipelineStep.FETCH), completion=Completion.NEVER,
                )
            ]
        if step is PipelineStep.PREPARE:
            return self._plan_prepare()
        if step is PipelineStep.GRID:
            return self._plan_grid()
        raise AssertionError(f"unreachable: {step}")

    def _execute(self, target: StepTarget) -> bool:
        if target.step is PipelineStep.FETCH:
            return self._execute_fetch(target)
        if target.step is PipelineStep.PREPARE:
            return self._execute_prepare(target)
        if target.step is PipelineStep.GRID:
            return self._execute_grid(target)
        raise AssertionError(f"unreachable: {target.step}")

    def _execute_fetch(self, target: StepTarget) -> bool:
        if not self.ctx.ssh_target:
            logger.warning("Fetch requires an HPC/remote target to be configured.")
            return False

        from src.data.common.fetch.driver import run_fetch

        return run_fetch(self, **self.cfg.raw.get("download", {}))

    # -- PREPARE ("vector") --------------------------------------------------

    def _raw_file(self, key: str) -> str:
        name = next(f.name for f in self.CONFIGURED_FILES if f.key == key)
        return os.path.join(self.output_root(PipelineStep.FETCH), name)

    def _plan_prepare(self) -> List[StepTarget]:
        hdi_file, wb_file = self._raw_file("hdi"), self._raw_file("worldbank")
        has_hdi, has_wb = os.path.exists(hdi_file), os.path.exists(wb_file)
        if not has_hdi and not has_wb:
            return []
        inputs = tuple(f for f, present in ((hdi_file, has_hdi), (wb_file, has_wb)) if present)
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key="country_classifications",
                output_path=os.path.join(self.output_root(PipelineStep.PREPARE), "classifications.parquet"),
                inputs=inputs, completion=Completion.PATH_EXISTS,
                meta={"has_hdi": has_hdi, "has_wb": has_wb},
            )
        ]

    def _execute_prepare(self, target: StepTarget) -> bool:
        if not self.cfg.override and os.path.exists(target.output_path):
            logger.info("Skipping country classifications processing, output already exists: %s", target.output_path)
            return True

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)

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
            return False

        result_df.to_parquet(target.output_path, index=False)
        logger.info("Country classifications processing complete: %d countries", len(result_df))
        return True

    # -- GRID ("spatial") -----------------------------------------------------

    def _plan_grid(self) -> List[StepTarget]:
        classifications_parquet = os.path.join(self.output_root(PipelineStep.PREPARE), "classifications.parquet")
        if not os.path.exists(classifications_parquet):
            return []

        # REQUIRES=(("gadm", GRID),) -- resolve gadm's own layout directly
        # rather than importing gadm's class (docs/design/09-integrated-pipeline.md
        # §2: cross-source coupling is on artefact paths, never a class import).
        # Must mirror GadmSource._plan_grid()'s own grid_store_path() call
        # (same v2_family="country_id") so this keeps finding gadm's output
        # under layout=v2 too, not just the legacy layout.
        gadm_zarr = layout.grid_store_path(
            self.ctx.data_root,
            "misc",
            "countries_grid.zarr",
            namespace="gadm",
            grid_id=self.ctx.grid_id,
            layout=self.ctx.layout,
            v2_family="country_id",
        )
        if not os.path.exists(gadm_zarr):
            return []

        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.GRID, key="country_classifications",
                # v2_family intentionally omitted: this is a small per-GID
                # parquet table, not a `<family>.zarr` pixel-grid store, so it
                # doesn't participate in layout:v2's "one store per family"
                # zarr directory -- grid_store_path() falls back to the
                # legacy per-source path shape regardless of ctx.layout.
                output_path=layout.grid_store_path(
                    self.ctx.data_root,
                    self.cfg.data_path,
                    "classifications_by_gid0.parquet",
                    namespace=self.cfg.namespace,
                    grid_id=self.ctx.grid_id,
                    layout=self.ctx.layout,
                ),
                inputs=(classifications_parquet, gadm_zarr), completion=Completion.PATH_EXISTS,
                # value_cols vary by which of hdi/worldbank were available at
                # PREPARE time, so only the always-present join key is checked.
                meta=verify.verification_meta(self.cfg.raw, expected_vars=("GID_0",)),
            )
        ]

    def _execute_grid(self, target: StepTarget) -> bool:
        import pandas as pd

        if not self.cfg.override and os.path.exists(target.output_path):
            logger.info("Skipping country classifications GID_0 table, output already exists: %s", target.output_path)
            return True

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)

        from src.data.sources.misc.gadm import gid_mapping_path

        classifications_parquet, gadm_zarr = target.inputs
        classifications_df = pd.read_parquet(classifications_parquet)

        country_mapping_file = gid_mapping_path(self.ctx.data_root, self.ctx.grid_id, self.ctx.layout, "GID_0")
        if not os.path.exists(country_mapping_file):
            logger.error("Country mapping file not found: %s", country_mapping_file)
            return False

        from src.analysis.subsets.registry import load_country_registry

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
