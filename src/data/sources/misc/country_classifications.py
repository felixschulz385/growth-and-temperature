"""UNDP HDI + World Bank income-group country classifications: fetch +
prepare + grid.

docs/design/09-integrated-pipeline.md §7 (the misc split): the third of the
three sources `misc.py` used to bundle. Two independently-fetched origins
(HDI csv, World Bank xlsx) sharing one joined prepare+grid step -- kept
together deliberately (not a 4-way split) because they're joined into one
`iso3`-keyed table and written into one Zarr store via sequential
`mode='a'` writes; see the design doc for the full reasoning and the
recorded escape hatch to split further later. `REQUIRES` on gadm's GRID
output is the first real use of that mechanism in this codebase -- it needs
GADM's country-id raster and `country_code_mapping.json` to rasterize onto.

Ports `src/data/download/sources/misc.py::MiscDataSource` (hdi + worldbank_
income_classes configured files) and `src/data/preprocess/sources/misc.py::
MiscPreprocessor`'s `_process_country_classifications_target`/
`_rasterize_country_classifications_target` (`stage="vector"` -> PREPARE,
`stage="spatial"` -> GRID). Output paths unchanged: `misc/processed/stage_1/
country_classifications/classifications.parquet`, `misc/processed/stage_2/
country_classifications/classifications_grid.zarr`.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
from pathlib import Path
from typing import List

from zarr.codecs import BloscCodec

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.misc._fetch import ConfiguredFile, ConfiguredFilesFetchMixin
from src.data.sources.misc.hdi import read_hdi
from src.data.sources.misc.worldbank import read_worldbank
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection

logger = logging.getLogger(__name__)

DEFAULT_HDI_URL = "https://hdr.undp.org/sites/default/files/2025_HDR/HDR25_Composite_indices_complete_time_series.csv"
DEFAULT_HDI_NAME = "HDR25.csv"
DEFAULT_WB_URL = "https://ddh-openapi.worldbank.org/resources/DR0095334/download"
DEFAULT_WB_NAME = "DR0095334.xlsx"


class CountryClassificationsSource(ConfiguredFilesFetchMixin, DataSource):
    """UNDP HDI + World Bank income-group classifications, joined on `iso3`
    and rasterized onto GADM's country-id grid."""

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
        import asyncio

        from src.data.common.fetch.async_downloader import run_async_download_workflow
        from src.data.common.hpc.client import HPCClient
        from src.data.common.index.unified_index import UnifiedDataIndex

        index = UnifiedDataIndex(
            bucket_name="", data_source=self, local_index_dir=self.ctx.local_index_dir,
            key_file=self.ctx.key_file, hpc_mode=bool(self.ctx.ssh_target),
        )
        index.build_index_from_source(data_source=self, rebuild=False, only_missing_entrypoints=True)
        index.save()

        hpc_client = HPCClient(target=self.ctx.ssh_target, key_file=self.ctx.key_file)
        return asyncio.run(
            run_async_download_workflow(
                data_source=self, index=index, hpc_client=hpc_client, context=self.ctx,
                config=dict(self.cfg.raw.get("download", {})),
            )
        )

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
                output_path=layout.grid_store_path(
                    self.ctx.data_root,
                    self.cfg.data_path,
                    "classifications_grid.zarr",
                    namespace=self.cfg.namespace,
                    grid_id=self.ctx.grid_id,
                    layout=self.ctx.layout,
                    v2_family="classifications",
                ),
                inputs=(classifications_parquet, gadm_zarr), completion=Completion.PATH_EXISTS,
            )
        ]

    def _execute_grid(self, target: StepTarget) -> bool:
        import pandas as pd
        import xarray as xr

        if not self.cfg.override and os.path.exists(target.output_path):
            logger.info("Skipping country classifications rasterization, output already exists: %s", target.output_path)
            return True

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)

        classifications_parquet, gadm_zarr = target.inputs
        classifications_df = pd.read_parquet(classifications_parquet)

        gadm_grid = xr.open_zarr(gadm_zarr, chunks="auto", consolidated=False)
        country_grid = gadm_grid.country.astype("int16").compute()

        country_mapping_file = os.path.join(os.path.dirname(gadm_zarr), "country_code_mapping.json")
        if not os.path.exists(country_mapping_file):
            logger.error("Country mapping file not found: %s", country_mapping_file)
            return False

        from src.analysis.subsets.registry import load_country_registry

        country_code_to_id = load_country_registry(Path(country_mapping_file)).country_to_id
        classifications_df["country_id"] = classifications_df["iso3"].map(lambda x: country_code_to_id.get(x, 0))

        classification_cols = [c for c in classifications_df.columns if c not in ("iso3", "country_id")]
        for col in classification_cols:
            country_ids_with_classification = classifications_df.query(f"{col} & country_id!=0").country_id.unique()
            classification_array = country_grid.isin(country_ids_with_classification)
            classification_array.attrs = {"description": f"{col} classification grid (True/False)"}
            if "crs" in country_grid.attrs:
                classification_array = classification_array.rio.write_crs(country_grid.attrs["crs"])
                classification_array = classification_array.odc.assign_crs(country_grid.attrs["crs"])

            single_ds = xr.Dataset(
                {col: classification_array},
                attrs={
                    "description": "Country classifications grid (HDI and World Bank income groups)",
                    "source": "UNDP HDI and World Bank income classifications",
                    "note": "Boolean values: True where classification applies, False otherwise",
                },
            )
            compressor = BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle", blocksize=0)
            encoding = {col: {"chunks": (512, 512), "compressors": compressor, "dtype": "bool"}}
            single_ds.to_zarr(target.output_path, mode="a", encoding=encoding, zarr_format=3, consolidated=False)
            del classification_array, single_ds

        gadm_grid.close()
        return True


registry.register(
    CountryClassificationsSource.ID, __name__, CountryClassificationsSource.__name__,
    CountryClassificationsSource.STEPS, requires=CountryClassificationsSource.REQUIRES,
)
