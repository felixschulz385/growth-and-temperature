"""Berman et al. mining conflict data: fetch (manual) + grid, no prepare.

docs/design/09-integrated-pipeline.md §5. Merges
`src/data/download/sources/manual.py::BermanMiningDataSource` (a manual
fetch -- prompts the user for a local file path, since ICPSR requires
authenticated manual download) and
`src/data/preprocess/sources/berman_mining.py::BermanMiningPreprocessor`
(`stage="spatial"` -> GRID). **No PREPARE step**: like PLAD, mining-point
gridding happens directly from the raw `.dta` file in one stage.

**No `REQUIRES` on gadm** -- correcting an earlier, unverified planning
assumption: this source only shares the VIIRS-derived geobox *cache location*
with osm/gadm (`get_or_create_geobox`, which builds the cache independently
from a VIIRS download if missing, never from GADM), not a hard dependency on
GADM's output.

**Two real bugs fixed here, not silently ported**: the old
`BermanMiningPreprocessor` defines `get_hpc_output_path` and `from_config`
twice each in the same class body (verified by direct source inspection,
pinned in tests/data/preprocess/sources/test_characterization_berman_mining.py)
-- the second definitions silently shadow the first, byte-identical dead
code. Not carried forward.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from zarr.codecs import BloscCodec

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection

logger = logging.getLogger(__name__)


class BermanMiningSource(DataSource):
    """Berman et al. mining conflict/resource-trap data -- gridded mine-count variables."""

    ID = "berman_mining"
    ALIASES = ("berman", "mining_conflict")
    STEPS = (PipelineStep.FETCH, PipelineStep.GRID)

    DATA_SOURCE_NAME = "berman_mining"
    has_entrypoints = False

    MANUAL_FILE = {
        "name": "BCRT_baseline.dta",
        "description": "Berman et al. Mining Conflict Resource Traps - Baseline Data",
        "url": "https://www.openicpsr.org/openicpsr/project/113068/version/V1/view",
        "subfolder": "baseline",
    }
    DOWNLOAD_INSTRUCTIONS = """
1. Visit: https://www.openicpsr.org/openicpsr/project/113068/version/V1/view
2. Log in or create an ICPSR account
3. Navigate to Data/BCRT_baseline.dta
4. Download the file to your local machine
5. Provide the path to the downloaded file below
"""

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="berman_mining")
        super().__init__(ctx, cfg)

        self.mining_data_path = cfg.raw.get("mining_data_path") or os.path.join(
            self.output_root(PipelineStep.FETCH), "baseline", "BCRT_baseline.dta"
        )
        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="berman_mining_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # RemoteFileCatalog contract (ports ManualDataSource verbatim)
    # ------------------------------------------------------------------

    def list_remote_files(self, entrypoint: Optional[dict] = None) -> List[tuple]:
        return [(f"{self.MANUAL_FILE['subfolder']}/{self.MANUAL_FILE['name']}", self.MANUAL_FILE["url"])]

    def get_file_hash(self, file_url: str) -> str:
        import hashlib

        return hashlib.md5(file_url.encode("utf-8")).hexdigest()

    def local_path(self, relative_path: str) -> str:
        return os.path.join("data", self.cfg.data_path, relative_path)

    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        return None

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        return []

    def _prompt_for_file_path(self) -> Optional[str]:
        from pathlib import Path

        print("\n" + "=" * 70)
        print(f"Manual Download Required: {self.MANUAL_FILE['name']}")
        print("=" * 70)
        print(f"\nDescription: {self.MANUAL_FILE['description']}")
        print(f"\nReference URL: {self.MANUAL_FILE['url']}")
        print(f"\nInstructions:\n{self.DOWNLOAD_INSTRUCTIONS}")
        print("\nPlease download this file manually and provide the path below.")
        print("Or press Enter to skip this file.")

        while True:
            file_path = input("\nFile path: ").strip()
            if not file_path:
                return None
            resolved = Path(file_path).expanduser().resolve()
            if not resolved.exists():
                print(f"Error: File not found: {resolved}")
                continue
            if not resolved.is_file():
                print(f"Error: Path is not a file: {resolved}")
                continue
            return str(resolved)

    def download(self, file_url: str, output_path: str, session: Any = None) -> None:
        import shutil

        source_path = self._prompt_for_file_path()
        if not source_path:
            raise FileNotFoundError(f"User skipped manual download for {self.MANUAL_FILE['name']}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(source_path, output_path)

    async def download_async(self, file_url: str, output_path: str, session: Any = None) -> None:
        import asyncio

        await asyncio.get_event_loop().run_in_executor(None, self.download, file_url, output_path, session)

    # ------------------------------------------------------------------
    # plan()/execute() dispatch
    # ------------------------------------------------------------------

    def _plan(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        if step is PipelineStep.FETCH:
            return [
                StepTarget(
                    source_id=self.ID, step=PipelineStep.FETCH, key="all",
                    output_path=self.output_root(PipelineStep.FETCH), completion=Completion.NEVER,
                )
            ]
        if step is PipelineStep.GRID:
            return [
                StepTarget(
                    source_id=self.ID, step=PipelineStep.GRID, key="all",
                    output_path=layout.grid_store_path(
                        self.ctx.data_root,
                        self.cfg.data_path,
                        "berman_mining_timeseries_reprojected.zarr",
                        namespace=self.cfg.namespace,
                        grid_id=self.ctx.grid_id,
                        layout=self.ctx.layout,
                        v2_family="berman_mining",
                    ),
                    completion=Completion.PATH_EXISTS,
                    meta={"year_range": self.cfg.year_range},
                )
            ]
        raise AssertionError(f"unreachable: {step}")

    def _execute(self, target: StepTarget) -> bool:
        if target.step is PipelineStep.FETCH:
            return self._execute_fetch(target)
        if target.step is PipelineStep.GRID:
            return self._execute_grid(target)
        raise AssertionError(f"unreachable: {target.step}")

    def _execute_fetch(self, target: StepTarget) -> bool:
        if not self.ctx.ssh_target:
            logger.warning("Fetch requires an HPC/remote target to be configured.")
            return False

        from src.data.common.fetch.driver import run_fetch

        return run_fetch(self, **self.cfg.raw.get("download", {}))

    # -- GRID ("spatial") -------------------------------------------------

    def _get_or_create_geobox(self):
        from src.data.common.geobox import get_target_geobox

        return get_target_geobox(self.ctx)

    def _create_mining_dataset(self, year_range: Optional[tuple]):
        import pandas as pd
        import xarray as xr

        mines = pd.read_stata(self.mining_data_path)
        variables = ["nb_mines_a", "nb_diamond"]
        mines_ds = xr.Dataset.from_dataframe(mines.set_index(["latitude", "longitude", "year"])[variables])
        mines_ds = mines_ds.rio.write_crs(4326)
        if year_range:
            mines_ds = mines_ds.sel(year=slice(year_range[0], year_range[1]))
        return mines_ds

    def _execute_grid(self, target: StepTarget) -> bool:
        import numpy as np
        import pandas as pd
        from odc.geo.xr import xr_reproject
        from src.data.sources.steps import is_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping spatial processing, output already exists: %s", target.output_path)
            return True

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)
        try:
            mines_ds = self._create_mining_dataset(target.meta.get("year_range"))
            if mines_ds is None:
                logger.error("Failed to create mining dataset")
                return False

            geobox = self._get_or_create_geobox()
            for var in mines_ds.data_vars:
                mines_ds[var] = mines_ds[var].fillna(255).astype(np.uint8, casting="unsafe")

            reprojected_ds = xr_reproject(mines_ds, geobox, resampling="nearest", dst_nodata=255)
            years = sorted(reprojected_ds.year.values)
            reprojected_ds = reprojected_ds.rename({"year": "time"})
            reprojected_ds["time"] = pd.to_datetime([f"{year}-12-31" for year in years])
            reprojected_ds = reprojected_ds.expand_dims("band").assign_coords(band=[1])
            dim_y, dim_x = geobox.dimensions
            reprojected_ds = reprojected_ds.assign_coords(
                {dim_y: geobox.coords[dim_y].values.round(5), dim_x: geobox.coords[dim_x].values.round(5)}
            )
            reprojected_ds = reprojected_ds.drop_vars(["spatial_ref"], errors="ignore")
            reprojected_ds = reprojected_ds.chunk({"time": 1, "band": 1, dim_y: 512, dim_x: 512})

            compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", blocksize=0)
            encoding = {
                var: {"chunks": (1, 512, 512, 1), "compressors": (compressor,), "dtype": "uint8", "fill_value": 255}
                for var in reprojected_ds.data_vars
            }
            reprojected_ds.to_zarr(target.output_path, mode="w", encoding=encoding, zarr_format=3, consolidated=False)
            return True
        except Exception:
            logger.exception("Error in Berman mining spatial processing")
            return False


registry.register(BermanMiningSource.ID, __name__, BermanMiningSource.__name__, BermanMiningSource.STEPS, aliases=BermanMiningSource.ALIASES)
