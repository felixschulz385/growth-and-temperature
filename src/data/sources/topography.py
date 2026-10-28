"""EarthEnv-Topography slope: fetch + prepare onto the 1 km grid.

Amatulli et al. 2018 (Sci Data 5:180040), built from GMTED2010: the mean of
250 m slopes (degrees) within each 5-arc-minute (~10 km) cell,
`slope_10KMmn_GMTEDmd.tif`, one global GeoTIFF of ~9 MB, so it can come
through the proxy as well as from the transfer node. The corridor
instrument's route cost reads it (docs/successors/corridors-prespec.md §3).
The 1 km product (`slope_1KMmd_GMTEDmd.tif`, ~700 MB) is a transfer-node
download if finer slope is ever needed.

PREPARE puts `slope_mean_10km` onto the canonical 1 km grid as a static
family (`prepared/topography/crs/<grid_id>/slope/`), nearest-neighbour, so
every 1 km pixel carries the slope of the ~10 km cell it falls in. That is a
pixel grid like any other source's: assembly can block-average it, and
`corridors` averages it per 10 km analysis cell. Like `ports`,
`download_async()` runs the `requests`-based `download()` in an executor so
the proxy environment variables apply.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, List

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry, verify
from src.data.sources.base import DataSource
from src.data.sources.misc._fetch import ConfiguredFile, ConfiguredFilesFetchMixin
from src.data.sources.steps import Completion, PipelineStep, StepTarget

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://data.earthenv.org/topography/slope_10KMmn_GMTEDmd.tif"
DEFAULT_NAME = "slope_10KMmn_GMTEDmd.tif"
VARIABLE = "slope_mean_10km"


class TopographySource(ConfiguredFilesFetchMixin, DataSource):
    """EarthEnv-Topography mean slope, static 1 km pixel grid."""

    ID = "topography"
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)

    DATA_SOURCE_NAME = "topography"
    #: bump to force a full reprocess (`run_tiled_prepare`'s `processing_version`)
    PROCESSING_VERSION = "1-tiled"

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="topography")
        super().__init__(ctx, cfg)

        url = cfg.raw.get("url", DEFAULT_URL)
        name = cfg.raw.get("name", DEFAULT_NAME)
        self.CONFIGURED_FILES: List[ConfiguredFile] = [ConfiguredFile(key="slope", url=url, name=name)]

    async def download_async(self, file_url: str, output_path: str, session: Any = None) -> None:
        """Synchronous `requests` download in an executor (module docstring)."""
        import asyncio

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.download, file_url, output_path, None)

    # -- PREPARE (GeoTIFF -> static 1 km grid) ------------------------------

    def _raw_file_path(self) -> str:
        return os.path.join(self.output_root(PipelineStep.FETCH), self.CONFIGURED_FILES[0].name)

    def _output_path(self) -> str:
        return layout.grid_store_path(
            self.ctx.data_root,
            self.cfg.data_path,
            grid_id=self.ctx.grid_id,
            family="slope",
            suffix="",  # cell_id-keyed parquet parts, not a Zarr store -- see grid_store_path docstring
        )

    def _plan_prepare(self) -> List[StepTarget]:
        raw_file = self._raw_file_path()
        if not os.path.exists(raw_file):
            return []
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key="slope",
                output_path=self._output_path(),
                inputs=(raw_file,), completion=Completion.MARKER,
                meta=verify.verification_meta(self.cfg.raw, expected_vars=(VARIABLE,), value_range=(0, 90)),
            )
        ]

    def _execute_prepare(self, target: StepTarget) -> bool:
        import numpy as np
        import rioxarray as rxr
        import xarray as xr

        from src.data.common.geobox import get_target_geobox
        from src.data.common.prepare.driver import run_tiled_prepare
        from src.data.common.raster.spatial import SpatialProcessor, sel_bbox
        from src.data.sources.steps import is_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping topography PREPARE -- already complete: %s", target.output_path)
            return True

        # ~9 MB: loaded once, eagerly, and clipped per tile.
        da = rxr.open_rasterio(target.inputs[0], mask_and_scale=True).squeeze("band", drop=True)
        ds = xr.Dataset({VARIABLE: da.astype(np.float32)}).rio.write_crs(da.rio.crs)

        geobox = get_target_geobox(self.ctx)
        processor = SpatialProcessor(hpc_root=self.ctx.data_root, target_geobox=geobox, default_nodata=np.nan)

        def raw_getter(tile, year):
            bbox = tile.geobox.pad(32, 32).extent.to_crs(ds.rio.crs).boundingbox
            clipped = sel_bbox(ds, bbox, y_dim="y", x_dim="x")
            if clipped.sizes.get("y", 0) == 0 or clipped.sizes.get("x", 0) == 0:
                # Outside the raster (it ends at 56°S): NaN tile, not a
                # failure. Built on tile.geobox itself so reprojecting it is
                # an identity.
                from odc.geo.xr import xr_zeros

                return xr.Dataset({VARIABLE: xr_zeros(tile.geobox, dtype="float32") * np.nan})
            return clipped

        return run_tiled_prepare(
            output_path=target.output_path,
            years=None,
            variables=[VARIABLE],
            target_geobox=geobox,
            processor=processor,
            raw_getter=raw_getter,
            resampling="nearest",
            processing_version=self.PROCESSING_VERSION,
            override=self.cfg.override,
        )


registry.register(TopographySource.ID, __name__, TopographySource.__name__, TopographySource.STEPS)
