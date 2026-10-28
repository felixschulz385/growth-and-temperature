"""GRIP4 road density: fetch + prepare onto the 1 km grid.

Meijer et al. 2018 (Environ. Res. Lett. 13:064006), Global Roads Inventory
Project v4: road length density (m/km²) per 5-arc-minute (~8 km) cell, by
road type -- tp1 highways, tp2 primary, tp3 secondary -- and in total. One
~2 MB zip per type (an ESRI ASCII grid without a CRS: WGS84 is assumed), so
it can come through the proxy as well as from the transfer node. Like
`ports`/`topography`, `download_async()` runs a `requests` download in an
executor so the proxy variables apply.

The roads are those mapped around 2018, so they are partly an outcome of
2002-2018 growth. The corridor fork uses them only to check that predicted
routes run along real roads (docs/successors/corridors-prespec.md), never as
an instrument or control.

PREPARE puts every type onto the canonical 1 km grid as one static family
(`prepared/grip/crs/<grid_id>/road_density/`), nearest-neighbour, columns
`grip_<type>_m_km2`.
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

BASE_URL = "https://dataportaal.pbl.nl/data/GRIP4"
ROAD_TYPES = ("tp1", "tp2", "tp3", "total")


def _variable(road_type: str) -> str:
    return f"grip_{road_type}_m_km2"


class GripSource(ConfiguredFilesFetchMixin, DataSource):
    """GRIP4 road length density by road type, static 1 km pixel grid."""

    ID = "grip"
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)

    DATA_SOURCE_NAME = "grip"
    #: bump to force a full reprocess (`run_tiled_prepare`'s `processing_version`)
    PROCESSING_VERSION = "1-tiled"

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="grip")
        super().__init__(ctx, cfg)

        base_url = cfg.raw.get("base_url", BASE_URL)
        self.road_types = tuple(cfg.raw.get("road_types", ROAD_TYPES))
        self.CONFIGURED_FILES: List[ConfiguredFile] = [
            ConfiguredFile(key=t, url=f"{base_url}/GRIP4_density_{t}.zip", name=f"GRIP4_density_{t}.zip")
            for t in self.road_types
        ]

    async def download_async(self, file_url: str, output_path: str, session: Any = None) -> None:
        """Synchronous `requests` download in an executor (module docstring)."""
        import asyncio

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.download, file_url, output_path, None)

    # -- PREPARE (zipped ASCII grids -> static 1 km grid) ------------------

    def _raw_files(self) -> List[str]:
        return [os.path.join(self.output_root(PipelineStep.FETCH), f.name) for f in self.CONFIGURED_FILES]

    def _output_path(self) -> str:
        return layout.grid_store_path(
            self.ctx.data_root,
            self.cfg.data_path,
            grid_id=self.ctx.grid_id,
            family="road_density",
            suffix="",  # cell_id-keyed parquet parts, not a Zarr store -- see grid_store_path docstring
        )

    def _plan_prepare(self) -> List[StepTarget]:
        raw_files = self._raw_files()
        if not all(os.path.exists(f) for f in raw_files):
            return []
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key="road_density",
                output_path=self._output_path(),
                inputs=tuple(raw_files), completion=Completion.MARKER,
                meta=verify.verification_meta(
                    self.cfg.raw, expected_vars=tuple(_variable(t) for t in self.road_types),
                ),
            )
        ]

    def _load(self, zip_path: str, road_type: str):
        """The density grid inside one zip (each zip also ships a land-area
        grid, which is skipped)."""
        import zipfile

        import numpy as np
        import rioxarray as rxr

        with zipfile.ZipFile(zip_path) as zf:
            (member,) = [n for n in zf.namelist() if n.endswith("_dens_m_km2.asc")]
        da = rxr.open_rasterio(f"/vsizip/{zip_path}/{member}", mask_and_scale=True).squeeze("band", drop=True)
        return da.rio.write_crs("EPSG:4326").astype(np.float32).rename(_variable(road_type))

    def _execute_prepare(self, target: StepTarget) -> bool:
        import numpy as np
        import xarray as xr
        from odc.geo.xr import xr_zeros

        from src.data.common.geobox import get_target_geobox
        from src.data.common.prepare.driver import run_tiled_prepare
        from src.data.common.raster.spatial import SpatialProcessor, sel_bbox
        from src.data.sources.steps import is_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping GRIP PREPARE -- already complete: %s", target.output_path)
            return True

        # Four ~9M-cell grids: loaded once, eagerly, and clipped per tile.
        ds = xr.merge([self._load(f, t) for f, t in zip(target.inputs, self.road_types)]).rio.write_crs("EPSG:4326")
        variables = [_variable(t) for t in self.road_types]

        geobox = get_target_geobox(self.ctx)
        processor = SpatialProcessor(hpc_root=self.ctx.data_root, target_geobox=geobox, default_nodata=np.nan)

        def raw_getter(tile, year):
            bbox = tile.geobox.pad(32, 32).extent.to_crs(ds.rio.crs).boundingbox
            clipped = sel_bbox(ds, bbox, y_dim="y", x_dim="x")
            if clipped.sizes.get("y", 0) == 0 or clipped.sizes.get("x", 0) == 0:
                return xr.Dataset({v: xr_zeros(tile.geobox, dtype="float32") * np.nan for v in variables})
            return clipped

        return run_tiled_prepare(
            output_path=target.output_path,
            years=None,
            variables=variables,
            target_geobox=geobox,
            processor=processor,
            raw_getter=raw_getter,
            resampling="nearest",
            processing_version=self.PROCESSING_VERSION,
            override=self.cfg.override,
        )


registry.register(GripSource.ID, __name__, GripSource.__name__, GripSource.STEPS)
