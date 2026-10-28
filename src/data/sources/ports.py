"""IMF PortWatch ports database: fetch + prepare, no grid.

One point per port (2,065 ports in October 2026) with AIS-based yearly
average vessel counts by ship type, counted from 2019 on. The corridor
instrument (`src/data/sources/corridors/`) uses it to pick the ports bulk
mines ship to (`vessel_count_dry_bulk`); see
docs/successors/corridors-prespec.md §2 for why not the World Port Index.

FETCH pages through PortWatch's own ArcGIS REST Feature Service
(`.../FeatureServer/0/query`, the same permanent-API reasoning as
`src/data/sources/ecoregions/source.py`) and writes one CSV. It downloads
with `requests`, which honours the `https_proxy` environment variables;
`src/data/common/fetch/driver.py`'s aiohttp session does not, so
`download_async()` runs the synchronous `download()` in an executor. The file
is ~1 MB, small enough to fetch through a proxy as well as from the transfer
node.

PREPARE normalises it to `prepared/ports/misc/ports.parquet`: `port_id`,
`port_name`, `iso3`, `lon`, `lat` and the `vessel_count_*` columns.
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

DEFAULT_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "PortWatch_ports_database/FeatureServer/0/query"
)
DEFAULT_NAME = "portwatch_ports.csv"
PAGE_SIZE = 1000

_KEEP_COLUMNS = {
    "portid": "port_id",
    "portname": "port_name",
    "ISO3": "iso3",
    "lon": "lon",
    "lat": "lat",
    "vessel_count_total": "vessel_count_total",
    "vessel_count_container": "vessel_count_container",
    "vessel_count_dry_bulk": "vessel_count_dry_bulk",
    "vessel_count_general_cargo": "vessel_count_general_cargo",
    "vessel_count_RoRo": "vessel_count_roro",
    "vessel_count_tanker": "vessel_count_tanker",
}


class PortsSource(ConfiguredFilesFetchMixin, DataSource):
    """IMF PortWatch port points with vessel counts by ship type."""

    ID = "ports"
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)

    DATA_SOURCE_NAME = "ports"

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="ports")
        super().__init__(ctx, cfg)

        url = cfg.raw.get("url", DEFAULT_URL)
        name = cfg.raw.get("name", DEFAULT_NAME)
        self.CONFIGURED_FILES: List[ConfiguredFile] = [ConfiguredFile(key="ports", url=url, name=name)]

    # _plan_fetch/_execute_fetch/_plan/_execute: inherited from
    # ConfiguredFilesFetchMixin.

    # -- FETCH download: ArcGIS REST query, paged through resultOffset ------

    def download(self, file_url: str, output_path: str, session: Any = None) -> None:
        import pandas as pd
        import requests

        s = session or requests.Session()
        rows: List[dict] = []
        offset = 0
        while True:
            params = dict(
                where="1=1", outFields="*", returnGeometry="false", f="json",
                resultOffset=offset, resultRecordCount=PAGE_SIZE,
            )
            resp = s.get(file_url, params=params, timeout=120)
            resp.raise_for_status()
            payload = resp.json()
            if "error" in payload:
                raise RuntimeError(f"PortWatch query failed at offset {offset}: {payload['error']}")
            features = payload.get("features", [])
            rows.extend(f["attributes"] for f in features)
            if not features or not payload.get("exceededTransferLimit"):
                break
            offset += len(features)

        if not rows:
            raise RuntimeError(f"PortWatch FETCH returned zero ports from {file_url}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        logger.info("Ports FETCH complete: %d ports -> %s", len(rows), output_path)

    async def download_async(self, file_url: str, output_path: str, session: Any = None) -> None:
        """Runs `download()` in an executor: `requests` honours the proxy
        environment variables, the driver's aiohttp session does not (module
        docstring). *session* is aiohttp-flavoured and not forwarded."""
        import asyncio

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.download, file_url, output_path, None)

    # -- PREPARE (CSV -> normalised parquet) --------------------------------

    def _raw_file_path(self) -> str:
        return os.path.join(self.output_root(PipelineStep.FETCH), self.CONFIGURED_FILES[0].name)

    def output_path(self) -> str:
        """Read by `corridors` through this path, never a class import
        (docs/design/09-integrated-pipeline.md §2)."""
        return os.path.join(self.output_root(PipelineStep.PREPARE, agg=layout.MISC_AGG), "ports.parquet")

    def _plan_prepare(self) -> List[StepTarget]:
        raw_file = self._raw_file_path()
        if not os.path.exists(raw_file):
            return []
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key="ports",
                output_path=self.output_path(),
                inputs=(raw_file,), completion=Completion.PATH_EXISTS,
            )
        ]

    def _execute_prepare(self, target: StepTarget) -> bool:
        import pandas as pd

        if not self.cfg.override and os.path.exists(target.output_path):
            logger.info("Skipping ports PREPARE, output already exists: %s", target.output_path)
            return True

        df = pd.read_csv(target.inputs[0])
        missing = set(_KEEP_COLUMNS) - set(df.columns)
        if missing:
            raise RuntimeError(f"PortWatch CSV lacks columns {sorted(missing)}; the service schema changed")
        df = df[list(_KEEP_COLUMNS)].rename(columns=_KEEP_COLUMNS)
        df = df.dropna(subset=["lon", "lat"])
        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)
        df.to_parquet(target.output_path, index=False)
        logger.info("Ports PREPARE complete: %d ports -> %s", len(df), target.output_path)
        return True


registry.register(PortsSource.ID, __name__, PortsSource.__name__, PortsSource.STEPS)
