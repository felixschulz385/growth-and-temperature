"""YCEO Surface Urban Heat Islands v4 (Chakraborty and Lee), the heat-island benchmark: fetch only.

The official copy is NASA SEDAC (DOI 10.7927/s5m5-zk14, CC BY 4.0): seven
files under ``https://data.earthdata.nasa.gov/.../sdei-yceo-sfc-uhi-v4/``.
They need an Earthdata login, which ``requests`` reads from ``~/.netrc``
(``machine urs.earthdata.nasa.gov``; never committed). FETCH downloads them
as published into ``raw/yceo_uhi/v4/`` and then writes two tables derived from
them into ``raw/yceo_uhi/v4/derived/``:

- ``yceo_clusters.gpkg`` -- the 10,136 urban cluster polygons of
  ``All_shp.shp`` (Natural Earth urban areas) with all their attributes:
  ``Code`` (unique; the cluster id used here), ``FID_1`` (also unique), and the
  2003-2018 mean SUHI (K) by period (annual, summer, winter) and time of day.
- ``yceo_cluster_year.parquet`` -- ``Code``, ``year`` (2003-2018), ``suhi_day``,
  ``suhi_night`` (K): the cluster-mean annual SUHI of
  ``UHI_yearly_averaged_<year>.tif``. That raster paints each cluster's mean over
  its whole polygon, so the value at one interior point
  (``representative_point()``) is the cluster's value. Where that pixel is empty
  (a polygon thinner than a 300 m pixel), the mean over the polygon's pixels
  is used, and ``from_polygon`` is True. Clusters the benchmark leaves
  empty stay NaN.

Yearly summer and winter SUHI are published only per pixel (the
``*_yearly_pixel`` files), not as cluster means; they stay as published. The
pixel products hold values only at each year's urban pixels (ESA CCI urban
class), so they also carry the yearly urban mask. The rural reference is not
published; the user guide defines it by a rule (``docs/data/yceo_uhi/README.md``).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
import tempfile
import zipfile
from typing import List, Tuple

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.base import DataSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection, TransferUnit

logger = logging.getLogger(__name__)

BASE_URL = "https://data.earthdata.nasa.gov/nasa-earth/human-dimensions/sedac-root/downloads"
DATA_DIR = "data/sdei/sdei-yceo-sfc-uhi-v4"
PREFIX = "sdei-yceo-sfc-uhi-v4"
FILES: Tuple[Tuple[str, str], ...] = (
    ("docs-repo", f"{PREFIX}-documentation.pdf"),
    (DATA_DIR, f"{PREFIX}-urban-cluster-means-shp.zip"),
    (DATA_DIR, f"{PREFIX}-uhi-all-averaged-geotiff.zip"),
    (DATA_DIR, f"{PREFIX}-uhi-monthly-averaged-geotiff.zip"),
    (DATA_DIR, f"{PREFIX}-uhi-yearly-averaged-2003-2018-geotiff.zip"),
    (DATA_DIR, f"{PREFIX}-uhi-yearly-pixel-2003-2018-geotiff.zip"),
    (DATA_DIR, f"{PREFIX}-summer-uhi-yearly-pixel-2003-2018-geotiff.zip"),
    (DATA_DIR, f"{PREFIX}-winter-uhi-yearly-pixel-2003-2018-geotiff.zip"),
)
YEARS = tuple(range(2003, 2019))
DERIVED = "derived"
CLUSTERS = "yceo_clusters.gpkg"
CLUSTER_YEAR = "yceo_cluster_year.parquet"


class YceoUhiSource(DataSource):
    """YCEO SUHI v4 from SEDAC, plus per-cluster tables derived from it."""

    ID = "yceo_uhi"
    ALIASES = ("yceo",)
    STEPS = (PipelineStep.FETCH,)

    DATA_SOURCE_NAME = "yceo_uhi"
    DEFAULT_TRANSFER_MODE = "manual"  # fetched on a workstation, pushed with `data transfer`
    has_entrypoints = False

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="yceo_uhi/v4")
        super().__init__(ctx, cfg)
        self.base_url = cfg.raw.get("base_url", BASE_URL)

    # -- layout -------------------------------------------------------------

    def _raw(self, *parts: str) -> str:
        return os.path.join(self.output_root(PipelineStep.FETCH), *parts)

    def _outputs(self) -> List[str]:
        return [self._raw(name) for _, name in FILES] + [
            self._raw(DERIVED, CLUSTERS),
            self._raw(DERIVED, CLUSTER_YEAR),
        ]

    def transfer_units(self, step: PipelineStep) -> List[TransferUnit]:
        """One unit per published file and per derived table present locally."""
        self._require_step(step)
        units = []
        for path in self._outputs():
            if os.path.isfile(path):
                rel = os.path.relpath(path, self.ctx.data_root).replace(os.sep, "/")
                units.append(TransferUnit(unit_id=os.path.basename(path), local_path=path, remote_path=rel))
        return units

    # -- plan/execute -------------------------------------------------------

    def _plan(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        missing = [os.path.basename(p) for p in self._outputs() if not os.path.isfile(p)]
        return [
            StepTarget(
                source_id=self.ID,
                step=PipelineStep.FETCH,
                key="all",
                output_path=self.output_root(PipelineStep.FETCH),
                completion=Completion.NEVER,
                meta={"missing": missing},
            )
        ]

    def _execute(self, target: StepTarget) -> bool:
        os.makedirs(self._raw(DERIVED), exist_ok=True)
        for subdir, name in FILES:
            path = self._raw(name)
            if os.path.isfile(path) and not self.cfg.override:
                continue
            self._download(f"{self.base_url}/{subdir}/{name}", path)
        if self.cfg.override or not os.path.isfile(self._raw(DERIVED, CLUSTERS)):
            self._write_clusters()
        if self.cfg.override or not os.path.isfile(self._raw(DERIVED, CLUSTER_YEAR)):
            self._write_cluster_year()
        return True

    @staticmethod
    def _download(url: str, path: str) -> None:
        """Earthdata download: ``requests`` takes the login from ~/.netrc on the
        redirect to urs.earthdata.nasa.gov, and the session keeps its cookie."""
        import requests

        logger.info("Downloading %s", url)
        tmp = path + ".part"
        with requests.Session() as session, session.get(url, stream=True, timeout=(60, 600)) as r:
            r.raise_for_status()
            if "html" in r.headers.get("Content-Type", ""):
                raise RuntimeError(f"{url}: got an HTML page, not the file (Earthdata login in ~/.netrc?)")
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        os.replace(tmp, path)

    def _read_clusters(self):
        import geopandas as gpd

        zpath = self._raw(f"{PREFIX}-urban-cluster-means-shp.zip")
        clusters = gpd.read_file(f"zip://{zpath}!All_shp.shp")
        if not clusters["Code"].is_unique:
            raise RuntimeError("cluster Code is not unique")
        return clusters.sort_values("Code").reset_index(drop=True)

    def _write_clusters(self) -> None:
        path = self._raw(DERIVED, CLUSTERS)
        tmp = path + ".part.gpkg"
        self._read_clusters().to_file(tmp, layer="clusters", driver="GPKG")
        os.replace(tmp, path)

    def _write_cluster_year(self) -> None:
        import numpy as np
        import pandas as pd
        import rasterio
        from rasterio.features import geometry_mask
        from rasterio.windows import Window, from_bounds

        clusters = self._read_clusters()
        points = clusters.geometry.representative_point()
        coords = list(zip(points.x, points.y))
        zpath = self._raw(f"{PREFIX}-uhi-yearly-averaged-2003-2018-geotiff.zip")
        rows = []
        tmpdir = tempfile.mkdtemp(prefix="yceo_", dir=self.cfg.temp_dir or None)
        try:
            with zipfile.ZipFile(zpath) as zf:
                for year in YEARS:
                    member = f"UHI_yearly_averaged_{year}.tif"
                    zf.extract(member, tmpdir)
                    tif = os.path.join(tmpdir, member)
                    with rasterio.open(tif) as src:
                        nodata = src.nodata
                        # sample in raster row order, so each strip is decompressed once
                        order = np.argsort([src.index(x, y)[0] for x, y in coords], kind="stable")
                        vals = np.full((len(coords), 2), np.nan)
                        sampled = src.sample([coords[i] for i in order], indexes=[1, 2])
                        for i, v in zip(order, sampled):
                            vals[i] = v
                        vals[(vals == nodata) | ~np.isfinite(vals)] = np.nan
                        from_polygon = np.zeros(len(coords), dtype=bool)
                        for i in np.where(np.isnan(vals).all(axis=1))[0]:
                            geom = clusters.geometry.iloc[i]
                            w = from_bounds(*geom.bounds, transform=src.transform)
                            c0, r0 = int(np.floor(w.col_off)), int(np.floor(w.row_off))
                            c1 = int(np.ceil(w.col_off + w.width)) + 1
                            r1 = int(np.ceil(w.row_off + w.height)) + 1
                            c0, r0 = max(c0, 0), max(r0, 0)
                            c1, r1 = min(c1, src.width), min(r1, src.height)
                            win = Window(c0, r0, c1 - c0, r1 - r0)
                            block = src.read([1, 2], window=win).astype(float)
                            inside = ~geometry_mask([geom], out_shape=block.shape[1:], transform=src.window_transform(win), all_touched=True)
                            block[(block == nodata) | ~np.isfinite(block)] = np.nan
                            sel = block[:, inside]
                            if sel.size and np.isfinite(sel).any():
                                vals[i] = np.nanmean(sel, axis=1)
                                from_polygon[i] = True
                    os.remove(tif)
                    rows.append(pd.DataFrame({
                        "Code": clusters["Code"].to_numpy(), "year": year,
                        "suhi_day": vals[:, 0], "suhi_night": vals[:, 1], "from_polygon": from_polygon,
                    }))
                    logger.info("YCEO %d: %d clusters with a value (%d from the polygon)",
                                year, int(np.isfinite(vals[:, 0]).sum()), int(from_polygon.sum()))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        out = pd.concat(rows, ignore_index=True)
        path = self._raw(DERIVED, CLUSTER_YEAR)
        out.to_parquet(path + ".part", index=False)
        os.replace(path + ".part", path)


registry.register(
    YceoUhiSource.ID,
    __name__,
    YceoUhiSource.__name__,
    YceoUhiSource.STEPS,
    aliases=YceoUhiSource.ALIASES,
)
