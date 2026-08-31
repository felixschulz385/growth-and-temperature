"""EOG VIIRS global gas-flare survey: fetch + prepare.

A standalone companion to `eog_viirs` (src/data/sources/eog/source.py). The
Earth Observation Group publishes an annual inventory of upstream gas
flares detected by VIIRS Nightfire -- one `.xlsx` per year from 2017, plus
a single combined file covering 2012-2016
(https://eogdata.mines.edu/products/vnf/global_gas_flare.html). Each row is
one flare's representative location (lat/lon) with volume/temperature
stats.

This source rasterizes those points onto the canonical EPSG:6933 EASE grid
as a **distance-banded `uint8` flag** (`flare_band`), one value per pixel,
so downstream analysis can exclude or down-weight flare-contaminated
nighttime-radiance cells without a rebuild:

    0  no flare within 5 km
    1  a flare point within 5 km
    2  a flare point within 2 km
    3  a flare point falls in this very pixel

Distances are geodesic (WGS84), measured from the survey point -- EASE6933
is equal-area, not equidistant, so the buffers are built as geodesic
circles in lon/lat and then reprojected, not as metric buffers in the grid
CRS.

**Why a separate source, not another `eog_viirs` column:** the flare
inventory is a plain-HTTP vector download parsed from spreadsheets and
rasterized from geometry (`reproject=False`), a completely different ingest
from `eog_viirs`'s Selenium-authenticated raster composites. Keeping it
standalone means the flag is independently rebuildable and "removable
downstream" == simply not joining `eog_flare` into the assembled panel.

**2012-2016:** the survey has no per-year breakdown for those years, only
the one combined file. Its flares are broadcast identically onto each of
2012-2016 (flares are persistent infrastructure); 2017 onward are
genuinely annual.

**Scope:** this is the upstream (`d.7` slope model) inventory. Refinery /
LNG / industrial "downstream" flares -- often inside cities -- are not in
it, so `flare_band == 0` does not guarantee a cell is flare-free.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import numpy as np

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection
from src.data.sources import verify

logger = logging.getLogger(__name__)


class EogFlareSource(DataSource):
    """EOG VIIRS global gas-flare survey -> distance-banded `flare_band` grid.

    FETCH   -- plain HTTPS download of the annual (2017+) and combined
               (2012-2016) flare `.xlsx` files, filenames discovered by
               scraping the product download page.
    PREPARE -- rasterize each year's flare points onto the canonical geobox
               as a geodesic distance-banded `uint8` flag (`reproject=False`
               -- geometry is rasterized straight onto each tile).
    """

    ID = "eog_flare"
    ALIASES = ("eog_gas_flare", "viirs_flare")
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)
    DEFAULT_TRANSFER_MODE = "auto"

    DATA_SOURCE_NAME = "eog_flare"
    has_entrypoints = True
    STATIC_ENTRYPOINTS = True  # get_all_entrypoints() is year-range-derived, no network call
    RAW_LISTING_DEPTH = 1  # flat .xlsx filenames under the raw root

    #: Bump to force a full PREPARE reprocess (`run_tiled_prepare`).
    PROCESSING_VERSION = "1"

    OUTPUT_COLUMN = "flare_band"

    #: (band value, radius in metres) for the two ring bands, widest first.
    #: An in-cell hit (the flare point's own pixel) is IN_CELL_BAND.
    RING_BANDS: Tuple[Tuple[int, float], ...] = ((1, 5000.0), (2, 2000.0))
    IN_CELL_BAND = 3

    #: Years with no per-year file -- served by the one combined survey.
    COMBINED_YEARS = (2012, 2013, 2014, 2015, 2016)

    _XLSX_HREF_RE = re.compile(r"""href=["']([^"']*?flaring[^"']*?\.xlsx)["']""", re.IGNORECASE)
    #: 4-digit years plausible for this product (2012 first VIIRS year on).
    _YEAR_RE = re.compile(r"(?<!\d)(20(?:1[2-9]|[2-3]\d))(?!\d)")

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            raise ValueError("'data_path' (or 'output_path') is required.")
        super().__init__(ctx, cfg)
        self.base_url: str = cfg.raw.get("base_url") or (
            "https://eogdata.mines.edu/products/vnf/global_gas_flare.html"
        )
        year_range = cfg.raw.get("year_range") or [2012, 2021]
        self.year_start, self.year_end = int(year_range[0]), int(year_range[1])

        from src.data.common import tiling

        self.tile_size = int(cfg.raw.get("tile_size", tiling.DEFAULT_TILE_SIZE))
        self.circle_vertices = int(cfg.raw.get("circle_vertices", 64))

        self._links_cache: Optional[List[Tuple[str, str]]] = None
        self._links_cache_ts: Optional[float] = None
        self._links_cache_ttl = 3600
        #: abs xlsx path -> (N, 2) float array of (lon, lat) flare points.
        self._points_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # RemoteFileCatalog contract
    # ------------------------------------------------------------------

    def local_path(self, relative_path: str) -> str:
        return os.path.join("data", self.DATA_SOURCE_NAME, relative_path)

    # get_file_hash: inherited from DataSource.

    def _scrape_xlsx_links(self) -> List[Tuple[str, str]]:
        """`(filename, absolute_url)` for every flare `.xlsx` linked from the
        product download page. Cached for an hour -- the page is small but
        `required_files()` hits this once per entrypoint."""
        now = time.time()
        if self._links_cache is not None and now - (self._links_cache_ts or 0) < self._links_cache_ttl:
            return self._links_cache

        import requests

        links: List[Tuple[str, str]] = []
        try:
            resp = requests.get(self.base_url, timeout=60)
            resp.raise_for_status()
            seen = set()
            for href in self._XLSX_HREF_RE.findall(resp.text):
                url = urljoin(self.base_url, href)
                name = os.path.basename(url.split("?", 1)[0])
                if name and name not in seen:
                    seen.add(name)
                    links.append((name, url))
        except requests.RequestException:
            logger.exception("Could not scrape EOG flare download page %s", self.base_url)

        self._links_cache = links
        self._links_cache_ts = now
        return links

    @classmethod
    def _years_for_filename(cls, filename: str) -> List[int]:
        """Which calendar year(s) a flare filename covers. A `YYYY-YYYY`
        span (the 2012-2016 combined file) expands to every year in the
        span; a lone `YYYY` token is that one year."""
        span = re.search(r"(20\d{2})[-_](20\d{2})", filename)
        if span:
            lo, hi = int(span.group(1)), int(span.group(2))
            if lo <= hi:
                return list(range(lo, hi + 1))
        years = [int(y) for y in cls._YEAR_RE.findall(filename)]
        return years[:1]

    def list_remote_files(self, entrypoint: Optional[dict] = None) -> List[Tuple[str, str]]:
        links = self._scrape_xlsx_links()
        if entrypoint is None:
            return links
        want = entrypoint.get("year")
        matches = [(n, u) for (n, u) in links if want in self._years_for_filename(n)]
        if len(matches) > 1:
            logger.warning(
                "Multiple EOG flare files match year %s: %s -- using %s",
                want, [n for n, _ in matches], matches[0][0],
            )
        return matches[:1]

    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        years = self._years_for_filename(os.path.basename(relative_path))
        return {"year": years[0]} if years else None

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        return [{"year": y} for y in range(self.year_start, self.year_end + 1)]

    async def download_async(self, source_url: str, output_path: str, session: Any = None) -> None:
        import aiohttp

        from src.data.common.fetch.http import download_with_retries

        await asyncio.sleep(0.3)
        if session is None:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=600, connect=60)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as sess:
                await download_with_retries(sess, source_url, output_path)
        else:
            await download_with_retries(session, source_url, output_path)

    # ------------------------------------------------------------------
    # plan()/execute() dispatch
    # ------------------------------------------------------------------

    def _plan(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        if step is PipelineStep.FETCH:
            return self._plan_fetch()
        if step is PipelineStep.PREPARE:
            return self._plan_prepare(selection)
        raise AssertionError(f"unreachable: {step}")

    def _execute(self, target: StepTarget) -> bool:
        if target.step is PipelineStep.FETCH:
            return self._execute_fetch(target)
        if target.step is PipelineStep.PREPARE:
            return self._execute_prepare(target)
        raise AssertionError(f"unreachable: {target.step}")

    # -- FETCH --------------------------------------------------------------

    def _plan_fetch(self) -> List[StepTarget]:
        return [
            StepTarget(
                source_id=self.cfg.source_id,
                step=PipelineStep.FETCH,
                key="all",
                output_path=self.output_root(PipelineStep.FETCH),
                completion=Completion.NEVER,
            )
        ]

    def _execute_fetch(self, target: StepTarget) -> bool:
        from src.data.common.fetch.driver import run_fetch

        return run_fetch(self, **self.cfg.raw.get("download", {}))

    # -- PREPARE ----------------------------------------------------------

    def _resolve_source_file_path(self, file_path: str) -> str:
        if os.path.isabs(file_path) or (self.ctx.data_root and file_path.startswith(self.ctx.data_root)):
            return file_path
        return os.path.join(self.output_root(PipelineStep.FETCH), file_path)

    def _raw_xlsx_by_year(self) -> Dict[int, str]:
        """`{year: relpath}` from the fetched raw dir. The 2012-2016
        combined file resolves to the same relpath for each of those years;
        a genuine per-year file wins over the combined one if both exist."""
        raw_root = self.output_root(PipelineStep.FETCH)
        if not os.path.isdir(raw_root):
            return {}
        out: Dict[int, str] = {}
        combined: Dict[int, str] = {}
        for dirpath, _dirnames, filenames in os.walk(raw_root):
            for fname in filenames:
                if not fname.lower().endswith(".xlsx"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, fname), raw_root)
                years = self._years_for_filename(fname)
                target = combined if len(years) > 1 else out
                for year in years:
                    target.setdefault(year, rel)
        for year, rel in combined.items():
            out.setdefault(year, rel)
        return out

    def _output_path(self) -> str:
        return layout.grid_store_path(
            self.ctx.data_root,
            self.cfg.data_path,
            grid_id=self.ctx.grid_id,
            family="eog_flare",
            suffix="",
        )

    def _plan_prepare(self, selection: TargetSelection) -> List[StepTarget]:
        files_by_year = self._raw_xlsx_by_year()
        years = sorted(
            year
            for year in files_by_year
            if selection.matches_year(year) and selection.matches_key(str(year))
        )
        if not years:
            return []
        raw_files = {year: files_by_year[year] for year in years}
        return [
            StepTarget(
                source_id=self.cfg.source_id,
                step=PipelineStep.PREPARE,
                key="all",
                output_path=self._output_path(),
                inputs=tuple(dict.fromkeys(raw_files[year] for year in years)),
                completion=Completion.MARKER,
                meta={
                    "years": years,
                    "raw_files": raw_files,
                    **verify.verification_meta(
                        self.cfg.raw,
                        expected_vars=(self.OUTPUT_COLUMN,),
                        value_range=(0, self.IN_CELL_BAND),
                        # Most tiles have no flare at all -- an all-zero
                        # sample is the correct, non-degenerate answer.
                        sparse_vars=(self.OUTPUT_COLUMN,),
                    ),
                },
            )
        ]

    def _load_flare_points(self, xlsx_path: str) -> np.ndarray:
        """`(N, 2)` array of `(lon, lat)` for one survey file, cached per
        path (so the 2012-2016 combined file is parsed once). Column names
        have drifted across survey versions -- matched case-insensitively on
        a `lat`/`lon` substring."""
        if xlsx_path in self._points_cache:
            return self._points_cache[xlsx_path]

        import pandas as pd

        df = pd.read_excel(xlsx_path, engine="openpyxl")
        cols = {str(c).strip().lower(): c for c in df.columns}

        def _find(*needles: str) -> Optional[str]:
            for lc, orig in cols.items():
                if any(n in lc for n in needles):
                    return orig
            return None

        lat_col = _find("latitude", "lat")
        lon_col = _find("longitude", "lon")
        if lat_col is None or lon_col is None:
            raise ValueError(
                f"{os.path.basename(xlsx_path)}: no lat/lon columns found in {list(df.columns)}"
            )

        lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy()
        lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy()
        ok = (
            np.isfinite(lat)
            & np.isfinite(lon)
            & (np.abs(lat) <= 90.0)
            & (np.abs(lon) <= 180.0)
            & ~((lat == 0.0) & (lon == 0.0))
        )
        pts = np.column_stack([lon[ok], lat[ok]]).astype("float64")
        logger.info("EOG flare %s: %d flare point(s)", os.path.basename(xlsx_path), len(pts))
        self._points_cache[xlsx_path] = pts
        return pts

    def _geodesic_circle(self, lon: float, lat: float, radius_m: float):
        """A closed shapely polygon approximating the geodesic circle of
        *radius_m* around `(lon, lat)` on the WGS84 ellipsoid."""
        from pyproj import Geod
        from shapely.geometry import Polygon

        geod = Geod(ellps="WGS84")
        az = np.linspace(0.0, 360.0, self.circle_vertices, endpoint=False)
        out_lon, out_lat, _ = geod.fwd(
            np.full(az.shape, lon), np.full(az.shape, lat), az, np.full(az.shape, radius_m)
        )
        return Polygon(np.column_stack([out_lon, out_lat]))

    def _rasterize_tile(self, tile, year: int, raw_files: Dict[int, str]):
        """One tile's `flare_band` raster on `tile.geobox` -- the
        `raw_getter` for `run_tiled_prepare(reproject=False)`. Always
        returns a Dataset (all-zero when no flare is near), never `None`."""
        import xarray as xr
        from odc.geo.geom import Geometry
        from odc.geo.xr import rasterize
        from shapely.geometry import MultiPoint
        from shapely.ops import unary_union

        tile_geobox = tile.geobox
        dim_y, dim_x = tile_geobox.dims
        band = np.zeros(tile_geobox.shape, dtype=np.uint8)

        pts = self._load_flare_points(self._resolve_source_file_path(raw_files[year]))
        if len(pts):
            # Filter to this tile's extent + a generous halo (>= widest ring)
            # so a flare just outside the tile still bands its edge pixels.
            ll = tile_geobox.extent.to_crs("EPSG:4326").boundingbox
            pad = 0.15  # deg; ~16 km lat, comfortably over the 5 km ring
            in_box = (
                (pts[:, 0] >= ll.left - pad)
                & (pts[:, 0] <= ll.right + pad)
                & (pts[:, 1] >= ll.bottom - pad)
                & (pts[:, 1] <= ll.top + pad)
            )
            near = pts[in_box]
        else:
            near = pts

        if len(near):
            crs = str(tile_geobox.crs)
            for band_value, radius_m in self.RING_BANDS:
                circles = unary_union(
                    [self._geodesic_circle(lon, lat, radius_m) for lon, lat in near]
                )
                geom = Geometry(circles, crs="EPSG:4326").to_crs(crs)
                mask = rasterize(geom, tile_geobox).values.astype(bool)
                band[mask] = band_value
            in_cell = Geometry(MultiPoint([tuple(p) for p in near]), crs="EPSG:4326").to_crs(crs)
            band[rasterize(in_cell, tile_geobox).values.astype(bool)] = self.IN_CELL_BAND

        return xr.Dataset({self.OUTPUT_COLUMN: ((dim_y, dim_x), band)})

    def _execute_prepare(self, target: StepTarget) -> bool:
        from src.data.common.geobox import get_target_geobox
        from src.data.common.prepare.driver import run_tiled_prepare
        from src.data.common.raster.spatial import SpatialProcessor
        from src.data.sources.steps import is_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping PREPARE -- already complete: %s", target.output_path)
            return True

        years: List[int] = target.meta["years"]
        raw_files: Dict[int, str] = target.meta["raw_files"]
        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)

        target_geobox = get_target_geobox(self.ctx)
        processor = SpatialProcessor(hpc_root=self.ctx.data_root, target_geobox=target_geobox)
        try:
            return run_tiled_prepare(
                output_path=target.output_path,
                years=years,
                variables=[self.OUTPUT_COLUMN],
                target_geobox=target_geobox,
                processor=processor,
                raw_getter=lambda tile, year: self._rasterize_tile(tile, year, raw_files),
                tile_size=self.tile_size,
                reproject=False,
                processing_version=self.PROCESSING_VERSION,
                override=self.cfg.override,
            )
        except Exception:
            logger.exception("Error processing EOG flare PREPARE target")
            return False


registry.register(
    EogFlareSource.ID,
    __name__,
    EogFlareSource.__name__,
    EogFlareSource.STEPS,
    aliases=EogFlareSource.ALIASES,
)
