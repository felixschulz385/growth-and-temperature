"""GAIA v1.0 annual impervious share on the canonical 1 km grid: fetch only.

GAIA (Gong et al. 2020, *RSE* 236:111510; CC BY 4.0) maps, per 30 m Landsat
pixel, the year it became impervious (> 50 % impervious). Earth Engine asset
``Tsinghua/FROM-GLC/GAIA/v10``, band ``change_year_index``: 34 = 1985 (and
earlier) ... 1 = 2018. Pixels never impervious by 2018, and the ocean, are
masked; the asset has no other "no data".

**Year rule: impervious in year Y  <=>  change_year_index >= 2019 - Y.**
The catalogue's example ("1990 = value greater than 29") contradicts its own
table (29 = 1990). Strict ">" would leave 1985 empty and never count the
2018 pixels (value 1), which exist; the values present are exactly 1-34,
with 34 the largest class in old city cores (the pre-1985 stock). See
``docs/data/gaia/README.md``.

FETCH computes, in Earth Engine, the share of each 1 km cell that is
impervious in each year 1985-2018:

- masked pixels -> code 0, in GAIA's native 30 m EPSG:4326 projection;
- ``reduceResolution(fixedHistogram)`` to the canonical EASE-Grid 2.0 1 km
  grid: the area-weighted weight of each year code per cell (``maxPixels``
  must cover a cell's inputs, about 1,100 at the equator and 2,300 at the
  grid's 60 deg edge);
- for each year Y, the weight of codes >= 2019 - Y over the cell's total
  (``build_image()``; identical to a per-year mean of a 0/1 image);
- share x 10,000, rounded, ``uint16``. The share is of the whole cell, water
  included. Nodata 65535 is declared but never written: GAIA cannot tell
  "no data" from "not impervious", so every cell gets a value.

Earth Engine does not parse ``EPSG:6933``, so requests carry its WKT. The
pixel grid is the canonical geobox's (``canonical_ease_geobox()``), checked
against the constants below at start-up. Each land tile is requested in
``block_size`` blocks with ``ee.data.computePixels``, from ``workers``
processes (a 256 block exceeds
Earth Engine's limit on the 30 m intermediate) and written as one compressed
GeoTIFF, ``raw/gaia/v10/ix=<row>/iy=<col>/gaia_imp_share_1985_2018.tif``
(``ix`` is the tile row, as on the HPC), via a temp file and ``os.replace``.
Re-running fetches only missing tiles.

Earth Engine is imported inside FETCH only, so the HPC can import this module
without ``ee``. The Earth Engine Cloud project comes from
``sources.gaia.ee_project`` in a git-ignored ``*.local.yaml`` or from the
``EE_PROJECT`` environment variable; it is never committed. PREPARE is added
on the HPC.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.base import DataSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection, TransferUnit

logger = logging.getLogger(__name__)

ASSET = "Tsinghua/FROM-GLC/GAIA/v10"
BAND = "change_year_index"
YEARS: Tuple[int, ...] = tuple(range(1985, 2019))
SCALE = 10_000
NODATA = 65535
FILENAME = "gaia_imp_share_1985_2018.tif"

#: Canonical 1 km grid (EPSG:6933), asserted against canonical_ease_geobox().
GRID_X0, GRID_Y0, PIXEL = -17_368_000, 6_352_000, 1000
GRID_WIDTH, GRID_HEIGHT = 34_736, 12_704
TILE = 2048

#: The 98 tiles (row, col) that contain land on the HPC's canonical grid.
LAND_TILES: Tuple[Tuple[int, int], ...] = (
    *((0, c) for c in range(17)),
    *((1, c) for c in (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)),
    *((2, c) for c in (0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)),
    *((3, c) for c in (0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16)),
    *((4, c) for c in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16)),
    *((5, c) for c in (0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)),
    *((6, c) for c in (5, 6, 7, 8, 15)),
)


def year_code(year: int) -> int:
    """GAIA's ``change_year_index`` for *year* (34 = 1985 ... 1 = 2018)."""
    return 2019 - year


def tile_window(row: int, col: int) -> Tuple[int, int, int, int]:
    """(row0, col0, height, width) in grid pixels, clipped at the grid edge."""
    r0, c0 = row * TILE, col * TILE
    return r0, c0, min(TILE, GRID_HEIGHT - r0), min(TILE, GRID_WIDTH - c0)


def _ease_wkt() -> str:
    import pyproj

    return pyproj.CRS.from_epsg(6933).to_wkt("WKT1_GDAL")


def _check_grid() -> None:
    """The constants above must be the canonical geobox's, exactly."""
    from src.data.common.geobox.canonical import canonical_ease_geobox

    gbox = canonical_ease_geobox()
    a = gbox.affine
    expected = (PIXEL, 0.0, GRID_X0, 0.0, -PIXEL, GRID_Y0)
    got = (a.a, a.b, a.c, a.d, a.e, a.f)
    if tuple(gbox.shape) != (GRID_HEIGHT, GRID_WIDTH) or not np.allclose(got, expected):
        raise RuntimeError(f"canonical grid changed: shape {gbox.shape}, affine {got}; expected {expected}")


class GaiaSource(DataSource):
    """GAIA v1.0 impervious share per 1 km cell and year, from Earth Engine.

    FETCH -- one 34-band uint16 GeoTIFF per land tile (module docstring).
    """

    ID = "gaia"
    STEPS = (PipelineStep.FETCH,)

    DATA_SOURCE_NAME = "gaia"
    DEFAULT_TRANSFER_MODE = "manual"  # fetched on a workstation, pushed with `data transfer`
    has_entrypoints = False

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="gaia/v10")
        super().__init__(ctx, cfg)
        raw = cfg.raw
        self.block_size = int(raw.get("block_size", 128))
        self.workers = int(raw.get("workers", 32))
        self.max_retries = int(raw.get("max_retries", 6))
        self.max_pixels = int(raw.get("max_pixels", 16384))
        tiles = raw.get("tiles")
        self.tiles: Tuple[Tuple[int, int], ...] = tuple(tuple(t) for t in tiles) if tiles else LAND_TILES
        if TILE % self.block_size:
            raise ValueError(f"block_size must divide {TILE}")

    # -- layout -------------------------------------------------------------

    def tile_path(self, row: int, col: int) -> str:
        return os.path.join(self.output_root(PipelineStep.FETCH), f"ix={row}", f"iy={col}", FILENAME)

    def missing_tiles(self) -> List[Tuple[int, int]]:
        return [t for t in self.tiles if not os.path.isfile(self.tile_path(*t))]

    def transfer_units(self, step: PipelineStep) -> List[TransferUnit]:
        """One unit per tile file present locally."""
        self._require_step(step)
        units = []
        for row, col in self.tiles:
            local = self.tile_path(row, col)
            if os.path.isfile(local):
                rel = os.path.relpath(local, self.ctx.data_root).replace(os.sep, "/")
                units.append(TransferUnit(unit_id=f"ix={row}/iy={col}", local_path=local, remote_path=rel))
        return units

    # -- plan/execute -------------------------------------------------------

    def _plan(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        return [
            StepTarget(
                source_id=self.ID,
                step=PipelineStep.FETCH,
                key="all",
                output_path=self.output_root(PipelineStep.FETCH),
                completion=Completion.NEVER,
                meta={"missing_tiles": len(self.missing_tiles()), "tiles": len(self.tiles)},
            )
        ]

    def _execute(self, target: StepTarget) -> bool:
        import multiprocessing as mp

        _check_grid()
        todo = self.missing_tiles()
        logger.info("GAIA: %d of %d tiles to fetch", len(todo), len(self.tiles))
        failed = []
        ctx = mp.get_context("spawn")
        with ctx.Pool(self.workers, initializer=_worker_init, initargs=(self._ee_project(), self.max_pixels)) as pool:
            for i, (row, col) in enumerate(todo, 1):
                t0 = time.time()
                try:
                    self._fetch_tile(pool, row, col)
                    logger.info("GAIA tile ix=%d/iy=%d done in %.0fs (%d/%d)", row, col, time.time() - t0, i, len(todo))
                except Exception:
                    logger.exception("GAIA tile ix=%d/iy=%d failed", row, col)
                    failed.append((row, col))
        if failed:
            logger.error("GAIA: %d tile(s) failed: %s", len(failed), failed)
        return not failed

    # -- Earth Engine -------------------------------------------------------

    def _ee_project(self) -> str:
        project = self.cfg.raw.get("ee_project") or os.environ.get("EE_PROJECT")
        if not project:
            raise RuntimeError("Set sources.gaia.ee_project in a git-ignored *.local.yaml, or EE_PROJECT")
        return project

    def _fetch_tile(self, pool, row: int, col: int) -> None:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import Affine

        tr0, tc0, th, tw = tile_window(row, col)
        out = np.empty((len(YEARS), th, tw), dtype=np.uint16)
        b = self.block_size
        jobs = [
            (tr0 + br, tc0 + bc, min(b, th - br), min(b, tw - bc), self.max_retries)
            for br in range(0, th, b)
            for bc in range(0, tw, b)
        ]
        for r0, c0, block in pool.imap_unordered(_worker_block, jobs):
            h, w = block.shape[1:]
            out[:, r0 - tr0:r0 - tr0 + h, c0 - tc0:c0 - tc0 + w] = block

        if (out > SCALE).any():
            raise RuntimeError(f"tile ix={row}/iy={col}: share above 1")
        if (np.diff(out.astype(np.int32), axis=0) < 0).any():
            raise RuntimeError(f"tile ix={row}/iy={col}: share decreases over the years")

        path = self.tile_path(row, col)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        profile = {
            "driver": "GTiff", "dtype": "uint16", "count": len(YEARS), "height": th, "width": tw,
            "crs": CRS.from_epsg(6933),
            "transform": Affine(PIXEL, 0, GRID_X0 + tc0 * PIXEL, 0, -PIXEL, GRID_Y0 - tr0 * PIXEL),
            "nodata": NODATA, "compress": "deflate", "predictor": 2, "zlevel": 6,
            "tiled": True, "blockxsize": 256, "blockysize": 256, "interleave": "band",
        }
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(out)
            for i, y in enumerate(YEARS, 1):
                dst.set_band_description(i, f"imp_{y}")
            dst.update_tags(
                source=ASSET, scale=str(SCALE),
                rule="impervious in year Y <=> change_year_index >= 2019 - Y; masked -> 0",
                aggregation="reduceResolution(fixedHistogram) from native 30 m, area-weighted; "
                "share of whole cell incl. water",
            )
        os.replace(tmp, path)


# -- worker processes ---------------------------------------------------------
# The Google API client's HTTP layer serializes threads, so blocks are fetched
# in processes (~38,000 cells/s with 32 on 2026-11-04, against ~2,000 with
# threads). Each worker initializes Earth Engine once.

_W: Dict[str, object] = {}


def build_image(ee, max_pixels: int):
    """34-band uint16 image of impervious share x SCALE on any requested grid.

    One area-weighted histogram of the year code per output cell (codes 0-34,
    masked pixels -> 0), then, for each year Y, the weight of codes
    >= 2019 - Y over the total weight. Identical to 34 per-year
    ``reduceResolution(mean)`` reductions (checked over Shanghai,
    2026-11-04: max difference 0) and about 4x faster.
    """
    gaia = ee.Image(ASSET).select(BAND)
    codes = gaia.unmask(0).setDefaultProjection(gaia.projection())
    hist = codes.reduceResolution(reducer=ee.Reducer.fixedHistogram(0, 35, 35), maxPixels=max_pixels)
    weight = hist.arraySlice(1, 1, 2).arrayProject([0])          # weight of codes 0..34
    total = weight.arrayReduce(ee.Reducer.sum(), [0]).arrayGet([0])
    imp = weight.arraySlice(0, 1, 35)                             # codes 1..34
    cum = imp.arrayAccum(0, ee.Reducer.sum())                     # cum[k-1] = weight of codes <= k
    total_imp = imp.arrayReduce(ee.Reducer.sum(), [0]).arrayGet([0])
    bands = []
    for y in YEARS:
        c = year_code(y)
        below = ee.Image(0) if c == 1 else cum.arrayGet([c - 2])  # codes < c
        bands.append(total_imp.subtract(below).divide(total).rename(f"imp_{y}"))
    return ee.Image.cat(bands).multiply(SCALE).round().toUint16()


def _worker_init(project: str, max_pixels: int) -> None:
    import warnings

    warnings.filterwarnings("ignore")
    import ee

    ee.Initialize(project=project)
    _W["ee"] = ee
    _W["image"] = build_image(ee, max_pixels)
    _W["wkt"] = _ease_wkt()


def _worker_block(job) -> Tuple[int, int, np.ndarray]:
    """(r0, c0, (len(YEARS), h, w) uint16) for grid pixels [r0:r0+h, c0:c0+w]."""
    r0, c0, h, w, max_retries = job
    ee = _W["ee"]
    request = {
        "expression": _W["image"],
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": w, "height": h},
            "affineTransform": {
                "scaleX": PIXEL, "shearX": 0, "translateX": GRID_X0 + c0 * PIXEL,
                "shearY": 0, "scaleY": -PIXEL, "translateY": GRID_Y0 - r0 * PIXEL,
            },
            "crsWkt": _W["wkt"],
        },
    }
    for attempt in range(max_retries + 1):
        try:
            arr = ee.data.computePixels(request)
            return r0, c0, np.stack([np.asarray(arr[f"imp_{y}"], dtype=np.uint16) for y in YEARS])
        except Exception as exc:  # EEException, HTTP and socket errors
            if attempt == max_retries:
                raise RuntimeError(f"block r0={r0} c0={c0}: {exc}") from exc
            wait = min(120.0, 2.0 ** attempt * 2) * (0.5 + random.random())
            logger.warning("block r0=%d c0=%d: %s; retry in %.0fs", r0, c0, str(exc)[:120], wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


registry.register(GaiaSource.ID, __name__, GaiaSource.__name__, GaiaSource.STEPS)
