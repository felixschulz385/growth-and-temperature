"""RESOLVE Ecoregions & Biomes: fetch + prepare + grid.

Dinerstein et al. 2017 "Ecoregions2017" (republished on Esri ArcGIS Hub as
"RESOLVE Ecoregions and Biomes"): one flat global polygon layer with three
attributes of increasing classification complexity -- `REALM` (8
biogeographic realms), `BIOME_NUM`/`BIOME_NAME` (14 WWF biomes), `ECO_ID`/
`ECO_NAME` (846 ecoregions) -- rasterized here as `realm_id`/`biome_id`/
`eco_id`, mirroring GADM's per-level `GID_N` id-grid pattern
(`src/data/sources/misc/gadm.py`) this module is structurally modeled on.

Unlike GADM's per-level files, RESOLVE ships one layer: every polygon carries
all three attributes at once, so PREPARE has no per-level loop, and GRID
rasterizes each polygon's boundary mask *once* and reuses it to paint all
three id-grids, rather than GADM's one-rasterize-call-per-level-per-polygon
(different geometries per level there; here the geometry is the same).

Also produces a second GRID target: the area-weighted dominant class (per
REALM/BIOME/ECO_ID) for each of GADM's own level-3 administrative units,
via a vector-vector overlay (`overlay.compute_dominant_classes`) rather than
raster zonal-mode -- exact polygon-area weighting is cheaper and more
accurate here than rasterizing both layers at 1km and counting pixels
(`src/data/assemble/geometry.py`'s `zonal_reduce_odc`/`assemble_geometry_weighted`
only support raster-cell reducers, no polygon overlay). Written as a tiny
`GID_3`-keyed parquet sidecar for `TileProcessor`'s `join_on` mechanism to
merge at assembly time, same pattern as `snl_mining`'s admin-count tables and
`country_classifications`'s `classifications_by_gid0.parquet`. `REQUIRES` on
gadm's PREPARE (level-3 simplified polygons AND `GID_3_code_mapping.json`,
both produced by gadm's PREPARE -- see module docstring in
`src/data/sources/misc/gadm.py`), resolved via
`layout.output_root()`/`gadm.gid_mapping_path()` directly, never a class
import (docs/design/09-integrated-pipeline.md §2).

There is no separate GRID step. Both `ecoregions_grid` and
`gadm_gid3_dominant` are PREPARE targets, each independently
resumable/executable (`_execute_prepare` dispatches on `target.key`).
`REQUIRES` is scoped to this source's own PREPARE step; `gadm_gid3_dominant`
only appears once gadm's level-3 output actually exists, so the
`ecoregions_grid` target alone runs unblocked before gadm exists.

FETCH pulls straight from RESOLVE's own ArcGIS REST Feature Service --
`.../FeatureServer/0/query` -- rather than any static file: the Esri Hub
"Export Data" link is a `replicafilescache` artifact of one specific export
click that 302-redirects to an Azure SAS URL expiring about an hour after
being minted, unusable as a config value a pipeline re-runs against
indefinitely. The REST query endpoint is the permanent, versioned API.

Verified live against the real service while building this: the bare
FeatureServer layer URL (`ESRIJSON:.../FeatureServer/0`, letting GDAL's
ESRIJSON driver auto-discover pagination) fails outright (it 404s/HTML's on
`f=json`-less requests), and a full un-paginated `outFields=*&f=json` query
(letting GDAL or the server return everything in one response) stalls --
individual ecoregion polygon sizes are wildly non-uniform (a Pacific atoll
vs. a continent-spanning biome), and an unpaginated response is both huge and
apparently subject to an upstream ~16MiB response-size cap that silently
truncates mid-JSON past that point. `download()` therefore pages through
`resultOffset` itself (`page_size` config, default 25), halving the page
size and retrying on any request/parse failure rather than trusting one
constant to always fit under that cap. The service also rate-limits "large
geometry" queries (confirmed live: `{"error":{"code":429,...,"details":
["...maximum allowed (60) per Minute. Retry after 60 sec."]}}`) -- delivered
as HTTP 200 with an error JSON body, not a real 429 status, so
`raise_for_status()` never sees it; `_rate_limit_wait_seconds()` sniffs for
it explicitly and sleeps the stated duration before retrying the same page
(not a parse failure -- shrinking page size would not fix a quota wait).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.common.raster.spatial import reproject_for_tile_overlap
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.misc._fetch import ConfiguredFile, ConfiguredFilesFetchMixin
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection
from src.data.sources import verify

logger = logging.getLogger(__name__)

#: RESOLVE Ecoregions 2017's own ArcGIS REST Feature Service (verified live:
#: layer 0, "Biomes and Ecoregions 2017", 847 polygon features, fields match
#: `_REQUIRED_COLUMNS` exactly). `geometryPrecision=5` (~1.1m at the equator)
#: trims payload size well below PREPARE's own `simplify_tolerance` (~111m
#: default) without losing anything that survives simplification anyway.
DEFAULT_URL = (
    "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/"
    "Resolve_Ecoregions/FeatureServer/0/query"
    "?where=1=1&outFields=REALM,BIOME_NUM,BIOME_NAME,ECO_ID,ECO_NAME"
    "&geometryPrecision=5&f=geojson"
)
DEFAULT_NAME = "resolve_ecoregions_2017.gpkg"
DEFAULT_PAGE_SIZE = 25

#: RESOLVE attribute column -> output id-grid variable name, in increasing
#: classification complexity (REALM -> BIOME_NUM -> ECO_ID).
CLASS_COLUMNS: Dict[str, str] = {"REALM": "realm_id", "BIOME_NUM": "biome_id", "ECO_ID": "eco_id"}
_REQUIRED_COLUMNS = ("REALM", "BIOME_NUM", "BIOME_NAME", "ECO_ID", "ECO_NAME")


def _rate_limit_wait_seconds(content: bytes) -> Optional[int]:
    """Esri delivers this service's rate-limit error as HTTP 200 + a JSON
    `{"error": {"code": 429, ...}}` body, not a real 4xx status -- confirmed
    live, see module docstring. Cheap pre-check (size + substring) before
    attempting a real JSON parse, since a legitimate feature page can be
    several MB and this must not cost a second full parse of it."""
    if len(content) > 2000 or b'"error"' not in content:
        return None
    try:
        payload = json.loads(content)
    except (ValueError, UnicodeDecodeError):
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    if not error or error.get("code") != 429:
        return None
    for detail in error.get("details", []):
        match = re.search(r"Retry after (\d+)\s*sec", str(detail))
        if match:
            return int(match.group(1))
    return 60


class EcoregionsSource(ConfiguredFilesFetchMixin, DataSource):
    """RESOLVE Ecoregions & Biomes -- simplified vector layer + rasterized
    realm/biome/ecoregion id grid, plus a GADM-GID_3 dominant-biome table."""

    ID = "ecoregions"
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)
    # gadm's PREPARE builds the simplified level-3 vectors and
    # GID_3_code_mapping.json directly; PipelineStep.GRID doesn't exist
    # anywhere. Scoped to this source's own PREPARE step.
    REQUIRES = ((PipelineStep.PREPARE, "gadm", PipelineStep.PREPARE),)

    DATA_SOURCE_NAME = "ecoregions"
    #: bump to force a full reprocess (`run_tiled_prepare`'s `processing_version`)
    PROCESSING_VERSION = "1-tiled"

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="misc")
        if cfg.namespace is None:
            cfg = dataclasses.replace(cfg, namespace="ecoregions")
        super().__init__(ctx, cfg)

        url = cfg.raw.get("url", DEFAULT_URL)
        name = cfg.raw.get("name", DEFAULT_NAME)
        self.CONFIGURED_FILES: List[ConfiguredFile] = [ConfiguredFile(key="ecoregions", url=url, name=name)]
        self.page_size = int(cfg.raw.get("page_size", DEFAULT_PAGE_SIZE))

        self.simplify_tolerance = cfg.raw.get("simplify_tolerance", 0.001)
        self.gadm_gid3_level = cfg.raw.get("gadm_gid3_level", "ADM_3")
        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="ecoregions_processor_")
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

    # -- FETCH download: ArcGIS REST query, manually paginated -- overrides
    # ConfiguredFilesFetchMixin's plain streaming-GET download()/download_async()
    # (see module docstring for why a plain download doesn't work here).

    def download(self, file_url: str, output_path: str, session: Any = None) -> None:
        import io

        import geopandas as gpd
        import pandas as pd
        import pyogrio
        import requests

        # GDAL's GeoJSON driver refuses to parse any single object past its
        # own `OGR_GEOJSON_MAX_OBJ_SIZE` ceiling ("GeoJSON object too
        # complex/large") -- confirmed live: one RESOLVE biome polygon alone
        # trips it even at `page_size=1`, so no amount of the halving retry
        # below can ever get past it (unlike a genuine truncated/oversized
        # *response*, which halving does fix). Raising it to unlimited is
        # safe here -- this source's own paging already bounds page byte
        # size independently, this just stops GDAL from second-guessing a
        # single legitimately huge polygon.
        pyogrio.set_gdal_config_options({"OGR_GEOJSON_MAX_OBJ_SIZE": "0"})

        s = session or requests.Session()
        page_size = self.page_size
        offset = 0
        frames: List["gpd.GeoDataFrame"] = []

        while True:
            page_url = f"{file_url}&resultOffset={offset}&resultRecordCount={page_size}"
            # The request itself (not just the parse below) is inside this
            # try -- a mid-transfer network failure on a large page (e.g.
            # `IncompleteRead`, confirmed live on a giant continent-spanning
            # biome polygon page) used to propagate straight out of
            # `download()` uncaught, burning one of this unit's limited
            # `manifest.record_failure` retry attempts -- and each of those
            # restarts this whole method from offset 0, discarding every
            # page already paged through. Treating a network failure the
            # same as a parse failure (halve and retry *this* offset) fixes
            # both: no wasted top-level attempt, and no lost progress.
            try:
                resp = s.get(page_url, timeout=120)
                resp.raise_for_status()
                content = resp.content

                retry_wait = _rate_limit_wait_seconds(content)
                if retry_wait is not None:
                    logger.warning(
                        "Ecoregions FETCH rate-limited at offset %d, waiting %ds before retrying",
                        offset, retry_wait,
                    )
                    time.sleep(retry_wait)
                    continue

                page = gpd.read_file(io.BytesIO(content))
            except Exception as exc:
                if page_size <= 1:
                    raise
                page_size = max(1, page_size // 2)
                logger.warning(
                    "Ecoregions FETCH page at offset %d failed (%s), halving page size to %d and retrying",
                    offset, exc, page_size,
                )
                continue

            if page.empty:
                break
            frames.append(page)
            offset += len(page)
            if len(page) < page_size:
                break
            # Proactive throttle -- the service caps "large geometry" queries
            # at 60/minute (confirmed live, module docstring); spacing
            # requests keeps a full run under that by construction rather
            # than relying on 429-triggered backoff to keep us honest.
            time.sleep(1.1)

        if not frames:
            raise RuntimeError(f"Ecoregions FETCH returned zero features from {file_url}")

        combined = pd.concat(frames, ignore_index=True)
        combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=frames[0].crs)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined.to_file(output_path, driver="GPKG")
        logger.info("Ecoregions FETCH complete: %d features (page_size=%d) -> %s", len(combined), page_size, output_path)

    async def download_async(self, file_url: str, output_path: str, session: Any = None) -> None:
        """The paginated `requests`-based loop above has no natural async
        form worth hand-rolling with aiohttp -- run it in a thread-pool
        executor instead, mirroring esacci.py's identical wrapping of a
        synchronous SDK call (`cdsapi.Client().retrieve`). *session*, if
        given, is aiohttp-flavored (this driver's own convention) and
        incompatible with the `requests.Session` `download()` builds
        internally -- not forwarded, since there is only ever one configured
        file here, so there is no cross-file connection pool to share."""
        import asyncio

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.download, file_url, output_path, None)

    # -- PREPARE -- phase 1 (shared): simplified .gpkg (single flat layer).
    # Phase 2: tiled rasterization (`ecoregions_grid`) and, once GADM's
    # level-3 output exists, the GID_3 dominant-biome table
    # (`gadm_gid3_dominant`) -- two independently resumable/executable
    # targets, exposed as one PREPARE step.

    def _raw_file_path(self) -> str:
        return os.path.join(self.output_root(PipelineStep.FETCH), self.CONFIGURED_FILES[0].name)

    def _vector_path(self) -> str:
        # MISC_AGG: a PREPARE-stage vector intermediate that feeds both the
        # CRS_AGG grid rasterization below and the ADM_AGG dominant-biome
        # table, but isn't itself GID-keyed or an admin boundary file --
        # "everything else" (src/data/sources/layout.py's crs/adm/misc
        # split), same judgment call as osm's land_polygons_simplified.gpkg.
        return os.path.join(
            self.output_root(PipelineStep.PREPARE, agg=layout.MISC_AGG), "ecoregions_simplified.gpkg"
        )

    def _ecoregions_grid_path(self) -> str:
        return layout.grid_store_path(
            self.ctx.data_root,
            self.cfg.data_path,
            grid_id=self.ctx.grid_id,
            family="ecoregions",
            suffix="",  # cell_id-keyed parquet parts, not a Zarr store -- see grid_store_path docstring
        )

    def _gadm_gid3_file(self) -> str:
        return os.path.join(
            layout.output_root(
                self.ctx.data_root, "misc", PipelineStep.PREPARE, namespace="gadm", agg=layout.ADM_AGG
            ),
            f"gadm_level{self.gadm_gid3_level}_simplified.gpkg",
        )

    def _dominant_biome_path(self) -> str:
        # A small per-GID (GID_3) parquet table, not a pixel-grid store --
        # ADM_AGG (no readers anywhere in src/ today; kept for future use,
        # same rationale as country_classifications.py's _output_path()).
        return os.path.join(
            layout.output_root(
                self.ctx.data_root, self.cfg.data_path, PipelineStep.PREPARE,
                namespace=self.cfg.namespace, agg=layout.ADM_AGG,
            ),
            "dominant_biome_by_gid3.parquet",
        )

    def _plan_prepare(self) -> List[StepTarget]:
        raw_file = self._raw_file_path()
        if not os.path.exists(raw_file):
            return []

        targets = [
            StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key="ecoregions_grid",
                output_path=self._ecoregions_grid_path(),
                inputs=(raw_file,), completion=Completion.MARKER,
                # No value_range: 0 = "no polygon at this pixel" (coastal/ocean
                # gaps), else a sequential id -- same rationale as gadm's own
                # GID_N variables (gadm.py:238-245).
                meta={
                    "raw_file": raw_file,
                    **verify.verification_meta(self.cfg.raw, expected_vars=tuple(CLASS_COLUMNS.values())),
                },
            )
        ]

        gadm_gid3_file = self._gadm_gid3_file()
        if os.path.exists(gadm_gid3_file):
            from src.data.sources.misc.gadm import gid_mapping_path

            gadm_gid3_mapping = gid_mapping_path(self.ctx.data_root, self.ctx.grid_id, "GID_3")
            if os.path.exists(gadm_gid3_mapping):
                targets.append(
                    StepTarget(
                        source_id=self.ID, step=PipelineStep.PREPARE, key="gadm_gid3_dominant",
                        output_path=self._dominant_biome_path(),
                        inputs=(raw_file, gadm_gid3_file, gadm_gid3_mapping), completion=Completion.PATH_EXISTS,
                        meta={
                            "raw_file": raw_file,
                            "gadm_gid3_file": gadm_gid3_file,
                            "gadm_gid3_mapping": gadm_gid3_mapping,
                            **verify.verification_meta(
                                self.cfg.raw, expected_vars=("GID_3", "dominant_biome_num", "biome_area_frac")
                            ),
                        },
                    )
                )
            else:
                logger.info(
                    "GADM GID_3 mapping not found at %s -- skipping dominant-biome table "
                    "until `data run --source gadm --step prepare` has completed.",
                    gadm_gid3_mapping,
                )
        else:
            logger.info(
                "GADM level-3 polygons not found at %s -- skipping dominant-biome table "
                "until `data run --source gadm --step prepare` has completed.",
                gadm_gid3_file,
            )
        return targets

    def _discover_prepare(self) -> List[StepTarget]:
        return self._plan_prepare()

    def _ensure_vector_file(self, raw_file: str) -> Optional[str]:
        """Phase 1, shared by both PREPARE targets -- whichever target runs
        first builds it; resumable via plain existence check since it's a
        whole-file output, not a zarr region."""
        import geopandas as gpd

        vector_path = self._vector_path()
        if not self.cfg.override and os.path.exists(vector_path):
            return vector_path

        os.makedirs(os.path.dirname(vector_path), exist_ok=True)

        if zipfile.is_zipfile(raw_file):
            extract_dir = os.path.join(self.temp_dir, "ecoregions_extracted")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(raw_file) as zf:
                zf.extractall(extract_dir)
            candidates = sorted(Path(extract_dir).glob("*.shp")) + sorted(Path(extract_dir).glob("*.gpkg"))
            if not candidates:
                logger.error("No .shp/.gpkg found inside ecoregions archive: %s", raw_file)
                return None
            source_path = str(candidates[0])
        else:
            source_path = raw_file

        gdf = gpd.read_file(source_path, engine="pyogrio")
        missing = [c for c in _REQUIRED_COLUMNS if c not in gdf.columns]
        if missing:
            logger.error(
                "Ecoregions source missing expected column(s) %s -- field names may differ from the "
                "RESOLVE 2017 schema this source assumes. Available columns: %s",
                missing, sorted(gdf.columns),
            )
            return None

        gdf_simplified = gdf.copy()
        gdf_simplified["geometry"] = gdf_simplified.geometry.simplify(
            tolerance=self.simplify_tolerance, preserve_topology=True
        )
        gdf_simplified.to_file(vector_path, driver="GPKG")
        logger.info("Ecoregions processing complete: %d polygons -> %s", len(gdf_simplified), vector_path)
        return vector_path

    def _execute_prepare(self, target: StepTarget) -> bool:
        if target.key == "gadm_gid3_dominant":
            return self._execute_gid3_dominant(target)
        return self._execute_ecoregions_grid(target)

    @staticmethod
    def _rasterize_tile(
        gdf: "gpd.GeoDataFrame",  # noqa: F821 -- geopandas imported lazily by caller
        code_to_id: Dict[str, Dict],
        tile,
    ) -> "xr.Dataset":
        """One tile's rasterized realm/biome/ecoregion id grid, on
        `tile.geobox` directly -- the `raw_getter` for
        `run_tiled_prepare(years=None, reproject=False, ...)`. `gdf`/
        `code_to_id` are built once by the caller and closed over, not
        reloaded per tile.

        Always returns a dataset, one column per `CLASS_COLUMNS` variable,
        even when no polygon overlaps this tile: the untouched-pixel
        default (0 = "no polygon at this pixel") is the correct output, and
        `run_tiled_prepare` would otherwise record a legitimate "nothing
        here" tile as a retryable failure.
        """
        import shapely.geometry
        import xarray as xr
        import numpy as np
        from odc.geo.geom import Geometry
        from odc.geo.xr import rasterize

        tile_geobox = tile.geobox
        tile_bounds = tile_geobox.boundingbox
        tile_polygon = shapely.geometry.box(tile_bounds.left, tile_bounds.bottom, tile_bounds.right, tile_bounds.top)

        tile_arrays = {var: np.zeros(tile_geobox.shape, dtype=np.uint32) for var in CLASS_COLUMNS.values()}
        # `gdf.sindex.query()` bbox-prunes candidates first (the index is
        # built lazily on first access and cached by geopandas across
        # calls, so this doesn't rebuild it per tile) -- a plain
        # `.intersects()` scan here would test every one of the ~14,000
        # RESOLVE polygons against every one of the ~100+ output tiles,
        # unlike snl_mining's analogous per-tile spatial join, which gets
        # this for free via DuckDB's rtree index.
        candidate_idx = gdf.sindex.query(tile_polygon, predicate="intersects")
        overlap = gdf.iloc[candidate_idx]
        # One rasterize() call per polygon, reused for all three id-grids --
        # REALM/BIOME_NUM/ECO_ID all live on the same geometry (unlike
        # GADM's genuinely distinct per-level polygons), so there's no need
        # to rasterize the same boundary three times.
        for _, row in overlap.iterrows():
            geom = Geometry(row.geometry, crs=str(gdf.crs))
            mask = rasterize(geom, tile_geobox)
            for col, var in CLASS_COLUMNS.items():
                value = code_to_id[col][row[col]]
                tile_arrays[var] = np.where(mask, value, tile_arrays[var])

        return xr.Dataset({var: (tile_geobox.dims, arr) for var, arr in tile_arrays.items()})

    def _execute_ecoregions_grid(self, target: StepTarget) -> bool:
        from src.data.common.geobox import get_target_geobox
        from src.data.common.prepare.driver import run_tiled_prepare
        from src.data.common.raster.spatial import SpatialProcessor
        from src.data.sources.steps import is_complete, mark_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping ecoregions rasterization, output already exists: %s", target.output_path)
            return True

        import geopandas as gpd

        vector_path = self._ensure_vector_file(target.inputs[0])
        if vector_path is None:
            return False

        gdf = gpd.read_file(vector_path, engine="pyogrio")
        missing = [c for c in CLASS_COLUMNS if c not in gdf.columns]
        if missing:
            logger.error("Prepared ecoregions file missing column(s) %s: %s", missing, vector_path)
            return False

        code_to_id: Dict[str, Dict] = {
            col: {code: i + 1 for i, code in enumerate(sorted(gdf[col].unique()))} for col in CLASS_COLUMNS
        }

        with self._dask_client() as client:
            dashboard_link = getattr(client, "dashboard_link", None)
            if dashboard_link:
                logger.info("Created Dask client for ecoregions rasterization: %s", dashboard_link)

            geobox = get_target_geobox(self.ctx)

            # Reproject once, up front -- same CRS-mismatch pitfall gadm hit
            # (commit f653033): the per-tile overlap prefilter compares raw
            # geometries against a tile polygon already in the target
            # geobox's CRS, with no reprojection of its own. See
            # reproject_for_tile_overlap()'s docstring for details.
            gdf = reproject_for_tile_overlap(gdf, geobox.crs)

            processor = SpatialProcessor(hpc_root=self.ctx.data_root, target_geobox=geobox)
            ok = run_tiled_prepare(
                output_path=target.output_path,
                years=None,
                target_geobox=geobox,
                processor=processor,
                raw_getter=lambda tile, year: self._rasterize_tile(gdf, code_to_id, tile),
                reproject=False,
                processing_version=self.PROCESSING_VERSION,
                override=self.cfg.override,
            )
            if not ok:
                return False

        output_dir = os.path.dirname(target.output_path)
        os.makedirs(output_dir, exist_ok=True)
        for col, var in CLASS_COLUMNS.items():
            with open(os.path.join(output_dir, f"{var}_code_mapping.json"), "w") as f:
                json.dump({str(k): v for k, v in code_to_id[col].items()}, f, indent=2, default=str)

        # Redundant with run_tiled_prepare's own internal mark_complete() in
        # the real path (harmless double-write of the same marker file) --
        # kept explicit so this method's own completion contract doesn't
        # depend on what run_tiled_prepare happens to delegate to internally
        # (e.g. tests stubbing it directly).
        mark_complete(target.output_path)
        return True

    # _dask_client: inherited from DataSource (src/data/sources/base.py).

    # -- GID_3 dominant-biome table -----------------------------------------

    def _execute_gid3_dominant(self, target: StepTarget) -> bool:
        import geopandas as gpd

        if not self.cfg.override and os.path.exists(target.output_path):
            logger.info("Skipping GID_3 dominant-biome table, output already exists: %s", target.output_path)
            return True

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)
        raw_file, gadm_gid3_file, gadm_gid3_mapping_file = target.inputs

        ecoregions_file = self._ensure_vector_file(raw_file)
        if ecoregions_file is None:
            return False

        with open(gadm_gid3_mapping_file) as f:
            code_to_id: Dict[str, int] = json.load(f)

        from src.data.common.geobox import get_target_geobox
        from src.data.sources.ecoregions import overlay

        geobox = get_target_geobox(self.ctx)
        ecoregions_gdf = gpd.read_file(ecoregions_file, engine="pyogrio")
        gadm_gid3_gdf = gpd.read_file(gadm_gid3_file, engine="pyogrio")

        out_df = overlay.compute_dominant_classes(
            gadm_gid3_gdf, ecoregions_gdf, gid_col="GID_3", crs=geobox.crs, code_to_id=code_to_id,
        )
        out_df.to_parquet(target.output_path, index=False)
        logger.info(
            "GID_3 dominant-biome table complete: %d admin unit(s) -> %s", len(out_df), target.output_path
        )
        return True


registry.register(
    EcoregionsSource.ID, __name__, EcoregionsSource.__name__, EcoregionsSource.STEPS,
    requires=EcoregionsSource.REQUIRES,
)
