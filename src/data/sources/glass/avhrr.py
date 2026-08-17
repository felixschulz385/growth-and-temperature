"""GLASS AVHRR LST (Daily 0.05D): fetch + prepare.

docs/design/12-glass-modis-rebuild.md §0/§4: split out of the former single
`GlassSource` (which branched internally on `data_source_kind` between MODIS
and AVHRR). This module is a **mechanical extraction** of the AVHRR-only
code paths -- behavior is unchanged from before the split; only the MODIS
branches/attributes have been deleted. See `src/data/sources/glass/modis.py`
for GLASS-MODIS's rebuilt (raw-MODIS-shaped) pipeline, which this module has
nothing to do with anymore.

FETCH   -- static per-(year, day) target list, attempt-and-log, no crawl/
           entrypoint cache (docs/design/11-glass-static-fetch.md). One
           global file per day, no tile dimension.
PREPARE -- one annual zarr per year with annual+monthly LST statistics
           (mean/median/std/max/min/rolling/threshold counts/valid-count),
           then GLASS's own bespoke tiled reprojection onto the canonical
           EPSG:4326 geobox (32px-halo pad, "mode" resampling), region-
           written into one shared multi-year zarr. Does not use the shared
           `SpatialProcessor` -- this bespoke path predates it and is kept
           as-is (unlike GLASS-MODIS, which now does use it).
"""

from __future__ import annotations

import calendar
import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from odc.geo.geom import clip_lon180
from odc.geo.xr import xr_reproject
from zarr.codecs import BloscCodec

from src.data.common.raster.spatial import write_crs_and_grid_mapping_encoding
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection
from src.data.sources import verify

logger = logging.getLogger(__name__)


def daterange_doy(start: Tuple[int, int], end: Tuple[int, int]):
    """(year, day) pairs from *start* through *end* inclusive, leap-aware.
    Neither bound needs to land on a calendar-year edge -- the first/last
    year are clipped to the given day, interior years use their own real
    365/366-day count (docs/design/11-glass-static-fetch.md §2)."""
    (y0, d0), (y1, d1) = start, end
    for year in range(y0, y1 + 1):
        days_in_year = 366 if calendar.isleap(year) else 365
        first_day = d0 if year == y0 else 1
        last_day = d1 if year == y1 else days_in_year
        for day in range(first_day, last_day + 1):
            yield year, day


class GlassAvhrrSource(DataSource):
    """GLASS LST, AVHRR (one global file/day).

    FETCH   -- crawl + download the configured `base_url` directory tree.
    PREPARE -- one annual zarr per year with annual+monthly LST statistics
               (mean/median/std/max/min/rolling/threshold counts/valid-
               count). GRID -- chunked tile-by-tile reprojection onto the
               canonical EPSG:4326 geobox, region-written into one shared
               multi-year zarr. Does not use the shared `SpatialProcessor`.
    """

    ID = "glass_avhrr"
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)

    BUCKET_NAME = "growthandheat"
    AVHRR_PATH_PREFIX = "glass/LST/AVHRR/0.05D/"
    VARIABLE_NAME = "LST"

    # GRID-output verification: the LST summary stats (not the "gt30C"/"lt0C"/
    # "valid_count" day-count vars _calculate_statistics also writes), scaled
    # (scale_factor=0.01) Kelvin, raw values masked to [20000, 35000] before
    # storage per _calculate_statistics -- i.e. ~200-350K physically. "std" is
    # excluded from the range check (via range_vars below): it's a spread,
    # not an absolute temperature, so it isn't on the same Kelvin scale.
    _STAT_VARS = ("mean", "median", "std", "max", "min", "rollmax3", "rollmin3")
    _RANGE_VARS = ("mean", "median", "max", "min", "rollmax3", "rollmin3")
    _LST_VALUE_RANGE = (150, 350)

    DATA_SOURCE_NAME = "glass"

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        self.path_prefix = self.AVHRR_PATH_PREFIX

        if cfg.data_path is None:
            import dataclasses

            cfg = dataclasses.replace(cfg, data_path=self.path_prefix.rstrip("/"))  # old GlassPreprocessor default
        super().__init__(ctx, cfg)

        self.base_url: Optional[str] = cfg.raw.get("base_url")
        if not self.base_url:
            raise ValueError("'base_url' is required.")

        # FETCH target space (docs/design/11-glass-static-fetch.md §2): static
        # (year, day-of-year) bounds, not derived from `cfg.year_range` (that
        # implies a calendar-year-aligned span; these bounds start/end
        # mid-year). No default -- every real config sets this explicitly.
        day_range = cfg.raw.get("day_range")
        if not day_range or "start" not in day_range or "end" not in day_range:
            raise ValueError("'day_range' with 'start'/'end' [year, day] pairs is required.")
        self.day_range_start: Tuple[int, int] = tuple(day_range["start"])
        self.day_range_end: Tuple[int, int] = tuple(day_range["end"])

        self.version = cfg.raw.get("version", "v1")
        self.chunk_size = cfg.raw.get("chunk_size") or {"band": 1, "x": 500, "y": 500}
        self.dashboard_port = cfg.raw.get("dashboard_port", ctx.dashboard_port)

        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="glass_AVHRR_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

        # In-run-only memoization of one (year) directory listing across many
        # sibling targets (every day sharing an AVHRR year) -- never
        # persisted to disk (docs/design/11-glass-static-fetch.md §4.3):
        # `source.execute()` is called once per target from the same
        # long-lived instance (src/cli/data/handlers.py::handle_run's
        # `for target in targets` loop), so this dict lives exactly as long
        # as one FETCH run needs.
        self._listing_cache: Dict[Any, List[Tuple[str, str]]] = {}

    def output_root(self, step: PipelineStep, *, namespace: str | None = None) -> str:
        """Overrides the base default: GLASS's output root is keyed by the
        fixed AVHRR `path_prefix` constant, not `cfg.data_path` (which
        exists only for index-file naming, matching old
        `GlassPreprocessor.get_hpc_output_path` using `self.path_prefix`).

        `agg=CRS_AGG` for PREPARE specifically: `layout.output_root()`
        requires an explicit `agg` for PREPARE (`src/data/sources/layout.py`'s
        CRS_AGG/ADM_AGG/MISC_AGG physical-layout split, added independently
        of this rebuild). GLASS-AVHRR's PREPARE output (`_annual_zarr_path()`'s
        per-year daily->annual stats zarr) is a pixel-grid raster store --
        just not yet reprojected onto the canonical grid, which is what the
        final `_grid_output_path()` (routed through GRID, not PREPARE, so it
        never needs `agg`) does -- so CRS_AGG is the locally-consistent
        bucket, unambiguous here since AVHRR has no non-pixel-grid PREPARE
        output that would need ADM_AGG/MISC_AGG instead."""
        return layout.output_root(
            self.ctx.data_root,
            self.path_prefix,
            step,
            namespace=namespace,
            grid_id=self.ctx.grid_id,
            agg=layout.CRS_AGG if step is PipelineStep.PREPARE else None,
        )

    def download(self, file_url: str, output_path: str, session: Any = None) -> None:
        import requests

        s = session or requests.Session()
        r = s.get(file_url, stream=True)
        r.raise_for_status()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # ------------------------------------------------------------------
    # plan()/execute() dispatch
    # ------------------------------------------------------------------

    def _plan(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        if step is PipelineStep.FETCH:
            return self._plan_fetch(selection)
        if step is PipelineStep.PREPARE:
            return self._plan_prepare(selection)
        raise AssertionError(f"unreachable: {step}")

    def _execute(self, target: StepTarget) -> bool:
        if target.step is PipelineStep.FETCH:
            return self._execute_fetch(target)
        if target.step is PipelineStep.PREPARE:
            return self._execute_prepare(target)
        raise AssertionError(f"unreachable: {target.step}")

    # -- FETCH: static per-(year, day) target list, attempt-and-log,
    # no crawl/entrypoint cache (docs/design/11-glass-static-fetch.md) -------

    def _index_existing_fetch_files(self, relative_paths: List[str]) -> Dict[tuple, str]:
        """Already-downloaded files on disk, keyed by the same `(year, day)`
        tuple `_plan_fetch()` generates targets for -- reuses
        `_parse_avhrr_filenames` (already needed by PREPARE's
        `_group_daily_files`), just re-keyed here for a plan-time completion
        lookup instead of a year-keyed grouping. One bulk parse up front
        rather than a per-target existence check."""
        df = self._parse_filenames(relative_paths)
        if df.empty:
            return {}
        keys = zip(df["year"], df["day"])
        return dict(zip(keys, df["path"]))

    def _pending_path(self, raw_root: str, year: int, day: int) -> str:
        """Deterministic placeholder for a not-yet-downloaded target's
        `StepTarget.output_path` -- only used for the `Completion.PATH_EXISTS`
        check (guaranteed not to exist yet, real filenames always start with
        "GLASS..."), never as an actual download destination
        (`_execute_fetch()` computes the real path itself once the remote
        listing resolves the true, unpredictable filename)."""
        return os.path.join(raw_root, str(year), f"pending.{year}{day:03d}.hdf")

    def _plan_fetch(self, selection: TargetSelection) -> List[StepTarget]:
        from src.data.common.fetch.manifest import resolve_fetch_listing
        from src.data.common.statusfile import STATUS_SUBDIR

        raw_root = self.output_root(PipelineStep.FETCH)
        # `transfer_mode=auto` (default for glass_avhrr) pushes each fetched
        # file to HPC right after FETCH and doesn't keep it locally
        # indefinitely -- same reasoning as MODIS's own `_plan_fetch()`, see
        # its comment. `selection.local_only` (`data summary`'s deliberately
        # network-free targets) forces the local listing regardless of
        # transfer_mode.
        listing, from_remote = resolve_fetch_listing(self, raw_root, allow_remote=not selection.local_only)
        status_prefix = f"{STATUS_SUBDIR}/"
        relative_paths = [rel for rel in listing if not rel.startswith(status_prefix)]
        found = self._index_existing_fetch_files(relative_paths)

        targets = []
        for year, day in daterange_doy(self.day_range_start, self.day_range_end):
            if not selection.matches_year(year):
                continue
            key = f"{year}/{day:03d}"
            if not selection.matches_key(key):
                continue
            found_key = (year, day)
            existing_rel = found.get(found_key)
            output_path = (
                os.path.join(raw_root, existing_rel)
                if existing_rel is not None
                else self._pending_path(raw_root, year, day)
            )
            if from_remote:
                completion = Completion.PRECOMPUTED
                meta = {"year": year, "day": day, "tile": None, "complete": existing_rel is not None}
            else:
                completion = Completion.PATH_EXISTS
                meta = {"year": year, "day": day, "tile": None}
            targets.append(
                StepTarget(
                    source_id=self.cfg.source_id,
                    step=PipelineStep.FETCH,
                    key=key,
                    output_path=output_path,
                    completion=completion,
                    meta=meta,
                )
            )
        return targets

    def _listing_url(self, year: int, day: int) -> str:
        """Year directory (`<year>/`, no day subdirectory -- confirmed
        against the real site, docs/design/11-glass-static-fetch.md §3)."""
        return f"{self.base_url}{year}/"

    @staticmethod
    def _list_single_directory(url: str) -> List[Tuple[str, str]]:
        """Non-recursive: one GET, parse one directory listing page's links
        -- every `(raw_href, absolute_url)` pair, no filtering (mirrors EOG's
        `_CrawlerMixin._list_single_directory()`, `src/data/sources/eog/
        crawler.py`, the same non-recursive shape already used elsewhere in
        this codebase for exactly this purpose)."""
        import requests
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        res = requests.get(url)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        results = []
        for link in soup.find_all("a"):
            href = link.get("href")
            if not href or href in ("../", "./"):
                continue
            results.append((href, urljoin(url, href)))
        return results

    def _listing_for(self, year: int, day: int) -> List[Tuple[str, str]]:
        """Memoized per §4.3 -- one real GET per year, shared across every
        sibling target that scope resolves, for the lifetime of this source
        instance (one FETCH run)."""
        cache_key = year
        if cache_key not in self._listing_cache:
            self._listing_cache[cache_key] = self._list_single_directory(self._listing_url(year, day))
        return self._listing_cache[cache_key]

    @staticmethod
    def _match_in_listing(
        listing: List[Tuple[str, str]], year: int, day: int, tile: Optional[str] = None
    ) -> Optional[Tuple[str, str]]:
        """The one listing entry matching this target's `(year, day)` --
        filenames always embed `A{year}{day:03d}` verbatim, e.g.
        `GLASS08B31.V40.A1982001.2021259.hdf` -- the trailing processing-date
        segment is the only unpredictable part, which this doesn't need to
        know."""
        token = f"A{year}{day:03d}"
        needle = f".{token}."
        for href, url in listing:
            if needle in href:
                return href, url
        return None

    def _downloaded_path(self, raw_root: str, year: int, day: int, href: str) -> str:
        return os.path.join(raw_root, str(year), href)

    def _execute_fetch(self, target: StepTarget) -> bool:
        import requests

        from src.data.common.fetch import manifest
        from src.data.sources.steps import is_complete

        status_dir = self.output_root(PipelineStep.FETCH)
        # `target.output_path` is only a reliable "what got written" answer
        # when it was resolved from a real local/remote listing entry at
        # plan time (`_plan_fetch()`) -- an outstanding target's is the
        # synthetic `_pending_path()` placeholder, since the real filename's
        # trailing processing-date is unpredictable until the listing below
        # resolves it. `handle_run` (`src/cli/data/handlers.py`) reads this
        # attribute, falling back to `target.output_path`, to push the right
        # file immediately after a successful fetch instead of a guaranteed-
        # wrong placeholder path.
        self._last_fetch_output_path = None
        if not self.cfg.override and is_complete(target):
            self._last_fetch_output_path = target.output_path
            return True

        year, day = target.meta["year"], target.meta["day"]
        try:
            listing = self._listing_for(year, day)
        except requests.HTTPError as exc:
            # A 404 on the listing itself (the day/year directory never
            # existed) is a real, permanent absence, not a transient
            # error -- distinguished from other request failures (timeout,
            # 5xx) below, which stay retryable.
            permanent = exc.response is not None and exc.response.status_code == 404
            manifest.record_failure(status_dir, target.key, f"listing fetch failed: {exc}", permanent=permanent)
            return False
        except requests.RequestException as exc:
            manifest.record_failure(status_dir, target.key, f"listing fetch failed: {exc}")
            return False

        match = self._match_in_listing(listing, year, day)
        if match is None:
            # Listing loaded fine but this day genuinely isn't in it -- a
            # real absence (sensor gap), not a transient error. No point
            # retrying against a directory that will never populate.
            manifest.record_failure(status_dir, target.key, "not present in remote listing", permanent=True)
            return False

        href, url = match
        output_path = self._downloaded_path(status_dir, year, day, href)
        try:
            self.download(url, output_path)
        except Exception as exc:
            manifest.record_failure(status_dir, target.key, f"download failed: {exc}")
            return False

        manifest.clear_failure(status_dir, target.key)
        self._last_fetch_output_path = output_path
        return True

    # -- PREPARE ("annual") -----------------------------------------------

    def _resolve_source_file_path(self, file_path: str) -> str:
        if os.path.isabs(file_path) or (self.ctx.data_root and file_path.startswith(self.ctx.data_root)):
            return file_path
        # Route through output_root(FETCH) rather than hand-building
        # "<path_prefix>/raw/..." -- FETCH output actually lives at
        # "raw/<data_path>/..." (src/data/sources/layout.py).
        return os.path.join(self.output_root(PipelineStep.FETCH), file_path)

    def _parse_filenames(self, filenames: List[str]) -> pd.DataFrame:
        """Expected format: GLASS08B31.V40.A1982001.2021259.hdf"""
        result = []
        for filename in filenames:
            try:
                basename = os.path.basename(filename)
                if not basename.endswith(".hdf"):
                    continue
                year_day_match = basename.split(".")[2]
                if not (year_day_match.startswith("A") and len(year_day_match) == 8):
                    continue
                result.append(
                    {"path": filename, "year": int(year_day_match[1:5]), "day": int(year_day_match[5:8])}
                )
            except (IndexError, ValueError) as exc:
                logger.warning("Could not parse filename %s: %s", filename, exc)
        return pd.DataFrame(result)

    def _group_daily_files(self, selection: TargetSelection) -> List[Dict[str, Any]]:
        """Live ground truth for which daily files exist per year: a crawl of
        FETCH's raw output directory (`snapshot_local_listing`, the same
        primitive FETCH's own driver uses). Called both to plan PREPARE
        (`_plan_prepare`) and, again, to execute it (`_execute_prepare`
        re-derives rather than trusting a StepTarget snapshot, since a
        group's daily file list can be large)."""
        from src.data.common.fetch.manifest import snapshot_local_listing
        from src.data.common.statusfile import STATUS_SUBDIR

        raw_root = self.output_root(PipelineStep.FETCH)
        listing = snapshot_local_listing(raw_root)
        status_prefix = f"{STATUS_SUBDIR}/"
        relative_paths = [rel for rel in listing if not rel.startswith(status_prefix)]

        files_df = self._parse_filenames(relative_paths)
        if files_df.empty:
            return []
        if selection.year_range:
            files_df = files_df[files_df["year"].between(selection.year_range[0], selection.year_range[1])]
        elif selection.years:
            files_df = files_df[files_df["year"].isin(selection.years)]

        groups: List[Dict[str, Any]] = []
        for year, group in files_df.groupby("year"):
            key = str(year)
            if not selection.matches_key(key):
                continue
            groups.append({"year": int(year), "grid_cell": "global", "key": key, "files": group["path"].tolist()})
        return groups

    def _annual_zarr_path(self, group: Dict[str, Any]) -> str:
        # TODO(GLASS rework): GLASS is slated to move to a new reprojection
        # workflow later; revisit whether these per-tile annual zarrs (and
        # `_ensure_annual_zarr` below) should become truly temporary instead
        # of persisted scratch, per the "only persist what's read downstream"
        # policy -- deferred for now because they double as a resumability
        # marker across process restarts (`marker_path` in
        # `_ensure_annual_zarr`).
        return os.path.join(self.output_root(PipelineStep.PREPARE), f"{group['year']}.zarr")

    def _plan_prepare(self, selection: TargetSelection) -> List[StepTarget]:
        """One PREPARE target per source ("all") -- `_execute_prepare`
        builds/reuses each year's daily->annual stats zarr internally, then
        reprojects them all into the final output in the same call (module
        docstring)."""
        groups = self._group_daily_files(selection)
        if not groups:
            return []
        years = sorted({g["year"] for g in groups})
        return [
            StepTarget(
                source_id=self.cfg.source_id,
                step=PipelineStep.PREPARE,
                key="all",
                output_path=self._grid_output_path(),
                completion=Completion.MARKER,
                meta={
                    "years_available": years,
                    "group_keys": [g["key"] for g in groups],
                    **verify.verification_meta(
                        self.cfg.raw,
                        expected_vars=self._STAT_VARS,
                        value_range=self._LST_VALUE_RANGE,
                        range_vars=self._RANGE_VARS,
                    ),
                },
            )
        ]

    def _dask_client(self):
        """Overrides `DataSource._dask_client`: GLASS supports a per-source
        config override of the dashboard port (`self.dashboard_port`, set in
        `__init__` from `cfg.raw` falling back to `ctx.dashboard_port`), so
        it can't use the shared base implementation as-is."""
        from src.data.common.dask.client import DaskClientContextManager

        return DaskClientContextManager(
            threads=self.ctx.dask_threads,
            memory_limit=self.ctx.dask_memory_limit,
            dashboard_port=self.dashboard_port,
            temp_dir=os.path.join(self.temp_dir, "dask_workspace"),
        )

    def _process_file_group_hpc(self, files: List[str], year: int, output_path: str, grid_cell: Optional[str] = None) -> bool:
        """Ported verbatim from GlassPreprocessor._process_file_group_hpc."""
        try:
            with self._dask_client() as client:
                if client is None:
                    logger.warning("Failed to initialize Dask client, proceeding without it")
                else:
                    dashboard_link = getattr(client, "dashboard_link", None)
                    if dashboard_link:
                        logger.info("Created Dask client for annual processing: %s", dashboard_link)

                files_with_day = []
                for file_path in files:
                    basename = os.path.basename(file_path)
                    year_day_match = basename.split(".")[2]
                    if year_day_match.startswith("A") and len(year_day_match) == 8:
                        day = int(year_day_match[5:8])
                        files_with_day.append((day, file_path))
                files_with_day.sort(key=lambda x: x[0])
                sorted_files = [f[1] for f in files_with_day]

                array_list = []
                days = []
                for file_path in sorted_files:
                    if not os.path.exists(file_path):
                        logger.warning("File does not exist: %s", file_path)
                        continue
                    basename = os.path.basename(file_path)
                    year_day_match = basename.split(".")[2]
                    day = int(year_day_match[5:8])
                    days.append(day)

                    chunks = {k: self.chunk_size[k] for k in ("band", "y", "x") if k in self.chunk_size}
                    ds = rxr.open_rasterio(file_path, decode_coords="all", chunks=chunks)
                    if hasattr(ds, "data_vars") and self.VARIABLE_NAME in ds.data_vars:
                        lst_data = ds[self.VARIABLE_NAME]
                    elif hasattr(ds, self.VARIABLE_NAME):
                        lst_data = getattr(ds, self.VARIABLE_NAME)
                    else:
                        logger.error("Could not find %s variable in %s", self.VARIABLE_NAME, file_path)
                        continue
                    array_list.append(lst_data)

                if not array_list:
                    logger.error("No valid files found for %s/%s", year, grid_cell)
                    return False

                combined_data = xr.concat(array_list, dim="day").rename({"day": "time"})
                combined_data = combined_data.assign_coords(
                    {"time": pd.to_datetime([f"{year}{day:03d}" for day in days], format="%Y%j")}
                )
                combined_data = combined_data.to_dataset(name=self.VARIABLE_NAME)

                logger.info("Calculating statistics with Dask")
                annual_stats, monthly_stats = self._calculate_statistics(combined_data)
                return self._create_annual_zarr_hpc(annual_stats, monthly_stats, output_path)
        except Exception:
            logger.exception("Error processing file group for %s/%s", year, grid_cell)
            return False

    def _calculate_statistics(self, data: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
        """Ported verbatim from GlassPreprocessor._calculate_statistics --
        INCLUDING its naive `resample(time="1YE").mean()` from raw daily
        data rather than from monthly_stats. GLASS-AVHRR keeps this
        behavior unchanged (docs/design/12-glass-modis-rebuild.md §0: only
        GLASS-MODIS's compositing was rebuilt, AVHRR's is untouched)."""
        mask = da.logical_and(data[self.VARIABLE_NAME] >= 20000, data[self.VARIABLE_NAME] <= 35000)
        masked = data.where(mask)
        rechunked = masked.chunk(
            {"time": -1, "x": self.chunk_size.get("x", 500), "y": self.chunk_size.get("y", 500)}
        )
        attrs = {"_FillValue": 0, "scale_factor": 0.01, "add_offset": 0.0}

        def format_output(arr):
            return arr.fillna(0).assign_attrs(attrs).astype(np.uint16, casting="unsafe")

        def format_output_count(arr):
            return arr.fillna(0).astype(np.uint16, casting="unsafe")

        annual_stats = xr.Dataset(
            {
                "mean": format_output(rechunked[self.VARIABLE_NAME].resample(time="1YE").mean()),
                "median": format_output(rechunked[self.VARIABLE_NAME].resample(time="1YE").median()),
                "std": format_output(rechunked[self.VARIABLE_NAME].resample(time="1YE").std()),
                "max": format_output(rechunked[self.VARIABLE_NAME].resample(time="1YE").max()),
                "min": format_output(rechunked[self.VARIABLE_NAME].resample(time="1YE").min()),
                "rollmax3": format_output(
                    rechunked[self.VARIABLE_NAME].rolling(time=3, center=True).mean().resample(time="1YE").max()
                ),
                "rollmin3": format_output(
                    rechunked[self.VARIABLE_NAME].rolling(time=3, center=True).mean().resample(time="1YE").min()
                ),
                "gt30C": format_output_count((rechunked[self.VARIABLE_NAME] > 30315).resample(time="1YE").sum()),
                "lt0C": format_output_count((rechunked[self.VARIABLE_NAME] < 27315).resample(time="1YE").sum()),
                "valid_count": format_output_count(mask.resample(time="1YE").sum()),
            }
        )

        monthly_stats = xr.Dataset(
            {
                "mean": format_output(data[self.VARIABLE_NAME].resample(time="1ME").mean()),
                "median": format_output(data[self.VARIABLE_NAME].resample(time="1ME").median()),
                "std": format_output(data[self.VARIABLE_NAME].resample(time="1ME").std()),
                "valid_count": mask.resample(time="1ME").sum().fillna(0).astype(np.uint16, casting="unsafe"),
            }
        )

        chunk_dict = {"time": 1}
        if "x" in self.chunk_size:
            chunk_dict["x"] = self.chunk_size["x"]
        if "y" in self.chunk_size:
            chunk_dict["y"] = self.chunk_size["y"]
        return annual_stats.chunk(chunk_dict), monthly_stats.chunk(chunk_dict)

    @staticmethod
    def _create_annual_zarr_hpc(annual_stats: xr.Dataset, monthly_stats: xr.Dataset, output_path: str) -> bool:
        try:
            chunks = {"x": 1000, "y": 1000, "time": 1}
            annual_stats = annual_stats.chunk(chunks)
            monthly_stats = monthly_stats.chunk(chunks)

            compressor = BloscCodec(cname="zstd", clevel=3, shuffle=2)
            encoding = {var: {"compressor": compressor} for var in annual_stats.data_vars}

            annual_output_path = output_path
            monthly_output_path = output_path.replace(".zarr", "_monthly.zarr")

            annual_stats.to_zarr(annual_output_path, mode="w", encoding=encoding, consolidated=True, compute=False).compute()
            monthly_encoding = {var: {"compressor": compressor} for var in monthly_stats.data_vars}
            monthly_stats.to_zarr(
                monthly_output_path, mode="w", encoding=monthly_encoding, consolidated=True, compute=False
            ).compute()
            return True
        except Exception:
            logger.exception("Error creating zarr file at %s.", output_path)
            return False

    def _grid_output_path(self) -> str:
        return layout.grid_store_path(
            self.ctx.data_root,
            self.path_prefix,
            grid_id=self.ctx.grid_id,
            family="glass_avhrr_lst",
        )

    def _ensure_annual_zarr(self, group: Dict[str, Any]) -> Optional[str]:
        """Build (or reuse) one year's daily->annual stats zarr -- the first
        phase of the PREPARE target's execution. Resumable the same way
        every other MARKER-completion output is: a sibling `.complete` file,
        checked directly rather than via a StepTarget (there isn't one per
        group)."""
        from src.data.sources.steps import marker_path

        annual_path = self._annual_zarr_path(group)
        if not self.cfg.override and os.path.exists(marker_path(annual_path)):
            return annual_path

        os.makedirs(os.path.dirname(annual_path), exist_ok=True)
        resolved_files = [self._resolve_source_file_path(f) for f in group["files"]]
        if not self._process_file_group_hpc(resolved_files, group["year"], annual_path, group["grid_cell"]):
            return None

        from src.data.sources.steps import mark_complete

        mark_complete(annual_path)
        return annual_path

    def _execute_prepare(self, target: StepTarget) -> bool:
        """First ensure every requested year's daily->annual stats zarr
        exists, then reproject them all into the final output, tile by
        tile, using GLASS's own bespoke reprojection path
        (`_process_years_chunked`/`_process_year_tiles`, ported verbatim
        from `GlassPreprocessor._process_spatial_target` and its chunked-tile
        helpers -- not the shared SpatialProcessor, see module docstring)."""
        from src.data.common.geobox import get_target_geobox
        from src.data.sources.steps import is_complete, mark_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping PREPARE -- already complete: %s", target.output_path)
            return True

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)

        years_available = set(target.meta["years_available"])
        group_keys = tuple(target.meta["group_keys"])
        groups = self._group_daily_files(TargetSelection(keys=group_keys))
        if not groups:
            logger.error("No daily file groups found for PREPARE (source=%s)", self.cfg.source_id)
            return False

        annual_paths = []
        for group in groups:
            annual_path = self._ensure_annual_zarr(group)
            if annual_path is None:
                logger.error("Failed to build annual stats zarr for %s/%s", group["year"], group["grid_cell"])
                return False
            annual_paths.append(annual_path)

        try:
            with self._dask_client() as client:
                dashboard_link = getattr(client, "dashboard_link", None)
                if dashboard_link:
                    logger.info("Created Dask client for spatial processing: %s", dashboard_link)

                import dask

                with dask.config.set(
                    {
                        "array.slicing.split_large_chunks": True,
                        "array.chunk-size": "512MB",
                        "optimization.fuse.active": False,
                        "distributed.comm.compression": "lz4",
                    }
                ):
                    try:
                        target_geobox = get_target_geobox(self.ctx)
                    except Exception:
                        logger.exception("Failed to get target geobox")
                        return False

                    if not os.path.exists(target.output_path):
                        if not self._create_empty_target_zarr(target.output_path, target_geobox, tuple(annual_paths)):
                            return False

                    ok = self._process_years_chunked(
                        annual_paths, target.output_path, target_geobox, sorted(years_available)
                    )
                    if ok:
                        mark_complete(target.output_path)
                    return ok
        except Exception:
            logger.exception("Error in GLASS spatial processing")
            return False

    def _create_empty_target_zarr(self, output_path: str, target_geobox, source_files: Tuple[str, ...]) -> bool:
        try:
            from src.data.common.raster.spatial import _NON_DATA_VAR_NAMES

            sample_ds = xr.open_zarr(source_files[0], mask_and_scale=False, chunks="auto", consolidated=False)
            # Exclude a leaked CRS grid-mapping variable (opened above
            # without decode_coords="all", so rioxarray's own "spatial_ref"
            # coordinate shows up in data_vars too) -- treating it as a real
            # data variable below would corrupt the actual CRS write_crs()
            # writes under the same name.
            variables = [v for v in sample_ds.data_vars.keys() if v not in _NON_DATA_VAR_NAMES]
            sample_attrs = sample_ds.attrs.copy()

            years = sorted(self._years_from_source_files(source_files))
            time_coords = pd.to_datetime([f"{year}-12-31" for year in years])

            ny, nx = target_geobox.shape
            dim_y, dim_x = target_geobox.dimensions
            y_coords = target_geobox.coords[dim_y].values.round(5)
            x_coords = target_geobox.coords[dim_x].values.round(5)

            data_vars = {}
            default_attrs = {"_FillValue": 0}
            packaging_attrs = {"scale_factor": 0.01, "add_offset": 0.0}
            for var in variables:
                var_attrs = sample_ds[var].attrs.copy() | default_attrs
                if "float" in str(sample_ds[var].dtype):
                    var_attrs |= packaging_attrs
                data_vars[var] = xr.DataArray(
                    da.zeros((len(years), 1, ny, nx), dtype=np.uint16, chunks=(1, 1, 512, 512)),
                    dims=["time", "band", dim_y, dim_x],
                    coords={"time": time_coords, "band": [1], dim_y: y_coords, dim_x: x_coords},
                    attrs=var_attrs,
                )
            sample_ds.close()

            empty_ds = xr.Dataset(data_vars, attrs=sample_attrs)
            compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", blocksize=0)
            base_encoding = {
                var: {"chunks": (1, 1, 512, 512), "compressors": (compressor,), "dtype": "uint16"} for var in variables
            }
            empty_ds, encoding = write_crs_and_grid_mapping_encoding(empty_ds, target_geobox, base_encoding)

            empty_ds.to_zarr(output_path, mode="w", encoding=encoding, compute=False, zarr_format=3, consolidated=False)
            return True
        except Exception:
            logger.exception("Error creating empty target zarr")
            return False

    @staticmethod
    def _years_from_source_files(source_files: Tuple[str, ...]) -> List[int]:
        years = set()
        for f in source_files:
            m = re.search(r"/(\d{4})/", f) or re.search(r"(\d{4})\.zarr", os.path.basename(f))
            if m:
                years.add(int(m.group(1)))
        return sorted(years)

    def _group_files_by_year(self, source_files: List[str]) -> Dict[int, List[str]]:
        files_by_year: Dict[int, List[str]] = {}
        for file_path in source_files:
            m = re.search(r"(\d{4})\.zarr", os.path.basename(file_path))
            if m:
                files_by_year.setdefault(int(m.group(1)), []).append(file_path)
        return files_by_year

    def _process_years_chunked(self, source_files: List[str], output_path: str, target_geobox, years_to_process: List[int]) -> bool:
        try:
            files_by_year = self._group_files_by_year(source_files)
            from odc.geo import GeoboxTiles

            tile_size = 2048
            tiles = GeoboxTiles(target_geobox, (tile_size, tile_size))

            for year in sorted(files_by_year):
                if year not in years_to_process:
                    continue
                year_files = files_by_year[year]
                logger.info("Processing year %s with %d files", year, len(year_files))

                if len(year_files) > 1:
                    prepare_root = layout.output_root(
                        self.ctx.data_root, self.path_prefix, PipelineStep.PREPARE,
                        agg=layout.CRS_AGG,
                    )
                    annual_temp_path = os.path.join(prepare_root, str(year), "temp_combined.tzarr")
                    if not self._aggregate_year_files(year_files, annual_temp_path, year):
                        logger.error("Failed to aggregate files for year %s", year)
                        return False
                    year_source = annual_temp_path
                else:
                    year_source = year_files[0]

                if not self._process_year_tiles(year_source, output_path, target_geobox, tiles, year, tile_size):
                    logger.error("Failed to process tiles for year %s", year)
                    return False
            return True
        except Exception:
            logger.exception("Error in chunked year processing")
            return False

    def _aggregate_year_files(self, year_files: List[str], temp_output_path: str, year: int) -> bool:
        try:
            if os.path.exists(temp_output_path):
                logger.info("Temporary output path already exists")
                return True

            logger.info("Aggregating %d files for year %s", len(year_files), year)
            datasets = []
            for file_path in year_files:
                ds = xr.open_zarr(file_path, decode_coords="all", chunks="auto")
                ds.coords["x"] = ds.coords["x"].astype("int")
                ds.coords["y"] = ds.coords["y"].astype("int")
                datasets.append(ds)

            combined = xr.combine_by_coords(datasets, combine_attrs="drop_conflicts", join="outer")
            x_coords = combined.coords["x"]
            y_coords = combined.coords["y"]
            nx, ny = len(x_coords), len(y_coords)

            variables = list(ds.data_vars.keys())
            coordinates = list(ds.coords.keys())
            data_vars = {}
            default_attrs = {"_FillValue": 0}
            packaging_attrs = {"scale_factor": 0.01, "add_offset": 0.0}
            for var in variables:
                var_attrs = combined[var].attrs.copy() | default_attrs
                if "float" in str(datasets[0][var].dtype):
                    var_attrs |= packaging_attrs
                data_vars[var] = xr.DataArray(
                    da.zeros((1, 1, ny, nx), dtype=np.uint16, chunks=(1, 1, 300, 300)),
                    dims=["band", "time", "y", "x"],
                    coords={"band": [1], "time": [pd.to_datetime(f"{year}-12-31")], "y": y_coords, "x": x_coords},
                    attrs=var_attrs,
                )
            combined_ds = xr.Dataset(data_vars)

            crs = 4326
            combined_ds = combined_ds.rio.write_crs(crs)

            compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", blocksize=0)
            encoding = {
                var: {"chunks": (1, 1, 300, 300), "compressors": (compressor,), "dtype": "uint16"} for var in variables
            } | {coord: {"compressors": (compressor,)} for coord in coordinates}

            combined_ds.to_zarr(temp_output_path, mode="w", encoding=encoding, zarr_format=3, consolidated=False, compute=False)

            for i, ds in enumerate(datasets):
                try:
                    ds_clean = ds.rio.write_crs(crs)
                    ds_clean = ds_clean.drop_vars(["spatial_ref"]).drop_attrs()
                    ds_clean = ds_clean.chunk({"band": 1, "time": 1, "y": 300, "x": 300})
                    ds_clean = ds_clean.isel(y=slice(None, None, -1))
                    if ds_clean.sizes["x"] == 0 or ds_clean.sizes["y"] == 0:
                        continue
                    ds_clean.to_zarr(temp_output_path, region="auto", align_chunks=True)
                except Exception:
                    logger.warning("Error processing region %d/%d for year %s", i + 1, len(datasets), year, exc_info=True)
                    continue

            combined_ds.close()
            for ds in datasets:
                ds.close()
            return True
        except Exception:
            logger.exception("Error aggregating year files for %s", year)
            return False

    def _process_year_tiles(self, year_source: str, output_path: str, target_geobox, tiles, year: int, tile_size: int) -> bool:
        try:
            year_ds = xr.open_zarr(year_source, consolidated=False, decode_coords="all")

            year_ds = year_ds.rio.write_crs(4326)
            year_ds = year_ds.sel(y=slice(None, None, -1))

            if year_ds.rio.crs is None:
                try:
                    year_ds = year_ds.rio.write_crs(year_ds.spatial_ref.crs_wkt)
                except Exception:
                    logger.warning("Error setting crs on dataset", exc_info=True)

            for ix in range(tiles.shape[0]):
                for iy in range(tiles.shape[1]):
                    try:
                        logger.info("Reprojecting tile [%d, %d] for year %s to: %s", ix, iy, year, output_path)
                        tile_geobox = tiles[ix, iy]

                        def extract_slice(ds, tg):
                            bbox = clip_lon180(tg.pad(32, 32).extent).to_crs(year_ds.rio.crs).boundingbox
                            return ds.sel(y=slice(bbox.bottom, bbox.top), x=slice(bbox.left, bbox.right))

                        clipped_ds = extract_slice(year_ds, tile_geobox).compute()
                        if clipped_ds.sizes["x"] == 0 or clipped_ds.sizes["y"] == 0:
                            continue

                        int_vars = [key for key, val in clipped_ds.dtypes.items() if np.issubdtype(val, np.integer)]
                        if int_vars:
                            clipped_ds[int_vars] = clipped_ds[int_vars].astype("float32").where(clipped_ds.valid_count > 0, np.nan)

                        reprojected_ds = xr_reproject(clipped_ds, tile_geobox, resampling="mode", dst_nodata=np.nan)
                        reprojected_ds = reprojected_ds.drop_vars(["spatial_ref"]).drop_attrs()
                        tile_dim_y, tile_dim_x = tile_geobox.dimensions
                        reprojected_ds.coords[tile_dim_x] = reprojected_ds.coords[tile_dim_x].round(5)
                        reprojected_ds.coords[tile_dim_y] = reprojected_ds.coords[tile_dim_y].round(5)
                        reprojected_ds = reprojected_ds.chunk({"band": 1, "time": 1, tile_dim_y: 512, tile_dim_x: 512})

                        reprojected_ds.to_zarr(output_path, region="auto", align_chunks=True, zarr_format=3, consolidated=False)
                        reprojected_ds.close()
                    except Exception:
                        logger.warning("Error processing tile [%d, %d] for year %s", ix, iy, year, exc_info=True)
                        continue

            year_ds.close()
            return True
        except Exception:
            logger.exception("Error processing tiles for year %s", year)
            return False


registry.register(
    "glass_avhrr",
    __name__,
    GlassAvhrrSource.__name__,
    GlassAvhrrSource.STEPS,
)
