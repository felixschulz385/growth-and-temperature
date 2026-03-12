"""
Preprocessor for ACAG (Atmospheric Composition Analysis Group) PM2.5 data.

Two-stage pipeline:
  Stage 1 (annual)  – Convert raw .nc/.nc4 files to annual zarr archives.
  Stage 2 (spatial) – Reproject to the project-wide unified geobox.
"""
import os
import re
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import dask.array as da
from zarr.codecs import BloscCodec

from gnt.data.preprocess.sources.base import AbstractPreprocessor
from gnt.data.preprocess.common.spatial import SpatialProcessor
from gnt.data.download.sources.acag import ACAGDataSource

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pa = None
    pq = None

logger = logging.getLogger(__name__)


class ACAGPreprocessor(AbstractPreprocessor):
    """
    HPC-mode preprocessor for ACAG annual PM2.5 surface concentration data.

    Processing stages
    -----------------
    1. **annual** – read one raw .nc/.nc4 file per year, extract the PM2.5
       variable (auto-detected), and write a standard annual zarr with
       dimensions ``(time=1, band=1, y, x)``.
    2. **spatial** – reproject all annual zarr files to the project-wide
       unified geobox and write a single multi-year zarr time series.

    Configuration keys (passed as ``**kwargs`` or via ``from_config``)
    ------------------------------------------------------------------
    stage : str
        ``"annual"`` or ``"spatial"``  (default ``"annual"``)
    year / year_range : int / [int, int]
        Year(s) to process (one of the two must be set).
    hpc_target : str
        Required – HPC root directory (may carry ``user@host:`` prefix).
    data_path : str
        Sub-path under *hpc_root* where the raw files live.
        Default: ``"acag/pm25"``.
    override : bool
        Re-process even when output already exists (default ``False``).
    dask_threads / dask_memory_limit : int / str
        Dask scheduler parameters (optional).
    temp_dir : str
        Working directory for temporary files.
    """

    # Candidate variable names used by different ACAG file versions
    _PM25_CANDIDATES = [
        "GWRPM25",       # V4 / V5 GWR product
        "PM25",          # generic
        "pm25",
        "PM2_5",
        "pm2_5",
        "Annual_PM2.5",  # some V6 flavours
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ---- Stage -------------------------------------------------------
        self.stage = kwargs.get("stage", "annual")
        if self.stage not in ("annual", "spatial"):
            raise ValueError(f"Unsupported stage '{self.stage}'. Use 'annual' or 'spatial'.")

        # ---- Years -------------------------------------------------------
        self.year = kwargs.get("year")
        self.year_range = kwargs.get("year_range")
        if self.year is None and self.year_range is None:
            raise ValueError("Either 'year' or 'year_range' must be specified.")

        if self.year is not None:
            self.years_to_process = [self.year]
        else:
            if not isinstance(self.year_range, (list, tuple)) or len(self.year_range) != 2:
                raise ValueError("'year_range' must be [start_year, end_year].")
            self.years_to_process = list(range(int(self.year_range[0]), int(self.year_range[1]) + 1))
            logger.info("Processing year range %d–%d (%d years)",
                        self.year_range[0], self.year_range[1], len(self.years_to_process))

        # ---- HPC paths ---------------------------------------------------
        hpc_target = kwargs.get("hpc_target")
        self.hpc_root = self._strip_remote_prefix(hpc_target)
        if not self.hpc_root:
            raise ValueError("'hpc_target' is required.")
        self.index_dir = os.path.join(self.hpc_root, "hpc_data_index")

        self.data_path = kwargs.get("data_path") or kwargs.get("output_path") or "acag/pm25"

        # ---- Other settings ----------------------------------------------
        self.override = kwargs.get("override", False)
        self.dask_threads = kwargs.get("dask_threads")
        self.dask_memory_limit = kwargs.get("dask_memory_limit")

        self.temp_dir = kwargs.get("temp_dir") or tempfile.mkdtemp(prefix="acag_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

        # ---- Data source (for metadata / path resolution) ----------------
        self.data_source = ACAGDataSource(
            file_extensions=kwargs.get("file_extensions"),
            output_path=self.data_path,
        )

        # ---- Parquet index path ------------------------------------------
        self._init_parquet_index_path()

        logger.info("Initialised ACAGPreprocessor  hpc_root=%s  data_path=%s  years=%d",
                    self.hpc_root, self.data_path, len(self.years_to_process))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _strip_remote_prefix(self, path: Optional[str]) -> Optional[str]:
        if isinstance(path, str):
            return re.sub(r"^[^@]+@[^:]+:", "", path)
        return path

    def _init_parquet_index_path(self):
        safe = self.data_path.replace("/", "_").replace("\\", "_")
        os.makedirs(self.index_dir, exist_ok=True)
        self.parquet_index_path = os.path.join(self.index_dir, f"parquet_{safe}.parquet")
        logger.debug("Parquet index path: %s", self.parquet_index_path)

    def _extract_year(self, path: str) -> Optional[int]:
        """Extract a four-digit year from a file path / name."""
        filename = os.path.basename(path)
        for pattern in (r"[._\-](\d{4})[._\-]", r"(\d{4})"):
            for m in re.finditer(pattern, filename):
                y = int(m.group(1))
                if 1990 <= y <= 2040:
                    return y
        return None

    def _resolve_raw_path(self, relative_path: str) -> str:
        """Turn the relative_path from the parquet index into a full path."""
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self.hpc_root, self.data_path, "raw", relative_path)

    def _initialize_dask_client(self):
        from gnt.data.common.dask.client import DaskClientContextManager
        return DaskClientContextManager(
            threads=self.dask_threads,
            memory_limit=self.dask_memory_limit,
            dashboard_port=8787,
            temp_dir=os.path.join(self.temp_dir, "dask_workspace"),
        )

    # ------------------------------------------------------------------
    # AbstractPreprocessor interface
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ACAGPreprocessor":
        return cls(**config)

    def get_hpc_output_path(self, stage: str) -> str:
        if stage == "annual":
            base = os.path.join(self.hpc_root, self.data_path, "processed", "stage_1")
        elif stage == "spatial":
            base = os.path.join(self.hpc_root, self.data_path, "processed", "stage_2")
        else:
            raise ValueError(f"Unknown stage: {stage}")
        return self._strip_remote_prefix(base)

    def get_preprocessing_targets(
        self, stage: str, year_range: Tuple[int, int] = None
    ) -> List[Dict[str, Any]]:
        """Build the list of targets by reading the parquet download index."""
        if not os.path.exists(self.parquet_index_path):
            logger.warning("Parquet index not found: %s", self.parquet_index_path)
            return []
        if not PANDAS_AVAILABLE:
            logger.error("pandas is not available – cannot read parquet index.")
            return []

        try:
            df = pd.read_parquet(self.parquet_index_path)

            # Filter downloaded / completed files
            if "status_category" in df.columns:
                df = df[df["status_category"] == "completed"]
            elif "download_status" in df.columns:
                df = df[df["download_status"] == "completed"]
            else:
                logger.warning("No status column found in parquet index.")
                return []

            if df.empty:
                logger.warning("No completed files in parquet index.")
                return []

            col = "relative_path" if "relative_path" in df.columns else None
            if col is None:
                logger.warning("'relative_path' column not found in parquet index.")
                return []

            file_paths = df[col].tolist()
            logger.info("Found %d completed files in parquet index.", len(file_paths))

            if stage == "annual":
                return self._gen_annual_targets(file_paths, year_range)
            elif stage == "spatial":
                return self._gen_spatial_targets(year_range)
            else:
                raise ValueError(f"Unknown stage: {stage}")

        except Exception:
            logger.exception("Error building preprocessing targets.")
            return []

    # ---- Target generation -------------------------------------------

    def _gen_annual_targets(
        self, file_paths: List[str], year_range: Optional[Tuple[int, int]]
    ) -> List[Dict]:
        """One target per year from the parquet index."""
        files_by_year: Dict[int, List[str]] = {}
        for rel_path in file_paths:
            clean = self._strip_remote_prefix(rel_path)
            year = self._extract_year(clean)
            if year is None:
                continue
            if year_range and not (year_range[0] <= year <= year_range[1]):
                continue
            files_by_year.setdefault(year, []).append(clean)

        targets = []
        for year in sorted(files_by_year):
            # Prefer .nc4 > .nc for resolution
            candidates = files_by_year[year]
            selected = next(
                (f for f in candidates if f.lower().endswith(".nc4")),
                next((f for f in candidates if f.lower().endswith(".nc")), candidates[0]),
            )
            targets.append({
                "year": year,
                "stage": "annual",
                "source_files": [selected],
                "output_path": os.path.join(self.get_hpc_output_path("annual"), f"{year}.zarr"),
                "dependencies": [],
                "metadata": {"source_type": "acag", "raw_candidates": len(candidates)},
            })

        logger.info("Generated %d annual targets.", len(targets))
        return targets

    def _gen_spatial_targets(self, year_range: Optional[Tuple[int, int]]) -> List[Dict]:
        """Single target that reprojection all available annual zarrs."""
        annual_files = self._list_annual_zarrs()
        if not annual_files:
            logger.warning("No annual zarr files found for spatial stage.")
            return []

        missing = set(self.years_to_process) - {f["year"] for f in annual_files}
        if missing:
            logger.warning("Missing annual zarr for years: %s", sorted(missing))

        return [{
            "stage": "spatial",
            "source_files": [f["zarr_path"] for f in annual_files],
            "output_path": os.path.join(
                self.get_hpc_output_path("spatial"), "acag_pm25_timeseries_reprojected.zarr"
            ),
            "dependencies": [f["zarr_path"] for f in annual_files],
            "metadata": {
                "source_type": "acag",
                "years_available": [f["year"] for f in annual_files],
                "years_requested": self.years_to_process,
                "missing_years": sorted(missing),
            },
        }]

    def _list_annual_zarrs(self) -> List[Dict]:
        """Return ``[{year, zarr_path}]`` for every ``<year>.zarr`` in stage_1."""
        annual_dir = self.get_hpc_output_path("annual")
        if not os.path.exists(annual_dir):
            return []
        results = []
        for fname in os.listdir(annual_dir):
            if fname.endswith(".zarr"):
                try:
                    year = int(os.path.splitext(fname)[0])
                    results.append({"year": year, "zarr_path": os.path.join(annual_dir, fname)})
                except ValueError:
                    pass
        return results

    # ------------------------------------------------------------------
    # process_target dispatcher
    # ------------------------------------------------------------------

    def process_target(self, target: Dict[str, Any]) -> bool:
        stage = target.get("stage")
        year = target.get("year")
        label = f"{stage}" + (f"/{year}" if year else "")
        logger.info("Processing target: %s", label)
        try:
            if stage == "annual":
                return self._process_annual_target(target)
            elif stage == "spatial":
                return self._process_spatial_target(target)
            else:
                logger.error("Unknown stage: %s", stage)
                return False
        except Exception:
            logger.exception("Error processing target %s.", label)
            return False

    # ------------------------------------------------------------------
    # Stage 1 – annual
    # ------------------------------------------------------------------

    def _process_annual_target(self, target: Dict[str, Any]) -> bool:
        year = target["year"]
        output_path = self._strip_remote_prefix(target["output_path"])

        if not self.override and os.path.exists(output_path):
            logger.info("Skipping year %d – output exists: %s", year, output_path)
            return True

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        raw_rel = target["source_files"][0]
        raw_abs = self._resolve_raw_path(raw_rel)

        try:
            ds = self._load_nc_as_dataset(raw_abs, year)
            if ds is None:
                logger.error("Failed to load %s for year %d.", raw_abs, year)
                return False

            success = self._write_annual_zarr(ds, output_path)
            if success:
                logger.info("Annual zarr written: %s", output_path)
            return success

        except Exception:
            logger.exception("Error in annual stage for year %d.", year)
            return False

    def _load_nc_as_dataset(self, file_path: str, year: int) -> Optional[xr.Dataset]:
        """
        Open a raw ACAG .nc/.nc4 (or HDF5) file, extract the PM2.5 variable,
        assign a time coordinate, normalise the spatial dimensions to
        ``(latitude, longitude)``, and apply ACAG-specific coordinate
        scaling.

        The function will first attempt to read the file with
        ``rioxarray.open_rasterio`` (which handles the HDF5 grid-style
        layout used by some ACAG products) and fall back to the classic
        ``xarray.open_dataset`` path otherwise.  In either case the
        following additional transformations are applied:

        * x/y grid indices are converted to degrees via ``lon = x*0.01 - 180``
          and ``lat = y*0.01 - 60``
        * dimensions/coords are renamed to ``latitude``/``longitude``
        * spatial dims are registered with rioxarray and a CRS is written
        * negative values are masked out and the array cast to ``float32``

        Returns an ``xr.Dataset`` with:
        - variable ``pm25``
        - dimensions ``(time, band, latitude, longitude)``
        - CRS written as WGS-84 (EPSG:4326)
        """
        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            return None

        try:
            da = rxr.open_rasterio(
                file_path,
                decode_coords="all",
                mask_and_scale=True,
                driver="HDF5",
            )
            logger.debug("Opened %s with rioxarray dims=%s", file_path, da.dims)
            # if the raster has multiple bands choose the first one and
            # drop the band dimension to match the single-variable API
            if "band" in da.dims and da.sizes.get("band", 1) > 1:
                da = da.isel(band=0)
            var_name = "pm25"
            # convert to a dataset immediately; this preserves the
            # DataArray attributes automatically as dataset attrs
            ds = da.to_dataset(name=var_name)
            ds.attrs["source_year"] = year
            ds.attrs["source_variable"] = var_name

            # ---- ACAG-specific coordinate handling -----------------------
            SP = 0.01
            if "x" in ds.coords:
                ds = ds.assign_coords(x=ds["x"] * SP - 180)
            if "y" in ds.coords:
                ds = ds.assign_coords(y=ds["y"] * SP - 60)

            # ---- Normalise spatial dim names ----------------------------
            dim_map = {}
            for d in ds.dims:
                dl = d.lower()
                if dl in ("lat", "latitude", "y"):
                    dim_map[d] = "latitude"
                elif dl in ("lon", "longitude", "x"):
                    dim_map[d] = "longitude"
            if dim_map:
                ds = ds.rename(dim_map)

            # set spatial dimensions explicitly and ensure CRS is known
            try:
                ds = ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
            except Exception:
                pass
            ds = ds.rio.write_crs("EPSG:4326")

            # drop invalid / negative values and reduce precision
            ds = ds.where(ds >= 0)
            ds = ds.astype("float32")

            # ---- Add time coordinate (annual → Dec-31) ------------------
            if "time" not in ds.dims:
                ds = ds.expand_dims(dim={"time": 1}).assign_coords(
                    time=[pd.Timestamp(f"{year}-12-31")]
                )

            # ---- Add band dimension for consistency with other sources ---
            if "band" not in ds.dims:
                ds = ds.expand_dims(dim={"band": 1}).assign_coords(band=[1])

            # Ensure canonical dim order
            ds = ds.transpose("time", "band", "latitude", "longitude")

            return ds

        except Exception:
            logger.exception("Error loading %s.", file_path)
            return None

    def _find_pm25_var(self, ds: xr.Dataset) -> Optional[str]:
        """Return the name of the PM2.5 variable, or None."""
        # Exact matches first
        for candidate in self._PM25_CANDIDATES:
            if candidate in ds.data_vars:
                return candidate
        # Case-insensitive fallback
        lower_map = {k.lower(): k for k in ds.data_vars}
        for candidate in self._PM25_CANDIDATES:
            if candidate.lower() in lower_map:
                return lower_map[candidate.lower()]
        # If only one variable, just use it
        if len(ds.data_vars) == 1:
            return next(iter(ds.data_vars))
        return None

    def _write_annual_zarr(self, ds: xr.Dataset, output_path: str) -> bool:
        """Write *ds* to *output_path* as a zarr v3 store."""
        try:
            ds = ds.chunk({"time": 1, "band": 1, "latitude": 512, "longitude": 512})

            compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", blocksize=0)
            encoding = {
                var: {
                    "compressors": (compressor,),
                    "chunks": (1, 1, 512, 512),
                }
                for var in ds.data_vars
            }

            ds.to_zarr(output_path, mode="w", encoding=encoding, zarr_format=3, consolidated=False)
            logger.info("Wrote annual zarr: %s", output_path)
            return True

        except Exception:
            logger.exception("Error writing annual zarr to %s.", output_path)
            return False

    # ------------------------------------------------------------------
    # Stage 2 – spatial reprojection
    # ------------------------------------------------------------------

    def _process_spatial_target(self, target: Dict[str, Any]) -> bool:
        output_path = self._strip_remote_prefix(target["output_path"])
        source_files: List[str] = target.get("source_files", [])

        if not source_files:
            logger.error("No source files listed in spatial target.")
            return False

        if not self.override and os.path.exists(output_path):
            logger.info("Skipping spatial stage – output exists: %s", output_path)
            return True

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            with self._initialize_dask_client() as client:
                if client is None:
                    logger.error("Failed to initialise Dask client.")
                    return False

                dashboard = getattr(client, "dashboard_link", None)
                if dashboard:
                    logger.info("Dask dashboard: %s", dashboard)

                spatial_processor = SpatialProcessor(
                    hpc_root=self.hpc_root,
                    temp_dir=self.temp_dir,
                    dask_client=client,
                )

                with spatial_processor.setup_dask_config():

                    def year_from_path(p: str) -> Optional[int]:
                        """Extract year from ``<year>.zarr`` filename."""
                        try:
                            return int(os.path.splitext(os.path.basename(p))[0])
                        except ValueError:
                            return None

                    def preprocess(ds: xr.Dataset) -> xr.Dataset:
                        """Ensure CRS is set before reprojection."""
                        if ds.rio.crs is None:
                            ds = ds.rio.write_crs("EPSG:4326")
                        return ds

                    def get_vars_and_attrs(file_path: str) -> Tuple[List[str], Dict]:
                        sample = xr.open_zarr(
                            file_path, mask_and_scale=False, chunks="auto", consolidated=False
                        )
                        variables = list(sample.data_vars.keys())
                        attrs = sample.attrs.copy()
                        sample.close()
                        return variables, attrs

                    success = spatial_processor.process_spatial_standard(
                        source_files=source_files,
                        output_path=output_path,
                        years_to_process=self.years_to_process,
                        year_pattern_func=year_from_path,
                        preprocess_func=preprocess,
                        get_variables_func=get_vars_and_attrs,
                    )

                    if success:
                        logger.info("ACAG spatial reprojection completed: %s", output_path)
                    return success

        except Exception:
            logger.exception("Error in ACAG spatial processing.")
            return False
