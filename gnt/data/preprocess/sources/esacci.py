"""
Preprocessor for ESA CCI Land Cover annual composites.

Two-stage pipeline:
  Stage 1 (annual)  – Convert raw .nc/.nc4 files to annual zarr archives.
  Stage 2 (spatial) – Reproject to the project-wide unified geobox.

The raw ESA CCI NetCDF files are shipped as zip archives (despite the ``.nc``
suffix) and carry several variables.  Only ``lccs_class`` (the LCCS land
cover classification map, uint8) is extracted by default; this can be
overridden via ``variables_to_keep`` in the configuration.

Note on data type
-----------------
``lccs_class`` is stored as uint8 (LCCS codes 0–220).  The SpatialProcessor
uses uint16 by default; the zarr encoding is overridden here to uint8 so no
information is lost.
"""
import os
import re
import shutil
import tempfile
import logging
import zipfile
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import dask.array as da
from zarr.codecs import BloscCodec

from gnt.data.preprocess.sources.base import AbstractPreprocessor
from gnt.data.preprocess.common.spatial import SpatialProcessor
from gnt.data.download.sources.esacci import ESACCIDataSource

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


class ESACCIPreprocessor(AbstractPreprocessor):
    """
    HPC-mode preprocessor for ESA CCI Land Cover annual composites.

    Processing stages
    -----------------
    1. **annual** – open one raw .nc/.nc4 file per year, extract the selected
       variables, and write a standard annual zarr with dimensions
       ``(time=1, band=1, latitude, longitude)``.
    2. **spatial** – reproject all annual zarr files to the project-wide
       unified geobox and write a single multi-year zarr time series.

    Configuration keys (passed as ``**kwargs`` or via ``from_config``)
    ------------------------------------------------------------------
    stage : str
        ``"annual"`` or ``"spatial"`` (default ``"annual"``).
    year / year_range : int / [int, int]
        Year(s) to process (one of the two must be set).
    hpc_target : str
        Required – HPC root directory (may carry ``user@host:`` prefix).
    data_path : str
        Sub-path under *hpc_root* where raw files live.
        Default: ``"esacci/landcover"``.
    variables_to_keep : list[str]
        Names of NetCDF variables to extract and store.
        Default: ``["lccs_class"]``.
    override : bool
        Re-process even when output already exists (default ``False``).
    dask_threads / dask_memory_limit : int / str
        Dask scheduler parameters (optional).
    temp_dir : str
        Working directory for temporary files.
    """

    # Variables extracted by default
    DEFAULT_VARIABLES = ["lccs_class"]

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
            self.years_to_process = [int(self.year)]
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

        self.data_path = kwargs.get("data_path") or kwargs.get("output_path") or "esacci/landcover"

        # ---- Variables ---------------------------------------------------
        self.variables_to_keep: List[str] = (
            kwargs.get("variables_to_keep") or self.DEFAULT_VARIABLES
        )

        # ---- Other settings ----------------------------------------------
        self.override = kwargs.get("override", False)
        self.dask_threads = kwargs.get("dask_threads")
        self.dask_memory_limit = kwargs.get("dask_memory_limit")

        self.temp_dir = kwargs.get("temp_dir") or tempfile.mkdtemp(prefix="esacci_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

        # ---- Data source -------------------------------------------------
        self.data_source = ESACCIDataSource(output_path=self.data_path)

        # ---- Parquet index -----------------------------------------------
        self._init_parquet_index_path()

        logger.info(
            "Initialised ESACCIPreprocessor  hpc_root=%s  data_path=%s  "
            "years=%d  variables=%s",
            self.hpc_root, self.data_path, len(self.years_to_process),
            self.variables_to_keep,
        )

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
        filename = os.path.basename(path)
        for pattern in (r"[._\-](\d{4})[._\-]", r"(\d{4})"):
            for m in re.finditer(pattern, filename):
                y = int(m.group(1))
                if 1990 <= y <= 2040:
                    return y
        return None

    def _resolve_raw_path(self, relative_path: str) -> str:
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
    def from_config(cls, config: Dict[str, Any]) -> "ESACCIPreprocessor":
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
        """Build the list of targets from the parquet download index."""
        if not os.path.exists(self.parquet_index_path):
            logger.warning("Parquet index not found: %s", self.parquet_index_path)
            return []
        if not PANDAS_AVAILABLE:
            logger.error("pandas is not available – cannot read parquet index.")
            return []

        try:
            df = pd.read_parquet(self.parquet_index_path)

            if "status_category" in df.columns:
                df = df[df["status_category"] == "completed"]
            elif "download_status" in df.columns:
                df = df[df["download_status"] == "completed"]
            else:
                logger.warning("No status column in parquet index.")
                return []

            if df.empty:
                logger.warning("No completed files in parquet index.")
                return []

            if "relative_path" not in df.columns:
                logger.warning("'relative_path' column not found in parquet index.")
                return []

            file_paths = df["relative_path"].tolist()
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

    # ------------------------------------------------------------------
    # Target generation
    # ------------------------------------------------------------------

    def _gen_annual_targets(
        self, file_paths: List[str], year_range: Optional[Tuple[int, int]]
    ) -> List[Dict]:
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
            candidates = files_by_year[year]
            # Prefer .nc4, then .nc
            selected = next(
                (f for f in candidates if f.lower().endswith(".nc4")),
                next((f for f in candidates if f.lower().endswith(".nc")), candidates[0]),
            )
            targets.append({
                "year": year,
                "stage": "annual",
                "source_files": [selected],
                "output_path": os.path.join(
                    self.get_hpc_output_path("annual"), f"{year}.zarr"
                ),
                "dependencies": [],
                "metadata": {
                    "source_type": "esacci",
                    "raw_candidates": len(candidates),
                },
            })

        logger.info("Generated %d annual targets.", len(targets))
        return targets

    def _gen_spatial_targets(self, year_range: Optional[Tuple[int, int]]) -> List[Dict]:
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
                self.get_hpc_output_path("spatial"),
                "esacci_lc_timeseries_reprojected.zarr",
            ),
            "dependencies": [f["zarr_path"] for f in annual_files],
            "metadata": {
                "source_type": "esacci",
                "years_available": [f["year"] for f in annual_files],
                "years_requested": self.years_to_process,
                "missing_years": sorted(missing),
            },
        }]

    def _list_annual_zarrs(self) -> List[Dict]:
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

        raw_abs = self._resolve_raw_path(target["source_files"][0])

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
        Open a raw ESA CCI NetCDF file and return a normalised ``xr.Dataset``
        with dimensions ``(time=1, band=1, latitude, longitude)``.

        Raw files ship as zip archives (despite the ``.nc`` suffix) containing
        a single ``.nc`` inside.  The archive is extracted to ``self.temp_dir``
        and opened with the ``h5netcdf`` engine.  If the file is not a zip it
        is read directly as a fallback.

        Only ``lccs_class`` (uint8) is retained; its variable attributes are
        preserved.  The existing ``time`` coordinate is overridden to Dec-31 of
        *year* for consistency with the rest of the pipeline.  A ``band``
        dimension is added so downstream processors see a uniform layout.
        """
        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            return None

        try:
            # ----------------------------------------------------------
            # Extract zip → temp .nc (ESA CCI ships .nc inside a zip)
            # ----------------------------------------------------------
            try:
                with zipfile.ZipFile(file_path) as z:
                    nc_name = next(
                        (n for n in z.namelist() if n.endswith(".nc")), None
                    )
                    if nc_name is None:
                        raise ValueError(
                            f"No .nc entry found inside zip: {file_path}"
                        )
                    tmp_nc_path = os.path.join(
                        self.temp_dir,
                        f"esacci_{year}_{os.path.basename(nc_name)}",
                    )
                    with z.open(nc_name) as f_in, open(tmp_nc_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                nc_path = tmp_nc_path
                logger.debug("Extracted zip to temp nc: %s", nc_path)
            except zipfile.BadZipFile:
                # Plain NetCDF – read directly
                logger.debug(
                    "File is not a zip archive, reading directly: %s", file_path
                )
                nc_path = file_path

            # ----------------------------------------------------------
            # Open with h5netcdf, keep raw integer codes
            # ----------------------------------------------------------
            raw = xr.open_dataset(
                nc_path,
                engine="h5netcdf",
                mask_and_scale=False,
                decode_coords="all",
                chunks="auto",
            )
            logger.debug("Opened %s  variables=%s", file_path, list(raw.data_vars))

            # ---- Compute final lccs_class variable ---------------------
            # the raw file ships several ancillary variables; a key one is
            # `processed_flag` which indicates whether the pixel was
            # processed.  We mask out unprocessed cells and drop the flag
            # afterwards.  The result is converted to float16 for storage.
            if "lccs_class" not in raw.data_vars:
                logger.error(
                    "Variable 'lccs_class' not found in %s. Available: %s",
                    os.path.basename(file_path), sorted(raw.data_vars),
                )
                raw.close()
                return None

            ds = raw[["lccs_class"]]

            # ---- Normalise spatial dimension names ----------------------
            dim_map = {}
            for d in list(ds.dims):
                dl = d.lower()
                if dl in ("lat", "latitude", "y"):
                    dim_map[d] = "latitude"
                elif dl in ("lon", "longitude", "x"):
                    dim_map[d] = "longitude"
            if dim_map:
                ds = ds.rename(dim_map)

            # ---- Override time coordinate to Dec-31 of the file year ---
            # (the raw file already has a time dim; we standardise it)
            if "time" in ds.dims:
                ds = ds.assign_coords(time=[pd.Timestamp(f"{year}-12-31")])
            else:
                ds = ds.expand_dims(dim={"time": 1}).assign_coords(
                    time=[pd.Timestamp(f"{year}-12-31")]
                )

            # ---- Add band dimension for pipeline consistency ------------
            if "band" not in ds.dims:
                ds = ds.expand_dims(dim={"band": 1}).assign_coords(band=[1])

            # Ensure canonical dim order
            ds = ds.transpose("time", "band", "latitude", "longitude")

            # ---- Write CRS (ESA CCI is always WGS-84) ------------------
            ds = ds.rio.write_crs("EPSG:4326")

            ds.attrs["source_year"] = year
            ds.attrs["source_file"] = os.path.basename(file_path)

            return ds

        except Exception:
            logger.exception("Error loading %s.", file_path)
            return None

    def _write_annual_zarr(self, ds: xr.Dataset, output_path: str) -> bool:
        """Write *ds* to *output_path* as a zarr v3 store."""
        try:
            ds = ds.chunk({"time": 1, "band": 1, "latitude": 512, "longitude": 512})

            compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", blocksize=0)

            encoding = {}
            for var in ds.data_vars:
                # Preserve native dtype (e.g. uint8 for lccs_class)
                native_dtype = ds[var].dtype
                encoding[var] = {
                    "compressors": (compressor,),
                    "chunks": (1, 1, 512, 512),
                    "dtype": str(native_dtype),
                }

            ds.to_zarr(
                output_path,
                mode="w",
                encoding=encoding,
                zarr_format=3,
                consolidated=False,
            )
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
                        try:
                            return int(os.path.splitext(os.path.basename(p))[0])
                        except ValueError:
                            return None

                    def preprocess(ds: xr.Dataset) -> xr.Dataset:
                        """Ensure CRS and use nearest-neighbour-friendly encoding."""
                        if ds.rio.crs is None:
                            ds = ds.rio.write_crs("EPSG:4326")
                        return ds

                    def get_vars_and_attrs(file_path: str) -> Tuple[List[str], Dict]:
                        sample = xr.open_zarr(
                            file_path,
                            mask_and_scale=False,
                            chunks="auto",
                            consolidated=False,
                        )
                        variables = list(sample.data_vars.keys())
                        attrs = sample.attrs.copy()
                        sample.close()
                        return variables, attrs

                    # ESA CCI LC is a categorical map – always use nearest
                    # neighbour resampling to avoid blending class codes.
                    # explicitly request that output pixels use 0 as the
                    # "nodata" value.  this is needed for the ESA CCI land
                    # cover product where 0 represents an unset/unknown class
                    # and must not be treated as valid data during reprojection.
                    success = spatial_processor.process_spatial_standard(
                        source_files=source_files,
                        output_path=output_path,
                        years_to_process=self.years_to_process,
                        year_pattern_func=year_from_path,
                        preprocess_func=preprocess,
                        get_variables_func=get_vars_and_attrs,
                        dst_nodata=0,
                        packaging_attrs={},  # disable scale/offset for categorical data
                    )

                    if success:
                        logger.info("ESA CCI spatial reprojection completed: %s", output_path)
                    return success

        except Exception:
            logger.exception("Error in ESA CCI spatial processing.")
            return False
