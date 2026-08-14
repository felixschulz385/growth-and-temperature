import os
import logging
import tempfile
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from zarr.codecs import BloscCodec
from odc.geo import GeoboxTiles
from odc.geo.xr import xr_reproject
from src.data.common.geobox.geobox import get_or_create_geobox

logger = logging.getLogger(__name__)

# Names rioxarray's CRS grid-mapping coordinate can leak into `data_vars`
# under when a sample zarr is opened without `decode_coords="all"`/`="all"`
# (every `get_variables_func` callback in the sources using this module
# does exactly that). If left in the `variables` list passed to
# `create_empty_target_zarr`, it gets treated as a real (time, band, y, x)
# data variable and zarr-encoded with the wrong chunked/packed encoding --
# which then silently corrupts the real CRS coordinate `.rio.write_crs()`
# tries to write under the same name right afterward, into a tiny
# metadata-less scalar. Filtered out wherever a variable list is assembled.
_NON_DATA_VAR_NAMES = {"spatial_ref", "crs", "grid_mapping"}


def write_crs_and_grid_mapping_encoding(ds: xr.Dataset, geobox, base_encoding: Dict[str, Dict[str, Any]]):
    """Write *geobox*'s CRS onto *ds* and return `(ds, encoding)`, where every
    entry in *base_encoding* has `"grid_mapping": "spatial_ref"` merged in.

    `.rio.write_crs()` records the CRS link as each *data variable's own*
    `encoding["grid_mapping"] = "spatial_ref"` -- not as an attr. A caller-
    supplied zarr `encoding=` dict passed to `to_zarr()` becomes each
    variable's *entire* encoding (not merged with what `write_crs()` just
    set), so an encoding dict built without this key silently drops the CRS
    link: the written store has a perfectly valid CRS on `"spatial_ref"` but
    no data variable points to it, so `.rio.crs` (and any grid_mapping-based
    CRS reader) returns `None` on every subsequent read.

    `create_empty_target_zarr` above applies this automatically for sources
    that go through `SpatialProcessor`; gadm/snl_mining/glass/berman_mining/
    ecoregions hand-roll their own zarr write path outside it (different
    variable shapes -- no `time`/`band` dims, multiple dtypes, etc. -- than
    that method assumes) and each independently needed this exact fix
    hand-copied in (found via `src.data.sources.verify` catching real "no
    CRS found" failures on HPC-produced GRID outputs). Centralized here so
    the next hand-rolled GRID zarr writer gets it by construction.
    """
    ds = ds.rio.write_crs(geobox.crs)
    ds.attrs["crs"] = str(geobox.crs)
    encoding = {var: {**enc, "grid_mapping": "spatial_ref"} for var, enc in base_encoding.items()}
    return ds, encoding


def reproject_for_tile_overlap(gdf, target_crs):
    """Reproject *gdf* to *target_crs* before testing per-tile overlap via a
    plain shapely `.intersects()`/`gdf.sindex.query()` against tile bounds
    built in that CRS.

    Comparing un-reprojected geometries (e.g. GADM/RESOLVE's native WGS84
    lon/lat degrees) against tile bounds built in a projected-meters CRS
    (e.g. EASE6933) is numerically incompatible (~1e7-magnitude meters vs
    +/-180/+/-90 degrees) and silently finds ~no overlap for ~every tile --
    no exception, just ~100%-null output (confirmed via
    `src.data.sources.verify` catching real ~100%-null GADM GRID output
    despite valid input geometries and a clean, no-exception run). Call this
    once, up front, on the whole layer -- not per tile.
    """
    return gdf.to_crs(target_crs)


class SpatialProcessor:
    """
    Common spatial processing utilities for reprojecting data to unified grids.
    
    This class provides shared functionality for spatial stage processing while
    allowing source-specific customization through callback functions.
    """
    
    def __init__(self, hpc_root: str, temp_dir: str = None, dask_client=None, default_nodata: Optional[float] = None,
                 target_geobox=None):
        """
        Initialize spatial processor.

        Args:
            hpc_root: HPC root directory for geobox
            temp_dir: Temporary directory for processing
            dask_client: Optional Dask client context manager
            default_nodata: if provided, this value will be used as the
                ``dst_nodata`` argument for every reprojection call made by
                this processor unless overridden at method call time.
            target_geobox: Optional pre-built GeoBox to reproject onto,
                overriding the default legacy EPSG:4326 grid
                (`get_or_create_geobox`). Additive per docs/design/05-
                migration.md §1 -- pass the canonical EPSG:6933 GeoBox
                (`src.data.common.geobox.get_or_create_canonical_geobox`) for
                sources landing on the new backbone grid; existing sources
                are unaffected since this defaults to ``None``.
        """
        self.hpc_root = hpc_root
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="spatial_processor_")
        self.dask_client = dask_client
        # store a default nodata for downstream operations
        self.default_nodata = default_nodata
        self._target_geobox_override = target_geobox
        # suppress verbose rasterio transformer warnings that crop up during
        # reprojection (e.g. CPLE_NotSupported XSCALE).  we only care about
        # errors.
        logging.getLogger("rasterio._env").setLevel(logging.ERROR)

    def get_target_geobox(self):
        """Get or create the target geobox for reprojection."""
        if self._target_geobox_override is not None:
            logger.info(f"Using overridden target geobox for reprojection: {self._target_geobox_override.shape}")
            return self._target_geobox_override
        try:
            target_geobox = get_or_create_geobox(self.hpc_root)
            logger.info(f"Using target geobox for reprojection: {target_geobox.shape}")
            return target_geobox
        except Exception as e:
            logger.error(f"Failed to get target geobox: {e}")
            raise
    
    def create_empty_target_zarr(
        self,
        output_path: str,
        target_geobox,
        years: List[int],
        variables: List[str],
        sample_attrs: Dict[str, Any] = None,
        variable_attrs_func: Callable[[str, Dict], Dict] = None,
        dst_nodata: Optional[float] = None,
        packaging_attrs: Optional[Dict[str, Any]] = None,
        dtype: str = "uint16",
        chunk_size: Tuple[int, int] = (512, 512),
    ) -> bool:
        """
        Create empty zarr file with target dimensions and metadata.

        Args:
            output_path: Path for output zarr
            target_geobox: Target geobox for spatial dimensions
            years: List of years for time dimension
            variables: List of variable names
            sample_attrs: Global attributes for the dataset
            variable_attrs_func: Function to get variable-specific attributes
            dst_nodata: Optional nodata value to use for the output arrays. If
                provided the default `_FillValue` attribute will be set to this
                value instead of the hard‑coded 65535.
            dtype: Storage dtype for the output arrays. Defaults to
                ``"uint16"`` (unchanged prior behaviour, matching the
                existing packed-integer sources). Pass e.g. ``"float32"``
                for sources writing already-physical (unpacked) values --
                forcing those through a uint16 store would silently
                truncate precision on the later region write.
            chunk_size: ``(y, x)`` zarr chunk shape. Defaults to ``(512, 512)``
                (unchanged prior behaviour). A caller writing per-output-tile
                regions (`process_tile_region`) must pass the same tile size
                used to build that tile grid (`src.data.common.tiling`) here
                -- a region write's slice must land on a chunk boundary, or
                zarr silently rewrites (and can corrupt, under concurrent
                writers) the neighboring tile's already-written chunk.

        Returns:
            bool: Success status
        """
        try:
            logger.info("Creating empty target zarr file")

            # Defensive filter, not just belt-and-suspenders: a caller's
            # variable list can carry a leaked CRS grid-mapping name (see
            # module-level `_NON_DATA_VAR_NAMES` comment) regardless of how
            # it built that list, and treating it as a real data variable
            # here would corrupt the real `spatial_ref` write below.
            variables = [v for v in variables if v not in _NON_DATA_VAR_NAMES]

            # Create time coordinates
            time_coords = pd.to_datetime([f"{year}-12-31" for year in sorted(years)])
            
            # Create empty dataset with target geobox dimensions. Dimension
            # names follow the geobox's own CRS-dependent axis names --
            # ('latitude', 'longitude') for a geographic CRS like the legacy
            # EPSG:4326 grid, ('y', 'x') for a projected CRS like the
            # canonical EPSG:6933 grid -- rather than hardcoding the
            # geographic names, which would silently mislabel a projected
            # grid's axes (docs/design/01-grid.md).
            ny, nx = target_geobox.shape
            dim_y, dim_x = target_geobox.dimensions
            y_coords = target_geobox.coords[dim_y].values.round(5)
            x_coords = target_geobox.coords[dim_x].values.round(5)
            np_dtype = np.dtype(dtype)

            # Create data variables with fill values and band dimension
            data_vars = {}

            # use provided nodata as fill value if given
            default_fill = dst_nodata if dst_nodata is not None else (65535 if np_dtype.kind in "ui" else np.nan)
            default_attrs = {"_FillValue": default_fill}
            if dst_nodata is not None:
                # also expose the attribute under common name
                default_attrs["nodata"] = dst_nodata
            # packaging attributes control data scaling/offset in zarr encoding.
            # zero-length dict disables packaging, leaving only fill values.
            if packaging_attrs is None:
                packaging_attrs = {
                    "scale_factor": 0.01,
                    "add_offset": 0.0,
                }
            
            for var in variables:
                # Get variable-specific attributes
                if variable_attrs_func:
                    var_attrs = variable_attrs_func(var, default_attrs.copy())
                else:
                    var_attrs = default_attrs.copy()
                    var_attrs.update(packaging_attrs)
                    
                data_vars[var] = xr.DataArray(
                    da.zeros((len(time_coords), 1, ny, nx), dtype=np_dtype, chunks=(1, 1, chunk_size[0], chunk_size[1])),
                    dims=['time', 'band', dim_y, dim_x],
                    coords={
                        'time': time_coords,
                        'band': [1],
                        dim_y: y_coords,
                        dim_x: x_coords
                    },
                    attrs=var_attrs
                )
            
            # Create empty dataset and copy global attributes.
            empty_ds = xr.Dataset(data_vars, attrs=sample_attrs or {})

            # Set CRS. Also stash a plain string CRS attr as a redundant
            # fallback (matching gadm's/osm's own defensive pattern) --
            # must come AFTER write_crs(), which strips any pre-existing
            # "crs" attr key itself. See the `grid_mapping` comment below
            # for why relying on write_crs() alone isn't safe here.
            empty_ds = empty_ds.rio.write_crs(target_geobox.crs)
            empty_ds.attrs["crs"] = str(target_geobox.crs)

            # Set up compression for Zarr output
            compressor = BloscCodec(cname="zstd", clevel=3, shuffle='bitshuffle', blocksize=0)
            encoding = {
                var: {
                    "chunks": (1, 1, chunk_size[0], chunk_size[1]),
                    "compressors": (compressor,),
                    "dtype": dtype,
                    # `.rio.write_crs()` above records the CRS link as each
                    # variable's own `encoding["grid_mapping"] = "spatial_ref"`
                    # -- NOT as an attr. Since this dict becomes each
                    # variable's *entire* zarr encoding (not merged with
                    # what write_crs() just set), leaving this out silently
                    # drops the link: the written store has a perfectly
                    # valid CRS on "spatial_ref" but no data variable points
                    # to it, so `.rio.crs` (and any grid_mapping-based CRS
                    # reader) returns None on every subsequent read. Found
                    # via src.data.sources.verify catching real "no CRS
                    # found" failures on HPC-produced acag/esacci/ntl_harm/
                    # eog GRID outputs.
                    "grid_mapping": "spatial_ref",
                }
                for var in variables
            }
            
            # Write empty zarr structure
            logger.info(f"Writing empty zarr structure to: {output_path}")
            empty_ds.to_zarr(
                output_path, 
                mode="w",
                encoding=encoding,
                compute=False,
                zarr_format = 3,
                consolidated = False
            )
            
            logger.info("Empty target zarr created successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Error creating empty target zarr: {e}")
            return False
    
    def setup_dask_config(self):
        """Set up Dask configuration for large array operations."""
        import dask
        return dask.config.set({
            'array.slicing.split_large_chunks': True,
            'array.chunk-size': '512MB',
            'optimization.fuse.active': False,
            'distributed.comm.compression': 'lz4',
        })
    
    def group_files_by_year(self, source_files: List[str], year_pattern_func: Callable[[str], Optional[int]]) -> Dict[int, List[str]]:
        """
        Group source files by year using a custom pattern function.
        
        Args:
            source_files: List of source file paths
            year_pattern_func: Function to extract year from file path
            
        Returns:
            Dict mapping year to list of file paths
        """
        files_by_year = {}
        
        for file_path in source_files:
            year = year_pattern_func(file_path)
            if year is not None:
                if year not in files_by_year:
                    files_by_year[year] = []
                files_by_year[year].append(file_path)
        
        return files_by_year
    
    def write_year_to_zarr(
        self,
        year_ds: xr.Dataset,
        output_path: str,
        year: int,
        target_geobox,
        preprocess_func: Callable[[xr.Dataset], xr.Dataset] = None,
        dst_nodata: Optional[float] = None,
        resampling: str = "nearest",
    ) -> bool:
        """
        Write a year's worth of data to zarr with reprojection.

        Args:
            year_ds: Source dataset for the year
            output_path: Output zarr path
            year: Year being processed
            target_geobox: Target geobox for reprojection
            preprocess_func: Optional preprocessing function
            dst_nodata: Optional destination nodata value that will be passed
                to :func:`odc.geo.xr.xr_reproject` as ``dst_nodata``. When
                ``None`` the reprojection library default ("auto") is used.
            resampling: Resampling method passed to
                :func:`odc.geo.xr.xr_reproject`. Defaults to ``"nearest"``,
                unchanged from this function's prior hardcoded behaviour.
                Radiance-like (flux) variables should pass ``"sum"``
                (area-weighted, flux-conserving) instead -- see
                docs/design/04-ingest.md §1. This must stay a per-call
                override, never a new hardcoded default here, since
                categorical variables (e.g. ESACCI land cover) require
                ``"nearest"``/``"mode"`` and would silently break under a
                different shared default.

        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Processing year {year} for spatial reprojection (resampling={resampling})")

            # Apply preprocessing if provided
            if preprocess_func:
                year_ds = preprocess_func(year_ds)

            # Reproject to target geobox; propagate custom nodata if given
            reproj_kwargs = {"resampling": resampling}
            if dst_nodata is not None:
                reproj_kwargs["dst_nodata"] = dst_nodata
            reprojected_ds = xr_reproject(year_ds, target_geobox, **reproj_kwargs)
            
            # Clean up dataset
            reprojected_ds = reprojected_ds.drop_vars(['spatial_ref'], errors='ignore').drop_attrs()

            # Transform coordinates -- dimension names follow the target
            # geobox's own axis names (see create_empty_target_zarr).
            dim_y, dim_x = target_geobox.dimensions
            reprojected_ds.coords[dim_x] = reprojected_ds.coords[dim_x].round(5)
            reprojected_ds.coords[dim_y] = reprojected_ds.coords[dim_y].round(5)

            # Rechunk for zarr writing
            reprojected_ds = reprojected_ds.chunk({'time': 1, 'band': 1, dim_y: 512, dim_x: 512})
            
            # Write to zarr
            reprojected_ds.to_zarr(
                output_path,
                region='auto',
                align_chunks=True,
                zarr_format=3,
                consolidated=False
            )
            
            logger.info(f"Successfully wrote year {year} to zarr")
            return True
            
        except Exception as e:
            logger.exception(f"Error writing year {year} to zarr: {e}")
            return False
    
    def process_tile_region(
        self,
        source_ds: xr.Dataset,
        output_path: str,
        tile,
        target_dims: Tuple[str, str],
        preprocess_func: Callable[[xr.Dataset], xr.Dataset] = None,
        dst_nodata: Optional[float] = None,
        resampling: str = "nearest",
    ) -> bool:
        """
        Reproject `source_ds` onto one output tile's own geobox
        (`tile.geobox`, a crop of the shared target geobox -- see
        `src.data.common.tiling`) and region-write it into the pre-created,
        chunk-aligned empty output zarr at `output_path`.

        Unlike `write_year_to_zarr` (which reprojects onto the *whole*
        target geobox and writes a full-extent region per year), this writes
        only the (row, col) slice `tile.y_slice`/`tile.x_slice` covers --
        the empty zarr's chunk boundaries must equal the tile grid's own
        boundaries (`create_empty_target_zarr(..., chunk_size=tile_size)`)
        or this region write silently touches a neighboring tile's chunk.

        `source_ds` should already cover `tile.geobox`'s extent plus
        whatever halo the caller's raw-getter reads for edge-effect-free
        resampling (halo handling is owned by each source's raw-getter, not
        this method) -- reprojecting a too-small `source_ds` just leaves
        nodata at the tile's edges, it does not raise.

        `region=` mixes an explicit slice per spatial dim with `"auto"` for
        every other dim (typically `time`/`band`) -- `"auto"` resolves by
        matching `source_ds`'s own coordinate labels (e.g. one year's
        timestamp) against the on-disk zarr's coords, the same mechanism
        `write_year_to_zarr`'s `region='auto'` already relies on for its
        non-spatial dims.
        """
        try:
            if preprocess_func:
                source_ds = preprocess_func(source_ds)

            reproj_kwargs = {"resampling": resampling}
            effective_nodata = dst_nodata if dst_nodata is not None else self.default_nodata
            if effective_nodata is not None:
                reproj_kwargs["dst_nodata"] = effective_nodata
            reprojected_ds = xr_reproject(source_ds, tile.geobox, **reproj_kwargs)

            reprojected_ds = reprojected_ds.drop_vars(['spatial_ref'], errors='ignore').drop_attrs()

            dim_y, dim_x = target_dims
            reprojected_ds.coords[dim_x] = reprojected_ds.coords[dim_x].round(5)
            reprojected_ds.coords[dim_y] = reprojected_ds.coords[dim_y].round(5)

            region: Dict[str, Any] = {dim_y: tile.y_slice, dim_x: tile.x_slice}
            for dim in reprojected_ds.dims:
                if dim not in region:
                    region[dim] = "auto"

            reprojected_ds.to_zarr(
                output_path,
                region=region,
                zarr_format=3,
                consolidated=False,
            )

            logger.info("Successfully wrote tile %s to zarr", getattr(tile, "id", f"({tile.row},{tile.col})"))
            return True

        except Exception as e:
            logger.exception(f"Error writing tile {getattr(tile, 'id', '?')} to zarr: {e}")
            return False

    def process_spatial_standard(
        self,
        source_files: List[str],
        output_path: str,
        years_to_process: List[int],
        year_pattern_func: Callable[[str], Optional[int]],
        preprocess_func: Callable[[xr.Dataset], xr.Dataset] = None,
        get_variables_func: Callable[[str], Tuple[List[str], Dict]] = None,
        dst_nodata: Optional[float] = None,
        packaging_attrs: Optional[Dict[str, Any]] = None,
        resampling: str = "nearest",
    ) -> bool:
        """
        Standard spatial processing workflow for simple cases.

        This handles the common case where each year has one file and minimal
        aggregation is needed.

        Args:
            source_files: List of source zarr files
            output_path: Output zarr path
            years_to_process: List of years to include
            year_pattern_func: Function to extract year from file path
            preprocess_func: Optional preprocessing function for each dataset
            get_variables_func: Function to get variables and attrs from sample file
            resampling: Resampling method for every year's reprojection
                (see `write_year_to_zarr`). Defaults to ``"nearest"``, this
                function's prior fixed behaviour; pass ``"sum"`` for
                radiance-like variables (docs/design/04-ingest.md §1).

        Returns:
            bool: Success status
        """
        try:
            # Get target geobox
            target_geobox = self.get_target_geobox()
            
            # Group files by year
            files_by_year = self.group_files_by_year(source_files, year_pattern_func)
            
            # Get variables and attributes from sample file
            if get_variables_func:
                variables, sample_attrs = get_variables_func(source_files[0])
            else:
                sample_ds = xr.open_zarr(source_files[0], mask_and_scale=False, chunks='auto')
                variables = list(sample_ds.data_vars.keys())
                sample_attrs = sample_ds.attrs.copy()
                sample_ds.close()
            
            # Create empty target zarr, carrying nodata setting if requested
            # If caller didn't supply a nodata value, fall back to the one
            # configured on the processor instance (if any).
            effective_nodata = dst_nodata if dst_nodata is not None else self.default_nodata

            if not self.create_empty_target_zarr(
                output_path,
                target_geobox,
                years_to_process,
                variables,
                sample_attrs,
                dst_nodata=effective_nodata,
                packaging_attrs=packaging_attrs,
            ):
                return False
            
            # Process each year
            for year in sorted(years_to_process):
                if year not in files_by_year:
                    logger.warning(f"No files found for year {year}")
                    continue
                
                year_files = files_by_year[year]
                if len(year_files) > 1:
                    logger.warning(f"Multiple files found for year {year}, using first: {year_files[0]}")
                
                # Open year dataset
                year_ds = xr.open_zarr(year_files[0], consolidated=False, decode_coords='all')
                
                # Write to output zarr
                effective_nodata = dst_nodata if dst_nodata is not None else self.default_nodata

                success = self.write_year_to_zarr(
                    year_ds,
                    output_path,
                    year,
                    target_geobox,
                    preprocess_func,
                    dst_nodata=effective_nodata,
                    resampling=resampling,
                )
                
                year_ds.close()
                
                if not success:
                    logger.error(f"Failed to process year {year}")
                    return False
            
            logger.info("Standard spatial processing completed successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Error in standard spatial processing: {e}")
            return False


def create_zarr_encoding(variables: List[str], chunks: Tuple[int, ...] = (1, 1, 512, 512)) -> Dict[str, Dict]:
    """Create standard zarr encoding configuration."""
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle='bitshuffle', blocksize=0)
    return {
        var: {
            "chunks": chunks, 
            "compressors": (compressor,),
            "dtype": "uint16"
        } 
        for var in variables
    }
