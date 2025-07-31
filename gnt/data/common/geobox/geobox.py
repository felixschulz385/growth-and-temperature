import os
import logging
import gzip
import shutil
import pickle
import pandas as pd
import rioxarray as rxr
from typing import Dict, List, Tuple

# Monkey patch for odc.geo.geobox bug fix
import odc.geo
from odc.geo.geobox import GeoBox, GeoboxTiles
from odc.geo import geom
from odc.geo.crs import CRS
from odc.geo.geom import Geometry
from affine import Affine

def _patched_geobox(crs, width=None, height=None, transform=None, resolution=None, anchor="top-left", **kw):
    """Patched version of odc.geo.geobox with unreleased bug fix."""
    if transform is not None and resolution is not None:
        raise ValueError("Cannot specify both transform and resolution")
    
    if transform is None and resolution is None:
        raise ValueError("Must specify either transform or resolution")
    
    if width is None or height is None:
        raise ValueError("Must specify both width and height")
    
    if transform is None:
        # Handle resolution parameter with bug fix
        if isinstance(resolution, (int, float)):
            resolution = (resolution, -resolution)
        elif len(resolution) == 1:
            resolution = (resolution[0], -resolution[0])
        elif len(resolution) == 2:
            res_x, res_y = resolution
            if res_y > 0:
                res_y = -res_y
            resolution = (res_x, res_y)
        
        # Handle anchor positioning
        if anchor == "top-left":
            transform = Affine.translation(0, 0) * Affine.scale(resolution[0], resolution[1])
        elif anchor == "center":
            transform = Affine.translation(-width * resolution[0] / 2, -height * resolution[1] / 2) * Affine.scale(resolution[0], resolution[1])
        else:
            raise ValueError(f"Unsupported anchor: {anchor}")
    
    return GeoBox(width=width, height=height, affine=transform, crs=crs)


def _patched_footprint(self, crs: CRS, buffer: float = 0, npoints: int = 100, wrapdateline: bool = False) -> Geometry:
    """Patched footprint method with antimeridian handling."""
    assert self.crs is not None
    ext = self.extent
    if buffer != 0:
        buffer = buffer * max(*self.resolution.xy)
        ext = ext.buffer(buffer)

    return ext.to_crs(
        crs,
        resolution=self._reproject_resolution(npoints),
        wrapdateline=wrapdateline,
    ).dropna()


def _patched_grid_intersect(self, src):
    """Patched grid_intersect method with antimeridian handling."""
    from odc.geo.geobox import GeoboxTiles
    
    # Check if we can use linear approach
    A = self._check_linear(src)
    if A is not None:
        return self._grid_intersect_linear(src, A)

    if src.base.crs == self.base.crs:
        src_footprint = src.base.extent
    else:
        # compute "robust" source footprint in CRS of self via epsg:4326
        try:
            src_footprint = (
                src.base.footprint(4326, 2) & self.base.footprint(4326, 2)
            ).to_crs(self.base.crs)
        except Exception:
            try:
                # Try using wrapdateline=True for more robust antimeridian handling
                src_fp_4326 = src.base.footprint(4326, 2, wrapdateline=True)
                self_fp_4326 = self.base.footprint(4326, 2, wrapdateline=True)

                # Compute intersection in 4326 with antimeridian-aware geometries
                intersection_4326 = src_fp_4326 & self_fp_4326
                src_footprint = intersection_4326.to_crs(self.base.crs)
            except Exception:
                # Final fallback: use source footprint directly without intersection
                try:
                    if self.base.crs is not None:
                        src_footprint = src.base.footprint(
                            self.base.crs, 2, wrapdateline=True
                        )
                    else:
                        src_footprint = src.base.extent
                except Exception:
                    # Last resort: use extent directly if available
                    if src.base.crs == self.base.crs:
                        src_footprint = src.base.extent
                    else:
                        # Use a very conservative approach: full world bounds
                        world_bounds = geom.box(-180, -90, 180, 90, "EPSG:4326")
                        if self.base.crs is not None:
                            src_footprint = world_bounds.to_crs(self.base.crs)
                        else:
                            src_footprint = world_bounds

    xy_chunks_with_data = list(self.tiles(src_footprint))
    deps: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

    for idx in xy_chunks_with_data:
        geobox = self[idx]
        deps[idx] = list(src.tiles(geobox.extent))

    return deps


# Apply monkey patches
odc.geo.geobox = _patched_geobox
GeoBox.footprint = _patched_footprint
GeoboxTiles.grid_intersect = _patched_grid_intersect

logger = logging.getLogger(__name__)

def get_or_create_geobox(hpc_root: str, output_dir: str = None, force_regenerate: bool = False):
    """
    Extract the geobox from a successful EOG VIIRS download and save it to a file.
    If the file is gzipped, unpack it first.
    
    Args:
        hpc_root: HPC root directory path
        output_dir: Directory to save the geobox pickle (optional)
        force_regenerate: If True, regenerate the geobox even if it exists
        
    Returns:
        geobox: The extracted geobox object.
        
    Raises:
        RuntimeError: If no successful VIIRS download is found.
    """
    # Default output directory
    if output_dir is None:
        output_dir = os.path.join(hpc_root, "misc", "processed", "stage_0", "misc")
    
    os.makedirs(output_dir, exist_ok=True)
    geobox_local = os.path.join(output_dir, "viirs_geobox.pkl")

    # If geobox exists and no forced regeneration, load it and apply the patch
    if os.path.exists(geobox_local) and not force_regenerate:
        logger.info(f"Loading geobox from {geobox_local} and applying monkeypatch")
        with open(geobox_local, 'rb') as f:
            old_geobox = pickle.load(f)
        
        # Regenerate using the patched constructor
        geobox = GeoBox(old_geobox.shape, old_geobox.affine, old_geobox.crs)
        return geobox

    # Generate new geobox from VIIRS data
    logger.info(f"Creating new geobox from VIIRS data")
    geobox = _create_geobox_from_viirs(hpc_root)
    
    # Save the geobox
    with open(geobox_local, 'wb') as f:
        pickle.dump(geobox, f)
    logger.info(f"Saved geobox to {geobox_local}")

    return geobox


def _create_geobox_from_viirs(hpc_root: str):
    """Create a geobox from VIIRS data."""
    parquet_path = os.path.join(hpc_root, "hpc_data_index/parquet_eog_viirs.parquet")
    if not os.path.exists(parquet_path):
        raise RuntimeError(f"Parquet index not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    # Try to find a successful download
    if 'status_category' in df.columns:
        ok = df[df['status_category'] == 'completed']
    elif 'download_status' in df.columns:
        ok = df[df['download_status'] == 'completed']
    else:
        raise RuntimeError("No status column found in parquet index")

    if ok.empty:
        raise RuntimeError("No successful VIIRS download found in index")

    # Use the first successful file
    viirs_local = os.path.join(hpc_root, "eog/viirs/raw", ok.iloc[0]['relative_path'])
    if not os.path.exists(viirs_local):
        raise RuntimeError(f"VIIRS file does not exist: {viirs_local}")

    # If the file is gzipped, unpack it to a temp location
    if viirs_local.endswith(".gz"):
        unpacked_path = viirs_local[:-3]
        if not os.path.exists(unpacked_path):
            logger.info(f"Unpacking gzipped VIIRS file: {viirs_local} -> {unpacked_path}")
            with gzip.open(viirs_local, 'rb') as f_in, open(unpacked_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        viirs_to_open = unpacked_path
    else:
        viirs_to_open = viirs_local

    # Open the file and extract the geobox
    viirs_data = rxr.open_rasterio(viirs_to_open, chunks="auto")
    geobox = viirs_data.odc.geobox
    
    return geobox
    viirs_data = rxr.open_rasterio(viirs_to_open, chunks="auto")
    geobox = viirs_data.odc.geobox
    
    return geobox
