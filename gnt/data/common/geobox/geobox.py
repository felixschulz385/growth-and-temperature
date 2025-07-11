import os
import logging
import gzip
import shutil
import pickle
import pandas as pd
import rioxarray as rxr

logger = logging.getLogger(__name__)

def get_or_create_geobox(hpc_root: str, output_dir: str = None):
    """
    Extract the geobox from a successful EOG VIIRS download and save it to a file.
    If the file is gzipped, unpack it first.
    
    Args:
        hpc_root: HPC root directory path
        output_dir: Directory to save the geobox pickle (optional)
        
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

    # If geobox already exists, load and return it
    if os.path.exists(geobox_local):
        logger.info(f"Geobox pickle already exists, loading from {geobox_local}")
        with open(geobox_local, 'rb') as f:
            geobox = pickle.load(f)
        return geobox

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

    with open(geobox_local, 'wb') as f:
        pickle.dump(geobox, f)
    logger.info(f"Saved geobox to {geobox_local}")

    return geobox
