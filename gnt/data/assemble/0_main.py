# module load Java/21.0.2

import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

import pyspark.pandas as ps
from pyspark.sql import SparkSession
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Get Spark configuration from environment variables (set by SLURM script)
spark_cpus = os.environ.get("SPARK_CPUS", "32")
spark_memory_gb = os.environ.get("SPARK_MEMORY_GB", "8")
driver_memory = f"{int(spark_memory_gb) // 2}g"
executor_memory = f"{int(spark_memory_gb) // 2}g"

# Start Spark session with custom configuration for memory and parquet handling
spark = SparkSession.builder \
    .appName("MergeAnalysis") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED") \
    .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS") \
    .config("spark.driver.memory", driver_memory) \
    .config("spark.executor.memory", executor_memory) \
    .config("spark.driver.cores", spark_cpus) \
    .config("spark.executor.cores", spark_cpus) \
    .config("spark.sql.ansi.enabled", "false") \
    .getOrCreate()

def merge():
    logger.info("Loading MODIS LST data")
    modis = ps.read_parquet(
        "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/glass/LST/MODIS/Daily/1KM/processed/stage_3/modis_tabular.parquet",
        index_col=["pixel_id", "ix", "iy", "year"],
        columns=["pixel_id", "ix", "iy", "year", "median"]
    )
    
    logger.info("Loading VIIRS Nighttime Lights data")
    viirs = ps.read_parquet(
        "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/eog/viirs/processed/stage_3/viirs_tabular.parquet",
        index_col=["pixel_id", "ix", "iy", "year"]
    )
    
    logger.info("Loading DVNL Nighttime Lights data")
    dvnl = ps.read_parquet(
        "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/eog/dvnl/processed/stage_3/viirs_dvnl_tabular.parquet",
        index_col=["pixel_id", "ix", "iy", "year"]
    )
    
    logger.info("Loading DMSP Nighttime Lights data")
    dmsp = ps.read_parquet(
        "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/eog/dmsp/processed/stage_3/dmsp_tabular.parquet",
        index_col=["pixel_id", "ix", "iy", "year"]
    )
    
    logger.info("Loading cluster variables")
    cluster_vars = ps.read_parquet(
        "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/misc/processed/stage_3/gadm_countries_grid_tabular.parquet",
        index_col=["pixel_id", "ix", "iy"]
    )
        
    logger.info("Loading land mask")
    land_mask = ps.read_parquet(
        "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/misc/processed/stage_3/osm_land_mask_tabular.parquet",
        index_col=["pixel_id", "ix", "iy"]
    )
    
    logger.info("Merging MODIS and DVNL datasets")
    merged = ps.merge(modis, dvnl, left_index=True, right_index=True, how="outer")
    
    logger.info("Merging with DMSP data")
    merged = ps.merge(merged, dmsp, left_index=True, right_index=True, how="outer")
    
    logger.info("Merging with VIIRS data")
    merged = ps.merge(merged, viirs, left_index=True, right_index=True, how="outer"dmsp)

    logger.info("Merging with cluster variables")
    merged = ps.merge(merged, cluster_vars, on=["pixel_id", "ix", "iy"])
    
    logger.info("Merging with land mask")
    merged = ps.merge(merged, land_mask, on=["pixel_id", "ix", "iy"])

    logger.info("Filtering to retain only land pixels")
    merged = merged[merged["land_mask"]]
    
    logger.info("Dropping 'land_mask' column")
    merged = merged.drop(columns=["land_mask"])
    
    logger.info("Demeaning temperature and light data annual and pixel-wise")
    try:
        for col in ["median", "dmsp", "viirs_dvnl", "viirs_annual"]:
            # Demean by year
            merged[f"{col}_demeaned"] = merged.groupby(["year"])[col].transform(lambda x: x - x.mean())
            # Demean by pixel after year demeaning
            merged[f"{col}_demeaned"] = merged.groupby(["pixel_id", "ix", "iy"])[f"{col}_demeaned"].transform(lambda x: x - x.mean())
    except Exception as e:
        logger.error(f"Error during demeaning: {e}")
    
    logger.info("Writing merged dataframe to parquet")
    merged.to_parquet(
        "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data/0_main.parquet",
        partition_cols=["ix", "iy"],
        compression="zstd"
    )
    logger.info("Merge and export process completed")

# Run the merge and export process
merge()
