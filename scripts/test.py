import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

import pyspark.pandas as ps
from pyspark.sql import SparkSession

# Start Spark
spark = SparkSession.builder \
    .appName("MergeAnalysis") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED") \
    .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.ansi.enabled", "false") \
    .getOrCreate()

def merge_analysis():
    land_mask = ps.read_parquet(
        "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/misc/processed/stage_3/osm_land_mask_tabular.parquet",
        index_col=None
    )

    # modis = ps.read_parquet("/scicore/.../modis_tabular.parquet", columns=["median", "pixel_id", "ix", "iy"])
    # dmsp  = ps.read_parquet("/scicore/.../dmsp_tabular.parquet")

    dvnl = ps.read_parquet(
        "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/eog/dvnl/processed/stage_3/viirs_dvnl_tabular.parquet",
        index_col=None
    )

    # merge on ["pixel_id", "ix", "iy"]
    dvnl = dvnl.merge(land_mask, on=["pixel_id", "ix", "iy"], how="inner")

    # filter by land_mask == True
    dvnl = dvnl[dvnl["land_mask"]]

    # print shape
    print((dvnl.shape[0], dvnl.shape[1]))

merge_analysis()
