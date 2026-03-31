import numpy as np
import pandas as pd
from duckreg import compressed_ols

m = compressed_ols(
    formula="modis_median ~ ntl_harm | pixel_id + year | 0 | country",
    data="/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/assembled/modis_subset.parquet",
    n_bootstraps=99,
    round_strata=5,
    seed=42,
    fe_method="mundlak",
    duckdb_kwargs={
        "temp_directory": "/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/assembled/scratch/duckdb_swap",
        "memory_limit": "48GB",
        "max_temp_directory_size": "1024GB",
        "enable_progress_bar": "true"
        },
)
results = m.summary()
restab = pd.DataFrame(
    np.c_[results["point_estimate"], results["standard_error"]],
    columns=["point_estimate", "standard_error"],
)

print(restab)