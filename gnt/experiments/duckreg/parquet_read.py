import numpy as np
import pandas as pd
from duckreg.estimators import DuckRegression

m = DuckRegression(
    db_name="",
    table_name="read_parquet('/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup/assembled/modis_subset.parquet/**/*.parquet')",
    formula="modis_median ~ ntl_harm",
    cluster_col="",
    n_bootstraps=0,
    seed=42,
)
m.fit()
m.fit_vcov()
results = m.summary()
restab = pd.DataFrame(
    np.c_[results["point_estimate"], results["standard_error"]],
    columns=["point_estimate", "standard_error"],
)