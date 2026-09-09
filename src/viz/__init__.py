"""Plotting engine for the GNT pipeline.

Two families:

* **Spatial** -- render EASE6933 ``cell_id`` PREPARE parquet grids
  (:func:`plot_grid`, :func:`load_matrix`).
* **Regression** -- DuckDB-computed descriptives on the assembled panel and
  diagnostics on a fitted model (:func:`plot_distribution`,
  :func:`plot_binscatter`, :func:`plot_corr_matrix`, :func:`plot_coefficients`,
  :func:`plot_residuals`), plus the stat builders and model adapters behind them.
"""

from src.viz.grid import MatrixResult, bbox_to_row_col, load_matrix
from src.viz.models import (
    FittedModel,
    as_fitted_model,
    from_duckreg,
    from_pyfixest,
    from_results_json,
)
from src.viz.plot import plot_grid
from src.viz.regplot import (
    plot_binscatter,
    plot_coefficients,
    plot_corr_matrix,
    plot_distribution,
    plot_residuals,
)
from src.viz.source import ResolvedSource, resolve_source
from src.viz.stats import (
    BinScatter,
    CorrMatrix,
    Histogram,
    binscatter,
    corr_matrix,
    histogram,
)

__all__ = [
    # spatial
    "MatrixResult",
    "bbox_to_row_col",
    "load_matrix",
    "plot_grid",
    # source
    "ResolvedSource",
    "resolve_source",
    # stat builders + result types
    "Histogram",
    "BinScatter",
    "CorrMatrix",
    "histogram",
    "binscatter",
    "corr_matrix",
    # model adapters
    "FittedModel",
    "as_fitted_model",
    "from_pyfixest",
    "from_results_json",
    "from_duckreg",
    # regression renderers
    "plot_distribution",
    "plot_binscatter",
    "plot_corr_matrix",
    "plot_coefficients",
    "plot_residuals",
]
