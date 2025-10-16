# streamreg

**streamreg** is a Python package for scalable, streaming, and parallelized regression analysis on large datasets. It supports ordinary least squares (OLS), two-stage least squares (2SLS/IV), and robust standard errors (including cluster-robust and two-way clustering), with a unified API for both in-memory and partitioned (parquet) data.

## Features

- **Streaming/Online Estimation**: Handles datasets that do not fit in memory by processing data in chunks.
- **Parallel Processing**: Efficiently utilizes multiple CPU cores for partitioned datasets.
- **Flexible Data Input**: Supports pandas DataFrames, single parquet files, and partitioned parquet datasets.
- **R-style Formula Interface**: Specify models using familiar R-style formulas, including interactions, polynomials, and instrument variables.
- **Cluster-Robust Standard Errors**: One-way and two-way clustering supported.
- **Feature Engineering**: Built-in support for polynomials, interactions, and custom transformations.
- **Comprehensive Output**: Standard regression tables, F-statistics, confidence intervals, and diagnostics.

## Installation

This package is part of the `growth-and-temperature` project and is not published on PyPI. To use it, clone the repository and install dependencies:

```bash
git clone <repo-url>
cd growth-and-temperature
pip install -r requirements.txt
```

## Usage

### OLS Example

```python
from gnt.analysis.streamreg.api import OLS

# Fit OLS with formula and cluster-robust SEs
model = OLS("y ~ x1 + x2 + I(x1^2)")
model.fit("data.parquet", cluster=["country", "year"])
print(model.summary())
```

### 2SLS Example

```python
from gnt.analysis.streamreg.api import TwoSLS

# Fit 2SLS with formula (endogenous and instruments)
model = TwoSLS("y ~ x1 + x2 | z1 + z2", endogenous=["x1"])
model.fit("data/", cluster="country")
print(model.summary())
```

### Command-Line Entrypoint

You can run analyses from the command line using the provided entrypoint:

```bash
python run.py analysis --config path/to/config.yaml --analysis-type online_rls --specification my_spec
```

## API Overview

- `OLS(formula, ...)`: Ordinary least squares estimator.
- `TwoSLS(formula, endogenous, ...)`: Two-stage least squares estimator.
- `.fit(data, cluster=...)`: Fit model to data (DataFrame, parquet file, or partitioned parquet directory).
- `.summary()`: Get regression summary as a pandas DataFrame.
- `.results_`: Access full results object (coefficients, SEs, F-stat, diagnostics, etc.).

## Supported Formula Syntax

- `y ~ x1 + x2`: Basic regression.
- `y ~ x1 + x2 + I(x1^2)`: Quadratic term.
- `y ~ x1 * x2`: Main effects and interaction.
- `y ~ x1 + x2 | z1 + z2`: 2SLS/IV regression (instruments after `|`).

## Output

- Regression tables (with significance stars)
- F-statistics and p-values
- Cluster diagnostics
- JSON, CSV, LaTeX, and Markdown outputs

## Development

- Python 3.8+
- Relies on `numpy`, `pandas`, `pyarrow`, `tqdm`, `scipy`, and `dataclasses`
- Designed for extensibility and integration with large-scale data workflows

## License

See the main repository for license information.

---

*streamreg is developed as part of the growth-and-temperature project for scalable, reproducible, and robust regression analysis on large datasets.*