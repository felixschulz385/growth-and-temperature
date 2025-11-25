# Analysis Results: TWFE MODIS VIIRS analysis for replication of Hodler & Raschky (2014)

**Generated:** 2025-11-04 14:16:32

## Overview

- **Model Type:** OLS
- **Observations:** 6,342,201,366
- **Features:** 2
- **R²:** 0.0000
- **Standard Errors:** two_way

## Files

| File | Description |
|------|-------------|
| `summary.txt` | Full formatted analysis report |
| `results.json` | Complete results in JSON format |
| `coefficients.csv` | Coefficient table (CSV) |
| `diagnostics.txt` | Detailed diagnostics and confidence intervals |
| `table.tex` | LaTeX-formatted results table |
| `config_snapshot.json` | Configuration used for this run |

## Specification

```
viirs_annual_twoway_demeaned ~ reg_fav_twoway_demeaned
```

## Quick Results

### Top 5 Coefficients by Magnitude

| Variable | Coefficient | p-value |
|----------|-------------|----------|
| reg_fav_twoway_demeaned | 0.0003 | 0.9319 |
| intercept | 0.0000 | 0.9997 |

## Cluster Information

### dim1

- Clusters: 2,419
- Cluster size: 9 - 313,142,193 (mean: 2621785.5)

### dim2

- Clusters: 9
- Cluster size: 704,668,051 - 704,695,295 (mean: 704689040.7)

---
*Analysis ID: 20251104_141632*
