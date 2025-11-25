# Analysis Results: TWFE MODIS NTLHARM analysis for replication of Hodler & Raschky (2014)

**Generated:** 2025-11-04 13:34:53

## Overview

- **Model Type:** OLS
- **Observations:** 14,798,601,216
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
ntl_harm_twoway_demeaned ~ reg_fav_twoway_demeaned
```

## Quick Results

### Top 5 Coefficients by Magnitude

| Variable | Coefficient | p-value |
|----------|-------------|----------|
| intercept | -0.0000 | 1.0000 |
| reg_fav_twoway_demeaned | -0.0108 | 0.9087 |

## Cluster Information

### dim1

- Clusters: 2,419
- Cluster size: 21 - 730,665,117 (mean: 6117553.8)

### dim2

- Clusters: 21
- Cluster size: 704,695,296 - 704,695,296 (mean: 704695296.0)

---
*Analysis ID: 20251104_133453*
