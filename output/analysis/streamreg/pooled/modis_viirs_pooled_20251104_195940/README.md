# Analysis Results: Pooled MODIS VIIRS analysis

**Generated:** 2025-11-04 19:59:40

## Overview

- **Model Type:** OLS
- **Observations:** 7,592,407,261
- **Features:** 2
- **R²:** 0.0005
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
median ~ viirs_annual
```

## Quick Results

### Top 5 Coefficients by Magnitude

| Variable | Coefficient | p-value |
|----------|-------------|----------|
| intercept | 284.2944*** | 0.0000 |
| viirs_annual | 0.0769*** | 0.0024 |

## Cluster Information

### dim1

- Clusters: 3,380
- Cluster size: 9 - 312,174,324 (mean: 2243038.8)

### dim2

- Clusters: 9
- Cluster size: 843,580,343 - 843,608,924 (mean: 843600806.8)

---
*Analysis ID: 20251104_195940*
