# Analysis Results: Pooled MODIS NTLHARM analysis

**Generated:** 2025-11-04 21:27:05

## Overview

- **Model Type:** OLS
- **Observations:** 17,716,471,755
- **Features:** 2
- **R²:** 0.0036
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
median ~ ntl_harm
```

## Quick Results

### Top 5 Coefficients by Magnitude

| Variable | Coefficient | p-value |
|----------|-------------|----------|
| intercept | 284.0101*** | 0.0000 |
| ntl_harm | 0.1238*** | 0.0073 |

## Cluster Information

### dim1

- Clusters: 3,381
- Cluster size: 21 - 728,474,925 (mean: 5232466.7)

### dim2

- Clusters: 21
- Cluster size: 843,595,164 - 843,868,314 (mean: 843641512.1)

---
*Analysis ID: 20251104_212705*
