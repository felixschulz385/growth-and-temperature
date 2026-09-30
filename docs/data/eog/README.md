# eog — EOG VIIRS nighttime lights and gas flares

| | |
|---|---|
| Config keys | `eog_viirs`, `eog_flare`; `eog_dmsp` and `eog_dvnl` are commented out |
| Modules | `src/data/sources/eog/source.py` (`EogSource`, VIIRS/DMSP/DVNL), `src/data/sources/eog/flare.py` (`EogFlareSource`) |
| Steps | FETCH, PREPARE · no `REQUIRES` |
| In panel | yes, both |

## eog_viirs — VIIRS annual composites (VNL v2.1)

**What it is.** The Earth Observation Group's annual VIIRS-DNB composites, 2012–2021
(`VIIRS_YEAR_RANGE`, hardcoded), 15 arc-second native resolution. 2012 is a partial year
(from April).

**Raw data (FETCH).** An authenticated Selenium session. Credentials come from
`orchestration/secrets/eog.credentials.json` (git-ignored) or `EOG_USERNAME`/`EOG_PASSWORD`. For
each year FETCH takes the one composite whose period ends in December, in three variants
(`average_masked`, `median_masked`, `cf_cvg`). Files land in `raw/eog/viirs/`.

**Prepared output.** `prepared/eog/viirs/crs/ease6933/eog_viirs_annual/ix=/iy=/part-<year>.parquet`

| Column | Meaning | PREPARE resampling | Panel aggregation |
|---|---|---|---|
| `viirs_annual_avg` | masked mean radiance (nW/cm²/sr), background, fire and aurora corrected | `sum` | `average` |
| `viirs_annual_median` | masked median radiance | `average` | `average` |
| `viirs_annual_cf_cvg` | number of cloud-free observations in the composite | `average` | `average` |

**Caveats.**

- Radiance is signed: background subtraction leaves small negative values over dark areas. Clip
  at 0 before taking logs.
- After `sum` resampling, flare and industrial cells can reach 10⁵–10⁶.
- `cf_cvg` is the treatment-side analogue of MODIS's valid counts. Low coverage makes the
  composite noisier, and coverage itself varies with cloud and haze.
- Only 10 years: a short panel, and the VIIRS-era first stage in `03_post_analysis.ipynb` (R5) was
  near zero.

## eog_flare — VIIRS Nightfire upstream gas-flare survey

**What it is.** EOG's annual inventory of upstream gas-flare locations: one spreadsheet per year
from 2017, plus one combined 2012–2016 file whose flares are copied onto each of those years.

**Raw data (FETCH).** Plain HTTPS download of the `.xlsx` files into `raw/eog/flare/`.

**Prepared output.** `prepared/eog/flare/crs/ease6933/eog_flare/ix=/iy=/part-<year>.parquet`,
rasterized directly (no resampling). The column is `flare_band` (uint8):

| Value | Meaning |
|---|---|
| 0 | no flare within 5 km |
| 1 | a flare within 5 km |
| 2 | a flare within 2 km |
| 3 | a flare point in this pixel |

Distances are geodesic. The panel aggregates with `max`.

**Caveats.** The survey covers upstream (oil and gas field) flares only. Refinery, LNG and other
industrial flares, often inside cities, are missing, so `flare_band == 0` does not guarantee a cell
is flare-free. There is no data before 2012.

## Disabled variants

`eog_dmsp` (DMSP-OLS 1992–2013) and `eog_dvnl` (DVNL 2013–2019) keep their code paths in
`source.py`. Re-enabling one is a config uncomment.
