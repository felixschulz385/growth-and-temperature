# Pollution Data Options for a Follow-up to Idea B

**Status: research note, 2026-10-29.** The `no2` (§3.1) and `ghs_pop` (§3.4) modules were
fetched on the transfer node (`orchestration/scripts/pollution-fetch.sh`) and prepared on
2026-10-30. Step 1 of §4 (night lights) ran on 2026-10-30, using `ghs_pop`. Its results are in
[`pollution-results.md`](pollution-results.md). `no2` is prepared but unused. MERRA-2 (§3.2) is
deferred: it needs an Earthdata login, which the pipeline does not have. It collects the data that
could take idea B ([`pollution-institutions-prespec.md`](pollution-institutions-prespec.md))
past its limits. Any use needs its own pre-spec block, and each new dataset needs a source module
(CLAUDE.md § "Data pipeline"). Facts were checked on the providers' pages on 2026-10-29 unless
marked **unverified**.

## 1. What the follow-up would fix

The main result (prespec §5, notebook `07_pollution.ipynb`) has four limits. Each points to
different data.

| Limit | Why it matters | Data that addresses it |
|---|---|---|
| **Narrow overlap.** DOSE has no poor strong democracies and few rich weak ones (G1). | Institutions can be compared only at 4,000–12,000 USD. | A growth measure that covers every country: **night lights** (§2) |
| **The path is national.** With country × year FE, no supported slope is positive. | It could be real, or it could be because PM2.5 drifts across regional borders. | An outcome that stays near its sources: **NO₂** (§3.1) |
| **Dust.** ACAG V6GL02.04 includes mineral dust. | Desert regions add natural PM2.5 that has nothing to do with output. | Aerosol species: **MERRA-2** (§3.2), or an older dust-free ACAG release |
| **Area weighting.** The pipeline has no population grid. | Exposure is about people, and the population-weighted check (5b) was really China vs India. | A population grid: **GHS-POP** (§3.4) |

## 2. Night lights

The pipeline already has three night-light sources, all in the assembled 10 km panel:

- `ntl_harm` (harmonized DMSP–VIIRS, 1992–2022, in DMSP-like digital numbers);
- `eog_viirs` (VIIRS annual composites, 2012–2021);
- `eog_flare` (VIIRS gas flares).

### Lights as the pollution measure: no

Lights record lighting and electrification, not emissions. A pollution path built from lights
would mostly show how closely lights follow output, which is why the retired project used them
as its growth measure. Artificial light at night is a pollutant in its own right, with work on its
ecological and health effects, but as an outcome against output it would be circular.

### Lights as the growth measure: yes, the cheapest step

Replacing DOSE output per person with lights restores the poor democracies, which are mostly in
sub-Saharan Africa and South Asia. It works at the grid cell, ADM2 or ADM1, and covers all of
ACAG's years in the sample (1998–2022). It needs no new data. The caveats a pre-spec would have to
handle:

- **Haze dims lights.** Aerosols attenuate the light reaching the satellite, so more PM2.5 lowers
  measured radiance. This biases the PM2.5–lights slope downwards, most where pollution is worst.
  - DNB radiance correlates with surface PM2.5 down to r = −0.83 at a Beijing site (Wang et al.
    2016).
  - Radiance falls with aerosol optical depth in Beijing and three other Chinese cities (IEEE
    JSTARS 2020).
  - In nine Asian cities, aerosols weaken lights as a proxy for output, and the effect differs by
    region (Remote Sensing 2025).
  - Annual composites average cloud-free nights but keep persistent haze. Possible handling: the
    sign of the bias stated in advance, a check that removes AOD from lights, or the haze-free
    part of the year.
- **Dark poor areas.** Many poor rural cells read zero, which is where the poor democracies are.
  The pre-spec has to say how zeros enter (ln(1 + lights), a share of lit cells, ADM aggregation).
- **Saturation and splice.** DMSP saturates in city cores, and `ntl_harm` splices DMSP to
  VIIRS in 2013/14.
- **The lights–output elasticity varies** with income and urbanization. The fork's notebook
  `06d_dose_twfe.ipynb` has an information cell relating ln lights to ln GRP within the
  fixed-effects model.
- **Already seen.** The retired project put ACAG PM2.5 into the 10 km panel as a channel
  (`04_channels.ipynb`, reduced form of PM2.5 on mines). A pre-spec should list what has been
  looked at.

### Gas flares: a real pollution source detected at night

`eog_flare` (VIIRS Nightfire) is in the pipeline. Flares emit black carbon, CO₂ and methane, but
they exist only in oil and gas regions. This is a niche outcome, not a general one.

## 3. Other pollution and population data

### 3.1 NO₂: the most informative second outcome

NO₂ lasts hours, so it stays near traffic, power plants and industry. If NO₂ shows a
within-country path where PM2.5 does not, transport explains the national-only result.

**Anenberg et al. (2022), global surface NO₂, 1 km.**
- **Years:** 1990, 1995, 2000 and every year 2005–2020.
- **Units and format:** ppb, annual GeoTIFFs.
- **Files and access:** one file per year on figshare (article 12968114, version 4), 0.29–0.62 GB
  each, about 8 GB in all. NetCDF copies are on a GWU Box folder, and there was a NASA HAQAST
  mirror (its README now returns HTTP 410).
- **License:** CC BY 4.0.
- **Method:** a land-use regression surface for 2010–2012 (5,220 monitors in 58 countries),
  scaled to other years with satellite and reanalysis NO₂ columns, including an in-house OMI
  level 3 product. It is adjusted for satellite overpass time and cloud cover, and corrected for a
  high bias in rural areas.
- **What region FE leave:** the base pattern is fixed, so with region FE the variation over time
  comes from the satellite scaling, at the satellite's resolution (about 10–25 km). That is
  coarse within a cell but fine for ADM1 means.
- **Check before use:** the 2018 and 2019 files are about 50 % larger than the others. Find out
  whether the scaling source changed then (for example to TROPOMI), which would be a break
  in the series.
- **Pipeline route:** too large for the proxy, so FETCH runs on the transfer node.

**Larkin et al. (2023), daily global land-use regression, 50 m, 2005–2019.** Cross-validated
R² is 0.63 for annual means. The volume is far beyond what an annual regional panel needs, and
whether it is downloadable was not checked. Not recommended.

**OMI OMNO2d (V004, "Ver 5.0, Col 004"), tropospheric NO₂ column, 0.25°, daily, 2004-10 to
now.** About 12 MB per day (≈ 4.4 GB a year) from NASA GES DISC, with an Earthdata login. It is
the raw satellite signal behind products like Anenberg's. It measures a column, not the surface.
OMI's row anomaly removes parts of the swath from 2007 on (known from the instrument's
literature; not restated on this catalogue page). It is useful only as a robustness check for a
column measure.

**TROPOMI NO₂ (2018–).** Sharper (about 5 km) but too short for a 1998–2018 path. Not
recommended as the main measure.

### 3.2 Removing dust

**MERRA-2 M2TMNXAER (v5.12.4), monthly aerosol diagnostics, 0.5° × 0.625°, 1980 to now.**
- **Content:** surface mass concentration for each aerosol species. PM2.5 is built as dust +
  sea salt + black carbon + organic carbon × organic-matter factor + sulfate × 132.14/96.06
  (Buchard et al. 2017; the factors are to be confirmed against the GES DISC FAQ).
- **Use here:** the dust (and sea-salt) share of PM2.5 per region-year, as a control or to
  remove it from ACAG.
- **Access:** NASA GES DISC with an Earthdata login. OPeNDAP can cut the request to the five
  species variables. **Unverified:** the variable names (expected `DUSMASS25`, `SSSMASS25`,
  `BCSMASS`, `OCSMASS`, `SO4SMASS`).
- **Caveats:**
  - About 50 km, which is coarse for small regions.
  - A reanalysis, not observations.
  - GMAO staff note that no species is purely anthropogenic: biomass burning and volcanoes add
    to all of them.

**ACAG releases.**
- **V6.GL.02.04** (in use): global, 0.01°, 1998–2023. No dust-free or component variant.
- **V5.GL.05.02**: global, 0.01°, 1998–2023, with an uncertainty grid. Also no dust-free
  variant. A second total-PM2.5 measure for robustness.
- **North America (V5.NA.05.02):** has component PM2.5, but no use for a global panel.
- **Older dust-free series:** an older ACAG series (V4.GL.03 era) had dust and sea salt removed,
  1998–2015. It may be in the SatPM2.5 archive (**unverified**). Its method differs from V6, so it
  would be a separate robustness check, not a replacement.
- **Summary files:** ACAG's population-weighted summaries cover countries, plus regions only in
  Canada, China, India and the United States. They do not replace a population grid.

### 3.3 Sulfur dioxide and emission inventories

**NASA MEaSUREs SO₂ point-source catalogue (MSAQSO2L4 V2; Fioletov et al. 2023, ESSD).**
- **Coverage:** 759 continuously emitting sources, 2005–2021: 477 power plants, 74 smelters, 102
  oil and gas, 106 volcanoes. Annual emissions from OMI, OMPS and TROPOMI.
- **Uncertainty:** ±10–20 kt/yr + 5 % to 2017, and 4–5 kt/yr from 2018.
- **Use:** SO₂ is the classic Kuznets-curve pollutant (global emissions fell about 50 % over
  2005–2021). These are point sources, so they would be aggregated to regions as counts and
  emissions. Most regions have none.

**EDGAR v8.1 air pollutants:** SO₂, NOₓ, CO, NMVOC, NH₃, PM10, PM2.5, BC and OC.
- **Coverage:** 1970–2022, by sector and country, with 0.1° grid maps. Large-scale biomass
  burning is excluded.
- **Problem:** it spreads national emissions onto the grid with proxy maps (**unverified** from
  this check; per EDGAR's methods these include population, roads and point-source databases).
  Regional variation is then partly built from proxies that correlate with output, so it is
  usable at country level but circular at regional level.

### 3.4 Population grids

**GHS-POP R2023A (JRC).**
- **Grid and epochs:** 1975–2030 in five-year epochs. 100 m and 1 km in Mollweide (EPSG:54009),
  and 3″ and 30″ in WGS84.
- **Files:** one global ZIP per epoch at 1 km, about 300 MB, from the JRC FTP
  (`jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E<year>_GLOBE_R2023A_54009_1000/V1-0/`).
  Also on the AWS Open Data registry.
- **License:** CC BY 4.0.
- **Use:** population-weighted region means of PM2.5 or NO₂. Years between epochs are
  interpolated, and the 1 km Mollweide grid is reprojected onto the EASE grid. Epochs 1995–2020
  are six files (about 1.8 GB), through the transfer node.

**WorldPop.** R2025A (alpha) covers 2015–2030 at 1 km, constrained, under CC BY 4.0. It starts
too late for 1998. WorldPop's older global 2000–2020 series was not re-checked
(**unverified**).

**GPWv4 (NASA SEDAC).** ACAG uses it for its population-weighted summaries. An alternative to
GHS-POP, with five-year steps from 2000 and an Earthdata login.

### 3.5 Not recommended

| Data | Why not |
|---|---|
| OpenAQ ground monitors | Sparse outside rich countries and mostly after 2015. Validation only. |
| ODIAC CO₂ | Spread across the map using night lights: circular with lights as the growth measure |
| OCO-2 XCO₂ | Too coarse and noisy for regional annual panels |

## 4. Suggested order

1. **PM2.5 on night lights**, all countries, ADM1 (and the 10 km panel). Drafted as
   [`pollution-lights-prespec.md`](pollution-lights-prespec.md) (confirmed 2026-10-30). This needs no new data
   and is the only step that addresses the overlap problem. The pre-spec has to fix how haze and
   zero lights are handled before any PM2.5 regression.
2. **NO₂ (Anenberg) as a second outcome**, on both DOSE output and lights. New module `no2`:
   FETCH through the transfer node, PREPARE onto the 1 km grid. It tests the transport
   explanation.
3. **GHS-POP** for population-weighted means of both outcomes. New module `ghs_pop`.
4. **MERRA-2 dust share** as a control or robustness check. New module `merra2_aer`.

Steps 2–4 can be built while step 1's pre-spec is written. None of them loads an outcome.

## Sources

- ACAG surface PM2.5 datasets (versions, summary files, GPWv4, GADM borders):
  <https://sites.wustl.edu/acag/datasets/surface-pm2-5/>
- Older ACAG dust- and sea-salt-removed series, 1998–2015: <https://osf.io/a6gc4/wiki>
- Anenberg et al. NO₂, figshare record (years, files, license):
  <https://figshare.com/articles/dataset/Global_surface_NO2_concentrations_1990-2020/12968114/4>
  (API: <https://api.figshare.com/v2/articles/12968114>)
- Anenberg et al. (2022), *Lancet Planetary Health* 6(1): e49–e58:
  <https://haqast.org/publications/long-term-trends-in-urban-no2-concentrations-and-associated-paediatric-asthma-incidence-estimates-from-global-datasets/>
- Larkin et al. (2023), global land-use regression for NO₂: <https://doi.org/10.3389/fenvs.2023.1125979>
- OMI OMNO2d V004: <https://www.earthdata.nasa.gov/data/catalog/ges-disc-omno2d-004>
- MERRA-2 M2TMNXAER: <https://www.earthdata.nasa.gov/fr/data/catalog/ges-disc-m2tmnxaer-5.12.4?page=6%2C0>;
  PM2.5 formula: <https://gmao.gsfc.nasa.gov/pubs/docs/Keller1364.pdf>; species caveat:
  <https://forum.earthdata.nasa.gov/viewtopic.php?p=24240>
- SO₂ catalogue: Fioletov et al. (2023), ESSD 15: 75–93,
  <https://essd.copernicus.org/articles/15/75/2023/essd-15-75-2023.pdf>; README:
  <https://docserver.gesdisc.eosdis.nasa.gov/public/project/MEaSUREs/Krotkov/README.MSAQSO2L4.V2.pdf>
- EDGAR v8.1 air pollutants: <https://edgar.jrc.ec.europa.eu/dataset_ap81>
- GHS-POP R2023A, JRC FTP:
  <https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2015_GLOBE_R2023A_54009_1000/V1-0/>;
  AWS registry: <https://registry.opendata.aws/ghs-pop/>
- WorldPop R2025A: <https://hub.worldpop.org/geodata/summary?id=80031>
- Haze and night lights: Wang et al. (2016), <https://arroma.uiowa.edu/docs/publication/paper_pdf/2016/Final_AE_Pring.pdf>;
  IEEE JSTARS (2020), <https://doaj.org/article/76e4cbd3a45d410f94e36fcf4a6002ef>;
  Zhang et al. (2025), AMT 18: 1787, <https://amt.copernicus.org/preprints/amt-2024-181/amt-2024-181.pdf>;
  Remote Sensing 17: 3492 (2025), <https://mdpi-res.com/d_attachment/remotesensing/remotesensing-17-03492/article_deploy/remotesensing-17-03492.pdf>
