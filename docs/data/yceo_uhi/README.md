# yceo_uhi — YCEO Surface Urban Heat Islands v4 (Chakraborty and Lee)

| | |
|---|---|
| Config key | `yceo_uhi` |
| Module | `src/data/sources/yceo_uhi.py` (`YceoUhiSource`) |
| Steps | FETCH only · no `REQUIRES` |
| In panel | no (benchmark of the heat-island successor's step 1 and ◇ G1, [`../../successors/heat-island-prespec.md`](../../successors/heat-island-prespec.md)) |

## What it is

Yearly surface urban heat island (SUHI) intensities, day and night, for 10,136 urban clusters,
2003–2018. The authors are Chakraborty and Lee, of the Yale Center for Earth Observation (YCEO).

- **Release:** NASA SEDAC, DOI 10.7927/s5m5-zk14, CC BY 4.0.
- **User guide:** DOI 10.7927/bb9y-9h12, September 2023, stored with the raw files.
- **Method paper:** Chakraborty and Lee (2019), *Int. J. Appl. Earth Obs. Geoinf.* 74:
  269–280, doi:10.1016/j.jag.2018.09.015.
- **Earth Engine copy:** `YALE/YCEO/UHI/*/v4`. It is rasters only, with no cluster ids,
  polygons or rural masks (checked 2026-11-04).

### Definitions (user guide §II and Appendix 1)

| | v4 |
|---|---|
| LST | MODIS 8-day composites **MOD11A2 (Terra) and MYD11A2 (Aqua)**, pixels with LST error ≤ 3 °C. "The Aqua and Terra products were combined to create daytime and nighttime values." The guide's sentence reads "MOD11A2 and MYD11A2 for Terra", but MOD is Terra and MYD is Aqua. The collection is not stated. |
| Overpass | Aqua about 13:30 and 01:30, Terra about 10:30 and 22:30. Day combines both daytime passes, and night both nighttime passes. |
| Urban cluster (extent) | Natural Earth "Urban areas" 1:10m (Kelso and Patterson 2018), one polygon per cluster, **fixed over the years** |
| Urban pixels | ESA CCI land cover urban class, year by year (2003–2018; v4 uses each year's own map), 300 m |
| **Rural reference** | **inside the same polygon:** the non-urban, non-water ESA CCI pixels, minus those whose GMTED2010 elevation differs by more than 50 m from the median elevation of the cluster's urban pixels. No buffer outside the polygon is used. |
| Cluster SUHI | mean LST of the urban pixels − mean LST of the rural pixels, for each cluster and year (K) |
| Pixel SUHI | a pixel's LST − the cluster's mean rural LST, at a nominal 300 m after downscaling |
| Seasons | summer is JJA in the northern hemisphere and DJF in the southern, and winter the opposite |

The guide says the method gives values similar to a 5 km rural buffer and lower than a 10 km
buffer (its footnote 3).

**The rural reference is not published as data.** It has to be rebuilt by the rule above from
ESA CCI (in the pipeline, `esacci`) and GMTED2010, inside the published polygons. ◇ G1 records
this.

## Raw data (FETCH)

`raw/yceo_uhi/v4/` holds the seven SEDAC files and the user guide, as published:

| File | Content |
|---|---|
| `…-documentation.pdf` | user guide |
| `…-urban-cluster-means-shp.zip` | `All_shp.shp`: 10,136 cluster polygons (EPSG:4326) with `Code`, `FID_1`, and the 2003–2018 mean SUHI by period (`Annual_`, `Summer_`, `Winter_`) and time of day (`day`, `nig`) |
| `…-uhi-all-averaged-geotiff.zip` | cluster means over 2003–2018, painted over each polygon |
| `…-uhi-monthly-averaged-geotiff.zip` | cluster means by calendar month |
| `…-uhi-yearly-averaged-2003-2018-geotiff.zip` | `UHI_yearly_averaged_<year>.tif`: the cluster mean of each year, bands `Daytime`, `Nightime` (sic) |
| `…-uhi-yearly-pixel-2003-2018-geotiff.zip` | pixel SUHI, annual, per year |
| `…-summer-` / `…-winter-uhi-yearly-pixel-2003-2018-geotiff.zip` | pixel SUHI by season, per year |

- The GeoTIFFs are EPSG:4326 at 0.0027° (about 300 m), 133,584 × 66,792 pixels, float32, LZW,
  in strips of one row (slow to read in windows).
- **The pixel products hold values only at each year's urban pixels,** so they also carry the
  yearly urban mask. Over Madrid in 2015, they cover 602 of the polygon's 2,435 km², and
  their mean (−0.57 K) is close to the cluster value (−0.54 K).

FETCH also writes two tables derived from the published files to `raw/yceo_uhi/v4/derived/`:

- `yceo_clusters.gpkg`, layer `clusters`: the polygons of `All_shp.shp` with all their
  attributes, sorted by `Code`.
  - **`Code` is the cluster id.** It is unique, and so is `FID_1`.
- `yceo_cluster_year.parquet`, with columns `Code`, `year`, `suhi_day`, `suhi_night` (K) and
  `from_polygon`:
  - the annual cluster mean of each year, read from `UHI_yearly_averaged_<year>.tif` at one
    interior point of the polygon;
  - that raster is constant over almost every polygon, so an interior pixel gives the cluster's
    value. Over all 10,136 polygons, min = max except for 45 in 2003 and 33 in 2018, most
    likely where neighbouring clusters' pixels meet;
  - where the point's pixel is empty, the mean over the polygon's pixels is used, and
    `from_polygon` is True.

Yearly summer and winter values are published only per pixel, not as cluster means.

**Route:** FETCH runs on a workstation with an Earthdata login in `~/.netrc` (never
committed). The files are then pushed with `data transfer --source yceo_uhi --step fetch`,
which deletes each local file once its push is verified. A re-run of FETCH therefore downloads
everything again. On sciCORE, the files are in `data_nobackup/raw/yceo_uhi/v4/`, and their
sizes match the downloads.

## Checks (2026-11-04)

- **Downloads:** all seven zips pass `unzip -t`, 2.8 GB in all. The module's `requests`
  download gives files identical to `curl --netrc`.
- **Coverage of `yceo_cluster_year.parquet`:** 162,176 rows (10,136 clusters × 16 years). Every
  cluster has a value in every year except `Code` 11996 in 2017 and 2018, and 3406, 8720 and
  8721 in 2018. No value needed the polygon fallback.
- **Against the Earth Engine copy:** for 2003 and 2018, day and night, the value read at the
  interior point equals Earth Engine's mean over the polygon exactly (difference 0.0) wherever
  the polygon is constant.
- **Against the shapefile:** the mean of the 16 yearly values correlates with `Annual_day` at
  0.9995 (night 0.9996), with a median absolute difference of 0.01 K. The shapefile's value is
  a pooled 2003–2018 composite, not a mean of yearly means. The 141 clusters that are empty
  in the shapefile have yearly values in the rasters.
- **Means over clusters and years:** 0.60 K by day and 0.44 K by night.

## Analysis caveats

- **Terra and Aqua combined.** The heat-island pre-spec assumed Aqua MYD11A2 alone. To match
  the benchmark, the replication has to combine MOD11A2 and MYD11A2 in the same way.
- **Extents are fixed.** The polygons don't change over 2003–2018, but the urban pixels inside
  them do (ESA CCI each year). The rural reference therefore shrinks as a cluster urbanizes.
- **The rural reference lies inside the urban polygon,** not in a buffer around it, so it
  includes peri-urban land. The guide warns against reading single clusters, especially per
  pixel.
- **Magnitudes fell with v2.** Moving from MODIS to ESA CCI land cover lowered mean SUHI by
  about 20 % against the 2019 paper. Over the shapefile's clusters, the 2003–2018 means are
  0.59 K by day and 0.44 K by night (141 clusters empty).
- **Missing clusters:** a cluster is empty if it has no urban or no rural pixels, or no usable
  LST.
