# Handoff: fetch GAIA and the YCEO heat-island benchmark from Earth Engine, push to sciCORE

You are working on the user's Mac in their clone of `growth-and-temperature`
(GitHub `felixschulz385/growth-and-temperature`). The analysis runs on the sciCORE HPC.
Its compute nodes have no internet, so this machine fetches the data and pushes it to the
HPC through the transfer node `transfer12.scicore.unibas.ch`. Another agent on the HPC
builds PREPARE and the analysis from what you deliver. **Your job ends when the raw files
are on the HPC and documented.**

## 0. Before anything else

1. `git fetch origin && git switch successor/heat-island && git pull`.
2. Read `CLAUDE.md` in full. Its hard rules apply to you:
   - pre-spec discipline;
   - "every new dataset is a source module, not a script";
   - commit only on this branch, never on `main`;
   - never force-push or rewrite history.
3. Read `docs/successors/heat-island-prespec.md`, at least §0, §2 and §7. It was confirmed
   on 2026-11-04.
   - **Do not edit it.** Report any construction choice or open question to the user in
     your final message, and the HPC agent logs it.
   - **Load no land-surface temperature (MODIS LST) of any kind.** The project's gates
     require that no LST is seen before G1.
4. Read one existing FETCH-only pattern end to end:
   - `src/data/sources/no2.py` (a module, `DEFAULT_TRANSFER_MODE`, `RAW_LISTING_DEPTH`);
   - `src/data/sources/registry.py` (`_SOURCE_MODULES`);
   - the `sources.no2` block in `orchestration/configs/data.yaml`;
   - `docs/data/no2/README.md`;
   - the `data transfer` command (`src/cli/data/handlers.py::handle_transfer`,
     `src/data/common/hpc/`).
5. Check this machine's `orchestration/configs/data.local.yaml` (git-ignored, per machine).
   - `paths.data_root` must be a local directory with room for a few GB.
   - `remote.ssh_target` must be
     `schulz0022@transfer12.scicore.unibas.ch:/scicore/home/meiera/schulz0022/projects/growth-and-temperature/data_nobackup`,
     with a working `key_file` or SSH agent.
   - Confirm `ssh schulz0022@transfer12.scicore.unibas.ch true` works without a prompt.
   - If the file is missing or wrong, ask the user. Don't invent credentials.
6. Use the repo's Python environment on this Mac (ask the user which one if it isn't obvious).
   `pip install earthengine-api` if missing.

## 1. Earth Engine access (needs the user)

- Ask the user for their Earth Engine Cloud project ID. Have them run
  `earthengine authenticate` themselves, because it opens a browser.
- Verify with `ee.Initialize(project=...)` and
  `ee.Image('Tsinghua/FROM-GLC/GAIA/v10').bandNames().getInfo()`.
- Never write the project ID, tokens or credentials into tracked files. Put the project ID in
  the git-ignored `data.local.yaml` or an environment variable, and read it from there.

## 2. Target grid (the HPC's canonical 1 km grid; must match exactly)

- **CRS:** EPSG:6933 (EASE-Grid 2.0 Global).
- **Pixel:** 1000 m.
- **Grid:** 34,736 columns × 12,704 rows. Affine `(1000, 0, -17368000, 0, -1000, 6352000)`,
  i.e. crsTransform `[1000, 0, -17368000, 0, -1000, 6352000]`.
- **Tiles:** 2048 × 2048 px, numbered by **row then column**: tile (row r, col c) covers pixel
  rows `2048r … 2048r+2047` and columns `2048c … 2048c+2047`, clipped at the grid edge. There
  are 7 rows and 17 columns of tiles.
- **On disk the HPC names a tile directory `ix=<row>/iy=<col>`.** Note that `ix` is the row.
  Use the same naming.
- **The 98 tiles with land** (`row,col`); fetch only these:

```
0,0 0,1 0,2 0,3 0,4 0,5 0,6 0,7 0,8 0,9 0,10 0,11 0,12 0,13 0,14 0,15 0,16
1,0 1,1 1,2 1,3 1,4 1,5 1,7 1,8 1,9 1,10 1,11 1,12 1,13 1,14 1,15 1,16
2,0 2,1 2,3 2,4 2,5 2,6 2,7 2,8 2,9 2,10 2,11 2,12 2,13 2,14 2,15 2,16
3,0 3,1 3,4 3,5 3,6 3,7 3,8 3,9 3,10 3,11 3,13 3,14 3,15 3,16
4,0 4,1 4,2 4,3 4,4 4,5 4,6 4,7 4,8 4,9 4,10 4,11 4,13 4,14 4,15 4,16
5,0 5,4 5,5 5,6 5,7 5,8 5,9 5,10 5,11 5,12 5,13 5,14 5,15 5,16
6,5 6,6 6,7 6,8 6,15
```

- **Check the grid before the bulk run:**
  - Request one small block over a city you can recognise, such as Beijing or Cairo.
  - Write it with its geotransform and confirm the corner coordinates match the affine above.
  - If Earth Engine rejects EPSG:6933, use the EASE-Grid 2.0 Global WKT, and say so in the
    report.

## 3. Task A: source module `gaia` (FETCH only)

**Content.**
- GAIA v1.0 (Gong et al. 2020, *RSE* 236:111510), Earth Engine asset
  `Tsinghua/FROM-GLC/GAIA/v10`, band `change_year_index`. Each 30 m pixel holds the year it
  became impervious (> 50 % impervious), coded 34 = 1985 … 1 = 2018.
- **Verify the encoding yourself.** Read the catalogue page and its sample code, and work out
  whether "impervious in year Y" is `>=` or `>` the code of Y. Getting this wrong shifts every
  year by one. State the rule you used and how you checked it.
- For every year 1985–2018 and every 1 km cell of the grid above, the **share of the cell's
  area that is impervious in that year**.
  - Use a 0/1 image with non-impervious and masked pixels set to 0.
  - Aggregate area-weighted to 1 km with `reduceResolution(ee.Reducer.mean(), maxPixels=…)`
    from GAIA's native 30 m projection.
  - Set `maxPixels` large enough for ~1,100 source pixels per output pixel, and check the
    result is not silently subsampled.
  - Water counts in the denominator: the share is of the whole cell. Record this in the source
    doc.
- **Encoding on disk:** `uint16`, share × 10,000 (0–10,000). Nodata 65535, only where GAIA has
  no data at all. 34 bands named `imp_1985` … `imp_2018`.

**Mechanics.**
- Request each land tile in sub-blocks with `ee.data.computePixels`, using the grid's exact
  crsTransform shifted to the block. A 512 × 512 block with 34 uint16 bands is ~18 MB, under
  the request limit. Shrink the blocks to 256 if requests time out.
- Run a few requests in parallel within Earth Engine's limits, with retries and backoff.
- Write **one compressed GeoTIFF per tile**:
  `raw/gaia/v10/ix=<row>/iy=<col>/gaia_imp_share_1985_2018.tif`, with the tile's exact
  geotransform. Write to a temp file and `os.replace` it, so a partial file is never seen.
- Re-running fetches only missing tiles.
- **If computePixels proves unworkable** (timeouts at any block size), fall back to batch
  exports (`Export.image.toCloudStorage` into a bucket of the user's project, same grid and
  encoding). Download from the bucket into the same raw layout. Ask the user before creating a
  bucket.

**Module conventions** (follow `no2.py`):
- `ID = "gaia"`; `STEPS = (PipelineStep.FETCH,)`. The HPC agent adds PREPARE.
- Register it in `registry.py`.
- Add a `sources.gaia` block to `orchestration/configs/data.yaml` with
  `data_path: "gaia/v10"`, and set the transfer mode so `data transfer --source gaia
  --step fetch` pushes the tiles.
- Earth Engine is imported lazily inside FETCH, so the HPC can import the module without
  `ee`.
- Write `docs/data/gaia/README.md`: what the data is, the encoding, caveats (the > 50 % rule
  misses densification inside a paved 30 m pixel; values only rise over time), the licence and
  the citation. Add a row to `docs/data/README.md`.

**Sanity checks** (report the numbers):
- Every cell's share is non-decreasing over the years.
- Values lie in [0, 1] after scaling.
- Global impervious area in 2018 is of the order the paper reports. Look the figure up and
  quote it with its source.
- A city tile looks right, and an ocean block is 0.
- Total size on disk.

## 4. Task B: the YCEO benchmark (investigate first, then source module `yceo_uhi`)

The pre-spec's step 1 replicates the per-city heat islands of YCEO Surface Urban Heat Islands
v4 (Chakraborty and Lee 2019), 2003–2018.

It needs, per urban cluster:
- **(i)** a stable cluster id;
- **(ii)** yearly annual day and night SUHI (summer and winter too if available);
- **(iii)** the urban extent;
- **(iv)** the rural reference area;
- **(v)** the definitions: which MODIS LST product and platform (Aqua and/or Terra,
  MOD11A2/MYD11A2), the overpass, and how urban and rural are delimited.

1. **Investigate the Earth Engine copies.** List everything under `YALE/YCEO/UHI/` (e.g.
   `UHI_yearly_averaged/v4`, `UHI_all_averaged/v4`, `UHI_yearly_pixel/v4`,
   `Summer_UHI_yearly_pixel/v4`, monthly). For each asset record the type
   (Image/ImageCollection/FeatureCollection), bands and properties, the resolution, and whether
   cluster ids or polygons exist. These are SUHI values, not LST, so inspecting them is
   allowed.
2. **The official copy is NASA SEDAC** (DOI 10.7927/s5m5-zk14). It is listed as "GeoTIFF,
   Shapefile", 7 files. It and its user guide (`sdei-yceo-sfc-uhi-v4-documentation.pdf`) need an
   **Earthdata login**. Ask the user whether they have one or will create one.
   - If yes, download the full v4 package and the user guide with their credentials (via
     `~/.netrc`, never committed).
   - The shapefiles are probably the cluster extents. Confirm that.
3. **Stop and report to the user before building the module** if the cluster extents or the
   rural references are not available as data, i.e. you would have to reconstruct them. The
   pre-spec's G1 has to record which source the extents come from.
4. Otherwise build `yceo_uhi` as a FETCH-only source module, like `gaia`:
   - Store the raw files as published under `raw/yceo_uhi/v4/`.
   - If you export from Earth Engine, store per-cluster tables (CSV or Parquet) plus the
     cluster geometries (GeoPackage).
   - Write `docs/data/yceo_uhi/README.md` (CC-BY-4.0, citation, definitions from the user
     guide: product, platform, overpass, urban and rural rules). Add a row to
     `docs/data/README.md`.
5. **Record from the user guide which LST product and platform v4 uses.** The pre-spec assumes
   Aqua MYD11A2, but the Earth Engine page says "MODIS 8-day Terra and Aqua". The HPC agent
   needs the exact answer to pick the replication's LST product.

## 5. Push to the HPC

- `python -m src.cli data transfer --config orchestration/configs/data.yaml --source gaia
  --step fetch` (likewise for `yceo_uhi`).
- Then confirm on the transfer node that the files arrived, and compare file counts and sizes:
  `ssh schulz0022@transfer12.scicore.unibas.ch 'find …/data_nobackup/raw/gaia/v10 -name "*.tif" | wc -l; du -sh …'`.
- Do not delete the local copies until the user agrees.

## 6. Git

- Commit the modules, config and docs on `successor/heat-island` in small commits. Never
  commit data, credentials or the project ID.
- Enable the repo's hooks once: `git config core.hooksPath .githooks`. Commit only after they
  pass.
- **The HPC agent may push to the same branch,** for example other new modules in
  `registry.py` and `data.yaml`. Before pushing: `git pull --rebase`, resolve any conflict by
  keeping both sides, then `git push`. Never force-push.
- Update `CLAUDE.md`'s "Fork and successor sources" list with one line per new source.

## 7. Out of scope

- PREPARE, the 1 km panel, any regression, or anything that reads LST.
- GHSL built-up, MYD11A2, elevation, WorldClim and ESA CCI: the HPC agent fetches those
  directly.
- Changes to `heat-island-prespec.md`, to `main`, or to the `successor/pollution-institutions`
  branch.

## 8. Final report to the user

- What was fetched, where it is on the HPC (paths, file counts, sizes), and what the sanity
  checks showed.
- The commit hashes pushed to `successor/heat-island`.
- The GAIA year-encoding rule you verified, and how.
- The YCEO findings: which assets exist, where the cluster extents and rural references come
  from, and the LST product, platform and overpass.
- Every construction choice you made, and every open question. The HPC agent turns these into
  the pre-spec's change log.
