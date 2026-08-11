# country_classifications — UNDP HDI + World Bank income-group classifications

Registry id `country_classifications` (`src/data/sources/misc/country_classifications.py`,
class `CountryClassificationsSource`). Steps: **fetch, prepare, grid**.
`REQUIRES`: `gadm` GRID (`REQUIRES = (("gadm", PipelineStep.GRID),)`). Config
key: `country_classifications`. `data_path: "misc"`, `namespace:
"country_classifications"` — third of the three sources split out of the old
combined `misc` source (module docstring, `docs/design/09-integrated-pipeline.md`
§7); kept as one joined source deliberately rather than split 4 ways, because
HDI and World Bank data are joined into a single `iso3`-keyed table before
GRID.

`REQUIRES` on `gadm`'s GRID output is (per the module docstring) the first
real use of that mechanism in this codebase: GRID needs gadm's
`GID_0_code_mapping.json` sidecar (see `docs/data/gadm/README.md`) to
translate `iso3` codes into gadm's integer `GID_0` ids.

**Not rasterized.** Every classification value is constant across all of a
country's pixels (it varies only by `GID_0`, never by pixel location), so
GRID writes a small `GID_0`-keyed parquet table instead of a pixel-grid zarr.
Assembly merges it onto rows via `src.data.assemble.processors.TileProcessor`'s
`join_on` mechanism, matching an existing `GID_0` column contributed by a
`gadm` dataset entry in the same assemble config.

## FETCH

Downloads two configured files via `ConfiguredFilesFetchMixin`
(`src/data/sources/misc/_fetch.py`); `_execute_fetch` requires `ctx.ssh_target`
(an HPC/remote target), else logs a warning and returns `False`.

| key | url config key / default | name config key / default |
|---|---|---|
| `hdi` | `hdi_url` / `https://hdr.undp.org/sites/default/files/2025_HDR/HDR25_Composite_indices_complete_time_series.csv` | `hdi_name` / `HDR25.csv` |
| `worldbank` | `worldbank_url` / `https://ddh-openapi.worldbank.org/resources/DR0095334/download` | `worldbank_name` / `DR0095334.xlsx` |

`data.yaml`'s `country_classifications:` block repeats these same defaults
verbatim.

- **Output path**: both files land directly under the same FETCH root (no
  per-origin subfolder in the current fetch driver — `run_fetch()` resolves
  one `raw_root()` for the whole source and each `ConfiguredFile` is written
  under its own `.name`):
  - legacy: `<data_root>/misc/raw/country_classifications`
  - v2: `<data_root>/raw/misc/country_classifications`
- **Format**: `HDR25.csv` (UNDP HDR composite-indices time series, CSV) and
  `DR0095334.xlsx` (World Bank "World By Income" workbook, xlsx).
- **Caveats**: no explicit URL-rotation/expiry caveat is written in code or
  `data.yaml` for either URL (unlike `ecoregions`' documented ArcGIS
  export-link expiry). Both default URLs are, however, versioned/dated by
  their own literal values — `2025_HDR` in the HDI URL, and a specific World
  Bank Data Catalog resource id (`DR0095334`) in the World Bank URL — so a
  future UNDP/World Bank release may require updating `hdi_url`/
  `worldbank_url` in `data.yaml`; this is an inference from the URL string
  itself, not a documented caveat in the code. A `data.yaml` comment notes
  the World Bank file's subfolder placement changed relative to the old
  pre-split `misc` config ("the old `misc` config downloaded the World Bank
  file into the `hdi` subfolder") — purely historical context, not a
  caveat about the current implementation.

## PREPARE

`_execute_prepare()` calls `read_hdi()` (`src/data/sources/misc/hdi.py`) if
the HDI file was fetched, `read_worldbank()` (`src/data/sources/misc/worldbank.py`)
if the World Bank file was fetched (tracked via `target.meta["has_hdi"]`/
`["has_wb"]`, set in `_plan_prepare()` from which raw files actually exist on
disk) — either input may be absent; if both are, PREPARE fails
(`logger.error("No data to process")`, returns `False`).

**`read_hdi(path)`** — parses `iso3` + `hdi_1990..hdi_2023` columns
(`encoding="latin1"`), melts to long form, buckets each `(iso3, year)` HDI
value into UNDP's four bands (`< 0.550` Low, `< 0.700` Medium, `< 0.800`
High, else Very High), then for each of `PANEL_YEARS = (1991, 1999, 2011)`
takes the last observation at-or-before that year per `iso3` and pivots to
one boolean column per `(band, year)`.

**`read_worldbank(path)`** — reads the `"Country Analytical History"` sheet
(`header=4`), renames the first column to `iso3`, drops the second column,
slices rows `[6:-2]` (trims header/footer rows), melts to long form, converts
2-digit fiscal-year columns to 4-digit years (`>50` → `19xx`, else `20xx` —
comment notes this rule is ported verbatim, not re-derived), drops rows coded
`".."`, then snapshots the same `PANEL_YEARS` the same way as HDI, mapping
World Bank's letter codes to `WB_LO` (`L`), `WB_LM` (`LM`/`LM*`), `WB_UM`
(`UM`), `WB_HI` (`H`).

If both inputs are present, the two wide tables are merged on `iso3`
(`how="left"`); World Bank boolean columns for `iso3` rows only present in
the HDI table are filled `False` rather than left null.

- **Output path**:
  - legacy: `<data_root>/misc/processed/stage_1/country_classifications/classifications.parquet`
  - v2: `<data_root>/prepared/misc/country_classifications/classifications.parquet`
- **Format**: parquet, one row per `iso3`.
- **Schema** (columns present depend on which of `has_hdi`/`has_wb` were
  available at PREPARE time — not a fixed set if one input is missing):

  | column | dtype | present when |
  |---|---|---|
  | `iso3` | string | always |
  | `HDI_LO_1991`, `HDI_ME_1991`, `HDI_HI_1991`, `HDI_VH_1991` | bool | `has_hdi` |
  | `HDI_LO_1999`, `HDI_ME_1999`, `HDI_HI_1999`, `HDI_VH_1999` | bool | `has_hdi` |
  | `HDI_LO_2011`, `HDI_ME_2011`, `HDI_HI_2011`, `HDI_VH_2011` | bool | `has_hdi` |
  | `WB_LO_1991`, `WB_LM_1991`, `WB_UM_1991`, `WB_HI_1991` | bool | `has_wb` |
  | `WB_LO_1999`, `WB_LM_1999`, `WB_UM_1999`, `WB_HI_1999` | bool | `has_wb` |
  | `WB_LO_2011`, `WB_LM_2011`, `WB_UM_2011`, `WB_HI_2011` | bool | `has_wb` |

- **Completion**: `Completion.PATH_EXISTS` (single file).
- **Caveats**: a code comment in both `hdi.py` and `worldbank.py` notes the
  melt/panel-year assembly uses plain bracket column assignment rather than
  `.loc[:, "year"] = ...`, specifically to avoid a `TypeError` under
  pandas ≥3.0's default `str` dtype for `melt()`-derived columns (this repo
  pins no pandas version). Not a data caveat, but explains an otherwise
  unusual code pattern if cross-checking against the old preprocessor.

## GRID

`_execute_grid()`: reads the PREPARE-stage parquet, resolves gadm's
`GID_0_code_mapping.json` via `gid_mapping_path(ctx.data_root, ctx.grid_id,
ctx.layout, "GID_0")` (imported from `src/data/sources/misc/gadm.py`; see
`docs/data/gadm/README.md`), loads it through
`src.analysis.subsets.registry.load_country_registry(...).country_to_id`,
maps each row's `iso3` to gadm's integer `GID_0` id (`0` if not found), then
**drops every row whose mapped `GID_0` is `0`** (i.e. any `iso3` absent from
gadm's mapping is silently excluded from the output — no warning is logged
for this drop, only a final row/column-count log line). `_plan_grid()`
requires gadm's grid zarr to already exist on disk before planning any
target at all.

- **Output path** (`layout.grid_store_path(...)` called **without**
  `v2_family`): the code comment explains this is deliberate — this is a
  small per-`GID_0` parquet table, not a `<family>.zarr` pixel-grid store, so
  it doesn't participate in layout:v2's "one store per family" zarr
  directory. `grid_store_path()` therefore falls back to its **legacy path
  shape regardless of `ctx.layout`** — i.e. even when `pipeline.layout: v2`
  is active, this output still lands at the *legacy-style* per-source path,
  not under `<data_root>/grid/<grid_id>/`:
  - `<data_root>/misc/processed/stage_2[_ease6933 if grid_id="ease6933"]/country_classifications/classifications_by_gid0.parquet`
  - (this exact path is used regardless of whether `pipeline.layout` is
    `legacy` or `v2`.)
- **Format**: parquet, one row per matched country.
- **Schema**:

  | column | dtype | meaning |
  |---|---|---|
  | `GID_0` | int | gadm's integer country id (from `GID_0_code_mapping.json`) |
  | *(remaining `value_cols`)* | bool | whichever `HDI_*`/`WB_*` columns PREPARE produced, carried through unchanged |

- **Completion**: `Completion.PATH_EXISTS` (single file).
- **Verification**: `verification_meta(expected_vars=("GID_0",))` — only the
  always-present join key is checked; a code comment explains `value_cols`
  vary by which of HDI/World Bank were available at PREPARE time, so they
  aren't pinned. `data.yaml`'s `country_classifications.verification` block
  declares the identical `expected_vars: ["GID_0"]` — present in config
  (unlike `gadm`/`ecoregions`, which omit the block entirely), but it
  matches the code default and so doesn't change behavior; no `value_range`
  is set anywhere (booleans have no numeric range to check).
- **Caveats**: countries whose `iso3` doesn't resolve via gadm's `GID_0`
  mapping are dropped silently (no per-row warning); this can happen for
  territories/entities present in the HDI or World Bank source data but
  absent from, or coded differently in, GADM's `ADM_0` layer.

**Not independently verifiable from code alone:** actual country coverage
(row count) of the joined classifications table, how many `iso3` rows get
dropped at GRID for lacking a `GID_0` match, and whether the HDI/World Bank
URLs currently in `data.yaml` still resolve (both are dated/versioned
endpoints, per the FETCH caveat above).
