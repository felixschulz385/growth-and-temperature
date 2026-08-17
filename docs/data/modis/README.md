# modis — MODIS night LST, streamed from Planetary Computer and gridded onto EPSG:6933

Registry id `modis`, class `ModisSource`, module `src/data/sources/modis/source.py`.
Aliases: `modis_lst`, `modis_robustness_11a1` (`ModisSource.ALIASES`) — both alias
strings route to the same class; which config block a run reads is selected by
`--source`/the `sources.<key>` config key, not by the alias mechanism.

`STEPS = (PipelineStep.FETCH, PipelineStep.GRID)` — **no PREPARE step**, unlike
every other multi-step source in the registry (`glass`, `eog`, `acag`, …). FETCH
here does the work a PREPARE stage would normally do elsewhere: it streams,
QC-masks, and annually composites MODIS data in one pass, rather than landing
untouched raw bytes first (see [FETCH](#fetch) below). `REQUIRES = ()` (inherited
default, not overridden) — MODIS has no cross-source dependency.

Config keys: `sources.modis` (primary) and `sources.modis_robustness_11a1`
(robustness arm), both `type: "modis"` in `orchestration/configs/data.yaml`.

## Config variants

| variant | product | platform | years | tiles | distinctive |
|---|---|---|---|---|---|
| `modis` | `21A2` (MYD21A2, 8-day TES LST+emissivity) | `aqua` | `year_range: [2002, 2025]` (contiguous) | every sinusoidal tile intersecting `lat_clip_deg: 60.0`, optionally narrowed by a `land_tiles` allowlist (unset in the checked-in config — see caveat below) | primary series; TES emissivity is land-cover-independent, which matters for this project's estimand (rationale in `07-modis-ingest.md` §1, not repeated here) |
| `modis_robustness_11a1` | `11A1` (MYD11A1, daily split-window LST+emissivity) | `aqua` | `years: [2004, 2014, 2023]` — explicit discrete years (early/mid/late Aqua mission), not a range | explicit 5-tile list: `h12v09` (Amazon), `h18v06` (Sahara), `h18v04` (Central Europe), `h22v03` (Siberian boreal, near the 60° clip edge), `h30v11` (Australian outback) | bounded same-methodology comparison arm, not a full parallel backfill — validates the 21A2 8-day valid-observation proxy against 11A1's true daily counts on a handful of biome-representative tiles/years |

Both variants share `qc_max_lst_error_k: 2.0` and
`stac_url: https://planetarycomputer.microsoft.com/api/stac/v1`. `product` picks
the STAC collection (`modis-21A2-061` / `modis-11A1-061`) and asset names via
`BAND_SPECS` in `source.py`; `ModisSource.__init__` rejects any other `product`
value. `self.years` (a discrete list) takes priority over `year_range` when both
would apply — `_plan_fetch`/`_plan_grid` compute `self.years or range(*year_range)`.
`data_path` defaults to `modis/<product>` when not set in config (both blocks here
leave it unset, so effectively `modis/21A2` and `modis/11A1`).

`modis`'s config block also carries a commented-out `tiles:`/`land_tiles:`
example and `transfer: {steps: ["fetch"]}` (orchestration-level: which steps get
pushed to HPC via the transfer mechanism).

## FETCH

Per-`(tile, year)` `StepTarget`s (`_plan_fetch`). For each target, `_execute_fetch`:

1. Bounding-boxes the sinusoidal tile into EPSG:4326 (`_tile_bbox_4326`, via
   `modis_util.tile_bounds_m` + a pyproj transform) and STAC-searches the
   collection for that bbox/year (`_search_items`), filtering to
   `properties.platform == self.platform` cross-checked against the `MOD`/`MYD`
   item-id prefix (a disagreement only logs a warning — the two signals have
   been checked to agree in 600 real items, so this is a tripwire, not an
   expected filter path).
2. Loads the configured bands via `odc.stac.load` (`_load_tile_year`),
   **manually** applies each band's `scale`/`offset`/`fill` from `BAND_SPECS`
   (`odc.stac.load` does not auto-apply STAC-declared scale/offset — confirmed
   empirically, see `docs/design/07a-modis-band-reference.md`).
3. Builds a QC-valid mask from the `qc` band (`modis_util.decode_qc_valid_mask`,
   product-specific bit layout — see [GRID caveats](#caveats) below) and
   month-first-then-annual composites `lst` and each emissivity/view band via
   the shared `composite_to_annual` helper (`src/data/common/raster/compositing.py`).
4. Writes one multi-band GeoTIFF per `(year, tile)`: annual `lst_night`,
   `valid_period_count_annual`, `valid_month_count_annual`, per-product
   emissivity/view bands, plus one band per month for `lst_night_monthly_MM`
   and `valid_period_count_monthly_MM` (`_write_annual_geotiff`).

**Output path** — FETCH here physically reuses the path shape a PREPARE stage
would use elsewhere (`ModisSource.output_root()` explicitly overrides the base
class to route FETCH through `layout.output_root(..., PipelineStep.PREPARE, ...)`
rather than `layout.raw_root()`'s bare-bytes convention every crawler-based FETCH
source uses):
- `<data_root>/prepared/<data_path>[/<namespace>]/<year>/<tile>.tif`

**Format**: one GeoTIFF per `(year, tile)`, float32, `nodata=NaN`, deflate-compressed,
band descriptions set to the variable/month names above (`dst.set_band_description`).
Not visible in `tiles.py`/`source.py` alone: the exact land-tile count actually
produced by a run — see caveat below.

**require_remote / ledger dependency.** Every FETCH `StepTarget` sets
`require_remote=True` (`_plan_fetch`) — per `docs/design/10-fetch-ledger.md`,
MODIS FETCH is the one case in this codebase where "complete" means more than
local-disk existence: FETCH streams from Planetary Computer off-cluster (needs
internet egress SLURM compute nodes may lack) and must be verified as pushed to
HPC via the ledger (`src/data/common/ledger/`, `SourceLedger.ensure_artifact`/
`set_local_state`) before GRID's SLURM job can trust the tile-year is actually
there. This doc does not re-derive the ledger mechanism itself — see
`docs/design/10-fetch-ledger.md`.

**Other caveats explicit in code:**
- `lat_clip_deg` (default 60.0) restricts the FETCH tile list at plan time via
  `modis_util.get_modis_sinusoidal_tiles`, not post-hoc filtering.
- `land_tiles` is an optional allowlist mechanism (`compute_land_tiles()` in
  `tiles.py`, driven by `scripts/compute_modis_land_tiles.py` against the `osm`
  source's land-polygon output) that further restricts tiles to land-covering
  ones. **Not populated in the checked-in `sources.modis` block** — confirmed
  from `data.yaml` (the `land_tiles:` line is commented out) — so a run today
  ingests every tile in the latitude band, ocean-only tiles included, unless an
  operator runs the script and adds the config. `docs/design/07b-modis-outstanding.md`
  already flags this as open ("tooled, not yet run"); still true against current
  `data.yaml`.
- QC handling happens at FETCH time, not GRID: `decode_qc_valid_mask` gates
  which pixels contribute to each composite before the GeoTIFF is written: see
  [GRID caveats](#caveats) for the per-product bit-layout status.

## GRID

Per-year `StepTarget`s (`_plan_grid`), one per year with at least one FETCH
GeoTIFF present. `_execute_grid` mosaics that year's per-tile GeoTIFFs
(`_mosaic_tiles`, `xr.combine_by_coords` over sinusoidal coordinates), then
reprojects onto the canonical target geobox via `SpatialProcessor`
(`nearest` resampling, `SPATIAL_RESAMPLING`), writing into a shared multi-year
zarr store one year-region at a time.

`output_root(GRID)` always forces `grid_id="ease6933"` regardless of
`ctx.grid_id`/`pipeline.grid` in `data.yaml` — a deliberately preserved MODIS-only
ad hoc case (module docstring, `docs/design/05-migration.md` §1).

**Output path** (`layout.grid_store_path`, `family=f"modis_lst_{product.lower()}"`):
- `<data_root>/grid/ease6933/modis_lst_<product>.zarr` (e.g.
  `modis_lst_21a2.zarr`, `modis_lst_11a1.zarr` — flat, no namespace, so the two
  config variants land in genuinely separate stores)

**Format**: Zarr (via `SpatialProcessor.create_empty_target_zarr`/
`write_year_to_zarr`), `dtype="float32"`, `dst_nodata=NaN`, dims include `time`
(one coordinate per year) over the canonical EPSG:6933 geobox.

**Variables** — `lst_night` is always written (per `_execute_fetch`'s `data_vars`
dict); the diagnostic/auxiliary bands are also mosaicked/reprojected whenever
present in the FETCH GeoTIFFs, but only `lst_night` is declared/checked by
`verification_meta`:

| variable | dtype | meaning | nodata | value_range (checked) |
|---|---|---|---|---|
| `lst_night` | float32 | annual mean night LST, Kelvin (scale/offset already applied at FETCH time — code comment: "no packed decode is needed here") | NaN | `[150, 350]` — both `sources.modis.verification` and `sources.modis_robustness_11a1.verification` in `data.yaml` declare `expected_vars: ["lst_night"]`, `value_range: [150, 350]`, matching the Python-side default passed by `_plan_grid`'s `verify.verification_meta(..., expected_vars=("lst_night",), value_range=(150, 350))` |
| `valid_period_count_annual`, `valid_month_count_annual`, `emis_29`\* /`emis_31`/`emis_32`, `view_angle`, `view_time` | float32 | diagnostics/auxiliary bands carried through mosaicking (see [FETCH](#fetch) for what each means) | NaN | not covered by `verification_meta`/`data.yaml`'s `verification:` block — no declared range check |

\* `emis_29` only exists for `21A2` (`BAND_SPECS["11A1"]` has no `emis_29` asset).

`verification_meta()` (`src/data/sources/verify.py`) only ever powers a *sampled*
sanity check (finite + in-range on a strided sample) opened via `data
summary`/the assembly gate — not a full-array pass. Actual observed value
distribution, tile/date coverage achieved by a real run, and zarr store size are
not knowable from code/config alone.

**TODO (needs live data):** actual land-tile count ingested by a real `modis`
run (the ~317-figure in `07a-modis-band-reference.md` is explicitly flagged
UNVERIFIED there, and `land_tiles` is unset — see FETCH caveats); observed
`lst_night` value distribution; zarr store sizes for `modis_lst_21a2.zarr` /
`modis_lst_11a1.zarr`; whether a full `modis` backfill (2002–2025 × full tile
list) has actually completed.

### Caveats

- **`Completion.NEVER` on GRID targets** (`_plan_grid`): unlike every other
  source's GRID step, MODIS's GRID never checks `override`/output-existence
  before writing a year into the shared multi-year zarr — it always re-runs
  that year. Preserved deliberately (module docstring cites
  `tests/data/preprocess/sources/test_characterization_modis.py` pinning the
  old behavior), not a bug.
- **`qc_max_lst_error_k`** (default `2.0`, both config blocks): the LST-error-K
  threshold `decode_qc_valid_mask` applies when building the FETCH-time valid
  mask — a configurable policy choice, not a layout fact, per
  `docs/design/07-modis-ingest.md` §6.
- **QC bit layout is confirmed for both products in the current code**, contrary
  to what a literal reading of `07a-modis-band-reference.md`'s "still open for
  MOD21A2" framing might suggest in isolation: `tiles.py`'s
  `_LST_ERROR_K_BY_BITS` dict has verified entries for both `"11A1"` and
  `"21A2"` (with the two products' bit-value-to-error-K polarity explicitly
  inverted and documented in the module comment), and
  `_QC_LAYOUT_CONFIRMED_PRODUCTS = frozenset(_LST_ERROR_K_BY_BITS)` covers both
  — the runtime "UNVERIFIED" warning in `decode_qc_valid_mask` only fires for a
  `product` outside that set, which neither `modis` nor `modis_robustness_11a1`
  is. `docs/design/07b-modis-outstanding.md` already reflects this as resolved
  (dated 2026-08-09) — flagging here only because `07a-modis-band-reference.md`
  read alone still frames MOD21A2's QC as "still open," which is stale relative
  to `07b` and the current code.
- Scale/offset/fill values in `BAND_SPECS` are applied once, manually, at FETCH
  time (`_load_tile_year`) — `odc.stac.load` does not auto-apply STAC-declared
  scale/offset (confirmed empirically per `07a-modis-band-reference.md`), so
  GRID reads already-physical values and does no further decoding.

## See also

- [`docs/design/07-modis-ingest.md`](../../design/07-modis-ingest.md) — product
  choice rationale (21A2 vs 11A1 vs rejected 11A2), the two-stage
  composite-in-native-sinusoidal/reproject-once architecture, the month-first
  compositing definition, and operational requirements (token refresh, dev
  cache, execution scripts).
- [`docs/design/07a-modis-band-reference.md`](../../design/07a-modis-band-reference.md) —
  authoritative per-band scale/offset/fill/range reference for both products,
  cited to primary sources (STAC + the MOD11/MxD21 user guides).
- [`docs/design/07b-modis-outstanding.md`](../../design/07b-modis-outstanding.md) —
  live checklist of resolved vs. still-open MODIS items (QC bit layout,
  land-tile allowlist, robustness-arm tile/year selection, HPC transfer
  throughput).
- [`docs/design/10-fetch-ledger.md`](../../design/10-fetch-ledger.md) — the
  `require_remote`/ledger mechanism MODIS FETCH depends on for HPC-verified
  completion (§7 specifically documents MODIS's FETCH placement).
