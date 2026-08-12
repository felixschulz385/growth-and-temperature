# 06 — Open Questions

Everything below could not be resolved from the repository or from a primary source verifiable
within this task, and is stated here rather than silently asserted, per the design brief's hard
constraint to flag rather than guess.

## 1. EASE-Grid 2.0 exact polar (y) extent — verified discrepancy, not resolved

A direct check against the installed PROJ database found the EPSG:6933 antimeridian (x) extent
agrees with NSIDC's published EASE2_G constant to millimetre precision (17,367,530.45 m computed vs.
17,367,530.44 m published). The **polar (y) extent disagrees by ~0.38%**: 7,342,230.14 m from a
direct ellipsoidal transform at φ=90° vs. 7,314,540.83 m in NSIDC's published grid definition. Likely
explanation: NSIDC's published raster grid is defined by an integer row/column count × a rounded
cell size (their documentation lists 34,704 cols × 14,616 rows at "1 km", cell size 1000.90 m) rather
than the literal ellipsoidal pole-to-pole distance — but this was not confirmed against NSIDC's
actual technical grid-definition reference (only their public documentation page).

**Resolution:** locate NSIDC's authoritative EASE2_G grid-definition technical document (not the
summary webpage) and reconcile. Until then, [`01-grid.md`](01-grid.md) §2 computes the extent
programmatically from the installed, pinned PROJ database rather than hardcoding either number — this
is a mitigation, not a resolution; the underlying discrepancy is still unexplained. **Not blocking for
|φ|≤60°** since the clip is far from the pole, but should be resolved before any work extends the
clip latitude closer to the poles.

## 2. Whether the currently-ingested GLASS product is clear-sky-only or all-weather gap-filled

The preprocessing code (`src/data/preprocess/sources/glass.py`) handles compositing and resampling,
but nothing in the repo states the underlying GLASS product's gap-filling methodology. This
materially affects the "resample as little as possible" reasoning in
[`04-ingest.md`](04-ingest.md) §2 and the GLASS-vs-MYD21 trade-off — a gap-filled product already
smooths over missing data in a way a raw swath product (MYD21) does not, which is a substantive
difference for the ring estimator's error properties, not a cosmetic one.

**Resolution:** check against GLASS's own product documentation/user guide (out of reach from the
repo alone).

## 3. Which MODIS collections are cloud-hosted at daily resolution, on which host, under what terms

Deferred by design to the companion prompt already present in this repository,
`PLAN_PROMPT_modis_ingest.md`, which targets Microsoft Planetary Computer's `modis-11A1-061`
collection specifically and is scoped to verify exactly this (asset lists, hosting region, signing
requirements, the 11A1/11A2/21A2 trade-off) before producing
`docs/design/07-modis-ingest.md` and `docs/design/07a-modis-band-reference.md`. That prompt is
written to read and conform to this document set once it exists.

**Resolution:** run that companion task next; it depends on the grid decisions in
[`01-grid.md`](01-grid.md) being settled, which they now are.

## 4. Missing environment/dependency pin file — format choice left open

Confirmed: no `pyproject.toml`, `environment.yml`, `requirements.txt`, `setup.py`, `poetry.lock`, or
`conda*.yml` exists anywhere in the repository, despite the README referencing one.
[`05-migration.md`](05-migration.md) §3 recommends fixing this as a prerequisite of the backbone
work, since the CRS/GeoBox design depends on exact odc-geo/pyproj/PROJ behavior. **What's left open
here is the format** (`pyproject.toml` + a lockfile vs. a conda `environment.yml`) — that choice
depends on HPC deployment conventions (`orchestration/slurm/*.sh` scripts currently do
`conda activate src` against a pre-existing, out-of-repo environment) that this design doc doesn't
have visibility into.

**Resolution:** a decision from whoever manages the HPC conda environment currently named `src`.

## 5. |φ| ≤ 60° clip's exact land-area / country impact

The clip was accepted in principle by the user as a scope restriction
([`00-backbone-overview.md`](00-backbone-overview.md)), on the computational (anisotropy-capping) and
data-quality (VIIRS/LST degradation) justifications already stated in
[`01-grid.md`](01-grid.md) §1. **What has not been computed here** is exactly which regions,
countries, or inhabited/economically-relevant land area fall outside the clip (e.g. most of
Scandinavia, Alaska, northern Canada, Siberia, Antarctica) — this is a substantive scientific-scope
question, not an engineering one, and belongs to explicit research-team sign-off rather than being
treated as settled by the computational argument alone.

**Resolution:** a GIS pass (existing GADM country rasters could support this) quantifying
population/GDP/land-area share excluded by the clip, reviewed by the research team.

## 6. Latitude-band elliptical kernel: axis-orientation sign convention — resolved during implementation

[`01-grid.md`](01-grid.md) §6 described the elliptical kernel correction in terms of
`scale_EW`/`scale_NS` but flagged that the precise direction of the correction (which axis
compresses vs. expands at a given latitude, in pixel space) was not verified numerically in the
design task and needed to be derived and confirmed at implementation time.

**Resolved.** `src/data/common/neighbourhood/kernels.py:anisotropy_scales` derives
`scale_EW = cos(φ_s)/cos(φ)`, `scale_NS = cos(φ)/cos(φ_s)` from the forward projection formulas
(ground-distance-per-map-unit ratios), giving semi-axes `a_EW = d·scale_EW`, `a_NS = d·scale_NS` for a
ground-circle of radius `d`. This was checked against an independent ground truth — not derived from
the same closed form — in
[`tests/data/common/neighbourhood/test_kernels.py`](../../tests/data/common/neighbourhood/test_kernels.py)'s
`test_elliptical_kernel_matches_geodesic_ground_truth`: for each of 5 test latitudes (0.5°, 15°, 30°,
45°, 59°), it rasterizes which pixels fall within `d` of a center point using WGS84 geodesic distance
(`pyproj.Geod.inv`), and confirms the analytic elliptical kernel agrees with that mask at IoU > 0.97.
All 5 latitudes pass, including one on each side of the 30° standard parallel and one near the 60°
clip edge — confirming both the magnitude and the orientation of the correction, not just its
existence.

## 7. Real Zarr compression ratios

[`02-storage.md`](02-storage.md) §6's sizing table uses illustrative, unmeasured compression
assumptions (2.5× for float32 fields, 5× for uint16 counts) to keep the arithmetic checkable against
the confirmed 5–20 TB budget. These are not measurements.

**Resolution:** measure actual `blosc`/`zstd` compression ratios on a representative subset of real
data (one tile, one year) before committing to the full ladder/variable set, and before the expensive
convolution stage runs at scale.

## 8. MODIS platform filter: `properties.platform` vs. `MOD`/`MYD` id prefix — RESOLVED (2026-08-09)

[`07-modis-ingest.md`](07-modis-ingest.md) §6 flagged this as the highest-consequence bug available in
the ingest component (mixing Terra/Aqua silently averages two different overpass times), since this
session only confirmed via STAC that both collections mix `aqua`/`terra` in `summaries.platform`
without confirming whether the two signals (`properties.platform` and the id's `MOD`/`MYD` prefix) ever
disagree.

**Resolved:** queried 600 real STAC items directly against the live Planetary Computer API (3
collections — `modis-21A2-061`, `modis-11A1-061`, `modis-11A2-061` — x 5 geographically spread regions
x 4 years spanning the mission, both platforms represented: 512 aqua / 88 terra items sampled). **Zero
disagreements** between `properties.platform` and the id prefix. Which signal is "authoritative" is
moot since they appear to always agree in practice; `_search_items`'s existing disagreement warning
(`src/data/sources/modis/source.py`) is kept as a live tripwire, not because a mismatch is expected.

## 9. QC_Night bit layout for the L3 gridded products (MOD11A1/A2, MOD21A2) — RESOLVED (2026-08-09)

**Resolved for MOD11A1** first: the user supplied the MOD11 V6.1 PDF (Collection-6 MODIS LST
Products Users' Guide, Wan, ERI/UCSB, June 2019) directly. Its Table 13 gives the 8-bit
`QC_Day`/`QC_Night` layout for MOD11A1 — mandatory QA (bits 1&0), data quality (bits 3&2), emissivity
error (bits 5&4), LST error category (bits 7&6: 00 ≤1K / 01 ≤2K / 10 ≤3K / 11 >3K) — which exactly
matches what `src/data/sources/modis/tiles.py::decode_qc_valid_mask` already implemented as an
unverified best guess.

**Resolved for MOD21A2** next: the user supplied the correct primary source, the *MxD21 LST&E User
Guide* (Hulley et al., JPL, March 2019, not the MOD11 guide — a different document, different
algorithm). Its Table 12 ("Bit flags defined in the QC_Day and QC_Night SDS in the MxD21A2 8-day
product") gives the same bit *positions* (1&0 mandatory QA, 7&6 LST accuracy) but **the opposite
meaning at bits 7&6**: MOD11 has increasing bit value = *worse* accuracy (00 ≤1K ... 11 >3K); MOD21
has increasing bit value = *better* accuracy (00 >2K poor ... 11 <1K excellent). Applying MOD11's
mapping to MOD21A2 data — which is what this module did before Table 12 was checked — would have
silently inverted the quality filter on the primary `21A2` product: keeping the worst-quality pixels
and discarding the best. `decode_qc_valid_mask` now takes a `product` argument selecting the correct
per-product bit-value-to-error-K mapping (`_LST_ERROR_K_BY_BITS` in
`src/data/sources/modis/tiles.py`), confirmed for both `"11A1"` and `"21A2"`. Pinned by
`tests/data/sources/modis/test_modis_qc.py`. Full detail in
[`07a-modis-band-reference.md`](07a-modis-band-reference.md).

## 10. MOD11A1/11A2 `Emis_31`/`Emis_32` offset — RESOLVED (2026-08-09)

The same user-supplied MOD11 V6.1 PDF's Table 9 ("The SDSs in the MOD11A1 product") confirms
`offset: 0.49` for `Emis_31`/`Emis_32`, matching MOD21A2's already-confirmed value. This caught a real
bug: `BAND_SPECS["11A1"]` in `src/data/sources/modis/source.py` had hardcoded `offset: 0.0` (an
unverified guess), silently shifting every MOD11A1 emissivity value by 0.49. Corrected, along with two
adjacent values the same table exposed as wrong: `Night_view_angl`'s fill (was `0`, guide says `255`)
and offset (was `0.0`, guide says `-65.0`), and `Night_view_time`'s fill (was `0`, guide says `255`).
Pinned by `tests/data/sources/modis/test_modis_qc.py`.

## 11. Whether `odc.stac.load` applies STAC `raster:bands` scale/offset automatically — RESOLVED (2026-08-09)

Stated as unverified in the original ingest brief. Getting this wrong would silently shift every scaled
band (LST, emissivity, view time) by its scale factor with no visible error.

**Resolved:** loaded a real `modis-21A2-061` item (`MYD21A2.A2015193.h18v05...`) via `odc.stac.load()`
with `odc-stac` 0.5.3, the same default kwargs `_load_tile_year` uses (no `dtype=` override), and
compared its `LST_Night_1KM` values against a raw `rasterio.open()` read of the same signed asset URL.
Both stayed `uint16` and matched exactly (e.g. `14875`, `14864`, `14879`, ...; declared
`raster:bands` scale on the asset is `0.02`, which would put real values in the ~150–350K range if
auto-applied — they didn't). **`odc.stac.load()` does NOT auto-apply STAC scale/offset** by default;
`_load_tile_year`'s manual `raw * scale + offset` (`src/data/sources/modis/source.py`) is required, not
a double-application bug.

## 12. Exact MODIS land-tile count and its reduction under the |φ|≤60° clip

`PLAN_PROMPT_modis_ingest.md`'s figure of ~317 land-covering sinusoidal tiles is a commonly-cited
literature figure, not independently re-derived this session.
[`07-modis-ingest.md`](07-modis-ingest.md) §3 additionally restricts the tile list to tiles
intersecting |φ|≤60°, which reduces that count by an unquantified amount.

**Resolution:** compute the actual sinusoidal `h`/`v` tile list intersecting land **and** |φ|≤60° at
implementation time (this is also the natural moment to tighten item 5's country/land-area impact
estimate, since both need the same GIS pass).

## 13. Exact Aqua-only temporal start date for MYD-prefixed collections — RESOLVED (2026-08-09)

All three STAC collections report a combined Terra+Aqua temporal extent starting 2000-02-16/18/24;
none report an Aqua-specific start. [`07-modis-ingest.md`](07-modis-ingest.md) §1 used an illustrative
~23-year Aqua era (2002–2025, since Aqua launched 2002-05-04) for its read-volume arithmetic, not
confirmed against the actual first available MYD11A1/MYD21A2 granule date.

**Resolved:** queried Planetary Computer directly for every `MYD`-prefixed item in the 2002-05-04
(launch) to 2002-08-01 window (1037/632/1049 items found for 21A2/11A1/11A2 respectively) and took
the true minimum by the acquisition date encoded in each item id (`A<year><day-of-year>`), not
relying on server-side sort order. Real earliest Aqua granule per collection:
- `modis-21A2-061` (the primary `modis` source's product): **2002-07-04**
- `modis-11A2-061`: **2002-07-04**
- `modis-11A1-061` (the `modis_robustness_11a1` arm's product): **2002-07-28**

All ~2 months after Aqua's 2002-05-04 launch (commissioning), consistent with the illustrative
estimate's order of magnitude. `data.yaml`'s `year_range: [2002, 2025]` already safely covers this —
2002 is simply a partial year, which `_execute_fetch` already handles gracefully (a tile-year with no
STAC items logs a warning and is skipped, `src/data/sources/modis/source.py`) — so no config change
was needed, only replacing the illustrative estimate with a confirmed one.

## 14. `glass-modis-preprocess-{annual,spatial}.sh` may already not match the current CLI — RESOLVED (2026-08-09)

`orchestration/slurm/glass-modis-preprocess-{annual,spatial}.sh` invoked
`run.py preprocess --config ... --source glass_modis --stage annual`, without the `run` subcommand that
`src/cli/preprocess/commands.py`'s `preprocess_cmd` subparser currently requires (only `run` is
registered, and `sub.required = True`). This looked like the scripts predated the CLI's modularization
into domain/subcommand pairs and might not currently execute as committed — noticed while reading these
scripts as the template for [`07-modis-ingest.md`](07-modis-ingest.md) §7's new scripts, not otherwise
investigated at the time.

**Resolved:** those two scripts no longer exist in the repo. They were superseded by
`orchestration/slurm/glass-modis-prepare.sh` and `glass-modis-grid.sh`, generated by
`orchestration/slurm/generate_slurm_scripts.py` from `jobs.yaml`'s `glass-modis-prepare`/
`glass-modis-grid` job entries, which correctly invoke the current `data run --source glass_modis
--step prepare`/`--step grid` (verified this subcommand registers and resolves against the source
registry). The only remaining `--stage`-flag text in the repo is inside
`validate-hard-gate-modis.sh`'s comparison-log echo strings (documenting the *old* invocation for a
before/after diff), not a live script that runs it.

**Resolution:** run `glass-modis-preprocess-annual.sh` (or just the equivalent `run.py` invocation) and
confirm whether it actually works today; if not, this is a pre-existing bug independent of the MODIS/
transfer work and should be fixed or reported separately.

## 15. Real HPC-transfer throughput and manifest schema

[`08-hpc-transfer.md`](08-hpc-transfer.md) §6: real transfer throughput/time for ~6,700 tile-year
annual composites over the scicore transfer node is not measurable from the repo, and whether
`UnifiedDataIndex`'s existing schema needs an addition for transfer-manifest use (vs. a namespace
convention on existing columns) is an implementation-time design call.

**Resolution:** measure real transfer throughput with a small representative batch of tile-year zarr
stores before relying on any time estimate; settle the manifest schema question during
`src/data/common/hpc/transfer.py`'s implementation, informed by whatever `UnifiedDataIndex` turns out
to support cleanly.
