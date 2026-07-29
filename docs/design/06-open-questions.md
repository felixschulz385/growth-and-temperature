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

## 8. MODIS platform filter: `properties.platform` vs. `MOD`/`MYD` id prefix — which is authoritative

[`07-modis-ingest.md`](07-modis-ingest.md) §6 flags this as the highest-consequence bug available in
the ingest component (mixing Terra/Aqua silently averages two different overpass times), but this
session only confirmed via STAC that both collections mix `aqua`/`terra` in `summaries.platform` — it
did not confirm which of `properties.platform` or the item id's `MOD`/`MYD` prefix is authoritative, or
whether they ever disagree.

**Resolution:** pull a sample of real STAC items for `modis-21A2-061` and confirm both signals agree
on a batch before trusting either alone; write this as a unit test against real (or cached) STAC items
before any large ingest run.

## 9. QC_Night bit layout for the L3 gridded products (MOD11A1/A2, MOD21A2) — not verified from a primary source

The only primary source successfully fetched this session for QC bit semantics, the MOD11 User Guide
(`icess.ucsb.edu/modis/LstUsrGuide/usrguide_mod11.html`), covers **only MOD11_L2** swath data, whose
16-bit QC layout is structurally inconsistent with the `uint8` (8-bit) `QC_Day`/`QC_Night` STAC-declared
dtype for the L3 gridded 11A1/11A2/21A2 products actually being ingested. The commonly-cited 8-bit
MOD11A1/A2 layout (mandatory QA / data quality / emissivity error / LST error bit-pairs) was **not**
independently confirmed this session, and no QC bit description was found for MOD21A2 at all. Direct
retrieval of the MOD11 V6.1 PDF (the correct primary source) failed to yield extractable bit-table text
via the tooling available this session. Full detail in
[`07a-modis-band-reference.md`](07a-modis-band-reference.md).

**Resolution:** extract the QC bit table directly from the MOD11 V6.1 PDF (a proper PDF parser, not a
web-fetch-and-summarize tool) or a peer-reviewed methods paper that reproduces it, and confirm a
parallel source for MOD21A2's QC field, before hardcoding any error-threshold logic. Until resolved,
[`07-modis-ingest.md`](07-modis-ingest.md) §6 requires the QC threshold to be a configurable parameter,
not a hardcoded constant, specifically to keep a wrong assumption cheap to fix.

## 10. MOD11A1/11A2 `Emis_31`/`Emis_32` offset — unconfirmed

STAC reports `scale: 0.002` for MOD11A1/11A2's `Emis_31`/`Emis_32` but no offset field. MOD21A2's
`Emis_29/31/32` was confirmed (via the NASA Earthdata catalog page) to need `offset: 0.49` in addition
to its own `scale: 0.002`. Whether MOD11's emissivity bands share that same offset, a different one, or
none, was not confirmed this session.

**Resolution:** decode a real `Emis_31` pixel from a sample MOD11A1 granule and check whether
`raw × 0.002` alone yields physically plausible emissivity (~0.9–1.0 over vegetated/urban land) or
whether an offset is required, before either family's emissivity band is used anywhere in the pipeline.

## 11. Whether `odc.stac.load` applies STAC `raster:bands` scale/offset automatically

Stated as unverified in the original ingest brief and not resolved this session. Getting this wrong
silently shifts every scaled band (LST, emissivity, view time) by its scale factor with no visible
error.

**Resolution:** load one real MODIS item with the pinned `odc-stac` version and compare the loaded
array's values against the STAC-declared `scale`/`offset` applied manually, before writing the ingest
loop against an assumption either way.

## 12. Exact MODIS land-tile count and its reduction under the |φ|≤60° clip

`PLAN_PROMPT_modis_ingest.md`'s figure of ~317 land-covering sinusoidal tiles is a commonly-cited
literature figure, not independently re-derived this session.
[`07-modis-ingest.md`](07-modis-ingest.md) §3 additionally restricts the tile list to tiles
intersecting |φ|≤60°, which reduces that count by an unquantified amount.

**Resolution:** compute the actual sinusoidal `h`/`v` tile list intersecting land **and** |φ|≤60° at
implementation time (this is also the natural moment to tighten item 5's country/land-area impact
estimate, since both need the same GIS pass).

## 13. Exact Aqua-only temporal start date for MYD-prefixed collections

All three STAC collections report a combined Terra+Aqua temporal extent starting 2000-02-16/18/24;
none report an Aqua-specific start. [`07-modis-ingest.md`](07-modis-ingest.md) §1 uses an illustrative
~23-year Aqua era (2002–2025, since Aqua launched 2002-05-04) for its read-volume arithmetic, but this
was not confirmed against the actual first available MYD11A1/MYD21A2 granule date.

**Resolution:** query the STAC API for the earliest `MYD`-prefixed item in each collection directly,
and use that as the real Aqua-era start for tile-year count arithmetic and time budgeting.

## 14. `glass-modis-preprocess-{annual,spatial}.sh` may already not match the current CLI

`orchestration/slurm/glass-modis-preprocess-{annual,spatial}.sh` invoke
`run.py preprocess --config ... --source glass_modis --stage annual`, without the `run` subcommand that
`src/cli/preprocess/commands.py`'s `preprocess_cmd` subparser currently requires (only `run` is
registered, and `sub.required = True`). This looks like the scripts predate the CLI's modularization
into domain/subcommand pairs and may not currently execute as committed — noticed while reading these
scripts as the template for [`07-modis-ingest.md`](07-modis-ingest.md) §7's new scripts, not otherwise
investigated.

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
