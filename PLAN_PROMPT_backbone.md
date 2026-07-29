# Agent prompt — plan the foundational backbone (v2 grid + neighbourhood engine)

> Paste this as the opening message to a coding agent in VS Code. **Enter plan mode.** The deliverable
> is a written plan, not code. Do not edit any source file during this task except to create the
> design documents named under "Deliverables".

---

## Your task

This repository (`growth-and-temperature`, the "GNT data system") already implements a working
pipeline: download → preprocess (stage_1 spatial / stage_2 annual) → assemble (stage_3 tabular
parquet) → analysis. The econometric design it was built for has changed, and the change invalidates
the **grid** the whole system rests on and requires a **new processing stage that does not yet exist**.

Produce a design plan for the foundational backbone of the revised system. Someone else (me) will
implement it afterwards, incrementally, with the existing pipeline kept working throughout.

---

## Research context you need in order to make good decisions

Read this section carefully; several design constraints follow from it and from nothing else.

**Outcome.** Night-time land surface temperature. **Regressor.** Night-time lights as a proxy for
local economic activity. **Panel.** Global, annual, ~1 km, roughly 1992–2023.

**Estimand.** The *total* effect of local economic activity on land surface temperature through all
channels — waste heat, growth-induced land-cover change, and growth-related pollution. Growth-induced
land-cover change is a **mediator, not a confounder**. This matters for the backbone because it means
time-varying land-cover layers (albedo, NDVI, tree cover, built-up) are needed as *mediators* in a
decomposition, not as controls in the baseline, and so must be carried through the pipeline on the
same grid and with the same neighbourhood treatment as the regressor.

**The specification the backbone must support.** Not an own-pixel regression. The estimating equation
is a *distance-ring* specification:

```
T_bt = μ_b + η_{c(b)t} + Σ_{r=0..R} β_r · L^(r)_bt + u_bt
```

where `μ_b` is a pixel fixed effect, `η_ct` a country×year effect, and `L^(r)_bt` is the **mean**
radiance over the annulus `[d_{r-1}, d_r)` around cell `b`. The parameter of interest is
`Σ_r β_r`, not `β_0`.

Why this matters for engineering: temperature responds to activity in a *neighbourhood*, not only in
the cell. With pixel fixed effects and only own-cell lights, the estimator recovers roughly
`(h/ρ)² ≈ 4%` of the true effect for 1 km cells and a 5 km spillover range — the neighbours are
differenced away along with the confounder. The ring terms put the neighbourhood back. **The
system currently has no neighbourhood computation at all** (verify: grep for `convolve`,
`annulus`, `ring_`, `neighborhood` in `src/` returns nothing relevant), so this is the principal
new capability.

**The key computational insight.** A ring mean is a *convolution*. Disc mean at radius `d` is the
field convolved with a disc kernel; an annulus is the difference of two nested discs. FFT convolution
over a raster is `O(N log N)`. Enumerating ~2,000 neighbours per cell for hundreds of millions of
cells is not tractable and must not be attempted. **All neighbourhood computation happens in raster
space, before tabularization.**

---

## The single most important finding you must design around

`src/data/assemble/constants.py` sets `DEFAULT_CRS = 4326`. The project grid is geographic
(lat/lon). **This is incompatible with the ring specification and must change.**

In EPSG:4326 a "1 km" cell is not 1 km on the ground, and its ground size varies with latitude, so a
disc of pixels is not a disc on the ground — the error is systematic in latitude, which correlates
with essentially every variable of interest. Rings are defined in kilometres; the grid must be metric
and equal-area.

**Do not silently substitute MODIS Sinusoidal as the fix.** It is equal-area, but it is severely
*sheared* away from its central meridian: meridians tilt by `arctan((λ−λ₀)·sin φ)`, and with MODIS's
global `λ₀ = 0` the shear reaches ~48° at (45°N, 90°E). A pixel disc becomes a skewed parallelogram.
Acceptable for a single-continent study; not for a worldwide one.

**Recommended starting point, which you should evaluate and may overrule with reasons:** a single
global GeoBox in **EPSG:6933** (EASE-Grid 2.0 global, Lambert cylindrical equal-area, standard
parallel 30°). It is equal-area, has **no shear**, and its distortion is a closed-form function of
latitude alone:

```
scale_EW = cos(φ_s)/cos(φ)      scale_NS = cos(φ)/cos(φ_s)      product = 1
```

so a true ground circle is an ellipse in pixel space with axis ratio `cos²(φ_s)/cos²(φ)` — 0.75 at
the equator, 1.0 at 30°, 3.0 at 60°. Because this depends only on the row index, the correction is a
**latitude-banded elliptical kernel**: precompute one kernel per narrow latitude band (bands chosen so
the ratio moves less than ~2%) and convolve band by band. Clipping the analysis to |φ| ≤ 60° caps the
anisotropy at 3:1 and is independently justified — VIIRS DNB is unusable at high latitude in summer
(no astronomical dark) and thermal LST degrades there.

The alternative is a per-tile local azimuthal equidistant CRS, where discs are exact. Evaluate it,
but note it forces per-tile CRS and therefore per-tile stores; state the trade-off explicitly.

---

## Other design decisions the plan must settle

### 1. Storage: when to write Zarr, and how to structure it

Write Zarr at points where computation is expensive **and** the output has more than one downstream
consumer. Argue for exactly two such points and specify both:

- **Analysis-ready grids** — after regridding onto the canonical GeoBox *and* after temporal
  compositing to annual. Never persist daily global data (see §3).
- **Nested disc sums and counts** — the convolution output, reused by every specification, ring
  partition, and robustness check.

Everything between (halo-padded arrays, intermediate reprojections) stays in memory / Dask graphs.

Specify, with justification:

- **One store per variable family, not one store per tile** — unless you adopt per-tile CRSs, in
  which case per-tile stores become mandatory. Make this dependency explicit; it is the same
  decision as the CRS decision.
- **Chunk shape == tile shape**, so `GeoboxTiles` gives the job partition for free. Chunk size is
  driven by `R_max`, not taste: halo overhead is `((C+2R)² − C²)/C²`, which is ~6% at C=2048 and ~25%
  at C=512 for R=30 cells. The repo already uses `DEFAULT_TILE_SIZE = 2048` and
  `DEFAULT_TILE_PADDING = 64` — check whether 64 is adequate for the chosen `R_max` and say so.
- **Parallel writes** via `to_zarr(..., region=...)` into a store initialised once with
  `compute=False`, so tile jobs write disjoint chunks.
- **Store nested disc sums + counts, not ring means.** Store `S_d` and `N_d` on a fixed radius ladder
  (e.g. 1, 2, 3, 4, 5, 7, 10, 14, 20, 30 km — widening with distance). Any ring partition is then a
  difference computed at query time, so ring edges can change without recomputing a convolution.
- **Share the count arrays.** `N_d` depends only on the validity mask, not on the variable, so store
  it once rather than once per variable. Counts fit in `uint16`. State the size saving.
- Compression and dtype choices; a sizing estimate in GB/TB for the global panel that I can check
  against disk budget **before** the expensive stage runs.

### 2. The neighbourhood (convolution) engine

This is the new subsystem. Specify its interface and where it lives in `src/`. It must:

- take a variable on the canonical grid plus a validity mask;
- convolve **both** the variable and the mask against each disc kernel (convolving the mask is what
  makes ring means correct under missing data instead of treating NaN as zero);
- use `GeoBox.buffered(R_max)` for halos — odc-geo gives this as a method, so tile padding should not
  be hand-indexed;
- handle the latitude-banded elliptical kernels from the CRS decision above;
- serve four consumers with one code path: lights, mediators, rasterized mine points ("mines within
  d km"), and the validity mask itself.

Also decide and justify: **do rings cross national borders?** They will by default, and a ring
straddling a border mixes two country×year fixed effects. Plan for computing both the unmasked and
own-country-masked variants — it costs one extra convolution against a country raster and avoids a
full re-run when a referee asks.

### 3. Ingest

Two changes from what the repo does now.

- **Lights: raw VIIRS, not the harmonized DMSP–VIIRS series.** The repo has an `ntl_harm` source.
  Harmonized series of that family convert VIIRS to DMSP-like by applying a power function for
  radiometric degradation **and a Gaussian low-pass filter for spatial degradation** — they blur the
  sharp sensor on purpose. That makes the effective resolution ~4–7 km across the whole panel, which
  is wider than the local channel the ring profile is meant to isolate, and it destroys a
  DMSP-vs-VIIRS sensor-contrast identification strategy that requires the blur radius to *differ*
  across the panel. Plan for `eog_viirs` (raw) as primary, raw DMSP for a 2012–13 overlap exercise,
  and `ntl_harm` retained only for an aggregated long-panel specification at ≥5 km cells.
  Resample lights to the grid by **area-weighted sum** (flux-conserving), never bilinear or nearest —
  bilinear would add a second blur on top of the sensor's.
- **LST: MODIS MYD21 rather than GLASS as primary.** MOD11/MYD11 assign emissivity from a *land-cover
  classification*; MOD21 retrieves it via temperature–emissivity separation. Since growth-induced
  land-cover change is part of our treatment effect, a land-cover-keyed emissivity assumption lets the
  treatment alter the outcome's measurement model. Resample the outcome as little as possible;
  prefer nearest-neighbour (repositions, does not smooth) over any averaging resampler.

Evaluate `odc-stac` + `pystac-client` against Microsoft Planetary Computer and/or LP DAAC's cloud
STAC as the acquisition path, since `odc.stac.load(items, geobox=...)` loads directly onto our
GeoBox and would replace hand-rolled reprojection. Compare against the repo's existing
`src/data/download/async_downloader.py` route and recommend one. **Verify collection availability
before recommending — do not assume a given MODIS product is cloud-hosted at daily resolution.**

**Efficiency constraint that dominates everything else here:** daily global 1 km night LST for 20+
years is multiple TB before any processing. **Composite to annual during ingest, streaming; never
land daily global data.** Persist the annual composite plus a per-cell-year **valid-observation
count** — that count is not a byproduct, it is the cloud-masking-selection diagnostic the design
requires.

### 4. Variables to carry

Plan the schema for these, on the canonical grid:

| Layer | Role | Note |
|---|---|---|
| LST night | outcome | plus QC |
| `View_Time` | control | night LST depends strongly on hours since sunset; overpass time varies across swath and drifts across the mission. Most commonly forgotten variable in this literature. |
| `View_Zenith` | control | real 1–2 K view-angle bias across MODIS's ±55° scan |
| `Emis_31`, `Emis_32` | mediator observable | arrives free with MOD21; independent land-surface signal not derived from a classification |
| valid-obs count | diagnostic | missingness by cell-year |
| VIIRS / DMSP radiance | regressor | area-weighted sum to grid |
| albedo, NDVI/EVI, tree cover, built-up | **mediators** | not baseline controls — see estimand above |
| AOD (repo has `acag`) | aerosol bound | stratify on it, do not condition |
| land-cover class | **strata only** | never a continuous mediator |
| country id, ADM2 id, land mask, mine points, leader birth region | static / identifiers | `misc`, `gadm`, `snl_mining`, `berman_mining`, `plad` sources exist |

Sub-1 km products (500 m albedo, 250 m tree cover) resample **up** by area-weighted mean, matching the
lights treatment. Never resample the 1 km outcome down to meet them.

### 5. Tabularization boundary

State the rule and where it is enforced in code: **tabularize once, at the end, after every
geometry-dependent operation, and only for cells that enter estimation.**

- Spatial **thinning** belongs here (every 3rd–5th cell). Information is bounded by the spatial
  correlation range, not pixel count, so thinning costs almost nothing in precision and saves a large
  multiple in rows. **Thin the rows, never the inputs** — a ring mean computed from a thinned raster
  is a different and wrong object. Make it structurally impossible to thin before convolution.
- The repo's existing `pixel_id` bit layout (`[ix:16 | iy:16 | local_pixel:32]`,
  `src/data/assemble/constants.py`) is the right idea — never join on floating-point coordinates.
  Assess whether it survives the CRS/grid change and whether 16 bits per tile axis suffices for the
  new global grid.
- Scale-curve and grid-shake robustness checks (block-reduce to coarser cells; `np.roll` + block-reduce
  for randomised grid origins) are **raster-space** operations. Say so explicitly, so they are never
  implemented as table operations.
- Do not materialise pixel dummies. Absorb the cell effect by within-cell demeaning; country×year
  remains a small dummy set. The repo already has `demean_columns` in the assemble config — check
  whether it is doing the right thing for the new design.

---

## What to read first

- `README.md` — note that it describes the *old* design (500 m grid, own-pixel two-way FE, GLASS LST,
  harmonized lights). Treat it as the "before" picture.
- `run.py` and `src/cli/` — the `download | preprocess | assemble | analysis` command surface.
- `src/data/common/geobox/` — existing GeoBox construction, including a monkeypatch
  (`geobox_patch.py`) for an odc-geo bug. Determine whether the patch is still needed on the pinned
  version.
- `src/data/assemble/{constants,tiles,geometry,workflow,processors}.py` — current tiling,
  `pixel_id`, and the parquet assembly.
- `src/data/preprocess/sources/` — the per-source preprocessors (`glass`, `eog`, `ntl_harm`,
  `esacci`, `acag`, `berman_mining`, `snl_mining`, `plad`, `misc`).
- `orchestration/configs/data.yaml` (+ `data.local.yaml`) — source and assembly configuration; the
  stage_1 / stage_2 / stage_3 path conventions.
- `orchestration/slurm/` — the HPC execution pattern the new stage must fit.

---

## Hard constraints

1. **Plan only. Write no implementation code.** Illustrative snippets of ≤10 lines are fine inside the
   design docs.
2. **The existing pipeline must keep running throughout the migration.** Propose the new grid and
   stage as additive, with an explicit cutover, not a rewrite in place. Name what breaks and when.
3. **Reuse the existing idioms** — config-driven sources in `data.yaml`, the stage_N path convention,
   the `run.py` subcommand surface, `GeoboxTiles`, SLURM scripts per stage. Do not invent a parallel
   architecture alongside them.
4. **Do not touch `src/analysis/`** in this task beyond noting the interface it consumes.
5. **Flag anything you cannot verify from the repo or from a primary source** rather than asserting it.
   Specifically: the EASE-Grid 2.0 extent constants (take them from the NSIDC grid definition, not
   from memory), which MODIS collections a given STAC host actually carries at daily resolution, and
   whether the GLASS product in use is clear-sky or all-weather gap-filled. There is no
   `environment.yml`/`pyproject.toml` at the repo root despite the README's install instructions —
   determine how the environment is actually specified and flag it if it is missing.

---

## Deliverables

Create these files and nothing else:

1. **`docs/design/00-backbone-overview.md`** — the argument in ≤2 pages: what changes, why, and the
   order of work. Lead with the CRS finding and its consequences.
2. **`docs/design/01-grid.md`** — canonical GeoBox definition, CRS decision with the alternatives
   evaluated and rejected in writing, tiling scheme, `pixel_id` scheme, latitude-band kernel registry,
   |φ| clipping decision, and the antimeridian/pole edge cases.
3. **`docs/design/02-storage.md`** — Zarr layout, chunking, dtypes, compression, region-write
   concurrency, the disc-ladder schema, and a **sizing table in GB/TB** for the global panel.
4. **`docs/design/03-neighbourhood-engine.md`** — module placement, public interface, kernel
   construction, halo strategy, missing-data handling, cross-border decision, and the four consumers.
5. **`docs/design/04-ingest.md`** — per-source acquisition and regridding plan including resampling
   method *per variable* with justification, the annual-compositing-at-ingest rule, and the STAC vs
   existing-downloader recommendation.
6. **`docs/design/05-migration.md`** — additive rollout: what is built first, what runs in parallel
   with the old grid, the cutover criteria, and what is retired. Include a dependency-ordered task
   list with rough effort sizing.
7. **`docs/design/06-open-questions.md`** — everything you could not resolve, each with the specific
   check that would resolve it and who/what is needed to run that check.

Each document should state **decisions with reasons**, and record rejected alternatives — I need to be
able to reconstruct why, six months from now, without re-deriving it.

---

## Before you write

Ask me about anything genuinely ambiguous whose answer would change the design — in particular the
final `R_max`, the target resolution (the README says 500 m; the current design discussion assumes
1 km, and the LST product is 1 km native), and the available disk and compute budget. Do not ask
about things you can determine by reading the repository.
