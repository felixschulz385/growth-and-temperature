# 04 — Ingest & Tabularization Boundary

## 1. Lights

**Decision: `eog_viirs` (raw VIIRS DNB) primary, raw DMSP for a 2012–13 overlap exercise,
`ntl_harm` (harmonized DMSP–VIIRS) retained only as input to a separate ≥5 km aggregated long-panel
specification** — not the primary 1 km ring-model panel.

Why raw VIIRS, not harmonized: the harmonized series in this family converts VIIRS to DMSP-like via
a power function (radiometric) plus a **Gaussian low-pass spatial filter** — a deliberate blur that
gives an effective resolution of ~4–7 km across the whole panel. That's wider than the local channel
the ring profile is meant to isolate, and it would destroy a DMSP-vs-VIIRS sensor-contrast
identification strategy that needs the blur radius to *differ* across the panel. None of this
requires new download-subsystem work — `eog.py` and the existing plugin-style
`src/data/download/sources/registry.py` already support `eog_viirs` and DMSP as registered sources.

**Resampling change — the one thing that must actually change in code.** Confirmed via direct
inspection: the shared reprojection helper, `SpatialProcessor.process_spatial_standard`
(`src/data/preprocess/common/spatial.py:232`), hardcodes `resampling="nearest"`. This is the code
path EOG/lights currently shares with `ntl_harm`, ESACCI, and ACAG. Nearest-neighbour resampling of a
radiance field is not flux-conserving and is exactly the wrong choice for a variable whose *ring
sums* are meant to represent locally-integrated economic activity — nearest-neighbour injects
aliasing noise that would be baked into every downstream ring sum.

**Decision: lights must resample by area-weighted sum (flux-conserving); this must be a
per-variable override, not a global default flip.** ESACCI (land cover, categorical) in particular
must stay nearest/mode-appropriate — flipping the shared default for all four consumers of
`process_spatial_standard` would silently break land-cover categorical resampling. This needs a
per-variable resampling-method parameter threaded through `SpatialProcessor`, e.g. a
`resampling: {variable: method}` override surfaced from `data.yaml`, not a constant edit.

**Flag:** it is not fully verified from static inspection alone whether `eog.py`'s own timeseries
reprojection call site actually routes through `process_spatial_standard`, or a separate path — trace
this with a runtime call before editing, rather than assuming.

## 2. LST

**Resampling principle (stated here, applies regardless of which product is chosen): prefer
nearest-neighbour over any averaging resampler.** LST is an intensive quantity (not additive/flux-like
the way radiance is) — area-weighted averaging blends physically distinct temperatures the way it
correctly aggregates flux for lights, but for LST that blending is not obviously the right operation.
Nearest-neighbour repositions without smoothing, preserving the interpretation of each output cell as
"the temperature at approximately this location" rather than a spatial blend.

**Product decision (GLASS vs. MODIS MYD21/11A1) — deliberately deferred, not resolved here.** A
companion prompt already present in this repository, `PLAN_PROMPT_modis_ingest.md`, is a dedicated
follow-up task that plans the MODIS LST ingest from Microsoft Planetary Computer's `modis-11A1-061`
collection in detail — verified collection facts (asset list, hosting region, signing requirements,
temporal extent), the 11A1-vs-11A2-vs-21A2 trade-off (including the emissivity-provenance argument:
11A1's `Emis_31`/`Emis_32` are land-cover classification lookups and cannot serve as an independent
mediator observable, while 21A2's temperature–emissivity-separation retrieval can), band-level
scale/offset/fill-value handling, QC bit unpacking, and operational design (compute-node network
access, token refresh, resumability). That prompt is explicitly written to read and conform to these
backbone documents once they exist. **This document does not duplicate or pre-empt that decision** —
see [`06-open-questions.md`](06-open-questions.md) item 3 for the cross-reference.

What this document does settle, because it's a backbone-level concern regardless of which LST
product wins: any LST source must (a) resample with nearest-neighbour as above, (b) composite to
annual at ingest time with a persisted valid-observation count (§4), and (c) land on the canonical
EPSG:6933 GeoBox before entering the neighbourhood engine, like every other variable.

## 3. STAC vs. the existing download subsystem

**Decision: evaluate and, if MODIS is adopted, recommend `odc-stac` + `pystac-client` as an
additive, source-specific ingest path — not a replacement for the existing download subsystem.**

The existing `AsyncHPCDownloader`/plugin registry (`src/data/download/`: index → poll → download
(aiohttp, semaphore-bounded) → tar → rsync-to-HPC → extract → verify → update-parquet-index) is
generic, already integrated with the HPC deployment target, and already works for every other source
in the repo (EOG, DMSP, ESACCI, ACAG, GLASS, mining sources). There is no argument for replacing it
broadly. STAC's specific value — spatial/temporal query filtering over a cloud-hosted per-scene/
per-day catalog — is a genuinely different access pattern from "download a versioned bulk archive and
extract it," which is what every current source looks like. Introducing STAC as one more entry in the
existing plugin-style `sources/registry.py` (wrapping `pystac-client`/`odc-stac` internally, still
funneling output through the same stage_1/stage_2/stage_3 conventions) is lower-risk than either
forcing MODIS through the bulk-archive pattern it doesn't fit, or migrating every source to STAC.

**Confirmed via repo-wide grep: zero existing use of `odc-stac`/`pystac-client`/STAC/Planetary
Computer/LP DAAC anywhere in `src/`** — this is a new dependency, not an extension of something
partially built. Budget for new-library risk (auth/rate-limit handling, new failure modes)
accordingly. Full verification of which MODIS collections are cloud-hosted at daily resolution, on
which host, under what terms, is explicitly the companion prompt's job — see
[`06-open-questions.md`](06-open-questions.md) item 3.

## 4. Annual compositing at ingest, streaming

**Decision: composite to annual (or the chosen temporal unit) at ingest/streaming time, never land
daily global data, and always emit a persisted valid-observation count alongside the composite.**

This mirrors the `S_d`/`N_d` pattern one level earlier in the pipeline: temporal compositing has
exactly the same missing-data structure problem as spatial convolution — a naive temporal mean over
days with different cloud/missing patterns is biased in a way correlated with seasonal cloud
climatology, which itself correlates with development. The valid-observation count is a required
diagnostic output, not an internal to discard.

**Build on existing prior art, don't invent from scratch.** GLASS's preprocessor already does
daily→annual/monthly compositing via `xr.Dataset.resample` (`src/data/preprocess/sources/glass.py`).
Generalize that pattern (or extract it into a shared helper) so any source needing temporal
compositing follows the same shape, and so the companion MODIS ingest plan can adopt it directly
rather than reinventing it.

## 5. Variables to carry (schema, on the canonical grid)

| Layer | Role | Note |
|---|---|---|
| LST night | outcome | plus QC; see §2 for the deferred product decision |
| `View_Time` | control | night LST depends strongly on hours since sunset; overpass time varies across swath and drifts across the mission |
| `View_Zenith` | control | real 1–2 K view-angle bias across MODIS's ±55° scan |
| `Emis_31`, `Emis_32` | mediator observable (product-dependent) | only a genuine independent land-surface signal if sourced from TES retrieval (MOD21/MYD21), not a land-cover-classification lookup (MOD11/MYD11) — see §2 |
| valid-obs count | diagnostic | first-class output of §4, not a byproduct |
| VIIRS / DMSP radiance | regressor | area-weighted sum to grid, §1 |
| albedo, NDVI/EVI, tree cover, built-up | **mediators** | not baseline controls — carried through the same neighbourhood treatment as the regressor ([`03-neighbourhood-engine.md`](03-neighbourhood-engine.md) §4) |
| AOD (`acag` source) | aerosol bound | stratify on it, do not condition on it in the baseline specification |
| land-cover class | **strata only** | never a continuous mediator |
| country id, ADM2 id, land mask, mine points, leader birth region | static/identifiers | existing `misc`, `gadm`, `snl_mining`, `berman_mining`, `plad` sources already produce these |

Sub-1 km products (500 m albedo, 250 m tree cover) resample **up** by area-weighted mean, matching
the lights treatment (§1) — never resample the 1 km outcome (LST) down to meet them.

## 6. Tabularization boundary

**Decision: tabularize exactly once, at the very end of the raster pipeline, after every raster-space
geometry operation (reprojection, temporal compositing, disc convolution) is complete, and only for
cells that will actually enter estimation** (after the land/validity mask and the |φ|≤60° clip are
applied). This follows directly from the key computational constraint in
[`00-backbone-overview.md`](00-backbone-overview.md): neighbourhood computation must happen in raster
space because tabular enumeration of ~2,000 neighbours/cell for hundreds of millions of cells is
intractable — so tabularization must be the *last* step, consuming already-convolved `S_d`/`N_d`
arrays, never an early step convolution has to work around.

**Thinning is row-level, strictly post-convolution — never pre-convolution.** Thinning (e.g.
subsampling every 3rd–5th pixel for a robustness check on effective sample size) changes *which
rows* enter the regression, not the neighbourhood field itself. A ring mean computed from a thinned
raster is a different, wrong object — the retained pixel's ring mean must still reflect its true,
full-resolution neighbourhood, including any thinned-out neighbours. Making thinning purely a
row-level, post-tabularization operation makes it structurally impossible to thin before convolution
by construction, and makes it cheap to test several thinning schemes against one already-convolved
output.

**`pixel_id` survives** ([`01-grid.md`](01-grid.md) §5) — tabularization keys rows by the same
`make_pixel_ids` construction as today, rebuilt against the new canonical GeoBox's tiling.

**Grid-shake / block-reduce robustness checks are raster-space operations**, for the same reason
thinning must be row-level — they change a pixel's spatial support, which changes what "the
neighbourhood" means for it, so they must run before tabularization, not as a table operation.
`src/data/assemble/utils.py`'s `DEFAULT_DERIVED_PIXEL_ID_RESOLUTIONS` and
`normalize_derived_pixel_id_specs`/`build_derived_pixel_id_mapping`/`add_derived_pixel_id_columns`
are the closest existing analog — a real, working mechanism in this repo for keying rows at multiple
derived resolutions off one canonical grid. **Caveat:** it is not literally block-reduce/`np.roll`
grid-shake (it remaps pixel ids to a coarser nominal resolution for join/dedup purposes, it doesn't
implement spatial aggregation or grid-origin-shift). Treat it as a pattern to imitate for *interface
design* (how derived-resolution columns get named/joined), not a component to reuse for the
robustness-check computation itself, which still needs to be built. Its hardcoded degree-based
resolutions (`"500m": 0.00417`, etc.) will also need a metric-CRS equivalent once the canonical grid
is EPSG:6933.

> **Update (assembly rework).** Grid-shake is now implemented in the assembly stage as a *whole-run
> origin shift*, applied once to the target GeoBox (`grid_shake.shift_geobox_origin`) before tiling —
> not as per-column `{var}__shake_N` reprojections. Each `assemble create --grid <coarse> --shake
> <preset>` variant is a full, identical-schema pass written to its own sibling table under
> `<output_root>/grid=<label>/shake=<base|s0|s1|…>/`. `shake=base` is always written; extra offsets
> are independent add-on passes/jobs that never touch `base`. It stays gated on downsampling (a
> native-resolution `--grid 1km` run ignores `--shake`).

**`demean_columns`/`assemble demean`: confirmed dead/broken configuration, do not resurrect as-is.**
`orchestration/configs/data.yaml` references `processing.demean_columns`, and the CLI has a `assemble
demean` subcommand wired to call `src.data.assemble.demean.run_workflow_with_config` — but
`src/data/assemble/demean.py` does not exist. Calling it today raises `ModuleNotFoundError`; there is
no working within-cell-demeaning implementation to preserve or extend. **Recommendation: do not
implement a literal "demean columns" primitive in the assembly stage.** Pixel-FE (`μ_b`) and
country×year-FE (`η_{c(b)t}`) absorption belongs to the estimation step — a proper high-dimensional-FE
within-transform applied jointly to the ring-mean columns and the FE structure — which is an
`src/analysis/` concern, out of scope for this backbone design per its hard constraints. **This
document specifies only the handoff interface**: tabularized rows keyed by `pixel_id` + `year`,
country id, and ring-mean columns `L^(1)_bt ... L^(R)_bt` with their `S_d`/`N_d` provenance available
for weighting/diagnostics.
