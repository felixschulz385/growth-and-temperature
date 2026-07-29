# 07 — MODIS LST Ingest

Replaces GLASS as the project's primary night-time LST source. This document assumes the backbone
grid/storage decisions in [`01-grid.md`](01-grid.md)/[`02-storage.md`](02-storage.md) are settled and
does not re-argue them; it also assumes and cites [`04-ingest.md`](04-ingest.md)'s backbone-level
ingest principles (nearest-neighbour resampling for LST, annual compositing at ingest with a
persisted valid-observation count, landing on the canonical EPSG:6933 GeoBox) rather than
re-deriving them. Band-level facts are in the companion
[`07a-modis-band-reference.md`](07a-modis-band-reference.md); this document cites it rather than
duplicating it.

This is **not a downloader**. Daily global 1 km night LST for the full Aqua record is multiple TB
before any processing; the correct object is a streaming ingest-and-composite stage that reads COGs
remotely, reduces to annual composites in flight (in native sinusoidal projection), and persists only
the composite plus its observation-count diagnostics.

## 1. Product decision: MYD21A2 primary, MYD11A1 robustness arm, MYD11A2 rejected

**Decision: `modis-21A2-061`, Aqua platform, night band, as the primary source. `modis-11A1-061`
(Aqua, night) as a mandatory but bounded-scope robustness arm. `modis-11A2-061` evaluated and
rejected — it is dominated by 11A1 for this project's purposes (see below).**

This differs from a literal reading of the originating brief, which defaulted toward 11A1-primary
with a 21A2 robustness arm "if 11A1 is chosen anyway." The brief explicitly invited overturning that
default on the merits; the STAC verification done for
[`07a-modis-band-reference.md`](07a-modis-band-reference.md) this session supports doing so.

**Why 21A2 over 11A1 — the estimand argument, now source-confirmed rather than asserted.** Both
products' emissivity provenance were verified directly against primary sources (not inferred):
MOD11/MYD11's `Emis_31`/`Emis_32` are a **land-cover-classification lookup**
(Snyder & Wan 1998, keyed to MOD12Q1 land cover — quoted verbatim in
[`07a-modis-band-reference.md`](07a-modis-band-reference.md)); MOD21/MYD21's `Emis_29/31/32` are a
genuine **TES (temperature-emissivity separation) physical retrieval**, independent of any land-cover
classification. Since growth-induced land-cover change is a mediator in this study's estimand
([`00-backbone-overview.md`](00-backbone-overview.md)), a classification-keyed emissivity lets the
treatment (land-cover change) silently alter the outcome's own measurement model — 11A1's emissivity
cannot serve as a mediator observable at all, only 21A2's can. This is the estimand-critical argument
and it dominates the choice.

**Why 21A2 over 11A1 — the operational argument.** 21A2 is an 8-day composite: roughly 46
composite-periods/year versus 365 daily scenes, an ≈8× reduction in COG reads. Illustrative arithmetic
(assumptions stated so it's checkable, tile count `~317` **UNVERIFIED** per
[`07a-modis-band-reference.md`](07a-modis-band-reference.md), Aqua-era span **illustrative** ~23 years
2002–2025 pending exact Aqua night-overpass start date, 7 assets/unit: `LST_Night_1KM`, `QC_Night`,
`Emis_29/31/32`, `View_Angle_Night`, `View_Time_Night`):

| Product | Units (tile × period) | Assets/unit | Total reads (illustrative) |
|---|---|---|---|
| MYD21A2 (8-day) | 317 × 23 × 46 ≈ 335,000 | 7 | **≈ 2.3 million** |
| MYD11A1 (daily) | 317 × 23 × 365 ≈ 2,660,000 | 7 | **≈ 18.6 million** |

An ≈8× reduction directly reduces the operational risk the original brief flagged as the dominant
engineering concern: SAS tokens expire in ~1 h, and a job an order of magnitude longer multiplies the
number of re-signing windows, retry surfaces, and rate-limit exposure. This tips the balance decisively
once the estimand argument is already settled in 21A2's favour — there is no scenario where accepting
21A2's few real costs (below) but paying 8× the operational risk for 11A1 would have been the better
trade.

**Real costs of 21A2, stated plainly, not hidden:**
1. **No native observation-count diagnostic.** Confirmed via STAC: 21A2 carries no
   `Clear_sky_days`/`Clear_sky_nights`-equivalent asset (11A2 does). §4 below specifies deriving a
   coarser-but-real diagnostic from `QC_Night` at 8-day-period granularity instead.
2. **Emissivity is day+night pooled**, not night-specific (confirmed via primary source in
   [`07a-modis-band-reference.md`](07a-modis-band-reference.md)) — acceptable since emissivity here is
   only ever a mediator observable, never fed back into the night-LST retrieval this pipeline consumes.
3. **8-day, not daily, temporal resolution** for the primary series — mitigated by the robustness arm.

**Why the robustness arm is still mandatory, and why it is bounded, not a full parallel backfill.**
The original brief is right that a same-methodology comparison is needed regardless of which product
wins, both to sanity-check the coarser 21A2 diagnostic against 11A1's true daily counts and to give a
literature-standard comparison point (11A1/MOD11 is the product most published LST-economics work
uses). Running it at full scope (23 years × 317 tiles) would reintroduce the exact 18.6M-read
operational risk 21A2 was chosen to avoid, for a check that doesn't need global/full-record coverage
to be informative. **Decision: run MYD11A1 on a bounded subset** — a handful of tiles spanning
different biomes/climates and cloud regimes (e.g. one tropical, one temperate, one arid tile) × 3–5
years spanning early/mid/late mission (to catch any sensor-degradation drift), sufficient to validate
(a) that 21A2's period-level valid-observation diagnostic tracks 11A1's true daily counts, and (b) that
annual composite values from the two products are consistent net of the known emissivity-provenance
difference. Exact tile/year selection is an implementation-time call once the tile list (§5) exists.

**Why 11A2 is rejected, not merely deprioritized.** 11A2 offers the same ≈8× read reduction as 21A2
**and** keeps the native `Clear_sky_days`/`Clear_sky_nights` diagnostic — but its emissivity is the
same classification lookup as 11A1, so it buys none of the estimand-critical benefit. Against 11A1 it
trades daily granularity (useful for the robustness arm's drift-detection purpose) for nothing this
project needs. There is no scenario in this design where 11A2 is preferred over both 21A2 (better
estimand fit) and 11A1 (better robustness-arm granularity) simultaneously — it is strictly dominated.

## 2. Architecture: composite in native sinusoidal, reproject once

Per [`04-ingest.md`](04-ingest.md) §2/§4 (already settled, not re-argued here): **Stage "annual"**
streams daily/8-day night assets per MODIS sinusoidal tile, applies QC, and reduces to an annual
composite **in native sinusoidal projection**; **Stage "spatial"** reprojects the annual composites
onto the canonical **EPSG:6933** GeoBox ([`01-grid.md`](01-grid.md)), following the additive
`stage_2_ease6933` path convention from [`05-migration.md`](05-migration.md) §1. Mirror
`ACAGPreprocessor`'s two-stage shape (`src/data/preprocess/sources/acag.py`) — cleanest existing
instance of this pattern — over `GlassPreprocessor`'s (`src/data/preprocess/sources/glass.py`), which
carries MODIS/AVHRR dual-source complexity this ingest doesn't need, though GLASS's compositing
implementation is still the prior art to generalize (§4).

**One architectural wrinkle specific to MYD21A2, not present in the original brief:** the primary
product is itself an *8-day composite of composites* — LP DAAC has already averaged clear daily scenes
within each 8-day window before we ever read a pixel (verified in
[`07a-modis-band-reference.md`](07a-modis-band-reference.md)). Stage "annual" is therefore compositing
21A2's 8-day composites into months into a year, not raw daily scenes into months into a year. §4
addresses the weighting this implies.

**Resolves the "compute nodes may lack outbound internet" blocking question by design, not by
investigation.** Stage "annual" requires internet egress to Planetary Computer (`westeurope`-hosted);
whether scicore's SLURM compute nodes have that access was flagged in the originating brief as
unknown and unverifiable from the repo, and remains unverified. Rather than block this design on that
fact, stage "annual" is designed to run on **whatever host has internet egress** — a scicore
login/transfer node, a workstation, or a cloud VM — and its output is pushed to scicore via the new
generic transfer capability ([`08-hpc-transfer.md`](08-hpc-transfer.md)) before stage "spatial" runs
there as an ordinary local SLURM job, unchanged in kind from how GLASS/ACAG work today. If compute-node
internet access is later confirmed, stage "annual" can also run there directly via a conventional
SLURM script with no design change — the transfer step becomes optional, not obsolete.

## 3. Tile list and the |φ|≤60° clip

**Decision: restrict the sinusoidal tile list to tiles intersecting |φ|≤60°, computed once at
build/init time (mirroring [`01-grid.md`](01-grid.md) §2's "compute, don't hardcode" convention),
rather than ingesting and later discarding high-latitude tiles.** This shrinks the ~317-land-tile
figure (itself **UNVERIFIED**, per [`07a-modis-band-reference.md`](07a-modis-band-reference.md)) by an
unquantified amount — Scandinavia, Alaska, northern Canada, Siberia, and Antarctica tiles drop out
entirely or partially. Exact count is an implementation-time computation, not asserted here; log it in
[`06-open-questions.md`](06-open-questions.md) alongside the existing clip-impact item.

Antimeridian handling for the sinusoidal→EPSG:6933 reprojection is **not** the vector/EPSG:4326
antimeridian problem [`01-grid.md`](01-grid.md) §7 describes (that logic covers vector inputs like
GADM boundaries pre-reprojection) — sinusoidal tiles are indexed by an integer `h`/`v` grid, not
degrees, so a tile spanning what would be ±180° in geographic coordinates is ordinary tile-boundary
mosaicking, not a coordinate discontinuity. No new antimeridian logic is needed beyond standard
multi-tile reprojection.

## 4. Compositing definition

**Decision: month-first, then annual, per [`04-ingest.md`](04-ingest.md)'s backbone rule — but
computed as a composite-of-composites for MYD21A2, which changes the weighting, not the principle.**

Concretely:
- Each 21A2 8-day period is assigned to the calendar month containing its **period start date**
  (a simple, stated rule — splitting a period's weight across a month boundary is not worth the
  complexity for an already-8-day-smoothed input; flag as a minor implementation choice, not a
  strongly-justified one).
- **Monthly composite = mean of that month's 8-day composite values, weighted by each period's
  `QC_Night`-valid pixel status** (a period contributes to the monthly mean only where its own
  `QC_Night` passes the configured threshold — see §5). This avoids compounding LP DAAC's own
  already-applied 8-day averaging with a second, differently-weighted averaging step that would double
  count or under-count partially-valid periods.
- **Annual composite = mean of that year's monthly composites**, over months with at least one valid
  contributing period. This is the same principle GLASS's existing prior art gets wrong in one
  respect worth calling out explicitly: `GlassPreprocessor._calculate_statistics`
  (`src/data/preprocess/sources/glass.py:740-806`) computes `annual_stats` via
  `rechunked[VARIABLE_NAME].resample(time="1YE").mean()` **directly from daily data**, not from its own
  `monthly_stats` — i.e. GLASS's current annual figure is exactly the naive equal-weighted annual mean
  the backbone design ([`00-backbone-overview.md`](00-backbone-overview.md)) warns is biased toward
  dry-season conditions in seasonally-cloudy regions. **Do not copy that pattern for MODIS.** Do copy
  its valid-count mechanism — `mask.resample(time="1YE").sum()` (`glass.py:785`) and the equivalent
  monthly line (`glass.py:793`) — which is the right shape for §5's diagnostic, just needs to key off
  `QC_Night` validity instead of GLASS's own mask.
- **Zero-valid-months propagate as excluded, not zero-filled**: a month with no valid contributing
  period is `NaN` in `monthly_mean` and is excluded from the annual mean's denominator (mean over
  available months, not over 12). The per-year count of valid months is itself persisted (§5) so this
  choice is auditable downstream, not silently baked in.
- **Mean, not median.** Since the 8-day inputs are themselves already period-means, a median-of-means
  is a different and harder-to-interpret statistical object than a mean-of-means; use mean at both the
  monthly and annual step for a single, traceable weighting scheme.

**Recommended generalization target for the shared compositing helper `04-ingest.md` §4 calls for**:
extract a `composite_to_annual(daily_or_periodic_da, valid_mask_da, freq="1YE")`-shaped helper (the
month-first, weighted-by-validity logic above) into `src/data/preprocess/common/` — GLASS's resample
calls are the pattern to imitate for xarray/Dask mechanics, not the annual-from-daily shortcut to
reuse as-is.

## 5. Persisted diagnostics: valid-observation count without a native daily count

Per [`04-ingest.md`](04-ingest.md) §4/§5, the valid-observation count is a required diagnostic output,
not an internal. For MYD21A2 (no native `Clear_sky_*` asset — §1, confirmed in
[`07a-modis-band-reference.md`](07a-modis-band-reference.md)):

- **Persist, per cell: annual mean, per-month means, per-month valid-*period*-count (0–≈4, since
  ~46 periods/year ÷ 12 months ≈ 3.8 periods/month), and annual valid-*period*-count (0–≈46).** This is
  a genuine, well-defined diagnostic — just at 8-day-period granularity rather than daily granularity,
  a real precision cost of the product choice (§1), quantifiable by the robustness arm rather than
  hand-waved.
- Period validity itself is derived from `QC_Night`, thresholded per §6's QC rule (currently
  unresolved — see below and [`06-open-questions.md`](06-open-questions.md)).
- The robustness arm (§1) should explicitly compare its true 11A1 daily valid-count against the
  implied `valid_periods × 8`-day proxy on the same tiles/years, to characterize how much information
  the coarser diagnostic loses — this is a concrete, checkable output of the robustness comparison, not
  just a sanity check on LST values themselves.

## 6. Correctness details

Per-band scale/offset/fill/QC handling is in
[`07a-modis-band-reference.md`](07a-modis-band-reference.md); this section covers cross-cutting rules.

- **Platform filter.** Confirmed via STAC: all three collections mix `aqua` and `terra` in
  `summaries.platform` — there is no separate MYD-only collection (matches the original brief's
  warning). Filter on `properties.platform == "aqua"` **and** cross-check against the `MYD` id prefix
  on a sample before trusting either signal alone (**UNVERIFIED** which is authoritative — flagged in
  [`07a-modis-band-reference.md`](07a-modis-band-reference.md)). This is the highest-consequence bug
  available in this component per the original brief; get the cross-check into a unit test against
  real STAC items before any large run.
- **QC threshold.** `QC_Night`'s bit layout is **not verified from a primary source this session**
  for the L3 gridded product (§ full detail and caveat in
  [`07a-modis-band-reference.md`](07a-modis-band-reference.md)) — implement the threshold as a
  **configurable** parameter (e.g. `qc_max_lst_error_k: 2` in the source's `data.yaml` block) so a
  wrong assumed bit layout is a one-line config fix, not a silent wrong mask baked into a completed
  ingest run. Do not hardcode a threshold until the bit layout is confirmed.
- **Fill values.** LST fill is 0 (confirmed, not Kelvin-zero — must be masked before any arithmetic).
  Emissivity fill is 0 (confirmed for both families). See
  [`07a-modis-band-reference.md`](07a-modis-band-reference.md) for the full per-band table, including
  the flagged-unverified MOD11 emissivity offset.
- **`odc.stac.load` scale/offset application.** **UNVERIFIED this session** whether `odc.stac.load`
  applies STAC `raster:bands` scale/offset automatically or whether the pipeline must apply
  `raw × scale + offset` manually — confirm against the installed `odc-stac` version's behaviour with a
  real read before writing the ingest loop; get this wrong and every band is silently off by its scale
  factor. Logged in [`06-open-questions.md`](06-open-questions.md).
- **Resampling, per variable, on stage "spatial"'s reprojection onto EPSG:6933:**

  | Variable | Resampling | Why |
  |---|---|---|
  | `LST_Night_1KM`/`1km` | nearest | Per [`04-ingest.md`](04-ingest.md) §2 backbone rule — LST is intensive, not additive; nearest repositions without blending physically distinct temperatures. |
  | `QC_Night` | nearest | Mandatory for flag/categorical bands — averaging a bit field is meaningless. |
  | `Emis_29/31/32` | nearest | Same intensive-quantity reasoning as LST; emissivity is a per-surface physical property, not a flux to conserve under averaging. |
  | `View_Time_Night`, `View_Angle_Night` (also carried as controls per [`04-ingest.md`](04-ingest.md) §5) | nearest | Same reasoning — these are point-in-time/point-in-scan-geometry readings, not additive quantities. |
  | valid-period-count (§5) | nearest | At this stage the field is being *repositioned* onto a ~1 km grid at essentially the same resolution, not spatially aggregated — aggregation of counts happens later, in the neighbourhood engine's own convolution ([`02-storage.md`](02-storage.md) §4-5), which is a different operation on the already-canonical-grid array. Do not average counts here. |

  This is the concrete fix [`04-ingest.md`](04-ingest.md) §1 already flags as necessary: today's
  shared `SpatialProcessor.process_spatial_standard` (`src/data/preprocess/common/spatial.py:232`)
  **hardcodes `resampling="nearest"`** — convenient for MODIS (every variable above wants nearest
  anyway), but MODIS's per-variable resampling override, once threaded through for lights' area-
  weighted-sum need, must not accidentally flip MODIS's own bands away from nearest by sharing a
  careless default.
- **Antimeridian/poles.** Covered in §3 — moot for this ingest's own reprojection step.

## 7. Operational requirements

- **Token refresh.** SAS tokens expire ~1 h (confirmed in the original brief, not re-verified this
  session — no expiry duration is stated in the STAC collection metadata itself). Re-sign
  (`planetary_computer.sign_inplace`) on a fixed interval meaningfully shorter than 1 h (e.g. every
  40 min) inside the long-running worker loop, not once at job start.
- **Concurrency and backoff.** Reads cross a network boundary to `westeurope`-hosted blob storage with
  unpublished rate limits (**UNVERIFIED** — Planetary Computer does not document exact req/s limits;
  behaviour must be inferred from observed 429/503 responses). Use exponential backoff with jitter,
  distinguishing 429/503 (retry) from 4xx auth/not-found (fail fast). Note this is a **different
  concurrency model** from the existing download subsystem's asyncio-semaphore-bounded
  `AsyncHPCDownloader` (`src/data/download/async_downloader.py`) — `odc.stac.load` with a Dask backend
  parallelizes via the Dask graph/worker pool, not asyncio; tune SLURM/worker Dask thread counts (if
  stage "annual" ever runs under SLURM, per §2) as the actual concurrency control, not a separate
  semaphore layered on top.
- **Resumability.** A manifest with per-tile-year (or per-tile/period, given §5's granularity) status,
  so failures retry individually and a killed job resumes. Reuse the existing parquet/sqlite
  `UnifiedDataIndex` machinery (`src/data/common/index/unified_index.py`) already used for the download
  subsystem's completed-file tracking, rather than inventing a second manifest format — this is also
  the transfer-manifest recommendation in [`08-hpc-transfer.md`](08-hpc-transfer.md), so the two share
  one index technology.
- **Zarr `region=` writes for idempotency.** Confirmed pattern already in use elsewhere in this repo
  (e.g. `spatial.py:248`'s `to_zarr(..., region='auto', align_chunks=True, zarr_format=3)`) — stage
  "annual" should write each tile-year's composite as its own region write into a store initialized
  once with `compute=False`, per [`02-storage.md`](02-storage.md) §3's parallel-write pattern, so a
  retried unit doesn't require rewriting the whole store.
- **A dev cache.** Land a small subset — 2–3 tiles × 2 years of MYD21A2 night assets — to local disk
  under `scratch_nobackup/modis_dev_cache/` (matching this repo's existing `scratch_nobackup/`
  convention) so development and tests don't hit the network or consume a SAS token.
- **Execution scripts.** Two scripts, not one, reflecting §2's resolution of the compute-node-internet
  question:
  - `orchestration/scripts/modis-ingest-annual.sh` (**not** under `orchestration/slurm/` by default,
    since it's designed to run on whichever host has internet egress, which may not be a SLURM compute
    node) — runs stage "annual", then `run.py preprocess transfer --source modis --stage annual`
    ([`08-hpc-transfer.md`](08-hpc-transfer.md)) to push results to scicore. If compute-node internet
    access is later confirmed, this can be resubmitted as a conventional SLURM script
    (`orchestration/slurm/modis-preprocess-annual.sh`, mirroring
    `glass-modis-preprocess-annual.sh`) with the transfer step simply skipped.
  - `orchestration/slurm/modis-preprocess-spatial.sh` — mirrors
    `glass-modis-preprocess-spatial.sh` exactly (same conda-env activation, memory-limit calculation,
    `run.py preprocess run --stage spatial` invocation pattern); its only new dependency is checking
    that the expected local annual zarrs are present (i.e. the transfer completed), an operational
    manifest check rather than a code dependency.

  Note for whoever implements this: the existing `glass-modis-preprocess-{annual,spatial}.sh` scripts
  invoke `run.py preprocess --config ... --source glass_modis --stage annual` — **without** the `run`
  subcommand `src/cli/preprocess/commands.py` currently requires (`preprocess_cmd` is a required
  subparser and only `run` is registered). This looks like the scripts predate the CLI's modularization
  into `preprocess run`/`preprocess transfer`-style subcommands and may already be broken as committed
  — worth a quick check before treating them as a working template. Logged in
  [`06-open-questions.md`](06-open-questions.md).

## 8. Cost/time estimate

Illustrative only, per the read-volume table in §1 (≈2.3M reads for the full MYD21A2 backfill,
≈2.3M×(bounded-subset fraction, small) for the robustness arm) — **not a measured throughput number**.
At an illustrative sustained rate of 20–50 concurrent COG reads and ~0.3–1 s/read over the
Basel↔westeurope network path (both figures unmeasured), a full historical backfill is order
**hours to low single-digit days**, not weeks — a direct consequence of the ≈8× read reduction from
choosing 21A2 over 11A1. An incremental annual update (one year, all tiles) is a small fraction of
that. **Measure real throughput on the dev cache subset (§7) before committing to a full-backfill
time budget or SLURM time allocation.**

## What to read next

[`07a-modis-band-reference.md`](07a-modis-band-reference.md) for band-level facts;
[`08-hpc-transfer.md`](08-hpc-transfer.md) for how stage "annual"'s output reaches scicore;
[`06-open-questions.md`](06-open-questions.md) for everything flagged UNVERIFIED above, consolidated.
