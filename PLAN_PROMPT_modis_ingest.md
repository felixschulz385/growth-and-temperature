# Agent prompt — plan the MODIS LST ingest from Microsoft Planetary Computer

> Paste as the opening message to a coding agent in VS Code. **Enter plan mode.** Deliverable is a
> written plan, not code. Depends on the canonical-grid decisions from `PLAN_PROMPT_backbone.md`;
> if those documents exist under `docs/design/`, read them first and conform to them.

---

## Your task

Plan the ingest of MODIS land surface temperature from Planetary Computer's
[`modis-11A1-061`](https://planetarycomputer.microsoft.com/dataset/modis-11A1-061) collection,
replacing GLASS as the project's primary LST source.

This is **not a downloader**. Daily global 1 km LST for the full record is roughly 8–11 TB, and we
will never land it. The correct object is a **streaming ingest-and-composite stage** that reads daily
COGs remotely, reduces to annual composites in flight, and persists only the composite plus its
observation-count diagnostics. Plan that, and be explicit where it conflicts with the repo's existing
download → preprocess separation.

---

## Why this data, and what the design needs from it

The panel is global, annual, ~1 km. The outcome is **night-time** LST; the regressor is night-time
lights (VIIRS DNB). The estimand is the *total* effect of local economic activity on surface
temperature, with growth-induced land-cover change treated as a **mediator, not a confounder**.

Three consequences for this component:

1. **Night, not day.** Daytime LST is dominated by albedo and evapotranspiration — the time-invariant
   surface signal we need fixed effects to absorb. Night-time LST is dominated by stored-heat release
   and the surface energy balance, where our channels operate.
2. **Aqua, not Terra.** Aqua's night overpass is ~01:30 local, within minutes of Suomi-NPP's VIIRS DNB
   overpass. Terra's night overpass is ~22:30. Overpass matching is a deliberate design choice, not a
   convenience — see the platform-filter warning below, because this is the easiest thing in the whole
   component to get silently wrong.
3. **Observation counts are a result, not a byproduct.** Thermal LST is clear-sky only, and cloud
   masking correlates with both urbanisation and temperature. The per-cell valid-observation count is
   a required diagnostic and must be persisted alongside the composite at both annual and monthly
   resolution.

---

## Verified facts about the collection — do not re-derive these

Confirmed against `https://planetarycomputer.microsoft.com/api/stac/v1/collections/modis-11A1-061`:

- **Collection id:** `modis-11A1-061`. Daily, 1 km. Temporal extent 2000-02-24 → present.
- **Terra *and* Aqua are in the same collection.** There is no separate MYD11A1 collection. You must
  filter by platform (item `properties.platform`, and/or the `MOD`/`MYD` prefix in the item id —
  verify which is reliable).
- **Assets:** `LST_Day_1km`, `LST_Night_1km`, `QC_Day`, `QC_Night`, `Emis_31`, `Emis_32`,
  `Clear_day_cov`, `Clear_night_cov`, `Day_view_angl`, `Night_view_angl`, `Day_view_time`,
  `Night_view_time`, plus `hdf` (full source granule) and `metadata`. Per-band assets are COGs.
- **Hosting:** `msft:region = westeurope`, storage account `modiseuwest`, container `modis-061`.
  Reads from Basel are geographically close but still cross-network — this is not colocated compute,
  and the plan must account for that in the concurrency and retry design.
- **Native grid:** MODIS Sinusoidal, tiled `hXXvYY`, 1200×1200 at 1 km. Items are one tile, one day.
- **Signing required:** assets need SAS tokens via `planetary_computer.sign_inplace` as a
  `pystac_client.Client.open(..., modifier=...)`. **Tokens expire (~1 h).** A multi-hour tile-year job
  that signs once at startup will fail partway through. Design for re-signing.

---

## Decision you must make and document: 11A1 vs 11A2 vs 21A2

I have specified `11A1` (daily), but PC also hosts `modis-11A2-061` (8-day) and `modis-21A2-061`
(8-day, 3-band emissivity). Evaluate all three, recommend one, and record the reasoning. Two
considerations that are not obvious:

**Read volume.** Roughly 317 land-covering sinusoidal tiles × ~21 Aqua-years. At daily resolution
that is ~7,665 days × 317 tiles × (4–7 assets) ≈ **10–17 million COG reads**. At 8-day resolution it
is ~46 composites/year — **8× fewer**. Compute the real numbers for each option and put them in the
plan, because this is the difference between a job that runs overnight and one that runs for weeks.
Note that `11A2` already carries `Clear_sky_days`/`Clear_sky_nights`, which may supply the
observation-count diagnostic without daily reads — verify the asset list before relying on it.

**Emissivity provenance.** This one bears directly on the estimand. MOD11's split-window retrieval
assigns emissivity from a **land-cover classification**; MOD21 retrieves it via temperature–emissivity
separation. Since growth-induced land-cover change is part of our treatment effect, a
land-cover-keyed emissivity assumption lets the treatment alter the outcome's measurement model. Two
implications to state plainly in the plan:

- `11A1`'s `Emis_31`/`Emis_32` are classification lookups, **not** an independent land-surface signal,
  and therefore cannot serve as a mediator observable. Only `21A2`'s TES emissivities can.
- If `11A1` is chosen anyway (for daily granularity), the plan must schedule a **`21A2` robustness
  arm** — the same pipeline run on 21A2 so the two can be compared. Do not treat this as optional.

Recommend on the merits. If your recommendation is not `11A1`, say so directly and give the trade-off;
I would rather change the target now than after the ingest runs.

---

## Architecture: composite in native projection, reproject once

Strong recommendation to evaluate — I believe this is right, but argue it rather than assuming it.

Do **not** reproject every daily scene onto the canonical grid. Instead:

- **Stage "annual":** per MODIS sinusoidal tile, per year, stream the daily night assets, apply QC,
  reduce to composites **in native sinusoidal**, write annual zarr.
- **Stage "spatial":** reproject the annual composites onto the project's canonical GeoBox.

This avoids ~2.4 M reprojections in favour of ~6,700, and it reprojects a temporally smoothed field
rather than noisy single-day scenes. It also maps directly onto the repo's existing convention — see
`src/data/preprocess/sources/acag.py` ("Stage 1 (annual) … Stage 2 (spatial)") and the paired
`orchestration/slurm/glass-modis-preprocess-{annual,spatial}.sh`. **Mirror the `glass_modis` source
structure**; this is a replacement for it, not a new architecture.

Resampling, per variable, with justification: prefer **nearest** for LST (repositions without
smoothing — never bilinear or average on the outcome, which would inject blur into the very field the
downstream ring estimator measures spatial decay on) and **nearest is mandatory** for QC and flag
bands. State the choice for each band explicitly.

---

## Compositing definition

Get this right; it is a methodological decision wearing an engineering costume.

- **Composite monthly first, then average months to annual.** A naive annual mean over clear-sky days
  is biased toward dry-season conditions wherever cloud cover is seasonal — which is most of the
  tropics, and correlated with development. Month-first partially corrects it.
- Persist, per cell: annual mean, **per-month means**, **per-month valid counts**, and annual valid
  count. The counts are outputs, not internals.
- Decide and justify mean vs median.
- Decide how months with zero valid observations propagate into the annual figure, and record it.

---

## Correctness details that are easy to get wrong

Enumerate the handling for each of these in the plan:

- **Platform filter.** Mixing Terra and Aqua silently averages two different overpass times. This is
  the highest-consequence bug available in this component.
- **Scale factors and offsets.** `LST_*_1km` is `uint16` with scale 0.02 → Kelvin; `*_view_time` has
  scale 0.1 (hours); `*_view_angl` carries an offset. Verify each against the product user guide
  rather than assuming — and state whether `odc.stac.load` applies them automatically or whether you
  must.
- **Fill values.** LST fill is 0, which is *not* NaN and will silently drag composites toward zero if
  treated as data.
- **QC bit unpacking.** `QC_Night` is a bit field: mandatory QA flags plus an LST error estimate.
  Specify the threshold (commonly error ≤ 1 K or ≤ 2 K) and make it configurable.
- **Antimeridian and polar tiles.** Sinusoidal tiles near the dateline and poles behave badly on
  reprojection. The repo already carries a monkeypatch for an odc-geo footprint/antimeridian bug in
  `src/data/common/geobox/geobox_patch.py` — check whether it is still needed on the pinned version.
- **Latitude clipping.** If the backbone plan clips analysis to |φ| ≤ 60°, the tile list should
  respect it rather than ingesting tiles that will be discarded.

---

## Operational requirements

- **Resumability.** ~6,700 tile-year units. A manifest with per-unit status, so failures retry
  individually and a killed job resumes. Zarr `region=` writes make units idempotent — confirm that
  and design for it.
- **Token refresh** on a schedule shorter than SAS expiry, inside long-running workers.
- **Concurrency and backoff.** Remote reads across a network boundary with rate limits. Specify
  concurrency, retry policy, and how transient 429/503 are distinguished from real failures.
- **A dev cache.** Land a small subset (a few tiles, a few years) to local disk so development and
  tests do not hit the network. Specify the subset and where it lives (`scratch_nobackup/`?).
- **SLURM.** New scripts following the existing naming, e.g. `modis-preprocess-annual.sh` and
  `modis-preprocess-spatial.sh`. Check whether the cluster's compute nodes have outbound network
  access at all — if they do not, the entire streaming design fails and you must plan a staged
  download on a login/transfer node instead. **Flag this as a blocking question if you cannot
  determine it from the repo.**
- **Cost/time estimate** for a full historical backfill, and for an incremental annual update.

---

## What to read first

- `src/data/preprocess/sources/glass.py` — the source being replaced; match its interface.
- `src/data/preprocess/sources/acag.py` — cleanest example of the annual → spatial two-stage pattern.
- `src/data/preprocess/sources/{base,factory}.py` — the source-registration contract.
- `src/data/download/{async_downloader,workflow_unified}.py` — the existing acquisition path; decide
  whether `odc-stac` replaces it here or plugs into it, and justify.
- `src/data/common/geobox/` — canonical GeoBox construction and the odc-geo monkeypatch.
- `orchestration/configs/data.yaml` — the `glass_modis` source block is the template for the new one.
- `orchestration/slurm/glass-modis-preprocess-*.sh` — the execution pattern to mirror.
- `docs/design/*` if present — the canonical grid definition takes precedence over anything here.

---

## Hard constraints

1. **Plan only.** No implementation. Snippets ≤10 lines inside the design doc are fine.
2. **Additive.** GLASS ingest keeps working until MODIS is validated; propose a comparison step
   (same cells, same years, GLASS vs MODIS) as the cutover gate.
3. **Reuse repo idioms** — `data.yaml` source config, the annual/spatial stage convention, the
   `run.py` subcommand surface, SLURM per stage, existing zarr path layout.
4. **Verify, don't assume.** Band scale factors, QC bit layouts, `Clear_*_cov` semantics, asset lists
   for 11A2/21A2, and whether PC's MODIS assets are all COG. Cite the product user guide or the STAC
   response for each. Flag anything unresolved rather than asserting it.
5. **No secrets in configs.** PC access goes through `planetary_computer` signing; if a subscription
   key is used, follow the existing `orchestration/secrets/` pattern.

---

## Deliverables

Create these and nothing else:

1. **`docs/design/07-modis-ingest.md`** — the full plan: product decision with alternatives evaluated,
   architecture, staging, compositing definition, band-by-band handling table (dtype, scale, fill,
   resampling, QC rule), operational design, and effort/runtime estimate.
2. **`docs/design/07a-modis-band-reference.md`** — a reference table of every band ingested, with
   its verified scale/offset/fill/units and the source citation for each. This is the artefact that
   prevents a whole class of silent bugs; make it standalone and complete.
3. Append to **`docs/design/06-open-questions.md`** (create if absent) — unresolved items, each with
   the specific check that would settle it.

State decisions with reasons and record rejected alternatives. I need to be able to reconstruct the
reasoning in six months without re-deriving it.

---

## Before you write

Ask me about anything whose answer would change the design — in particular whether **compute nodes
have outbound internet access**, the disk budget for the annual composites, whether we need the
full 2000–present record or only the VIIRS-overlap era (2012→), and whether the daytime LST bands are
wanted at all (I currently think not, but ingesting them later means re-reading everything).
