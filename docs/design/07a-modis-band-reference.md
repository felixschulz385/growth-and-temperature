# 07a — MODIS Band Reference

Verified band-level reference for every asset ingested from Microsoft Planetary Computer's MODIS
collections. Every scale/offset/fill/range value below is cited to its source; anything not
independently verified this session is marked **UNVERIFIED** rather than asserted. This table is the
artefact meant to prevent silent unit/scale bugs — treat it as standalone and authoritative for
implementation, not a summary of [`07-modis-ingest.md`](07-modis-ingest.md).

All three collections confirmed live on Planetary Computer via direct STAC fetch on 2026-07-29
(`https://planetarycomputer.microsoft.com/api/stac/v1/collections/<id>`):

| Collection | Title | Temporal extent | `msft:region` | `msft:storage_account` | `msft:container` |
|---|---|---|---|---|---|
| `modis-11A1-061` | MODIS Land Surface Temperature/Emissivity Daily | 2000-02-24 → present | `westeurope` | `modiseuwest` | `modis-061` |
| `modis-11A2-061` | MODIS Land Surface Temperature/Emissivity 8-Day | 2000-02-18 → present | `westeurope` | `modiseuwest` | `modis-061` |
| `modis-21A2-061` | MODIS Land Surface Temperature/3-Band Emissivity 8-Day | 2000-02-16 → present | `westeurope` | `modiseuwest` | `modis-061` |

All three carry both `aqua` and `terra` in `summaries.platform` — **confirms the platform-filter
requirement**: there is no separate MYD-only collection, matching the correctness-details warning in
the original ingest brief. Platform must be filtered per item (`properties.platform` and/or the
`MOD`/`MYD` id prefix — **UNVERIFIED which is authoritative; check both agree on a sample before
trusting either alone**).

Confirmed **404** for `modis-21A1D-061` (queried directly): **no daily TES-algorithm (MOD21A1D/N)
collection is hosted on Planetary Computer** — only the 8-day `modis-21A2-061` composite. This is
material to the product decision in [`07-modis-ingest.md`](07-modis-ingest.md) §1: a 21A2-based
design cannot fall back to a same-family daily product for a finer-grained observation count if the
8-day diagnostic proves too coarse.

No SAS-signing requirement or subscription-key note appears anywhere in the three collections'
metadata — this is expected (`planetary_computer.sign_inplace` is a client-side convention, not a
collection-level flag) and does not change the original brief's guidance to sign via
`pystac_client.Client.open(..., modifier=...)` and re-sign on a schedule shorter than the ~1 h token
lifetime.

## MOD/MYD11A1 (daily) and MOD/MYD11A2 (8-day) — shared band definitions

Both collections declare an identical band set (11A2's are the 8-day composite of 11A1's). Source:
direct STAC `item_assets` fetch for both collections, 2026-07-29.

| Asset | Dtype | Scale | Offset | Fill | Unit/Range | Source |
|---|---|---|---|---|---|---|
| `LST_Day_1km` / `LST_Night_1km` | `uint16` | 0.02 | 0 (**UNVERIFIED** — see below) | 0 | Kelvin | STAC (scale); fill confirmed via NASA Earthdata catalog page + MOD11_L2 guide, both stating fill=0 for the LST SDS |
| `QC_Day` / `QC_Night` | `uint8` | — | — | — (bit field, no single fill) | bit field | STAC dtype; **bit layout UNVERIFIED for the L3 gridded product** — see caveat below |
| `Emis_31` / `Emis_32` | `uint8` | 0.002 | **UNVERIFIED** (see below) | 0 | dimensionless (0–1 physical) | STAC (scale, dtype); MOD11_L2 guide states valid range 1–255, fill 0, but offset not stated there and STAC did not report an offset field for 11A1/11A2 specifically |
| `Day_view_angl` / `Night_view_angl` | `uint8` | 1 (none reported) | 0 | 0 | Degree, range 0–180 | STAC (unit); MOD11_L2 guide corroborates valid range 0–180, fill 0 (L2-guide figures, treated as corroborating not authoritative for L3 — see caveat) |
| `Day_view_time` / `Night_view_time` | `uint8` | 0.1 | 0 | 0 | Hours (local solar), raw range 0–240 → 0–24 h | STAC (scale); MOD11_L2 guide corroborates raw valid range 0–240, fill 0 |
| `Clear_day_cov` (11A1) / — | `uint16` | 0.0005 | — | **UNVERIFIED** | dimensionless coverage fraction | STAC |
| `Clear_night_cov` (11A1) / — | `uint16` | 0.0005 | — | **UNVERIFIED** | dimensionless coverage fraction | STAC |
| `Clear_sky_days` (11A2 only) | `uint8` | — | — | **UNVERIFIED** | count of clear days contributing to the 8-day composite, max 8 | STAC description: "the days in clear-sky conditions and with validate LSTs" |
| `Clear_sky_nights` (11A2 only) | `uint8` | — | — | **UNVERIFIED** | count of clear nights contributing, max 8 | STAC description: "the nights in clear-sky conditions and with validate LSTs" |
| `hdf` | `application/x-hdf` | — | — | — | full source granule | STAC |
| `metadata` | `application/xml` | — | — | — | FGDC metadata | STAC |

**Emissivity offset — flagged, not assumed.** MOD21A2's Emis_29/31/32 (below) has a confirmed
`offset: 0.49` in addition to `scale: 0.002` (`raw × 0.002 + 0.49` maps the uint8 range to
physically-plausible emissivity ≈0.49–1.0). MOD11A1/11A2's Emis_31/32 STAC entries report `scale:
0.002` but **no offset field was present in the fetched STAC response**, and the MOD11_L2 guide (which
covers a different product tier — see next caveat) doesn't state one either. Do not assume the two
products share the same offset. **Resolve before implementation**: decode a real Emis_31 pixel from a
sample MOD11A1 granule and confirm whether raw×0.002 alone produces physically valid emissivity
(~0.9–1.0 over vegetated/urban land) or whether an offset is needed. Logged in
[`06-open-questions.md`](06-open-questions.md).

**QC bit layout — the single most important unresolved item in this document.** The only primary
source successfully fetched this session, the MOD11 (L2) User Guide
(`icess.ucsb.edu/modis/LstUsrGuide/usrguide_mod11.html`), **explicitly and only covers MOD11_L2
swath data**, confirmed via its own title and table of contents. Its QC field is a **16-bit** layout
(bits 0–1 mandatory QA; 2–3 data quality; 4–5 cloud flag; 6–7 LST model number; 8–9 LST quality flag;
10–11 emissivity flag; 12–13 emissivity quality flag; 14–15 emissivity error category) — structurally
inconsistent with `QC_Day`/`QC_Night` being declared `uint8` (8-bit) for the gridded 11A1/11A2
products in the STAC response, confirming this L2 layout **does not apply** to the product actually
being ingested. The literature commonly cites an 8-bit MOD11A1/A2 QC layout (bits 0–1 mandatory QA:
00 good / 01 not produced-cloud / 10 not produced-other / 11 low quality; bits 2–3 data quality flag;
bits 4–5 emissivity error flag; bits 6–7 LST error flag: 00 ≤1K / 01 ≤2K / 10 ≤3K / 11 >3K) — **this
plan does not assert that layout as verified**. The direct MOD11 V6.1 PDF fetch (the correct primary
source for the L3 gridded product) failed to yield extractable bit-table text via the available
tooling (binary/table-layout PDF, saved locally for manual review at the WebFetch tool's cache path).
**Resolve before implementation**: pull the QC bit table from the MOD11 V6.1 PDF directly (a PDF
reader/parser, not the WebFetch tool used this session) or from a peer-reviewed methods paper that
reproduces it, before hardcoding a threshold. Logged in [`06-open-questions.md`](06-open-questions.md).

## MOD/MYD21A2 (8-day, TES) — band definitions

Source: direct STAC `item_assets` fetch, 2026-07-29, corroborated by the NASA Earthdata MOD21A2
catalog page for compositing/derivation wording.

| Asset | Dtype | Scale | Offset | Fill | Valid range / Unit | Source |
|---|---|---|---|---|---|---|
| `LST_Day_1KM` / `LST_Night_1KM` | `uint16` | 0.02 | 0 | 0 | 7500–65535 (raw) → 150–1310.7 K (envelope, not physical bound) | STAC + NASA Earthdata catalog page |
| `Emis_29` / `Emis_31` / `Emis_32` | `uint8` | 0.002 | **0.49** | 0 | 1–255 (raw) → ≈0.492–1.0 | STAC (scale); offset confirmed via NASA Earthdata catalog page |
| `QC_Day` / `QC_Night` | `uint8` | — | — | — (bit field) | bit field | STAC dtype; **bit layout UNVERIFIED — not stated on the fetched catalog page**, same caveat class as 11A1/11A2 |
| `View_Angle_Day` / `View_Angle_Night` | `uint8` | 1 (none reported) | 0 | — | Degree | STAC |
| `View_Time_Day` / `View_Time_Night` | `uint8` | 0.1 | 0 | — | Hours | STAC |
| `hdf` | `application/x-hdf` | — | — | — | full source granule | STAC |
| `metadata` | `application/xml` | — | — | — | FGDC metadata | STAC |

**No `Clear_sky_days`/`Clear_sky_nights`-equivalent asset exists for `modis-21A2-061`** — confirmed
absent from the STAC `item_assets` list (compare against 11A2's explicit inclusion of both, above).
This is the operational cost of the 21A2 recommendation in [`07-modis-ingest.md`](07-modis-ingest.md)
§1: the valid-observation-count diagnostic must be derived from `QC_Night` pixel validity at each
8-day composite's own granularity (up to ~46 composite-periods/year), not from a native
count-of-contributing-days field. See [`07-modis-ingest.md`](07-modis-ingest.md) §4 for how the annual
compositing stage constructs this diagnostic from `QC_Night` instead.

**Emissivity is a single day+night-pooled value, not separate.** Confirmed verbatim from the NASA
Earthdata MOD21A2 catalog page: "the values for the MODIS emissivity bands 29, 31, and 32 are the
average of both the nighttime and daytime acquisitions." Contrast with MOD11A1/11A2, whose `Emis_31`/
`Emis_32` are also single (not day/night-split) but for a different reason — the classification lookup
is time-of-day-invariant by construction, not temporally pooled. Neither product gives a
night-only emissivity; state this explicitly in [`07-modis-ingest.md`](07-modis-ingest.md) as an
accepted limitation of using emissivity purely as a mediator observable (not as an LST-retrieval
input we control).

## Compositing method — verified, not assumed

Both `MOD11A2` and `MOD21A2` are **simple averages of clear-sky daily pixels within the 8-day window**,
not single-best-day picks — both confirmed via direct quotes:

- MOD11A2 (NASA Earthdata catalog page): *"Each pixel value in the MOD11A2 is a simple average of all
  the corresponding MOD11A1 LST pixels collected within that 8-day period."*
- MOD21A2 (NASA Earthdata catalog page): *"The algorithm calculates the average from all the cloud
  free MOD21A1D and MOD21A1N daily acquisitions from the 8-day period."* (Note: this sentence implies
  daily `MOD21A1D`/`MOD21A1N` granules exist upstream at USGS/LP DAAC even though they are **not**
  separately hosted on Planetary Computer — confirmed 404 above. They are not a usable ingest path for
  this project regardless of upstream existence, since the plan is scoped to Planetary Computer.)

This means both 8-day products are themselves already a naive (unweighted, non-seasonally-adjusted)
temporal average — the same kind of bias the backbone design's month-first compositing rule exists to
correct, just pre-applied at 8-day granularity by USGS before the data ever reaches this pipeline. See
[`07-modis-ingest.md`](07-modis-ingest.md) §4 for how this composes with (rather than substitutes for)
the project's own month-then-annual compositing rule.

## Emissivity provenance — verified via primary source, both products

**MOD11 family (classification-based, confirmed).** Quoted verbatim from the MOD11_L2 User Guide
(`icess.ucsb.edu/modis/LstUsrGuide/usrguide_mod11.html`), which — while scoped to the L2 swath
product — describes the shared split-window algorithm's emissivity assignment method used across the
MOD11 family: *"Emissivities in bands 31 and 32 are estimated by the classification-based emissivity
method (Snyder and Wan, 1998) according to land cover types in the pixel determined by the input data
in quarterly Land Cover (MOD12Q1) and daily Snow Cover (MOD10_L2)."* This is the primary-source
confirmation the backbone docs' emissivity-provenance argument needed: MOD11/MYD11's `Emis_31`/
`Emis_32` are a **land-cover-classification lookup**, not an independent physical retrieval, and
therefore cannot serve as a mediator observable per [`04-ingest.md`](04-ingest.md) §5.

**MOD21 family (TES, confirmed).** Quoted verbatim from both the `modis-21A2-061` STAC collection
description and the NASA Earthdata catalog page (word-for-word identical text in both sources): *"The
MOD21 TES algorithm uses a physics-based algorithm to dynamically retrieve both the LST and spectral
emissivity simultaneously from the MODIS thermal infrared bands 29, 31, and 32."* This confirms
MOD21/MYD21's emissivity is a genuine independent land-surface signal, satisfying the mediator-
observable requirement.

## Land tile count (unchanged from the original brief, not independently re-verified)

`PLAN_PROMPT_modis_ingest.md`'s figure of **~317 land-covering sinusoidal tiles** (of 460 in the full
global `hXXvYY` grid) is a commonly-cited figure in the MODIS literature and was not independently
re-derived this session — flagged as **UNVERIFIED** here rather than silently inherited; confirm
against the actual sinusoidal tile grid + a land mask before finalizing tile lists, per
[`06-open-questions.md`](06-open-questions.md).
