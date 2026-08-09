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
`MOD`/`MYD` id prefix — **RESOLVED 2026-08-09**: queried 600 real items across all three collections,
zero disagreements between the two signals; see `06-open-questions.md` #8 for the full sample
breakdown).

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
| `Emis_31` / `Emis_32` | `uint8` | 0.002 | **0.49** (confirmed, see below) | 0 | dimensionless (0–1 physical) | STAC (scale, dtype); offset confirmed via Table 9 of the Collection-6 MODIS LST Products Users' Guide (Wan, 2019) |
| `Day_view_angl` / `Night_view_angl` | `uint8` | 1 (none reported) | **-65.0** | **255** | Degree, raw range 0–130 → -65–65°, negative = viewed from east | STAC (unit); scale/offset/fill/range confirmed via Table 9 of the Users' Guide (Wan, 2019) — corrects an earlier draft that had wrongly corroborated these against the MOD11_L2 (different product tier) guide instead |
| `Day_view_time` / `Night_view_time` | `uint8` | 0.1 | 0 | **255** | Hours (local solar), raw range 0–240 → 0–24 h | STAC (scale); fill confirmed via Table 9 of the Users' Guide (Wan, 2019) |
| `Clear_day_cov` (11A1) / — | `uint16` | 0.0005 | — | **UNVERIFIED** | dimensionless coverage fraction | STAC |
| `Clear_night_cov` (11A1) / — | `uint16` | 0.0005 | — | **UNVERIFIED** | dimensionless coverage fraction | STAC |
| `Clear_sky_days` (11A2 only) | `uint8` | — | — | **UNVERIFIED** | count of clear days contributing to the 8-day composite, max 8 | STAC description: "the days in clear-sky conditions and with validate LSTs" |
| `Clear_sky_nights` (11A2 only) | `uint8` | — | — | **UNVERIFIED** | count of clear nights contributing, max 8 | STAC description: "the nights in clear-sky conditions and with validate LSTs" |
| `hdf` | `application/x-hdf` | — | — | — | full source granule | STAC |
| `metadata` | `application/xml` | — | — | — | FGDC metadata | STAC |

**Emissivity offset — RESOLVED (2026-08-09).** MOD21A2's Emis_29/31/32 (below) has a confirmed
`offset: 0.49` in addition to `scale: 0.002` (`raw × 0.002 + 0.49` maps the uint8 range to
physically-plausible emissivity ≈0.49–1.0). MOD11A1's Table 9 ("The SDSs in the MOD11A1 product") in
the Collection-6 MODIS LST Products Users' Guide (Wan, ERI/UCSB, June 2019) confirms Emis_31/Emis_32
share that same `offset: 0.49` — the two products' emissivity bands use the same decode.
`src/data/sources/modis/source.py`'s `BAND_SPECS["11A1"]` previously hardcoded `offset: 0.0` for both
(an unverified guess that turned out wrong); corrected to `0.49`, pinned by
`tests/data/sources/modis/test_modis_qc.py`.

**QC bit layout — RESOLVED for MOD11A1, still open for MOD21A2 (2026-08-09).** The Collection-6 MODIS
LST Products Users' Guide (Wan, ERI/UCSB, June 2019) — the correct primary source, not the MOD11_L2
guide previously the only one reachable — gives Table 13 ("Bit flags defined for SDSs QC_day and
QC_Night in MOD11A1"): an 8-bit layout with bits 1&0 mandatory QA (00 good / 01 other-quality / 10
cloud / 11 other), bits 3&2 data quality, bits 5&4 emissivity error, bits 7&6 LST error category (00
≤1K / 01 ≤2K / 10 ≤3K / 11 >3K). This **exactly matches** the layout
`src/data/sources/modis/tiles.py::decode_qc_valid_mask` already implemented as its best guess — no
bit-shift/mask logic needed to change, only the UNVERIFIED status, now confirmed for `product="11A1"`
(pinned by `tests/data/sources/modis/test_modis_qc.py`).

**Still unverified for MOD21A2** (the primary `21A2` product): this guide's table of contents covers
only the MOD11 family (MOD11_L2/A1/A2/B1/B2/B3/C1/C2/C3). MOD21 is generated by the physics-based TES
algorithm (Wan and Li, 1997), a different algorithm from MOD11's split-window method, and its QC
semantics are not addressed anywhere in this guide. `decode_qc_valid_mask` still logs its UNVERIFIED
warning for any `product` other than `"11A1"`. **Resolve before a production 21A2 run**: find a
MOD21-specific primary source (e.g. the MOD21 ATBD) for its QC bit table. Logged in
[`06-open-questions.md`](06-open-questions.md).

## MOD/MYD21A2 (8-day, TES) — band definitions

Source: direct STAC `item_assets` fetch, 2026-07-29, corroborated by the NASA Earthdata MOD21A2
catalog page for compositing/derivation wording, and by the *MxD21 LST&E User Guide* (Hulley et al.,
JPL, March 2019) Table 11 ("The SDSs in the MxD21A2 8-day product") — the correct primary source for
this exact product, confirmed 2026-08-09.

| Asset | Dtype | Scale | Offset | Fill | Valid range / Unit | Source |
|---|---|---|---|---|---|---|
| `LST_Day_1KM` / `LST_Night_1KM` | `uint16` | 0.02 | 0 | 0 | 7500–65535 (raw) → 150–1310.7 K (envelope, not physical bound) | STAC + NASA Earthdata catalog page; confirmed exactly by Table 11 |
| `Emis_29` / `Emis_31` / `Emis_32` | `uint8` | 0.002 | **0.49** | 0 | 1–255 (raw) → ≈0.492–1.0 | STAC (scale); offset confirmed via NASA Earthdata catalog page and Table 11 |
| `QC_Day` / `QC_Night` | `uint8` | — | — | — (bit field) | bit field | **RESOLVED**: Table 12 of the MxD21 guide (see below) |
| `View_Angle_Day` / `View_Angle_Night` | `uint8` | 1 | **-65** | **255** | raw 0–130 → -65–65°, negative = viewed from east | Table 11 — corrects `BAND_SPECS["21A2"]` (`src/data/sources/modis/source.py`), which had `offset: 0.0`/`fill: None` (unmasked) before this table was checked |
| `View_Time_Day` / `View_Time_Night` | `uint8` | 0.1 | 0 | **255** | Hours | Table 11 — corrects the same spec's `fill: None` |
| `hdf` | `application/x-hdf` | — | — | — | full source granule | STAC |
| `metadata` | `application/xml` | — | — | — | FGDC metadata | STAC |

**QC bit layout — RESOLVED 2026-08-09.** Table 12 of the MxD21 guide gives bits 1&0 (mandatory QA,
00=good — same convention as MOD11) and bits 7&6 ("LST accuracy": 00 = >2K poor, 01 = 1.5–2K, 10 =
1–1.5K, 11 = <1K excellent). **Bits 7&6 sit at the same position as MOD11's "LST error" bits but mean
the opposite thing** — MOD11's convention has increasing bit value = *worse* accuracy; MOD21's has
increasing bit value = *better*. `decode_qc_valid_mask` (`src/data/sources/modis/tiles.py`) previously
applied MOD11's bit-value-to-error-K mapping uniformly to every product, which would have silently
inverted the quality filter for every MOD21A2 pixel — keeping the >2K-error pixels and discarding the
<1K ones. Fixed by giving `_LST_ERROR_K_BY_BITS` a per-product mapping (`"11A1"` and `"21A2"` both now
confirmed), selected via a `product` argument threaded from `ModisSource.product`. Pinned by
`tests/data/sources/modis/test_modis_qc.py`.

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
