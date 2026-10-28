# DOSE at ADM1: Pre-Specification

**Status: §§1–2 fixed 2026-10-28, before Gate D1 ran; D1 passed (§4). §3 (steps B and C) is a proposal; the
author confirms it before step B loads any temperature data.** This is Part 2 of
[`fork-corridors-dose.md`](fork-corridors-dose.md). Part 1 (corridors) stopped at Gate C1
([`corridors-prespec.md`](corridors-prespec.md) §8), so Part 2 uses the mine instrument only, and
the fork doc's step A is gone.

## 1. Data

- **DOSE v2.9** (Wenz et al. 2023; Zenodo 13773040), from the `dose` data module
  ([`../data/dose/README.md`](../data/dose/README.md)). DOSE codes are GADM 3.6. They are matched to
  GADM 4.1 by code and name within the country (rules in `src/data/sources/dose.py`). 1,596 of
  1,662 regions match; 58 have no match and 8 are ambiguous and dropped.
- **Mine instrument:** active SNL mines per GADM 4.1 region and year (`adm1_year_counts`, all
  commodities). These are the retired project's mines, counted per region instead of within
  20 km.

## 2. ◇ Gate D1: enough exposed regions (fixed before it ran)

- **DOSE coverage:** the region has `grp_pc_lcu_2015` and `pop` in at least 15 of the 19 years
  2002–2020.
- **Mine exposure:** the region's active-mine count changes at least once within 2002–2020. With
  region FE, only change within a region identifies anything.
- **Rule** (threshold confirmed by the author, 2026-10-28): **at least 150 regions that pass both,
  in at least 15 countries.** Then steps B and C may use the mine instrument. Below that, B and C
  report the fixed-effects association only, with no IV.
- Loads no temperature or lights data.

## 3. Steps B and C (proposal; confirm before B runs)

| | Step B | Step C |
|---|---|---|
| Unit | DOSE-matched GADM 4.1 region × year, 2002–2020 | same |
| Sample | regions passing D1's coverage rule | same |
| Outcome | mean night LST over the region's 10 km panel cells (the retired panel's sample and `BASE`), weighted by cell count; day LST second | same |
| Treatment | ln(1 + mean `ntl_harm` over the same cells) | ln(`grp_pc_lcu_2015`) |
| Instrument | active-mine count in the region; ln(1 + count) as robustness | same |
| FE | region + country × year | same |
| Control | DOSE precipitation `P_a`. **Never `T_a`, which is temperature.** | same |
| Clusters | region; country as robustness | same |

- **Region-years:** a region-year enters only if at least half the region's panel cells have
  night LST that year. That keeps the composition stable.
- **Per-step gate:** first stage of the treatment on the instrument, without LST. Continue to
  LST only if F ≥ 10. Inference on LST uses Anderson–Rubin sets, which hold under weak
  instruments.
- **Always reported:** the fixed-effects association of LST on the treatment, which is the Story 1
  comparison at the regional scale.
- **Reading:** if B and C disagree, the growth measure is responsible. If the retired cell result
  and B disagree, the unit is. The units differ (K per log point of GRP against lights), so compare
  signs and the overlap of AR sets, not magnitudes.
- **Known problem for C:** a mine's output counts in its region's GRP where it is produced, so
  the first stage of GRP on mines is partly mechanical.

## 4. Outcomes

**◇ Gate D1: passed** (2026-10-28, `output/notebooks/06_dose.ipynb`): **399 exposed regions in 48
countries**, against 150 in 15.

- 1,205 regions have DOSE output in at least 15 of the 19 years, and 474 of them contain an
  active mine at some point.
- Largest countries by exposed regions: US 35, Russia 33, China 30, Indonesia 26, Mexico 22,
  Turkey 20, Peru 19, Brazil 15, India 14. No single country dominates.
- Steps B and C may therefore use the mine instrument, once §3 is confirmed.

## Change log

None yet.
