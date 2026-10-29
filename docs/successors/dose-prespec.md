# DOSE at ADM1: Pre-Specification

**Status: §§1–2 fixed 2026-10-28, before Gate D1 ran; D1 passed (§4). §3 (steps B and C),
§3a (two more instruments, added at the author's request) and §3b (precipitation: DOSE `P_a`)
confirmed by the author 2026-10-29, before any first stage ran. Outcome (§4): all six first
stages fail, so there is no IV at ADM1. The fixed-effects association is negative with lights
(−0.26 K, p < 0.001) and with GRP (−0.13 K, p = 0.09). Part 2 is complete.** This is Part 2 of
[`fork-corridors-dose.md`](fork-corridors-dose.md). Part 1 (corridors) stopped at Gate C1
([`corridors-prespec.md`](corridors-prespec.md) §8), so the corridor instrument is gone, and with
it the fork doc's step A.

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

## 3. Steps B and C (confirmed 2026-10-29)

| | Step B | Step C |
|---|---|---|
| Unit | DOSE-matched GADM 4.1 region × year, 2002–2020 | same |
| Sample | regions passing D1's coverage rule | same |
| Outcome | mean night LST over the region's 10 km panel cells (the retired panel's sample and `BASE`), weighted by cell count; day LST second | same |
| Treatment | ln(1 + mean `ntl_harm` over the same cells) | ln(`grp_pc_lcu_2015`) |
| Instrument | active-mine count in the region (main); ln(1 + count) as robustness. Price shock and favoritism: §3a | same |
| FE | region + country × year | same |
| Control | annual precipitation, source open (§3b). **Never DOSE's `T_a`, which is temperature.** | same |
| Clusters | region; country as robustness | same |

- **Region-years:** a region-year enters only if at least half the region's panel cells have
  night LST that year. That keeps the composition stable.
- **Clarified 2026-10-29, before any first stage.**
  - **Weights:** the regression weight is the region's number of panel cells over 2002–2020,
    fixed over time. Each region's lights and LST are plain means over its cells.
  - **Same rows for B and C:** both steps use the region-years that have GRP and precipitation
    and pass the coverage rule above. So B → C changes only the growth measure.
  - **Code:** `src/analysis/dose.py`.
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

## 3a. Two more instruments (confirmed by the author 2026-10-29)

Added at the author's request before any step B or C first stage ran. Each instrument induces a
different kind of growth, so each gives its own estimate, not a sharper version of the others.

**Mine × world price, with exposure frozen at baseline.**

$$P_{rt} = \sum_{m \in r,\ \text{active in } 2001}\ \sum_c s_{mc} \ln p_{ct}$$

- **Mines:** SNL mines active in 2001, the year before the panel, placed in GADM 4.1 regions by
  point-in-polygon, as for the mine counts.
- **Shares $s_{mc}$:** `commodity_shares` from `snl_mining` where it exists. Otherwise the
  mine's primary commodity gets share 1.
- **Prices:** World Bank Pink Sheet, real, from `commodity_prices`. The 12 priced commodities
  are coal, copper, gold, iron ore, lead, nickel, phosphate, platinum, potash, silver, tin and
  zinc. A commodity without a price contributes nothing.
- **What identifies it:** with region FE, only world prices move $P_{rt}$. The growth it induces
  is existing mines producing, hiring and expanding more when prices rise, with no new
  openings. That is a different mix from the count, which is driven by openings and their
  footprint (the retired I4 found the footprint cells carry the cell-level identification).
- **Information, not gated:** the moving-exposure version, which counts every mine active in
  year $t$, as the retired `mine_priceshock` did. Also leave-one-commodity-out estimates: only 12
  price series drive $P$, so region clusters may understate the uncertainty (Adão, Kolesár and
  Morales 2019).

**Regional favoritism.**

$$F_{rt} = \text{share of year } t \text{ in which the country's leader was born in region } r$$

- **Leaders:** PLAD (Archigos effective leaders), from the `plad` module's raw table, with
  start and end dates.
- **Birth region:** PLAD's birthplace coordinates, placed in GADM 4.1 regions by
  point-in-polygon. PLAD's own `gid_1` codes follow an older GADM, and the coordinates avoid a
  second version match. Leaders born abroad favour no region.
- **What identifies it:** country × year FE compare the birth region with other regions in the
  same country-year. The growth it induces comes from public spending, transfers and
  infrastructure, with no mine involved: the fork's original aim of a second kind of growth.
- **Information, not gated:** lagged one year (the timing in Hodler and Raschky 2014) and an
  indicator version (favoured on any day of year $t$).
- **Prior:** weak. Favoritism had no first stage at cell level (retired R9, F ≈ 1.4). Hodler
  and Raschky found it at ADM1 with lights, mainly where institutions are weak, and DOSE leans
  towards richer countries.

**◇ Rules for all three instruments.**

- Each instrument is used alone (just identified), in step B and in step C, each with its own
  F ≥ 10 gate before LST. No pooled, over-identified estimate.
- The mine count stays the main instrument, the one compared with the retired cell result. The
  other two answer whether a different kind of growth gives a different answer.
- Before any first stage, report for information (no gate): the regions with a priced mine
  active in 2001, and the regions whose favoured status changes within 2002–2020, both within
  D1's coverage sample, with their country counts.
- All six first stages (three instruments × steps B and C) are reported, whichever pass.

## 3b. Precipitation control (decided 2026-10-29: DOSE `P_a`)

DOSE's `P_a` is ERA5 annual total precipitation, area-averaged over DOSE's regions (Wenz et al.
2023, Methods). **It stops in 2018.** For regions with GRP, it covers 1,266 regions in 2018 and
none in 2019 or 2020.

- **Decision (author, 2026-10-29):** steps B and C start with `P_a`. Region-years without it
  drop, so the estimation sample is 2002–2018.
- **Possible later extension:** a precipitation module of our own, aggregated to GADM 4.1
  regions, would restore 2019–2020. Its overlap with `P_a` over 2002–2018 would check it. Adding
  it is a change-log entry.

## 4. Outcomes

**◇ Gate D1: passed** (2026-10-28, `output/notebooks/06_dose.ipynb`): **399 exposed regions in 48
countries**, against 150 in 15.

- 1,205 regions have DOSE output in at least 15 of the 19 years, and 474 of them contain an
  active mine at some point.
- Largest countries by exposed regions: US 35, Russia 33, China 30, Indonesia 26, Mexico 22,
  Turkey 20, Peru 19, Brazil 15, India 14. No single country dominates.
- Steps B and C may therefore use the mine instrument, once §3 is confirmed.

**Exposure of the new instruments (information, 2026-10-29).** Regions whose instrument changes,
within D1's coverage sample: mine count 397 in 48 countries (D1's 399 also counted years without
DOSE output), mine × price 343 in 49, favoritism 163 in 53.

**◇ Step B and step C gates: all six fail. No IV at ADM1; no LST was loaded**
(2026-10-29, `output/notebooks/06b_dose_step_b.ipynb`, `06c_dose_step_c.ipynb`).

- **Sample:** 19,590 region-years, 1,182 regions in 58 countries, 2002–2018 (`P_a` ends in 2018).
- **First stages:** weighted by panel cells, region clusters, gate F ≥ 10.

| Instrument | Step B: ln lights, coefficient (SE) | F | Step C: ln GRP pc, coefficient (SE) | F |
|---|---|---|---|---|
| mine count (per active mine) | 0.00078 (0.00078) | **1.0** | 0.0014 (0.0007) | **4.3** |
| mine × price (per unit of Σ share × ln price) | 0.00051 (0.00027) | **3.5** | 0.0010 (0.0006) | **3.2** |
| favoritism (share of the year) | 0.0094 (0.0160) | **0.3** | 0.016 (0.019) | **0.7** |

- **Information, step B.** Every variant has F < 10 except one: mine × price with country
  clusters, F = 10.9. It is not the gate, and 48 country clusters with 12 price series are not a
  reliable basis for it.
  - ln(1 + count): F 0.9.
  - Mine count with country clusters: F 5.0.
  - Moving price exposure: F 1.2.
  - Favoritism: any day F 0.2, lagged one year F 0.9.
  - Leaving one commodity out: F 1.3 (gold) to 4.1 (nickel).
- **Information, step C.** Every variant has F < 5.
- **Reading.** At regional scale none of the three instruments moves the growth measures enough
  to identify anything.
  - **Mines are diluted:** a mine changes lights over a few cells, while the median region
    has 126 panel cells. With cell weights, half the weight sits in regions of 2,385 cells or
    more. Even GRP, where mine output counts mechanically, moves by only
    0.14 % per mine (t = 2.1).
  - **Favoritism shows no first stage,** as at cell level (retired R9). DOSE leans towards
    countries where Hodler and Raschky find little favoritism.
- **What remains of Part 2:** the fixed-effects association of LST on lights and on GRP at ADM1,
  which §3 reports in every case. It is descriptive, Story 1 at regional scale, and it is the
  first step that loads LST.
- Changing the weights, the unit or the instrument scaling now would be fitted to these results,
  so the IV half of Part 2 stops here.

**Fixed-effects associations (descriptive, as §3 always reports)** (2026-10-29,
`output/notebooks/06d_dose_twfe.ipynb`). The rows, FE, control and weights are those of the
first stages. Units are K per log point of the growth measure.

| Growth measure | Night LST, region clusters | Day LST, region clusters | Night LST, country clusters |
|---|---|---|---|
| B: ln(1 + mean lights) | **−0.257** (0.072), p < 0.001 | −0.293 (0.140), p = 0.04 | −0.257 (0.132), p = 0.06 |
| C: ln GRP per capita | **−0.125** (0.075), p = 0.09 | −0.209 (0.146), p = 0.15 | −0.125 (0.120), p = 0.30 |

- **Information:** within the same FE, ln lights rise 0.17 (0.03) per log point of GRP. The two
  measures share little of their year-to-year variation.
- **Reading.**
  - **The sign matches the retired cell result,** which was −0.030 with country × biome × year
    FE and −0.027 with country × year FE. Night LST falls where growth is faster, now between
    regions of the same country-year. Magnitudes are not comparable, because a log point of a
    region's mean lights is a different quantity from a log point of one cell's lights (§3).
  - **Growth measured without satellites points the same way:** GRP gives the same sign, about
    half the size, and is not significant at 5 %. The two confidence intervals overlap almost
    entirely. That is weak evidence that the association is not specific to lights.
  - **Inference is fragile:** with country clusters only the lights–night association is near
    significance.
  - **This is not a causal estimate.** All six first stages failed. Reverse causality fits the
    sign: hot years lower regional output in the data DOSE grew from (Kalkuhl and Wenz 2020),
    and a hot year also raises LST. That alone gives a negative association. The design cannot
    separate the two. Controlling for DOSE's `T_a` would remove the outcome, so it stays excluded.
- **Part 2 is complete.** No causal bound at ADM1. A regional association with the retired
  result's sign, seen with both measures and precise only with lights.

## Change log

None yet.
