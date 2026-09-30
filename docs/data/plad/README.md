# plad — Political Leaders' Affiliation Database (regional favoritism)

| | |
|---|---|
| Config key | `plad` (`admin_level: 2`) |
| Module | `src/data/sources/plad.py` (`PlaDSource`) |
| Steps | FETCH, PREPARE |
| Requires | `gadm` PREPARE (`GID_2_code_mapping.json`) |
| In panel | yes, joined on `(GID_2, year)`, `fillna: false` |

## What it is

The Political Leaders' Affiliation Database (Harvard Dataverse, `doi:10.7910/DVN/YUS575`): the
birth regions of national political leaders, with GADM `gid_1`/`gid_2` codes and tenure years.
It is used for a Hodler & Raschky (2014)-style regional-favoritism instrument.

## Raw data (FETCH)

The Dataverse `.dta` file. File id and name are hardcoded, because the Dataverse API sits behind a
bot-challenge WAF (see the module comment). Lands in `raw/plad/`.

## Prepared output (PREPARE)

`prepared/plad/adm/plad_adm2_reg_fav.parquet`: one row per favored `(GID_2, year)`, with columns
`GID_2` (gadm integer id), `year`, `reg_fav = True`. Years run 1980–2022. Leader spells are
expanded to one row per year; codes not found in gadm's mapping are dropped.

## Analysis caveats

- **Rare treatment.** `reg_fav` is true for about 0.5 % of pixel-years; about 2 % of pixels ever
  switch (`03_post_analysis.ipynb` R8).
- **Absorbed by pixel fixed effects.** Under pixel + country × year FE the first stage was dead
  (F ≈ 0.5), and the reduced form on LST was +0.09 K. The variation is at the region level, so the
  natural design is an ADM2 panel with region FE.
- **Favoritism works through infrastructure.** Roads, power and public buildings are themselves
  land-cover change, so exclusion is tied to the land-cover channel.
- Codes that don't match gadm are dropped silently. Check the match rate.
