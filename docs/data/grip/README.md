# grip — GRIP4 road density

| | |
|---|---|
| Config key | `grip` |
| Module | `src/data/sources/grip.py` |
| Steps | FETCH, PREPARE · no `REQUIRES` |
| In panel | no; used by the corridor route check R1 (`src/analysis/corridors.py`) |

## What it is

Global Roads Inventory Project v4 (Meijer et al. 2018, *Environ. Res. Lett.* 13:064006): road
length density in m/km² per 5-arc-minute (~8 km) cell, by type: tp1 highways, tp2 primary, tp3
secondary, and total.

## Raw data (FETCH)

`raw/grip/GRIP4_density_{tp1,tp2,tp3,total}.zip`, ~2 MB each (proxy or transfer node). Each zip
holds an ESRI ASCII grid without a CRS (WGS84 assumed) and a land-area grid, which is skipped.

## Prepared output (PREPARE)

Static pixel grid `prepared/grip/crs/ease6933/road_density/`, columns `grip_<type>_m_km2`,
nearest-neighbour onto the 1 km grid.

## Analysis caveats

- Roads mapped around 2018 are partly a result of 2002–2018 growth. Use them only to check the
  predicted routes (docs/successors/corridors-prespec.md §6a), never as a control or instrument.
- At 1 km this is a smoothed ~8 km field.
