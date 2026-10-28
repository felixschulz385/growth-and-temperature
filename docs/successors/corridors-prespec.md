# Corridors: Pre-Specification

**Status: fixed 2026-10-28, before any corridor exposure was built. Outcome (§8): R1 passed, Gate C1 failed (F = 0.08), so Part 1 stops.** This is step 0 of Part 1 in
[`fork-corridors-dose.md`](fork-corridors-dose.md). It replaces that page's "first thoughts" where
the two differ (clustering, destination rule, shock set). Changes after this date go in the change
log at the end, with a reason, and never after a result they could be fitted to.

## 1. Shock: bulk mines

- **Mines.** SNL properties in `active_mines` (the retired project's mine table) whose
  `primary_commodity` is bulk: Coal, Iron Ore, Bauxite, Phosphate, Potash, Manganese, Chromite,
  Ilmenite, Heavy Mineral Sands, Rutile, Zircon, Titanium. That is 3,414 mines; 1,858 open or close
  within 2002–2022.
- **Why bulk only.** A bulk mine ships millions of tonnes a year by truck or rail. A gold mine
  ships a few trucks a week. The corridor mechanism only exists for the first kind.
- **Shock.** $S_{mt} = 1$ if mine $m$ is active in year $t$ (opening year to closing year, or
  open-ended), as in `active_mines`.
- **Robustness:** all mines, every commodity.

## 2. Destination: dry-bulk ports

- **Ports.** IMF PortWatch ports with `vessel_count_dry_bulk` ≥ 50 a year (772 of 2,065 ports).
  The World Port Index was rejected: its harbour size measures shelter, not throughput (Port
  Hedland, the largest bulk export port, is "small"; Kamsar is "very small").
- **Caveat.** PortWatch counts ships from 2019 on, so the set of bulk ports partly reflects mines
  that opened during the panel. Only the port's location enters the instrument, and recentering
  (§5) absorbs the fixed geography. A robustness check uses ports with ≥ 50 ships of any type.
- **Assignment.** Each mine ships to the port reached at least cost (§3), not a fixed port.

## 3. Routes

- **Graph.** The analysis grid's own 10 km cells: the 10 × 10 blocks of 1 km pixels that assembly
  forms within each 2048-pixel tile (`src/data/assemble/sql_engine.py::_pixel_id_sql`). Routing
  on these cells makes the exposure join the panel's `pixel_id` exactly. A cell is passable if at
  least half its pixels are land (OSM land mask). Each cell connects to its 8 neighbours.
- **Cost.** Edge cost = geodesic length between cell centres × $(1 + \bar s / 5)$, where
  $\bar s$ is the mean slope of the two cells in degrees (EarthEnv-Topography, mean of
  GMTED2010 slopes per 10 km cell). A 5° mean slope doubles the cost. Borders cost nothing.
- **Solution.** One multi-source least-cost search from all destination ports. A mine's route is
  the path from its cell back to the port that reaches it at least cost. Every cell on that path
  belongs to the same port's catchment.
- **Snapping.** A mine or port in an impassable cell moves to the nearest passable cell, if that
  is at most 25 km away. Mines and ports farther away (north of 60°N, where the grid ends, or on
  small islands) are dropped.
- **Robustness:** flat cost (slope ignored).

## 4. Exposure

For cell $i$, year $t$ and band $b$ of distance along the route from the mine:

$$Z^{b}_{it} = \sum_{m \in \text{bulk}} \mathbb{1}[i \in \text{route}_m,\ d_{im} \in b]\ S_{mt}$$

- **Bands:** 20–100 km, 100–300 km, over 300 km along the route.
- **Main instrument:** $Z_{it} = Z^{20\text{–}100}_{it} + Z^{100\text{–}300}_{it}$. Far trunk
  segments merge routes from many mines near ports, so they mostly measure port hinterlands.
  The far band is reported, not used.
- **Robustness:** the indicator $\mathbb{1}[Z_{it} > 0]$.

## 5. Recentering

Opening and closing years are reassigned among mines in the same stratum (country of the mine ×
primary commodity). The expected exposure then has a closed form:

$$\mu_{it} = \sum_{m} \mathbb{1}[i \in \text{route}_m,\ d_{im} \in 20\text{–}300]\ \bar S_{k(m)t}$$

where $\bar S_{kt}$ is the share of stratum $k$'s mines active in year $t$. The instrument is
$\tilde Z_{it} = Z_{it} - \mu_{it}$ (Borusyak and Hull 2023).

## 6. Sample and comparison

- **The retired specification** (`src/analysis/spec.py`): years 2002–2022, `BASE`, the coverage
  rule, cell FE + country × biome × year FE.
- **Exclusions:**
  - cells within 20 km of any SNL mine active at some point in 2002–2022, of any commodity;
  - cells within 30 km of a destination port that receives at least one route.
- **Comparison at equal distance:** add distance-to-nearest-mine band × year FE. The bands are
  20–50, 50–100, 100–200, 200–300 and over 300 km, from any SNL mine of any commodity.
- **Clustering:** by destination port catchment (main) and by ADM1 (reported).

## 6a. ◇ R1, route check (added 2026-10-28, before it ran)

Do predicted routes run along real roads? Loads neither lights nor LST.

- **Data:** one row per cell outside the exclusions. Outcome ln(1 + GRIP4 highway + primary road
  density, m/km²). Regressor: on a `main` route at 20–300 km.
- **Comparison:** FE for country × distance-to-mine band; clustered by port catchment.
- **Rule:** go on to C1 if the coefficient is positive with t ≥ 3. Otherwise the routes are not
  corridors: revisit the cost surface (logged here) before C1.
- **Reported for information:** the same regression for the `flat_cost` routes, and balance of
  on-route against off-route cells on slope, land share and distance to the nearest port. These
  never switch the main specification.

GRIP4 roads are from about 2018 and partly a result of growth. R1 uses them only to check the
routes, never as a control.

## 7. Gate C1: first stage only

Regress `log_ntl` on $\tilde Z$ with the comparison above. No LST column is loaded.

- **Continue to LST only if F ≥ 70** with port clustering (F = t² of the single instrument). 70
  is the F that gave the mining bound of ±0.3 K.
- Reported for information, not as part of the gate: F with ADM1 clustering, the coefficient by
  band, and an event study around mine openings with three leads, which shows whether lights rise
  during construction.

## 8. Outcomes

Run 2026-10-28 on the `main` variant (`output/notebooks/05_corridors.ipynb`). No LST was loaded.

**◇ R1: passed.** Predicted routes run along real roads.

| Routes | On-route coefficient, ln(1 + highway + primary density) | SE | t | On-route cells |
|---|---|---|---|---|
| main (slope cost) | +0.547 | 0.073 | 7.5 | 17,141 of 1,130,150 |
| flat cost (information) | +0.443 | 0.068 | 6.5 | 21,292 |

Balance (information): on-route cells are 2.1° flatter (off-route mean 3.1°; SE 0.2), as expected
from routes that avoid slopes. Land share differs by +0.01, and distance to the nearest port by
−16 km (SE 62). Cell FE absorb these levels.

**◇ Gate C1: failed. Part 1 stops.**

| Clustering | Coefficient on $\tilde Z$ (log lights per active upstream mine) | SE | F |
|---|---|---|---|
| port catchment (gate) | 0.0026 | 0.0093 | **0.08** |
| ADM1 (information) | 0.0026 | 0.0082 | 0.10 |

- **Sample:** 23.8 million cell-years, 1.14 million cells; 12,088 cells have exposure that
  changes over time.
- **By band, not recentered (information):** 20–100 km +0.012 (SE 0.009, p = 0.18); 100–300 km
  +0.007 (SE 0.004, p = 0.06); over 300 km +0.0004 (SE 0.0008).
- **Not a recentering artefact.** Recentering removes most of the variation (exposure and its
  expectation correlate at 0.985 among exposed rows), but the raw band counts are weak too.
- The event study with leads was not run; it was information only, and with F < 1 there is no
  first stage to time.

**Reading.** Bulk freight along predicted routes does not raise lights in the cells it passes
through by a measurable amount: about 1% per active mine 20–100 km upstream, at most. The mining
bound (±0.3 K) remains the only causal bound. Transit-corridor growth yields none, because the
instrument does not move lights. Part 2's sequence (fork doc) loses its step A and the corridor
instrument in steps B and C; the mine instrument remains.

## Change log

- **2026-10-28, before any first stage.** Snapping capped at 25 km (§3). The first build snapped
  mines and ports north of 60°N to the grid edge, up to 2,200 km away; Murmansk drew 130 routes.
  Found by checking routes and snapping distances, with no lights or LST loaded.
