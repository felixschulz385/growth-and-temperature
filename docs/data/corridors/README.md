# corridors — bulk-mine transit-corridor exposure

| | |
|---|---|
| Config key | `corridors` |
| Module | `src/data/sources/corridors/source.py`, computation in `corridors/routing.py` |
| Steps | PREPARE only · `REQUIRES` snl_mining, ports, topography, osm, gadm (all PREPARE) |
| In panel | no; tables keyed by the 10 km analysis grid's `pixel_id`, joined at analysis time |

## What it is

Derived, not downloaded. Least-cost routes from bulk mines to dry-bulk ports, and an annual
count of active mines whose route passes each 10 km cell. This is the instrument of the fork on
branch `fork/corridors-dose`. Every rule (mines, ports, cost, bands, recentering, exclusions) is
fixed in [`../../successors/corridors-prespec.md`](../../successors/corridors-prespec.md); the
config keys mirror it.

## Prepared output (PREPARE)

Under `prepared/corridors/misc/`:

| File | Rows | Columns |
|---|---|---|
| `cells_10km.parquet` | analysis cells with any land (shared by all variants) | `pixel_id`, `br`/`bc` (lattice indices), `lon`/`lat` (centroid), `land_share`, `GID_0`, `slope` |
| `<variant>/ports.parquet` | destination ports | PortWatch columns, `node`, `snap_km`, `used` |
| `<variant>/mines.parquet` | SNL mines active at some point in 2002–2022 | `commodity`, `node`, `snap_km`, `GID_0`, `stratum`, `shock`, `routed`, `port_id` |
| `<variant>/routes.parquet` | (mine, route cell) | `pixel_id`, `property_id`, `dist_km` (along the route from the mine) |
| `<variant>/cell_flags.parquet` | every cell | `port_id` (catchment), `mine_dist_km`, `mine_dist_band` (0 = under 20 km), `near_mine`, `near_port` |
| `<variant>/exposure.parquet` | (on-route cell, year) | `z_20_100`, `z_100_300`, `z_300_plus`, `z_main`, `mu_main`, `z_main_rc` |

Cells missing from `exposure.parquet` have zero exposure. Variants: `main` (pre-specified),
`flat_cost`, `all_mines`, `any_ship_ports`.

**Why `pixel_id` tables, not a 1 km grid.** Routes and exposure exist only on the 10 km cells;
a 1 km grid would repeat each value 100 times in ~2,100 mostly-zero tile-year parts. The routing
graph is assembly's own block lattice, and `pixel_id` comes from assembly's SQL, so the tables
join `grid=10km` exactly (checked 2026-10-28: all 1,205,481 panel cells match). They are valid for
`resolution_m` only and unshaken grids only.

## First build (2026-10-28, `main`)

- 1,175,086 routing nodes; 3,256 bulk mines on the grid, 3,228 routed to 287 ports.
- After the exclusions, 12,097 cells have exposure that changes over time, in 202 port
  catchments. The ten largest catchments hold 41% of exposed cells.
- Routes look right for the big coal and ore regions (Shanxi → Tianjin/Huanghua, Hunter Valley →
  Newcastle). Least cost misses dedicated rail lines: Mpumalanga coal goes to Maputo, not Richards
  Bay.

## Analysis caveats

- **Predicted, not observed, routes.** No check against today's road or rail network yet (GRIP
  would be the source).
- **The grid ends at 60°N.** Mines and ports beyond it are dropped (25 km snapping cap), which
  removes Arctic Russia and northern Canada.
- **Lumpy clusters.** Port catchments are few and unequal in size; clustered inference by port
  rests on ~200 clusters dominated by a few.
