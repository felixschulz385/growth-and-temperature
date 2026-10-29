# ports — IMF PortWatch ports database

| | |
|---|---|
| Config key | `ports` |
| Module | `src/data/sources/ports.py` |
| Steps | FETCH, PREPARE · no `REQUIRES` |
| In panel | no; consumed by `corridors` PREPARE (destination ports) |

## What it is

IMF PortWatch's port list: 2,065 ports (October 2026), each with yearly average vessel counts by
ship type (container, dry bulk, general cargo, Ro-Ro, tanker) estimated from AIS ship tracking
since 2019.

## Raw data (FETCH)

`raw/ports/portwatch_ports.csv`, paged out of PortWatch's ArcGIS Feature Service. About 0.5 MB, so
it can come through the proxy as well as from the transfer node.

## Prepared output (PREPARE)

`prepared/ports/misc/ports.parquet`: `port_id`, `port_name`, `iso3`, `lon`, `lat`, and
`vessel_count_{total,container,dry_bulk,general_cargo,roro,tanker}`.

## Analysis caveats

- **Counts start in 2019.** Ports that became bulk ports because of mines that opened during
  2002–2022 count as bulk ports. The corridor instrument uses only port locations
  (docs/successors/corridors-prespec.md §2).
- **Why not the World Port Index.** Its harbour size measures shelter, not throughput: Port
  Hedland, the largest bulk export port, is "small", and Kamsar is "very small".
