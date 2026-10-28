"""Transit-corridor exposure: PREPARE only, derived from other sources.

docs/successors/corridors-prespec.md fixes every rule this module applies;
`routing.py` holds the computation. Inputs, all read by artefact path, never
a class import (docs/design/09-integrated-pipeline.md §2):

- `snl_mining`: `active_mines` in its prepared DuckDB (mine points, active
  years) and `properties.primary_commodity` in its raw DuckDB;
- `ports`: `prepared/ports/misc/ports.parquet`;
- `osm` land mask, `gadm` country ids and `topography` slope: 1 km pixel
  grids, aggregated here to the analysis grid's cells.

**Outputs are keyed by the analysis grid's `pixel_id`, not written as a 1 km
grid.** Routes and exposure are defined on the 10 km cells (`routing.py`
docstring), so a 1 km grid would only repeat each cell's value 100 times
across ~2,100 mostly-zero tile-year parts. The tables join onto the
assembled `grid=<resolution_m/1000>km` panel by `pixel_id` (and `year`), like
the admin-keyed sidecars join by `GID_N`. Under
`prepared/corridors/misc/`:

- `cells_<res>.parquet`: every analysis cell with land -- lattice indices,
  centroid, land share, `GID_0`, slope. Shared by all variants.
- `<variant>/`: `ports.parquet` (destination ports and their snapped
  nodes), `mines.parquet` (one row per mine: commodity, stratum, node, port,
  whether routed), `routes.parquet` (`pixel_id`, `property_id`, `dist_km`),
  `cell_flags.parquet` (per cell: `port_id` catchment, distance to nearest
  SNL mine and its band, `near_mine`, `near_port`) and `exposure.parquet`
  (per on-route cell-year: band counts, `z_main`, `mu_main`, `z_main_rc`).
  `exposure.parquet` is written last and is the target's completion path.

**Variants.** `variants:` in the config block maps a name to overrides of
the top-level rule keys (`commodities`, `port_count_column`,
`port_min_count`, `slope_scale_deg`); each is its own PREPARE target. `main`
is the pre-specified instrument; the others are its robustness checks.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, Dict, List

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection

logger = logging.getLogger(__name__)

#: corridors-prespec.md §1
DEFAULT_COMMODITIES = (
    "Coal", "Iron Ore", "Bauxite", "Phosphate", "Potash", "Manganese", "Chromite",
    "Ilmenite", "Heavy Mineral Sands", "Rutile", "Zircon", "Titanium",
)
_RULE_KEYS = ("commodities", "port_count_column", "port_min_count", "slope_scale_deg")


class CorridorsSource(DataSource):
    """Bulk-mine-to-port least-cost routes and corridor exposure on the
    analysis grid."""

    ID = "corridors"
    STEPS = (PipelineStep.PREPARE,)
    REQUIRES = (
        (PipelineStep.PREPARE, "snl_mining", PipelineStep.PREPARE),
        (PipelineStep.PREPARE, "ports", PipelineStep.PREPARE),
        (PipelineStep.PREPARE, "topography", PipelineStep.PREPARE),
        (PipelineStep.PREPARE, "osm", PipelineStep.PREPARE),
        (PipelineStep.PREPARE, "gadm", PipelineStep.PREPARE),
    )

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="corridors")
        super().__init__(ctx, cfg)
        raw = cfg.raw
        self.resolution_m = int(raw.get("resolution_m", 10000))
        self.years = tuple(raw.get("years", (2002, 2022)))
        self.min_land_share = float(raw.get("min_land_share", 0.5))
        self.max_snap_km = float(raw.get("max_snap_km", 25))
        self.bands_km = tuple(raw.get("bands_km", (20, 100, 300)))
        self.main_max_km = float(raw.get("main_max_km", 300))
        self.mine_exclusion_km = float(raw.get("mine_exclusion_km", 20))
        self.port_exclusion_km = float(raw.get("port_exclusion_km", 30))
        self.mine_distance_bands_km = tuple(raw.get("mine_distance_bands_km", (20, 50, 100, 200, 300)))
        self.base_rules: Dict[str, Any] = {
            "commodities": list(raw.get("commodities", DEFAULT_COMMODITIES)),
            "port_count_column": raw.get("port_count_column", "vessel_count_dry_bulk"),
            "port_min_count": float(raw.get("port_min_count", 50)),
            "slope_scale_deg": raw.get("slope_scale_deg", 5.0),
        }
        self.variants: Dict[str, Dict[str, Any]] = dict(raw.get("variants") or {"main": {}})
        for name, overrides in self.variants.items():
            unknown = set(overrides or {}) - set(_RULE_KEYS)
            if unknown:
                raise ValueError(f"corridors variant {name!r}: unknown keys {sorted(unknown)}")

    # -- input paths (other sources' artefacts) ------------------------------

    def _grid_glob(self, data_path: str, family: str) -> str:
        root = layout.grid_store_path(self.ctx.data_root, data_path, grid_id=self.ctx.grid_id, family=family, suffix="")
        return os.path.join(root, "ix=*", "iy=*", "part.parquet")

    def _snl_prepared_db(self) -> str:
        return os.path.join(
            layout.output_root(self.ctx.data_root, "snl_mining", PipelineStep.PREPARE, agg=layout.MISC_AGG),
            "snl_mining_prepared.duckdb",
        )

    def _snl_raw_db(self) -> str:
        return os.path.join(layout.raw_root(self.ctx.data_root, "snl_mining"), "database.duckdb")

    def _ports_path(self) -> str:
        return os.path.join(
            layout.output_root(self.ctx.data_root, "ports", PipelineStep.PREPARE, agg=layout.MISC_AGG), "ports.parquet"
        )

    # -- output paths ----------------------------------------------------------

    def _misc_root(self) -> str:
        return self.output_root(PipelineStep.PREPARE, agg=layout.MISC_AGG)

    def _cells_path(self) -> str:
        return os.path.join(self._misc_root(), f"cells_{self.resolution_m // 1000}km.parquet")

    def _variant_dir(self, variant: str) -> str:
        return os.path.join(self._misc_root(), variant)

    # -- plan/execute ----------------------------------------------------------

    def _plan(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        inputs = (self._snl_prepared_db(), self._snl_raw_db(), self._ports_path())
        if not all(os.path.exists(p) for p in inputs):
            return []
        keys = selection.keys or tuple(self.variants)
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key=name,
                output_path=os.path.join(self._variant_dir(name), "exposure.parquet"),
                inputs=inputs, completion=Completion.PATH_EXISTS,
                meta={"rules": {**self.base_rules, **(self.variants[name] or {})}},
            )
            for name in keys if name in self.variants
        ]

    def _execute(self, target: StepTarget) -> bool:
        if not self.cfg.override and os.path.exists(target.output_path):
            logger.info("Skipping corridors %s -- already complete: %s", target.key, target.output_path)
            return True
        self._run_variant(target.key, target.meta["rules"], target.output_path)
        return True

    # -- the computation -----------------------------------------------------

    def _connect(self):
        import duckdb

        con = duckdb.connect()
        con.execute(f"PRAGMA threads={int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 4))}")
        con.execute("SET enable_progress_bar = false")
        return con

    def _cells(self, con):
        """The analysis cells, built once and cached (shared by variants)."""
        import pandas as pd

        from src.data.assemble.sql_engine import GridFacts
        from src.data.common.geobox import get_target_geobox
        from src.data.sources.corridors import routing

        path = self._cells_path()
        if os.path.exists(path) and not self.cfg.override:
            return pd.read_parquet(path)
        g = GridFacts.build(float(self.resolution_m), (0.0, 0.0))
        cells = routing.cell_table(
            con,
            land_glob=self._grid_glob("misc", "land_mask"),
            country_glob=self._grid_glob("misc", "country_id"),
            slope_glob=self._grid_glob("topography", "slope"),
            g=g,
            affine=get_target_geobox(self.ctx).affine,
        )
        _write_parquet(cells, path)
        logger.info("Corridors cells: %d analysis cells with land -> %s", len(cells), path)
        return cells

    def _mines(self, con):
        """One row per SNL mine active at some point in the panel years, with
        its primary commodity; and the active mine-years."""
        con.execute(f"ATTACH '{self._snl_prepared_db()}' AS p (READ_ONLY)")
        con.execute(f"ATTACH '{self._snl_raw_db()}' AS r (READ_ONLY)")
        y0, y1 = self.years
        active = con.execute(f"""
            SELECT property_id, year FROM p.active_mines WHERE year BETWEEN {y0} AND {y1}
        """).df()
        mines = con.execute(f"""
            SELECT a.property_id, any_value(a.longitude) AS lon, any_value(a.latitude) AS lat,
                   any_value(a.opening_year) AS opening_year, any_value(a.closing_year) AS closing_year,
                   any_value(pr.primary_commodity) AS commodity
            FROM p.active_mines a LEFT JOIN r.properties pr USING (property_id)
            WHERE a.year BETWEEN {y0} AND {y1}
            GROUP BY a.property_id
        """).df()
        con.execute("DETACH p")
        con.execute("DETACH r")
        return mines, active

    def _run_variant(self, variant: str, rules: Dict[str, Any], exposure_path: str) -> None:
        import numpy as np
        import pandas as pd

        from src.data.sources.corridors import routing

        out_dir = self._variant_dir(variant)
        con = self._connect()
        cells = self._cells(con)
        mines, active = self._mines(con)
        con.close()

        # Routing nodes: cells that are at least half land.
        nodes = cells[cells["land_share"] >= self.min_land_share].reset_index(drop=True)
        node_index = routing.PointIndex(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
        graph = routing.build_graph(nodes, rules["slope_scale_deg"])

        # Destination ports (prespec §2), snapped to their nearest node.
        ports = pd.read_parquet(self._ports_path())
        ports = ports[ports[rules["port_count_column"]] >= rules["port_min_count"]].copy()
        ports["node"], ports["snap_km"] = node_index.nearest(ports["lon"], ports["lat"])
        # Beyond the grid (|lat| > 60) or far inland of any passable cell.
        ports = ports[ports["snap_km"] <= self.max_snap_km]
        # Several ports can snap to one node: the busiest one names it.
        ports = ports.sort_values(rules["port_count_column"], ascending=False)
        source_ports = ports.drop_duplicates("node")
        _, pred, src = routing.least_cost_tree(graph, source_ports["node"].to_numpy())
        port_of_node = pd.Series(source_ports["port_id"].to_numpy(), index=source_ports["node"].to_numpy())

        # Shock mines (prespec §1) and their strata (country x commodity, §5).
        mines["node"], mines["snap_km"] = node_index.nearest(mines["lon"], mines["lat"])
        mines["GID_0"] = nodes["GID_0"].to_numpy()[mines["node"]]
        commodities = rules["commodities"]
        on_grid = mines["snap_km"] <= self.max_snap_km
        shock = mines[on_grid] if commodities is None else mines[on_grid & mines["commodity"].isin(commodities)]
        routes = routing.trace_routes(
            shock["node"].to_numpy(), shock["property_id"].tolist(), pred, src,
            nodes["lon"].to_numpy(), nodes["lat"].to_numpy(),
        )
        routes["pixel_id"] = nodes["pixel_id"].to_numpy()[routes["node"].to_numpy()]
        mines["shock"] = mines["property_id"].isin(shock["property_id"])
        mines["routed"] = mines["property_id"].isin(routes["property_id"])
        mines["port_id"] = np.where(
            mines["routed"], port_of_node.reindex(src[mines["node"].to_numpy()]).to_numpy(), None
        )
        mines["stratum"] = mines["GID_0"].astype(str) + "|" + mines["commodity"].fillna("unknown")
        used_ports = set(mines.loc[mines["routed"], "port_id"])
        ports["used"] = ports["port_id"].isin(used_ports)

        # Per-cell flags (prespec §6): catchment, distance to any SNL mine,
        # nearness to a used destination port.
        flags = cells[["pixel_id"]].copy()
        cell_node, _ = node_index.nearest(cells["lon"], cells["lat"])
        flags["port_id"] = port_of_node.reindex(src[cell_node]).to_numpy()
        mine_index = routing.PointIndex(mines["lon"].to_numpy(), mines["lat"].to_numpy())
        _, d_mine = mine_index.nearest(cells["lon"], cells["lat"])
        flags["mine_dist_km"] = d_mine
        edges = list(self.mine_distance_bands_km) + [np.inf]
        flags["mine_dist_band"] = np.searchsorted(edges, d_mine, side="right")  # 0 = inside the first edge
        flags["near_mine"] = d_mine < self.mine_exclusion_km
        used = ports[ports["used"]]
        if len(used):
            _, d_port = routing.PointIndex(used["lon"].to_numpy(), used["lat"].to_numpy()).nearest(cells["lon"], cells["lat"])
        else:
            d_port = np.full(len(cells), np.inf)
        flags["near_port"] = d_port < self.port_exclusion_km

        exposure = routing.exposure(
            routes[["pixel_id", "property_id", "dist_km"]],
            active,
            mines.set_index("property_id")["stratum"],
            bands_km=self.bands_km,
            main_max_km=self.main_max_km,
            years=range(self.years[0], self.years[1] + 1),
        )

        _write_parquet(ports, os.path.join(out_dir, "ports.parquet"))
        _write_parquet(mines, os.path.join(out_dir, "mines.parquet"))
        _write_parquet(routes, os.path.join(out_dir, "routes.parquet"))
        _write_parquet(flags, os.path.join(out_dir, "cell_flags.parquet"))
        _write_parquet(exposure, exposure_path)  # last: the completion path
        logger.info(
            "Corridors %s: %d shock mines, %d routed, %d destination ports used, %d on-route cells -> %s",
            variant, len(shock), int(mines["routed"].sum()), len(used_ports),
            exposure["pixel_id"].nunique(), out_dir,
        )


def _write_parquet(df, path: str) -> None:
    """Temp file + os.replace, so Completion.PATH_EXISTS is safe."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


registry.register(
    CorridorsSource.ID, __name__, CorridorsSource.__name__, CorridorsSource.STEPS, requires=CorridorsSource.REQUIRES
)
