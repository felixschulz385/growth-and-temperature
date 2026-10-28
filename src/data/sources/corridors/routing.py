"""Least-cost routes from bulk mines to ports on the 10 km analysis grid, and
the corridor exposure built from them (docs/successors/corridors-prespec.md
§§3-5). Plain functions over DataFrames/arrays; `source.py` wires them into
PREPARE.

**The routing graph is the analysis grid itself.** Assembly coarsens the
1 km grid into F x F blocks that restart at every 2048-pixel tile origin
(`src/data/assemble/sql_engine.py::_pixel_id_sql`), so the 10 km cells are not
a regular global lattice: the last block of each tile is 8 pixels wide. The
blocks still form a regular lattice *in index space* -- block row
`(row // TS) * ceil(TS / F) + (row % TS) // F`, likewise for columns -- so
8-neighbour adjacency is read off that lattice, while edge lengths use each
block's true centroid. Every node carries the panel's own `pixel_id`, computed
with assembly's SQL, so exposure joins the panel exactly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0088


# ---------------------------------------------------------------------------
# Cells: one row per 10 km analysis cell, aggregated from 1 km grids
# ---------------------------------------------------------------------------

def cell_table(con, *, land_glob: str, country_glob: str, slope_glob: str, g, affine) -> pd.DataFrame:
    """Aggregate the 1 km land mask, country ids and slope to the analysis
    cells of grid *g* (`GridFacts`, unshaken). Returns `pixel_id`, lattice
    indices `br`/`bc`, centroid `lon`/`lat`, `land_share`, `GID_0` (modal
    non-zero id, 0 if none) and `slope` (mean degrees), for cells with any
    land."""
    from pyproj import Transformer

    from src.data.assemble.sql_engine import _pixel_id_sql

    W, TS, F = g.W, g.TS, g.F
    nb = -(-TS // F)  # blocks per full tile
    pid = _pixel_id_sql("cell_id", g, F, 0, 0)
    br = f"((cell_id // {W}) // {TS}) * {nb} + ((cell_id // {W}) % {TS}) // {F}"
    bc = f"((cell_id % {W}) // {TS}) * {nb} + ((cell_id % {W}) % {TS}) // {F}"

    land = con.execute(f"""
        SELECT {pid} AS pixel_id, {br} AS br, {bc} AS bc,
               avg(land_mask::DOUBLE) AS land_share,
               avg((cell_id % {W})::DOUBLE) AS col, avg((cell_id // {W})::DOUBLE) AS row
        FROM read_parquet('{land_glob}')
        GROUP BY ALL HAVING max(land_mask::INT) = 1
    """).df()
    country = con.execute(f"""
        SELECT {pid} AS pixel_id, mode(GID_0) AS GID_0
        FROM read_parquet('{country_glob}') WHERE GID_0 > 0 GROUP BY 1
    """).df()
    slope = con.execute(f"""
        SELECT {pid} AS pixel_id, avg(slope_mean_10km) AS slope
        FROM read_parquet('{slope_glob}') WHERE NOT isnan(slope_mean_10km) GROUP BY 1
    """).df()

    cells = land.merge(country, on="pixel_id", how="left").merge(slope, on="pixel_id", how="left")
    cells["GID_0"] = cells["GID_0"].fillna(0).astype("int64")
    cells["slope"] = cells["slope"].fillna(0.0)  # only south of 56°S, where EarthEnv ends

    # Centroid of the member pixels' centres, in EPSG:6933 metres -> lon/lat.
    x = affine.c + (cells["col"].to_numpy() + 0.5) * affine.a
    y = affine.f + (cells["row"].to_numpy() + 0.5) * affine.e
    lon, lat = Transformer.from_crs("EPSG:6933", "EPSG:4326", always_xy=True).transform(x, y)
    cells["lon"], cells["lat"] = lon, lat
    return cells.drop(columns=["col", "row"]).sort_values(["br", "bc"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _unit_vectors(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_r, lat_r = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)])


def _chord_to_km(chord: np.ndarray) -> np.ndarray:
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2, 0, 1))


@dataclass
class PointIndex:
    """Nearest-neighbour search on the sphere (KD-tree on unit vectors)."""

    lon: np.ndarray
    lat: np.ndarray

    def __post_init__(self):
        from scipy.spatial import cKDTree

        self._tree = cKDTree(_unit_vectors(np.asarray(self.lon), np.asarray(self.lat)))

    def nearest(self, lon, lat) -> Tuple[np.ndarray, np.ndarray]:
        """(index, great-circle distance in km) of the nearest point."""
        chord, idx = self._tree.query(_unit_vectors(np.asarray(lon), np.asarray(lat)))
        return idx, _chord_to_km(chord)


def geodesic_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    from pyproj import Geod

    _, _, d = Geod(ellps="WGS84").inv(lon1, lat1, lon2, lat2)
    return np.asarray(d) / 1000.0


# ---------------------------------------------------------------------------
# Graph and least-cost search
# ---------------------------------------------------------------------------

#: forward half of the 8-neighbourhood; the graph is undirected
_OFFSETS = ((0, 1), (1, -1), (1, 0), (1, 1))


def build_graph(nodes: pd.DataFrame, slope_scale_deg: float | None):
    """Undirected 8-neighbour graph over *nodes* (`br`, `bc`, `lon`, `lat`,
    `slope`, positional index = node id). Edge cost = geodesic km x
    (1 + mean slope / *slope_scale_deg*); flat when *slope_scale_deg* is
    None. Returns the cost matrix (CSR, one entry per undirected edge;
    route lengths are recomputed along each route, not kept here)."""
    from scipy.sparse import coo_matrix

    br = nodes["br"].to_numpy().astype(np.int64)  # DuckDB returns unsigned
    bc = nodes["bc"].to_numpy().astype(np.int64)
    n = len(nodes)
    lattice = np.full((br.max() + 2, bc.max() + 2), -1, dtype=np.int64)
    lattice[br, bc] = np.arange(n)

    lon, lat, slope = (nodes[c].to_numpy() for c in ("lon", "lat", "slope"))
    rows, cols, costs = [], [], []
    for dr, dc in _OFFSETS:
        r2, c2 = br + dr, bc + dc
        ok = (r2 >= 0) & (c2 >= 0) & (c2 < lattice.shape[1])
        a = np.nonzero(ok)[0]
        b = lattice[r2[ok], c2[ok]]
        keep = b >= 0
        a, b = a[keep], b[keep]
        length = geodesic_km(lon[a], lat[a], lon[b], lat[b])
        factor = 1.0 if slope_scale_deg is None else 1.0 + (slope[a] + slope[b]) / 2.0 / slope_scale_deg
        rows.append(a)
        cols.append(b)
        costs.append(length * factor)
    rows, cols, costs = np.concatenate(rows), np.concatenate(cols), np.concatenate(costs)
    logger.info("Routing graph: %d nodes, %d undirected edges", n, len(rows))
    return coo_matrix((costs, (rows, cols)), shape=(n, n)).tocsr()


def least_cost_tree(graph, source_nodes: Sequence[int]):
    """One multi-source Dijkstra: for every node, the cheapest source and the
    predecessor towards it. Returns (cost, predecessors, source node)."""
    from scipy.sparse.csgraph import dijkstra

    cost, pred, src = dijkstra(
        graph, directed=False, indices=np.asarray(source_nodes), min_only=True, return_predecessors=True
    )
    return cost, pred, src


def trace_routes(
    start_nodes: np.ndarray, mine_ids: Sequence[str], pred: np.ndarray, src: np.ndarray, lon: np.ndarray, lat: np.ndarray
) -> pd.DataFrame:
    """Follow predecessors from each mine's node to its source (port) node.
    Returns one row per (mine, route node): `property_id`, `node`,
    `dist_km` (geodesic distance travelled from the mine's node). Mines on
    nodes no port reaches are left out."""
    out_mine: List[str] = []
    out_node: List[np.ndarray] = []
    out_dist: List[np.ndarray] = []
    for mine, start in zip(mine_ids, start_nodes):
        if src[start] < 0:
            continue
        path = [start]
        node = start
        while pred[node] >= 0:
            node = pred[node]
            path.append(node)
        path = np.asarray(path)
        step = geodesic_km(lon[path[:-1]], lat[path[:-1]], lon[path[1:]], lat[path[1:]]) if len(path) > 1 else np.array([])
        out_mine.extend([mine] * len(path))
        out_node.append(path)
        out_dist.append(np.concatenate([[0.0], np.cumsum(step)]))
    if not out_node:
        return pd.DataFrame(columns=["property_id", "node", "dist_km"])
    return pd.DataFrame(
        {"property_id": out_mine, "node": np.concatenate(out_node), "dist_km": np.concatenate(out_dist)}
    )


# ---------------------------------------------------------------------------
# Exposure and recentering
# ---------------------------------------------------------------------------

def band_label(lo: float, hi: float | None) -> str:
    return f"z_{int(lo)}_{int(hi)}" if hi is not None else f"z_{int(lo)}_plus"


def exposure(
    routes: pd.DataFrame,
    active: pd.DataFrame,
    strata: pd.Series,
    *,
    bands_km: Sequence[float],
    main_max_km: float,
    years: Iterable[int],
) -> pd.DataFrame:
    """Corridor exposure per (pixel_id, year).

    *routes*: `pixel_id`, `property_id`, `dist_km`. *active*:
    `property_id`, `year`, one row per active mine-year. *strata*:
    `property_id` -> stratum label (country x commodity).

    Columns: one count per distance band (`bands_km` edges, the last band
    open-ended); `z_main` = count over [bands_km[0], main_max_km);
    `mu_main`, its expectation when opening/closing years are reassigned
    within strata (Borusyak and Hull 2023: the stratum's share of active
    mines in the year); `z_main_rc` = `z_main - mu_main`. Only cells that lie
    on some route at distance >= bands_km[0] appear; every other cell is
    zero on all columns.
    """
    years = list(years)
    edges = list(bands_km) + [np.inf]
    r = routes[routes["dist_km"] >= edges[0]].copy()
    r["band"] = pd.cut(r["dist_km"], bins=edges, right=False, labels=False)
    r["main"] = r["dist_km"] < main_max_km

    act = active[active["year"].isin(years)][["property_id", "year"]].drop_duplicates()
    act["s"] = 1.0

    # Counts by band: route rows x active years of that mine.
    ra = r.merge(act, on="property_id")
    counts = (
        ra.pivot_table(index=["pixel_id", "year"], columns="band", values="s", aggfunc="sum", fill_value=0.0)
        .rename(columns={i: band_label(edges[i], None if np.isinf(edges[i + 1]) else edges[i + 1]) for i in range(len(edges) - 1)})
    )
    z_main = ra[ra["main"]].groupby(["pixel_id", "year"])["s"].sum().rename("z_main")

    # Expected exposure: stratum share active per year, over routed mines.
    routed = pd.Index(r["property_id"].unique())
    k = strata.reindex(routed).rename("stratum").reset_index().rename(columns={"index": "property_id"})
    grid = k.merge(pd.DataFrame({"year": years}), how="cross")
    grid = grid.merge(act, on=["property_id", "year"], how="left").fillna({"s": 0.0})
    share = grid.groupby(["stratum", "year"])["s"].mean().rename("s_bar").reset_index()
    rm = r[r["main"]][["pixel_id", "property_id"]].merge(k, on="property_id").merge(share, on="stratum")
    mu = rm.groupby(["pixel_id", "year"])["s_bar"].sum().rename("mu_main")

    # Every on-route cell gets every year, including years with no active mine.
    cells = pd.Index(r["pixel_id"].unique(), name="pixel_id")
    full = pd.MultiIndex.from_product([cells, years], names=["pixel_id", "year"])
    out = pd.DataFrame(index=full).join(counts).join(z_main).join(mu).fillna(0.0)
    for i in range(len(edges) - 1):
        col = band_label(edges[i], None if np.isinf(edges[i + 1]) else edges[i + 1])
        if col not in out:
            out[col] = 0.0
    out["z_main_rc"] = out["z_main"] - out["mu_main"]
    return out.reset_index()
