"""Region-level instruments for the fork's DOSE analysis: PREPARE only.

docs/successors/dose-prespec.md §3a fixes the rules. One row per GADM 4.1
first-level region (`GID_1`) and year, for the three instruments steps B and C
use one at a time:

- `mine_count`: SNL mines active in the region that year. Same assignment as
  `snl_mining`'s `adm1_year_counts` (point in the simplified GADM polygon, no
  snapping), so it reproduces the count Gate D1 used.
- `price_frozen`: sum over the mines active in `baseline_year` of
  share x ln(real world price) of each priced commodity they produce. Only
  world prices move it. `price_moving` uses the mines active in each year
  instead (information only).
- `fav_share`: share of the year in which the country's leader was born in the
  region (PLAD). `fav_any` (favoured on any day) and `fav_share_lag1` are
  information only.

Inputs, all read by artefact path (docs/design/09-integrated-pipeline.md §2):

- `snl_mining`: `active_mines` and `commodity_shares` in its prepared DuckDB,
  `properties.primary_commodity` in its raw DuckDB. A mine without
  `commodity_shares` rows gets share 1 on its primary commodity.
- `commodity_prices`: `prepared/commodity_prices/misc/commodity_prices.parquet`.
- `plad`: the raw PLAD table (tab-separated despite its `.dta` name, see
  `plad.py`). Birthplaces are placed by their coordinates, not PLAD's `gid_1`
  codes, which follow an older GADM. A birthplace outside every simplified
  polygon (coastal towns) goes to the nearest region of the same country
  within `birthplace_snap_km`. Leaders flagged `foreign_leader` favour no region.
- `gadm`: `gadm_levelADM_1_simplified.gpkg`.

Outputs, under `prepared/adm1_instruments/adm/`:

- `adm1_price_exposure.parquet`: `GID_1`, `commodity`, `weight` (sum of the
  baseline mines' shares). Lets the analysis leave one commodity out.
- `adm1_instruments.parquet`: every region x year, zeros where nothing is
  exposed. Written last; the target's completion path.

The tables hold region aggregates only. The per-mine assignment stays in
memory: mine-level data is confidential (CLAUDE.md).
"""

from __future__ import annotations

import calendar
import dataclasses
import logging
import os
from typing import List

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection

logger = logging.getLogger(__name__)


class Adm1InstrumentsSource(DataSource):
    """Mine count, mine x world price and regional favoritism per GADM 4.1
    region and year."""

    ID = "adm1_instruments"
    STEPS = (PipelineStep.PREPARE,)
    REQUIRES = (
        (PipelineStep.PREPARE, "snl_mining", PipelineStep.PREPARE),
        (PipelineStep.PREPARE, "commodity_prices", PipelineStep.PREPARE),
        (PipelineStep.PREPARE, "plad", PipelineStep.PREPARE),
        (PipelineStep.PREPARE, "gadm", PipelineStep.PREPARE),
    )

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="adm1_instruments")
        super().__init__(ctx, cfg)
        raw = cfg.raw
        self.years = tuple(raw.get("years", (2002, 2022)))
        self.baseline_year = int(raw.get("baseline_year", 2001))
        self.birthplace_snap_km = float(raw.get("birthplace_snap_km", 25))

    # -- input paths (other sources' artefacts) ------------------------------

    def _snl_prepared_db(self) -> str:
        return os.path.join(
            layout.output_root(self.ctx.data_root, "snl_mining", PipelineStep.PREPARE, agg=layout.MISC_AGG),
            "snl_mining_prepared.duckdb",
        )

    def _snl_raw_db(self) -> str:
        return os.path.join(layout.raw_root(self.ctx.data_root, "snl_mining"), "database.duckdb")

    def _prices_path(self) -> str:
        return os.path.join(
            layout.output_root(self.ctx.data_root, "commodity_prices", PipelineStep.PREPARE, agg=layout.MISC_AGG),
            "commodity_prices.parquet",
        )

    def _plad_path(self) -> str | None:
        root = layout.raw_root(self.ctx.data_root, "plad")
        if not os.path.isdir(root):
            return None
        names = sorted(n for n in os.listdir(root) if "plad" in n.lower() and n.lower().endswith(".dta"))
        return os.path.join(root, names[-1]) if names else None

    def _gadm_path(self) -> str:
        gadm_dir = layout.output_root(
            self.ctx.data_root, "misc", PipelineStep.PREPARE, namespace="gadm", agg=layout.ADM_AGG
        )
        return os.path.join(gadm_dir, "gadm_levelADM_1_simplified.gpkg")

    # -- output paths ----------------------------------------------------------

    def _adm_root(self) -> str:
        return self.output_root(PipelineStep.PREPARE, agg=layout.ADM_AGG)

    # -- plan/execute ----------------------------------------------------------

    def _plan(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        plad = self._plad_path()
        inputs = (self._snl_prepared_db(), self._snl_raw_db(), self._prices_path(), self._gadm_path())
        if plad is None or not all(os.path.exists(p) for p in inputs):
            return []
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key="all",
                output_path=os.path.join(self._adm_root(), "adm1_instruments.parquet"),
                inputs=(*inputs, plad), completion=Completion.PATH_EXISTS,
            )
        ]

    def _execute(self, target: StepTarget) -> bool:
        if not self.cfg.override and os.path.exists(target.output_path):
            logger.info("Skipping adm1_instruments -- already complete: %s", target.output_path)
            return True
        self._build(target.output_path)
        return True

    # -- the computation -----------------------------------------------------

    def _connect(self):
        import duckdb

        con = duckdb.connect()
        con.execute(f"PRAGMA threads={int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 4))}")
        con.execute("SET enable_progress_bar = false")
        con.execute("INSTALL spatial; LOAD spatial")
        return con

    def _build(self, out_path: str) -> None:
        import pandas as pd

        con = self._connect()
        con.execute(f"""
            CREATE TABLE regions AS
            SELECT GID_0, GID_1, geom FROM ST_Read('{self._gadm_path()}') WHERE GID_1 IS NOT NULL
        """)
        y0, y1 = self.years
        years = pd.DataFrame({"year": range(y0, y1 + 1)})
        grid = con.execute("SELECT GID_0, GID_1 FROM regions").df().merge(years, how="cross")

        mines = self._mine_regions(con)
        shares = self._mine_shares(con, mines)
        prices = pd.read_parquet(self._prices_path(), columns=["commodity", "year", "ln_price_real"])
        con.execute("ATTACH '%s' AS p (READ_ONLY)" % self._snl_prepared_db())
        active = con.execute("SELECT DISTINCT property_id, year FROM p.active_mines").df()
        con.execute("DETACH p")
        active = active.merge(mines[["property_id", "GID_1"]], on="property_id")

        # mine_count
        count = (active[active.year.between(y0, y1)].groupby(["GID_1", "year"]).size().rename("mine_count"))

        # price exposure frozen at the baseline year, and the moving version
        priced = shares[shares.commodity.isin(prices.commodity.unique())]
        base = active.loc[active.year == self.baseline_year, ["property_id", "GID_1"]]
        exposure = base.merge(priced, on="property_id").groupby(["GID_1", "commodity"], as_index=False)["share"].sum()
        exposure = exposure.rename(columns={"share": "weight"})
        frozen = (exposure.merge(prices, on="commodity").assign(v=lambda d: d.weight * d.ln_price_real)
                  .groupby(["GID_1", "year"])["v"].sum().rename("price_frozen"))
        moving = (active[active.year.between(y0, y1)].merge(priced, on="property_id")
                  .merge(prices, on=["commodity", "year"])
                  .assign(v=lambda d: d.share * d.ln_price_real)
                  .groupby(["GID_1", "year"])["v"].sum().rename("price_moving"))

        fav = self._favoritism(con)

        out = grid.set_index(["GID_1", "year"]).join([count, frozen, moving, fav])
        fill = {c: 0 for c in ("mine_count", "price_frozen", "price_moving", "fav_share", "fav_any", "fav_share_lag1")}
        out = out.fillna(fill).reset_index()
        out["mine_count"] = out.mine_count.astype("int32")
        out["fav_any"] = out.fav_any.astype("int8")

        os.makedirs(self._adm_root(), exist_ok=True)
        exposure.to_parquet(os.path.join(self._adm_root(), "adm1_price_exposure.parquet"), index=False)
        out.to_parquet(out_path, index=False)
        logger.info(
            "adm1_instruments: %d region-years; %d regions with a priced baseline mine; "
            "%d favoured region-years -> %s",
            len(out), exposure.GID_1.nunique(), int((out.fav_share > 0).sum()), out_path,
        )
        con.close()

    def _mine_regions(self, con):
        """Each SNL mine's region: point in the simplified polygon, as
        `snl_mining`'s `adm1_year_counts` does. Mines outside every polygon are
        left out, as there."""
        con.execute("ATTACH '%s' AS p (READ_ONLY)" % self._snl_prepared_db())
        con.execute("ATTACH '%s' AS r (READ_ONLY)" % self._snl_raw_db())
        mines = con.execute("""
            WITH m AS (
                SELECT property_id, any_value(longitude) AS lon, any_value(latitude) AS lat
                FROM p.active_mines GROUP BY property_id
            )
            SELECT m.property_id, g.GID_1, pr.primary_commodity
            FROM m JOIN regions g ON ST_Intersects(g.geom, ST_Point(m.lon, m.lat))
            LEFT JOIN r.properties pr USING (property_id)
        """).df()
        con.execute("DETACH p")
        con.execute("DETACH r")
        # a point on a shared border falls in two polygons: keep one
        return mines.drop_duplicates("property_id")

    def _mine_shares(self, con, mines):
        """`commodity_shares` where a mine has them, else share 1 on its
        primary commodity (canonical keys, `commodities.normalize_commodity`)."""
        import pandas as pd

        from src.data.sources.commodities import normalize_commodity

        con.execute("ATTACH '%s' AS p (READ_ONLY)" % self._snl_prepared_db())
        listed = con.execute("SELECT property_id, commodity, share FROM p.commodity_shares").df()
        con.execute("DETACH p")
        rest = mines.loc[~mines.property_id.isin(listed.property_id) & mines.primary_commodity.notna()]
        rest = pd.DataFrame({
            "property_id": rest.property_id,
            "commodity": rest.primary_commodity.map(lambda c: normalize_commodity(c, source="snl")),
            "share": 1.0,
        }).dropna(subset=["commodity"])
        return pd.concat([listed, rest], ignore_index=True)

    def _favoritism(self, con):
        """`fav_share`, `fav_any` and `fav_share_lag1` per (GID_1, year), from
        the leaders' birthplaces and their days in office."""
        import pandas as pd

        plad = pd.read_table(self._plad_path())
        plad = plad[(plad.foreign_leader.astype(str) != "1") & plad.latitude.notna() & plad.longitude.notna()]
        plad = plad[["plad_id", "gid_0", "latitude", "longitude", "startdate", "enddate"]].copy()
        con.register("leaders", plad)
        placed = con.execute("""
            SELECT l.plad_id, g.GID_1 FROM leaders l
            JOIN regions g ON ST_Intersects(g.geom, ST_Point(l.longitude, l.latitude))
        """).df().drop_duplicates("plad_id")
        # the rest: nearest region of the same country, within the snap distance
        snapped = con.execute(f"""
            WITH rest AS (SELECT * FROM leaders WHERE plad_id NOT IN (SELECT plad_id FROM placed)),
            d AS (
                SELECT l.plad_id, g.GID_1,
                       ST_Distance(ST_Transform(ST_Point(l.longitude, l.latitude), 'EPSG:4326', 'EPSG:6933', true),
                                   ST_Transform(g.geom, 'EPSG:4326', 'EPSG:6933', true)) / 1000 AS km
                FROM rest l JOIN regions g ON g.GID_0 = l.gid_0
            )
            SELECT plad_id, arg_min(GID_1, km) AS GID_1, min(km) AS km FROM d GROUP BY plad_id
        """).df()
        snapped = snapped[snapped.km <= self.birthplace_snap_km]
        logger.info(
            "PLAD birthplaces: %d in a polygon, %d snapped (<= %.0f km), %d dropped",
            len(placed), len(snapped), self.birthplace_snap_km, len(plad) - len(placed) - len(snapped),
        )
        regions = pd.concat([placed, snapped[["plad_id", "GID_1"]]])
        spells = plad.merge(regions, on="plad_id")

        y0, y1 = self.years
        rows = []
        for s in spells.itertuples():
            start, end = pd.Timestamp(s.startdate), pd.Timestamp(s.enddate)
            for year in range(max(start.year, y0 - 1), min(end.year, y1) + 1):
                lo = max(start, pd.Timestamp(year, 1, 1))
                hi = min(end, pd.Timestamp(year, 12, 31))
                days = (hi - lo).days + 1
                if days > 0:
                    rows.append((s.GID_1, year, days / (366 if calendar.isleap(year) else 365)))
        fav = pd.DataFrame(rows, columns=["GID_1", "year", "fav_share"]).groupby(["GID_1", "year"]).sum()
        fav["fav_share"] = fav.fav_share.clip(upper=1.0)
        fav["fav_any"] = 1
        lag = fav.fav_share.rename("fav_share_lag1").reset_index().assign(year=lambda d: d.year + 1)
        fav = fav.join(lag.set_index(["GID_1", "year"]), how="outer")
        return fav[fav.index.get_level_values("year").to_series().between(y0, y1).to_numpy()]


registry.register(
    Adm1InstrumentsSource.ID, __name__, Adm1InstrumentsSource.__name__,
    Adm1InstrumentsSource.STEPS, requires=Adm1InstrumentsSource.REQUIRES,
)
