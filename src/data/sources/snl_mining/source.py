"""SNL/S&P Global mining property tables: prepare + grid, no fetch.

docs/design/09-integrated-pipeline.md §6: newly registered (today it's
unregistered on the download side -- notebooks + README only, because its
acquisition is a manual S&P Global `.xls` export plus an OpenAI batch-
enrichment loop, genuinely not automatable). `FETCH` is declared absent
rather than faked; the notebooks that produce the stage-0 DuckDB this source
depends on move to `src/data/sources/snl_mining/notebooks/` so the whole
lifecycle -- including its manual part -- lives in one directory.
`src/data/sources/snl_mining/scraper/` is a related but separate,
interactively-run tool (Selenium-driven Capital IQ screener/profile scraper,
credentials + browser required): it automates the export click-through, not
the acquisition step itself, and is deliberately not wired into `STEPS` for
the same reason -- login/browser-session fragility means it still needs a
human at the keyboard, not the unattended per-file resumability `FETCH`
promises.

**The one place this migration genuinely improves a source's resumability,
not just relocates its code**: the old `SnlMiningPreprocessor.process_target`
does DuckDB feature-building (`_prepare_duckdb_features` -- buffer tables,
ADM count tables, R-tree indexes) and tiled rasterization together inside one
`stage="spatial"` call. Here they are split into an explicit, independently
resumable PREPARE step (writes `prepared_db_path`) and GRID step (reads it),
matching the PREPARE/GRID vocabulary used everywhere else and letting a
retried rasterization skip the expensive DuckDB rebuild.

Ports `src/data/preprocess/sources/snl_mining.py::SnlMiningPreprocessor`.
`REQUIRES` on gadm's PREPARE (reads `misc/processed/stage_1/gadm/
gadm_levelADM_{1,2}_simplified.gpkg`, the same GADM artefact PLAD depends on)
**and** gadm's GRID (reads `GID_1`/`GID_2_code_mapping.json`, to translate
this source's own admin-count tables into gadm's integer ids -- see below).

**Admin-polygon mine counts are no longer rasterized.** `mine_count_adm1`/
`mine_count_adm2` are constant across every pixel of their containing ADM
polygon for a given year -- they vary only by `GID_1`/`GID_2` and year, never
by pixel location -- so GRID now writes them as tiny `(GID_N, year)`-keyed
parquet sidecars instead of full pixel-grid zarr variables. Assembly merges
them directly onto rows via
`src.data.assemble.processors.TileProcessor`'s `join_on` mechanism. The
radius-buffer counts (`mine_count_10km`/`20km`/`50km`) are genuinely
per-pixel (a pixel's count of mines within a fixed-radius circle varies
continuously with location) and stay in the rasterized zarr unchanged.

**`mine_priceshock_{10,20,50}km`**: a second, float family of the same
radius-buffer variables, added to test whether mineral price shocks fuel
local conflict (Berman et al. 2017, "This Mine Is Mine!"). `REQUIRES` on
`commodity_prices`'s PREPARE output (a small (commodity, year) -> real-price
lookup table, resolved via `layout.output_root(...)` directly -- not a
framework-injected path, mirroring how `_default_admin_variables()` resolves
gadm's own PREPARE output below) plus a user-owned `commodity_shares` table
inside the stage-0 `raw_db` DuckDB (contract: `(property_id VARCHAR,
commodity VARCHAR, share DOUBLE)`, one row per `(property_id, commodity)`,
static across a mine's active years, `commodity` already normalized via
`src.data.sources.commodities.normalize_commodity(..., source="snl")`).
`_create_mine_priceshock_table()` builds a per-`(property_id, year)` value =
`SUM(share * ln_price_real)`, joined into the *same* `mine_buffers_{R}km`
tables as a second `value_priceshock` column (not a parallel table -- one
`ST_Buffer`/rtree build serves both variables per radius). Unlike the count
variables (fill=0, uint16), an unmatched-commodity mine-year is SQL `NULL`,
not `0` -- carried through to the raster as `NaN` (`float32`), since 0 is
itself a legitimate price-shock value and must not be confused with "no
priced mine nearby".
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from src.data.common.geobox import get_target_geobox
from src.data.common.raster.spatial import write_crs_and_grid_mapping_encoding
from src.data.common.years import MAX_PLAUSIBLE_YEAR as _MAX_PLAUSIBLE_YEAR, MIN_PLAUSIBLE_YEAR as _MIN_PLAUSIBLE_YEAR
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import commodities, layout, registry
from src.data.sources.base import DataSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection
from src.data.sources import verify

if TYPE_CHECKING:
    from src.data.common.ledger.store import ArtifactRow

logger = logging.getLogger(__name__)

DEFAULT_RADIUS_VARIABLES = {
    "mine_count_10km": {"radius_km": 10, "table_name": "mine_buffers_10km", "value_column": "value", "dtype": "uint16"},
    "mine_count_20km": {"radius_km": 20, "table_name": "mine_buffers_20km", "value_column": "value", "dtype": "uint16"},
    "mine_count_50km": {"radius_km": 50, "table_name": "mine_buffers_50km", "value_column": "value", "dtype": "uint16"},
    "mine_priceshock_10km": {
        "radius_km": 10, "table_name": "mine_buffers_10km", "value_column": "value_priceshock", "dtype": "float32",
    },
    "mine_priceshock_20km": {
        "radius_km": 20, "table_name": "mine_buffers_20km", "value_column": "value_priceshock", "dtype": "float32",
    },
    "mine_priceshock_50km": {
        "radius_km": 50, "table_name": "mine_buffers_50km", "value_column": "value_priceshock", "dtype": "float32",
    },
    # No "radius_km" -- the discriminator that routes this variable through
    # _create_polygon_table (real mine footprints) instead of
    # _create_buffer_table (synthetic ST_Buffer circles) in _execute_prepare.
    "mine_polygon_count": {"table_name": "mine_polygons", "value_column": "value", "dtype": "uint16"},
}

#: nodata/fill sentinel per rasterized dtype -- `0` is a legitimate value for
#: `mine_priceshock_*` (a mine-year with genuinely zero-priced-exposure
#: coverage), so it can't share `mine_count_*`'s `0` fill/nodata convention;
#: `NaN` is used instead (see module docstring).
DTYPE_FILL_VALUES: Dict[str, float] = {"uint16": 0, "float32": float("nan")}

#: Scraper tables fused into PREPARE's mine identity/location/closing-year
#: (docs/data/snl_mining/README.md's fusion section): `mines` is the identity
#: backbone (adds mine-only-in-scraper records the manual `.xls` export never
#: had); `detail_location_map_claims__location.decimal_degrees` fills
#: location gaps; `detail_discoveries_milestones__milestones`'s
#: `event_type='Actual Closure'` rows fill closing-year gaps. No scraped
#: equivalent exists for *opening* year -- confirmed against live data, no
#: 'Actual Startup' milestone type exists among 49 real event types -- so
#: opening_year fusion stops at COALESCE(manual, llm-imputed), unchanged.
SCRAPED_MINES_TABLE = "mines"
SCRAPED_LOCATION_TABLE = "detail_location_map_claims__location"
SCRAPED_CLOSURE_MILESTONES_TABLE = "detail_discoveries_milestones__milestones"


class SnlMiningSource(DataSource):
    """SNL/S&P Global mining property tables -- gridded metric-radius buffer
    counts (zarr) + per-GID containing-ADM-polygon counts (parquet, merged
    directly during assembly rather than rasterized)."""

    ID = "snl_mining"
    STEPS = (PipelineStep.PREPARE, PipelineStep.GRID)
    REQUIRES = (
        ("gadm", PipelineStep.PREPARE), ("gadm", PipelineStep.GRID),
        ("commodity_prices", PipelineStep.PREPARE),
    )

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="snl_mining")
        super().__init__(ctx, cfg)

        aggregation = cfg.raw.get("aggregation", {}) or {}

        # stage-0 is a pre-pipeline manual export (module docstring), not a
        # FETCH/PREPARE/GRID artefact in the pipeline's own sense -- but it is
        # this source's raw input, so it lives under output_root(FETCH) (->
        # layout.raw_root()) like every other source's downloaded bytes,
        # respecting ctx.layout instead of hardcoding the legacy shape.
        # "database.duckdb" is shared with the scraper (scraper/config.py's
        # DEFAULT_DB_PATH): the manual-import notebook's properties/
        # property_texts/property_llm_years/etc. tables and the scraper's own
        # mines/detail_* tables live in one file -- both are "parsed xlsx",
        # just via different intake paths.
        self.duckdb_path = self._resolve_path(
            cfg.raw.get("duckdb_path")
            or os.path.join(self.output_root(PipelineStep.FETCH), "database.duckdb")
        )
        prepared_db_override = cfg.raw.get("prepared_db_path", aggregation.get("prepared_db_path"))
        self.prepared_db_path = (
            self._resolve_path(prepared_db_override)
            if prepared_db_override
            # PREPARE's own output -- route through output_root() like every
            # other source so this respects ctx.layout (docs/design/09-integrated-pipeline.md
            # §14's v2 rename) instead of hardcoding the legacy stage_1 shape.
            else os.path.join(self.output_root(PipelineStep.PREPARE), "snl_mining_prepared.duckdb")
        )

        self.properties_table = cfg.raw.get("properties_table", "properties")
        self.llm_years_table = cfg.raw.get("llm_years_table", "property_llm_years")
        self.work_history_table = cfg.raw.get("work_history_table", "property_work_history_events")
        # User-owned table inside raw_db (module docstring): per-mine
        # commodity production shares, joined against commodity_prices'
        # attached price table to build mine_priceshock.
        self.commodity_shares_table = cfg.raw.get("commodity_shares_table", "commodity_shares")

        # commodity_prices' PREPARE output -- resolved directly via
        # layout.output_root(), exactly like _default_admin_variables() below
        # resolves gadm's PREPARE output; REQUIRES is ordering/scheduling
        # metadata only (docs/design/09-integrated-pipeline.md §2), it never
        # injects a path.
        commodity_prices_path_override = cfg.raw.get("commodity_prices_path")
        self.commodity_prices_path = (
            self._resolve_path(commodity_prices_path_override)
            if commodity_prices_path_override
            else os.path.join(
                layout.output_root(self.ctx.data_root, "commodity_prices", PipelineStep.PREPARE, layout=self.ctx.layout),
                "commodity_prices.parquet",
            )
        )

        self.latitude_column = cfg.raw.get("latitude_column", "latitude")
        self.longitude_column = cfg.raw.get("longitude_column", "longitude")
        self.opening_year_column = cfg.raw.get("opening_year_column", "actual_start_up_year")
        self.closing_year_column = cfg.raw.get("closing_year_column", "actual_closure_year")
        self.llm_opening_year_column = cfg.raw.get("llm_opening_year_column", "llm_opening_year")
        self.llm_closing_year_column = cfg.raw.get("llm_closing_year_column", "llm_closing_year")

        self.metric_crs = cfg.raw.get("metric_crs", aggregation.get("metric_crs", "ESRI:54009"))
        self.tile_size = int(cfg.raw.get("tile_size", aggregation.get("tile_size", 2048)))
        self.output_filename = cfg.raw.get(
            "output_filename", aggregation.get("output_filename", "snl_mining_timeseries_reprojected.zarr")
        )

        radius_variables = aggregation.get("radius_variables") or DEFAULT_RADIUS_VARIABLES
        admin_variables = aggregation.get("admin_variables") or self._default_admin_variables()

        self.buffer_tables = {
            variable: (
                spec.get("table_name", f"{variable}_buffer"),
                # None (radius_km omitted) marks a polygon-sourced variable
                # (e.g. mine_polygon_count) -- built via _create_polygon_table
                # from real geometry instead of _create_buffer_table's
                # ST_Buffer circles (see _execute_prepare's dispatch split).
                int(spec["radius_km"]) * 1000 if spec.get("radius_km") is not None else None,
                spec.get("value_column", "value"),
                spec.get("dtype", "uint16"),
            )
            for variable, spec in radius_variables.items()
        }
        self.admin_tables = {
            variable: {
                "table_name": spec["table_name"],
                "geometry_path": self._resolve_path(spec["geometry_path"]),
                "code_column": spec["code_column"],
            }
            for variable, spec in admin_variables.items()
        }
        # Rasterized zarr variables: radius-buffer counts only (genuinely
        # per-pixel). Admin-polygon counts (self.admin_tables) are exported as
        # per-GID parquet sidecars instead -- see module docstring.
        self.output_variables = list(
            cfg.raw.get("output_variables", aggregation.get("output_variables", list(self.buffer_tables)))
        )

        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="snl_mining_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.ctx.data_root, path)

    def _default_admin_variables(self) -> Dict[str, Dict[str, str]]:
        """Cross-source reference to gadm's own PREPARE output (REQUIRES on
        gadm's PREPARE, see module docstring). Resolved through
        `layout.output_root()` -- not hardcoded to the legacy `misc/processed/
        stage_1/gadm` shape -- so this keeps finding gadm's simplified vector
        files under `ctx.layout="v2"` too, matching how
        `CountryClassificationsSource._plan_grid()` resolves its own
        cross-source gadm reference (src/data/sources/misc/country_classifications.py)."""
        gadm_prepare_dir = layout.output_root(
            self.ctx.data_root, "misc", PipelineStep.PREPARE, namespace="gadm", layout=self.ctx.layout
        )
        return {
            "mine_count_adm1": {
                "table_name": "adm1_year_counts",
                "geometry_path": os.path.join(gadm_prepare_dir, "gadm_levelADM_1_simplified.gpkg"),
                "code_column": "GID_1",
            },
            "mine_count_adm2": {
                "table_name": "adm2_year_counts",
                "geometry_path": os.path.join(gadm_prepare_dir, "gadm_levelADM_2_simplified.gpkg"),
                "code_column": "GID_2",
            },
        }

    # ------------------------------------------------------------------
    # plan()/execute() dispatch
    # ------------------------------------------------------------------

    def _plan(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        if step is PipelineStep.PREPARE:
            return self._plan_prepare()
        if step is PipelineStep.GRID:
            return self._plan_grid()
        raise AssertionError(f"unreachable: {step}")

    def _discover(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        """Ground truth for `data reconcile` -- see gadm.py's identical
        `_discover()` for the full rationale. This source's targets are
        singletons with no year/key selection to apply; `selection` is
        accepted for interface symmetry only."""
        if step is PipelineStep.PREPARE:
            return self._discover_prepare()
        if step is PipelineStep.GRID:
            return self._discover_grid()
        raise AssertionError(f"unreachable: {step}")

    def _execute(self, target: StepTarget) -> bool:
        if target.step is PipelineStep.PREPARE:
            return self._execute_prepare(target)
        if target.step is PipelineStep.GRID:
            return self._execute_grid(target)
        raise AssertionError(f"unreachable: {target.step}")

    # -- PREPARE: DuckDB feature build (newly split out of the old single
    # "spatial" stage -- see module docstring) ------------------------------

    def _plan_prepare(self) -> List[StepTarget]:
        """Ledger-backed fast path. `inputs` (`self.duckdb_path`,
        `self.commodity_prices_path`) are plain config-derived instance
        attributes, not values discovered by a live crawl -- no need to
        persist them in `meta`, `build_target` just reads them off `self`
        directly. Falls back to `_discover_prepare()` -- today's exact live
        logic -- if no ledger is configured yet, or `data reconcile
        --step prepare` hasn't populated one yet."""

        def build_target(row: "ArtifactRow", _ledger: Any) -> Optional[StepTarget]:
            if row.local_path is None:
                return None
            return StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key=row.unit_id,
                output_path=row.local_path, inputs=(self.duckdb_path, self.commodity_prices_path),
                completion=Completion.PATH_EXISTS, meta=row.meta,
            )

        targets = self._plan_from_ledger(PipelineStep.PREPARE, TargetSelection(), build_target)
        if targets is not None:
            return targets
        logger.warning(
            "No ledger for source='%s' step='prepare' -- falling back to live discovery; "
            "run `data reconcile --source %s --step prepare` for faster planning.",
            self.ID, self.ID,
        )
        return self._discover_prepare()

    def _discover_prepare(self) -> List[StepTarget]:
        if not os.path.exists(self.duckdb_path):
            logger.warning(
                "SNL mining stage-0 DuckDB not found at %s -- run "
                "src/data/sources/snl_mining/notebooks/snl_mining_manual_xls_to_duckdb.ipynb first.",
                self.duckdb_path,
            )
            return []
        if not os.path.exists(self.commodity_prices_path):
            logger.warning(
                "commodity_prices PREPARE output not found at %s -- run "
                "`data run --source commodity_prices --step prepare` first "
                "(REQUIRES, see module docstring).",
                self.commodity_prices_path,
            )
            return []
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key="all",
                output_path=self.prepared_db_path, inputs=(self.duckdb_path, self.commodity_prices_path),
                completion=Completion.PATH_EXISTS,
            )
        ]

    def _connect_duckdb(self, path: str):
        import duckdb

        con = duckdb.connect(path)
        try:
            con.execute("LOAD spatial;")
        except Exception:
            con.execute("INSTALL spatial;")
            con.execute("LOAD spatial;")
            try:
                con.execute("SET geometry_always_xy = true;")
            except Exception:
                logger.debug("DuckDB geometry_always_xy setting unavailable; continuing")
        return con

    def _get_or_create_geobox(self):
        return get_target_geobox(self.ctx)

    def _execute_prepare(self, target: StepTarget) -> bool:
        from src.data.sources.steps import is_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping SNL mining DuckDB preparation, already exists: %s", target.output_path)
            return True
        if not os.path.exists(self.duckdb_path):
            raise FileNotFoundError(f"SNL mining DuckDB not found: {self.duckdb_path}")
        for variable, table_spec in self.admin_tables.items():
            if not os.path.exists(table_spec["geometry_path"]):
                raise FileNotFoundError(f"Admin geometry for {variable} not found: {table_spec['geometry_path']}")

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)
        geobox = self._get_or_create_geobox()
        raster_crs = str(geobox.crs)

        con = self._connect_duckdb(self.prepared_db_path)
        try:
            con.execute(f"ATTACH '{self.duckdb_path}' AS raw_db (READ_ONLY)")
            llm_years_available = self._raw_table_exists(con, self.llm_years_table)
            if not llm_years_available:
                logger.warning("LLM years table %s not found; falling back to observed years only", self.llm_years_table)
            start_year, end_year = self._determine_year_bounds(con, llm_years_available)

            self._create_active_mines_table(con, start_year, end_year, raster_crs, llm_years_available)
            self._create_commodity_shares_table(con)
            self._create_mine_priceshock_table(con)
            # Two output variables (mine_count_*, mine_priceshock_*) can share
            # one physical buffer table (different value_column, same
            # table_name/radius) -- build each distinct table_name once, not
            # once per variable, to avoid redundantly rebuilding identical
            # ST_Buffer/ST_Transform geometry work. Polygon-sourced variables
            # (radius_m is None, e.g. mine_polygon_count) go through
            # _create_polygon_table instead of _create_buffer_table -- both
            # sets of table names still feed the shared rtree-index pass.
            radius_tables: Dict[str, int] = {}
            polygon_table_names: set[str] = set()
            for table_name, radius_m, _value_column, _dtype in self.buffer_tables.values():
                if radius_m is None:
                    polygon_table_names.add(table_name)
                else:
                    radius_tables[table_name] = radius_m
            for table_name, radius_m in radius_tables.items():
                self._create_buffer_table(con, table_name, radius_m, raster_crs)
            for table_name in polygon_table_names:
                self._create_polygon_table(con, table_name, raster_crs)
            for table_spec in self.admin_tables.values():
                self._create_admin_count_table(con, table_spec["table_name"], table_spec["geometry_path"], table_spec["code_column"], raster_crs)
            self._create_rtree_indexes(con, {**radius_tables, **dict.fromkeys(polygon_table_names)})
            self._verify_rtree_queries(con)
            con.execute("DETACH raw_db")
            self._write_back_shared_tables()
            return True
        except Exception:
            logger.exception("Error preparing SNL mining DuckDB features")
            return False
        finally:
            con.close()

    def _raw_table_exists(self, con, table_name: str) -> bool:
        row = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_catalog='raw_db' AND table_schema='main' AND table_name=? LIMIT 1",
            [table_name],
        ).fetchone()
        return row is not None

    #: Sanity bound for auto-detected year bounds -- guards against a single
    #: garbled opening_year/closing_year value (e.g. "150" typo'd for "1950",
    #: or a bad LLM-imputed year) dragging MIN()/MAX() to an implausible
    #: value that later becomes an actual zarr time coordinate. pandas can't
    #: represent a year that far from the present as a nanosecond Timestamp,
    #: so a store built on top of it fails not at PREPARE time (where the
    #: bad value actually lives) but much later and far more cryptically,
    #: deep inside `to_zarr(region="auto")`'s CF-time auto-region-detection
    #: (an `OutOfBoundsDatetime` chained into an unrelated missing-`cftime`
    #: ImportError). Confirmed on real data: a year of 150 reached
    #: `_rasterize_tiles_to_zarr` this way and crashed GRID, not PREPARE.
    #: Sourced from src.data.common.years -- shared across any source that
    #: infers time bounds from untrusted/enriched content, not just this one.
    MIN_PLAUSIBLE_YEAR = _MIN_PLAUSIBLE_YEAR
    MAX_PLAUSIBLE_YEAR = _MAX_PLAUSIBLE_YEAR

    def _llm_year_exprs(self, llm_years_available: bool) -> Tuple[str, str, str]:
        """`(llm_open_expr, llm_close_expr, llm_join)` SQL fragments for
        falling back to LLM-imputed opening/closing years -- shared by
        `_determine_year_bounds` and `_create_active_mines_table` so the two
        can't independently drift on how that fallback is expressed."""
        if not llm_years_available:
            return "NULL", "NULL", ""
        return (
            f"y.{self.llm_opening_year_column}",
            f"y.{self.llm_closing_year_column}",
            f"LEFT JOIN raw_db.main.{self.llm_years_table} AS y USING (property_id)",
        )

    def _fusion_ctes(self) -> str:
        """`scraped_location`/`scraped_closure` CTE definitions (no leading
        `WITH`), shared by `_determine_year_bounds` and
        `_create_active_mines_table` so the two can't independently drift on
        what "latitude"/"longitude"/"closing_year" mean (module-level
        constants docstring). `decimal_degrees` is a free-text "lat, lon"
        string (e.g. `"12.345, -67.890"`); `TRY_CAST` -> `NULL` on
        anything that doesn't parse, same silent-fallback philosophy as
        `_llm_year_exprs`. A mine can have multiple 'Actual Closure' rows
        (reopened-then-reclosed); `MAX` takes the most recent one."""
        # `sl`/`sc` keep the column named `mine_id` (not `property_id`) so
        # `_llm_year_exprs`' `USING (property_id)` join stays unambiguous --
        # it would otherwise have three same-named `property_id` columns in
        # scope (p/sl/sc) to choose from instead of just `p`'s.
        return rf"""
            scraped_location AS (
                SELECT mine_id,
                    TRY_CAST(TRIM(split_part(decimal_degrees, ',', 1)) AS DOUBLE) AS s_latitude,
                    TRY_CAST(TRIM(split_part(decimal_degrees, ',', 2)) AS DOUBLE) AS s_longitude
                FROM raw_db.main.{SCRAPED_LOCATION_TABLE}
                WHERE decimal_degrees IS NOT NULL AND decimal_degrees != ''
            ),
            scraped_closure AS (
                SELECT mine_id,
                    MAX(TRY_CAST(regexp_extract(period, '(\d{{4}})', 1) AS INTEGER)) AS s_closure_year
                FROM raw_db.main.{SCRAPED_CLOSURE_MILESTONES_TABLE}
                WHERE event_type = 'Actual Closure'
                GROUP BY mine_id
            )
        """

    def _fused_mines_from_clause(self) -> str:
        """`mines` (scraped identity backbone) LEFT JOINed against the manual
        properties table and both fusion CTEs, keyed on `property_id` ==
        `mine_id` (same id space -- verified: every manual `properties` row
        has a matching `mines` row)."""
        return f"""
            FROM raw_db.main.{SCRAPED_MINES_TABLE} AS m
            LEFT JOIN raw_db.main.{self.properties_table} AS p ON p.property_id = m.mine_id
            LEFT JOIN scraped_location AS sl ON sl.mine_id = m.mine_id
            LEFT JOIN scraped_closure AS sc ON sc.mine_id = m.mine_id
        """

    def _fused_latitude_expr(self) -> str:
        return f"COALESCE(CAST(p.{self.latitude_column} AS DOUBLE), sl.s_latitude)"

    def _fused_longitude_expr(self) -> str:
        return f"COALESCE(CAST(p.{self.longitude_column} AS DOUBLE), sl.s_longitude)"

    def _fused_closing_year_expr(self, llm_close_expr: str) -> str:
        return f"COALESCE(p.{self.closing_year_column}, sc.s_closure_year, {llm_close_expr})"

    def _determine_year_bounds(self, con, llm_years_available: bool) -> Tuple[int, int]:
        if self.cfg.year_range:
            return int(self.cfg.year_range[0]), int(self.cfg.year_range[1])

        llm_open_expr, llm_close_expr, llm_join = self._llm_year_exprs(llm_years_available)
        latitude_expr = self._fused_latitude_expr()
        longitude_expr = self._fused_longitude_expr()
        open_year = f"COALESCE(p.{self.opening_year_column}, {llm_open_expr})"
        # Bounds detection keeps the original extra fallback tier to open_year
        # (never NULL here, unlike _create_active_mines_table's closing_year,
        # which stays NULL on purpose so a still-active mine can be detected
        # downstream via "CASE WHEN closing_year IS NULL").
        close_year = f"COALESCE({self._fused_closing_year_expr(llm_close_expr)}, {open_year})"
        plausible = f"BETWEEN {self.MIN_PLAUSIBLE_YEAR} AND {self.MAX_PLAUSIBLE_YEAR}"
        # CASE-to-NULL (which MIN/MAX ignore), not a WHERE filter -- a WHERE
        # filter would drop the whole row before `count(*) FILTER` below
        # ever saw it, undercounting exclusions.
        query = f"""
            WITH {self._fusion_ctes()}
            SELECT
                CAST(MIN(CASE WHEN {open_year} {plausible} THEN {open_year} END) AS INTEGER) AS start_year,
                CAST(MAX(CASE WHEN {close_year} {plausible} THEN {close_year} END) AS INTEGER) AS end_year,
                count(*) FILTER (WHERE NOT ({open_year} {plausible})) AS excluded_count
            {self._fused_mines_from_clause()}
            {llm_join}
            WHERE {open_year} IS NOT NULL
              AND {latitude_expr} IS NOT NULL AND {longitude_expr} IS NOT NULL
        """
        start_year, end_year, excluded_count = con.execute(query).fetchone()
        if start_year is None or end_year is None:
            raise ValueError("Unable to infer mining year range from stage 0 tables")
        if excluded_count:
            logger.warning(
                "Excluded %d propert(ies) with an opening_year outside the plausible range "
                "[%d, %d] from SNL mining year-bounds detection -- likely a data-entry error "
                "(e.g. '150' instead of '1950'). Their own year is still clamped into "
                "[%d, %d] when building active_mines, not dropped entirely.",
                excluded_count, self.MIN_PLAUSIBLE_YEAR, self.MAX_PLAUSIBLE_YEAR, start_year, end_year,
            )
        return int(start_year), max(int(end_year), int(start_year))

    def _create_active_mines_table(self, con, start_year: int, end_year: int, raster_crs: str, llm_years_available: bool) -> None:
        llm_open_expr, llm_close_expr, llm_join = self._llm_year_exprs(llm_years_available)
        latitude_expr = self._fused_latitude_expr()
        longitude_expr = self._fused_longitude_expr()
        close_year_expr = self._fused_closing_year_expr(llm_close_expr)
        query = f"""
            CREATE OR REPLACE TABLE active_mines AS
            WITH {self._fusion_ctes()},
            canonical_mines AS (
                SELECT
                    CAST(m.mine_id AS VARCHAR) AS property_id,
                    {longitude_expr} AS longitude,
                    {latitude_expr} AS latitude,
                    CAST(COALESCE(p.{self.opening_year_column}, {llm_open_expr}) AS INTEGER) AS opening_year,
                    CAST({close_year_expr} AS INTEGER) AS closing_year,
                    ST_Point({longitude_expr}, {latitude_expr}) AS point_wgs84
                {self._fused_mines_from_clause()}
                {llm_join}
                WHERE COALESCE(p.{self.opening_year_column}, {llm_open_expr}) IS NOT NULL
                  AND {latitude_expr} IS NOT NULL AND {longitude_expr} IS NOT NULL
            ),
            bounded_mines AS (
                SELECT property_id, longitude, latitude, opening_year, closing_year,
                    CASE WHEN closing_year IS NULL THEN {end_year} ELSE closing_year END AS closing_year_effective,
                    point_wgs84
                FROM canonical_mines
                WHERE opening_year <= {end_year} AND COALESCE(closing_year, {end_year}) >= {start_year}
            ),
            expanded AS (
                SELECT m.property_id, yr.range::INTEGER AS year, m.longitude, m.latitude, m.opening_year, m.closing_year, m.point_wgs84
                FROM bounded_mines AS m,
                LATERAL range(GREATEST(m.opening_year, {start_year}), LEAST(m.closing_year_effective, {end_year}) + 1) AS yr(range)
            )
            SELECT property_id, year, longitude, latitude, opening_year, closing_year, point_wgs84,
                ST_Transform(point_wgs84, 'EPSG:4326', '{self.metric_crs}', true) AS point_metric,
                ST_Transform(point_wgs84, 'EPSG:4326', '{raster_crs}', true) AS point_raster
            FROM expanded
        """
        con.execute(query)

    #: `Contained(...)` unit variants seen in `detail_reserves_resources`
    #: (whitespace already stripped from the raw `metric` string) -> a
    #: tonnes-per-unit multiplier. Verified against live data: each of the
    #: ~63 real commodity labels uses exactly one of these four units
    #: consistently (oz troy for precious metals, ct metric carat for
    #: diamonds, lbs avoirdupois for uranium, tonnes for everything else) --
    #: no per-commodity mixed units, so this conversion needs no density
    #: lookup, just the standard mass-unit factors.
    _CONTAINED_TONNES_PER_UNIT = {
        "Contained(tonnes)": 1.0,
        "Contained(oz)": 31.1034768 / 1e6,  # troy oz -> tonnes
        "Contained(lbs)": 0.45359237 / 1e3,  # avoirdupois lb -> tonnes
        "Contained(ct)": 0.0002 / 1e3,  # metric carat -> tonnes
    }

    def _create_commodity_shares_table(self, con) -> None:
        """Builds `commodity_shares`: one row per `(property_id, commodity)`,
        `share` in `[0, 1]` summing to `1` per mine, consumed by
        `_create_mine_priceshock_table`.

        If the user has placed a real table at
        `raw_db.main.{self.commodity_shares_table}` (the original
        "user-owned" contract -- module docstring), it's copied as-is and
        nothing is auto-derived; the copy lands in this (writable) prepared-db
        connection even though `raw_db` itself is `ATTACH`ed `READ_ONLY`, so
        the override escape hatch still works. Otherwise `commodity_shares`
        is auto-derived from `detail_reserves_resources`'s
        `category='Total Reserves & Resources'` `Contained(...)` rows,
        converted to one common tonnes basis (`_CONTAINED_TONNES_PER_UNIT`)
        and normalized via `commodities.normalize_commodity(..., source="snl")`
        -- several raw labels (e.g. individual REEs) collapse onto the same
        canonical key, so normalization happens *before* the share ratio is
        computed, not after, or those mines' shares wouldn't sum to 1.
        """
        if self._raw_table_exists(con, self.commodity_shares_table):
            con.execute(f"CREATE OR REPLACE TABLE commodity_shares AS SELECT * FROM raw_db.main.{self.commodity_shares_table}")
            return

        import pandas as pd

        raw_labels = [
            row[0]
            for row in con.execute(
                "SELECT DISTINCT commodity FROM raw_db.main.detail_reserves_resources "
                "WHERE category = 'Total Reserves & Resources' "
                "AND regexp_replace(metric, '\\s', '', 'g') LIKE 'Contained(%' "
                "AND commodity IS NOT NULL"
            ).fetchall()
        ]
        mapping_rows: List[Tuple[str, str]] = []
        unmapped: List[str] = []
        for label in raw_labels:
            canonical = commodities.normalize_commodity(label, source="snl")
            (mapping_rows if canonical else unmapped).append((label, canonical) if canonical else label)

        if unmapped:
            logger.warning(
                "commodity_shares: %d raw commodity label(s) have no canonical mapping "
                "(commodities.normalize_commodity(source='snl') returned None) and are "
                "excluded from every mine's shares: %s -- add a mapping in commodities.py "
                "if these should count.",
                len(unmapped), sorted(unmapped),
            )

        mapping_df = pd.DataFrame(mapping_rows, columns=["raw_commodity", "canonical_commodity"])
        con.register("_commodity_alias_map", mapping_df)
        try:
            unit_case = " ".join(
                f"WHEN regexp_replace(r.metric, '\\s', '', 'g') = '{unit}' THEN {factor}"
                for unit, factor in self._CONTAINED_TONNES_PER_UNIT.items()
            )
            query = f"""
                CREATE OR REPLACE TABLE commodity_shares AS
                WITH normalized AS (
                    SELECT
                        r.mine_id AS property_id,
                        a.canonical_commodity AS commodity,
                        r.value AS amount,
                        CASE {unit_case} END AS tonnes_per_unit
                    FROM raw_db.main.detail_reserves_resources AS r
                    JOIN _commodity_alias_map AS a ON a.raw_commodity = r.commodity
                    WHERE r.category = 'Total Reserves & Resources'
                      AND regexp_replace(r.metric, '\\s', '', 'g') LIKE 'Contained(%'
                      AND r.value IS NOT NULL
                ),
                tonnage AS (
                    SELECT property_id, commodity, SUM(amount * tonnes_per_unit) AS tonnes
                    FROM normalized
                    WHERE tonnes_per_unit IS NOT NULL
                    GROUP BY property_id, commodity
                )
                SELECT property_id, commodity,
                    tonnes / SUM(tonnes) OVER (PARTITION BY property_id) AS share
                FROM tonnage
                WHERE tonnes > 0
            """
            con.execute(query)
        finally:
            con.unregister("_commodity_alias_map")

    def _create_mine_priceshock_table(self, con) -> None:
        """Builds `mine_priceshock`: one row per `(property_id, year)`, `value
        = SUM(share * ln_price_real)` over the mine's commodity shares
        (`commodity_shares`, built by `_create_commodity_shares_table` just
        before this method runs) joined against `commodity_prices`'s
        prepared price table (read directly via `read_parquet()`, not
        `ATTACH` -- a parquet file isn't an attachable DuckDB database).

        `LEFT JOIN ... GROUP BY SUM(...)` is load-bearing: a commodity with no
        price match contributes SQL `NULL` (ignored by `SUM`), and if *every*
        commodity for a mine is unmatched, `value` is `NULL` for that
        `(property_id, year)` -- deliberately distinct from `0`, which is a
        legitimate price-shock value in its own right (see
        `_rasterize_tiles_to_zarr`'s NaN-fill handling). An empty
        `commodity_shares` table (no user override, no derivable
        reserves/resources data) degrades gracefully to an empty
        `mine_priceshock` via the same `JOIN`, no separate empty-table branch
        needed.
        """
        escaped_prices_path = self.commodity_prices_path.replace("'", "''")
        query = f"""
            CREATE OR REPLACE TABLE mine_priceshock AS
            SELECT m.property_id, m.year, SUM(s.share * p.ln_price_real) AS value
            FROM (SELECT DISTINCT property_id, year FROM active_mines) AS m
            JOIN commodity_shares AS s USING (property_id)
            LEFT JOIN read_parquet('{escaped_prices_path}') AS p
                ON p.commodity = s.commodity AND p.year = m.year
            GROUP BY m.property_id, m.year
        """
        con.execute(query)

    def _create_buffer_table(self, con, table_name: str, radius_m: int, raster_crs: str) -> None:
        query = f"""
            CREATE OR REPLACE TABLE {table_name} AS
            WITH buffered AS (
                SELECT m.property_id, m.year, 1::INTEGER AS value, ps.value AS value_priceshock,
                    ST_MakeValid(ST_Buffer(m.point_metric, {radius_m})) AS geometry_metric
                FROM active_mines AS m
                LEFT JOIN mine_priceshock AS ps USING (property_id, year)
            )
            SELECT property_id, year, value, value_priceshock, geometry_metric,
                ST_Transform(geometry_metric, '{self.metric_crs}', '{raster_crs}', true) AS geometry_raster
            FROM buffered WHERE geometry_metric IS NOT NULL
        """
        con.execute(query)

    def _create_polygon_table(self, con, table_name: str, raster_crs: str) -> None:
        """Builds a polygon-sourced variable's table (e.g. `mine_polygons`,
        backing `mine_polygon_count`) from real scraped mine footprints
        instead of a synthetic `ST_Buffer` circle -- structurally parallel to
        `_create_buffer_table`, but geometry comes from
        `mine_property_geometries` (`geometry_kind='property'`: the tightest,
        most accurate single-mine footprint, verified median ~300m across;
        `linked`-kind claims polygons are out of scope for now).

        A plain `JOIN` (not `LEFT JOIN`) excludes the ~43% of active-mine-
        years without a `property`-kind polygon -- same "just don't
        contribute a row" pattern `_create_buffer_table` uses for
        `geometry_metric IS NOT NULL`. One static polygon is reused for every
        active year of a given mine, exactly like buffer-circle geometry
        already is (mine location doesn't change year to year).
        """
        query = f"""
            CREATE OR REPLACE TABLE {table_name} AS
            WITH mine_polygons_wgs84 AS (
                SELECT mine_id AS property_id,
                    ST_MakeValid(ST_GeomFromText(geometry_wkt)) AS geometry_wgs84
                FROM raw_db.main.mine_property_geometries
                WHERE geometry_kind = 'property'
            )
            SELECT m.property_id, m.year, 1::INTEGER AS value,
                ST_Transform(g.geometry_wgs84, 'EPSG:4326', '{raster_crs}', true) AS geometry_raster
            FROM active_mines AS m
            JOIN mine_polygons_wgs84 AS g USING (property_id)
        """
        con.execute(query)

    def _write_back_shared_tables(self) -> None:
        """Copies `commodity_shares` from `prepared_db_path` back into the
        shared `database.duckdb` (the user asked for "a table", a first-class
        visible artefact -- not just a PREPARE-internal intermediate) and
        exports it to CSV alongside the scraper's own `csv/` exports.

        `database.duckdb` may be locked by a concurrently-running scraper
        process (DuckDB doesn't support concurrent writers) -- this is
        best-effort: log a warning and move on rather than failing PREPARE,
        since `_create_mine_priceshock_table`'s actual dependency is the
        `prepared_db_path` copy, already written before this runs.
        """
        import duckdb

        try:
            shared_con = duckdb.connect(self.duckdb_path)
        except Exception:
            logger.warning(
                "Could not open %s to write back commodity_shares (likely locked by a "
                "concurrent scraper run) -- skipping, PREPARE's own copy is unaffected.",
                self.duckdb_path,
            )
            return
        try:
            shared_con.execute(f"ATTACH '{self.prepared_db_path}' AS prepared_db (READ_ONLY)")
            shared_con.execute("CREATE OR REPLACE TABLE commodity_shares AS SELECT * FROM prepared_db.main.commodity_shares")
            shared_con.execute("DETACH prepared_db")
            csv_dir = os.path.join(os.path.dirname(self.duckdb_path), "csv")
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, "commodity_shares.csv").replace("'", "''")
            shared_con.execute(f"COPY commodity_shares TO '{csv_path}' (HEADER, DELIMITER ',')")
        except Exception:
            logger.exception("Error writing commodity_shares back to %s -- PREPARE's own copy is unaffected.", self.duckdb_path)
        finally:
            shared_con.close()

    def _detect_gpkg_geometry_column(self, con, gpkg_path: str, code_column: str) -> str:
        escaped_path = gpkg_path.replace("'", "''")
        describe_rows = con.execute(f"DESCRIBE SELECT * FROM ST_Read('{escaped_path}')").fetchall()
        column_names = {str(row[0]) for row in describe_rows}
        for candidate in ("geom", "geometry", "wkb_geometry"):
            if candidate in column_names:
                return candidate
        raise ValueError(f"Could not find a geometry column in {gpkg_path}; available columns: {sorted(column_names)}")

    def _create_admin_count_table(self, con, table_name: str, gpkg_path: str, code_column: str, raster_crs: str) -> None:
        escaped_path = gpkg_path.replace("'", "''")
        geometry_column = self._detect_gpkg_geometry_column(con, gpkg_path, code_column)
        query = f"""
            CREATE OR REPLACE TABLE {table_name} AS
            WITH admin_polygons AS (
                SELECT CAST({code_column} AS VARCHAR) AS adm_code,
                    ST_MakeValid(ST_Transform({geometry_column}, 'EPSG:4326', '{raster_crs}', true)) AS geometry_raster
                FROM ST_Read('{escaped_path}')
                WHERE {code_column} IS NOT NULL AND {geometry_column} IS NOT NULL
            ),
            assignments AS (
                SELECT m.year, a.adm_code FROM active_mines AS m
                JOIN admin_polygons AS a ON ST_Intersects(a.geometry_raster, m.point_raster)
            ),
            counts AS (SELECT year, adm_code, COUNT(*)::INTEGER AS value FROM assignments GROUP BY 1, 2)
            SELECT c.year, c.adm_code, c.value, a.geometry_raster
            FROM counts AS c JOIN admin_polygons AS a USING (adm_code)
        """
        con.execute(query)

    def _create_rtree_indexes(self, con, radius_tables: Dict[str, int]) -> None:
        index_specs = [(f"idx_{t}_rtree", t) for t in radius_tables]
        index_specs += [(f"idx_{s['table_name']}_rtree", s["table_name"]) for s in self.admin_tables.values()]
        for index_name, table_name in index_specs:
            con.execute(f"DROP INDEX IF EXISTS {index_name}")
            con.execute(f"CREATE INDEX {index_name} ON {table_name} USING RTREE (geometry_raster)")

    def _verify_rtree_queries(self, con) -> None:
        try:
            geobox = self._get_or_create_geobox()
            bounds = geobox.boundingbox
            sample_sql = f"""
                EXPLAIN SELECT value FROM mine_buffers_10km
                WHERE year = (SELECT MIN(year) FROM active_mines)
                  AND ST_Intersects(geometry_raster, ST_GeomFromText(
                      'POLYGON(({bounds.left} {bounds.bottom}, {bounds.right} {bounds.bottom}, {bounds.right} {bounds.top}, {bounds.left} {bounds.top}, {bounds.left} {bounds.bottom}))'))
            """
            plan_text = "\n".join(str(row) for row in con.execute(sample_sql).fetchall())
            if "RTREE_INDEX_SCAN" not in plan_text:
                logger.info("DuckDB EXPLAIN did not report RTREE_INDEX_SCAN for SNL mining tile fetch.")
        except Exception:
            logger.debug("Skipping DuckDB R-tree plan verification (EXPLAIN not stable on this build).")

    # -- GRID: tiled rasterization from the prepared DuckDB ------------------

    def _plan_grid(self) -> List[StepTarget]:
        """Ledger-backed fast path. `meta["years"]` (persisted by
        `_discover_grid()` below, exactly like gadm's GRID persists
        `level_files` -- see gadm.py's `_plan_grid()`) is read straight back
        instead of re-opening `prepared_db_path` and re-running `SELECT
        DISTINCT year FROM active_mines` -- this is the one source whose
        live discovery queries a source-specific DuckDB rather than crawling
        the filesystem, so avoiding that on the fast path is the whole point
        of this migration for this source."""

        def build_target(row: "ArtifactRow", _ledger: Any) -> Optional[StepTarget]:
            years = row.meta.get("years")
            if not years or row.local_path is None:
                return None
            return StepTarget(
                source_id=self.ID, step=PipelineStep.GRID, key=row.unit_id,
                output_path=row.local_path, inputs=(self.prepared_db_path,),
                completion=Completion.PATH_EXISTS, meta=row.meta,
            )

        targets = self._plan_from_ledger(PipelineStep.GRID, TargetSelection(), build_target)
        if targets is not None:
            return targets
        logger.warning(
            "No ledger for source='%s' step='grid' -- falling back to live discovery; "
            "run `data reconcile --source %s --step grid` for faster planning.",
            self.ID, self.ID,
        )
        return self._discover_grid()

    def _discover_grid(self) -> List[StepTarget]:
        if not os.path.exists(self.prepared_db_path):
            return []
        con = self._connect_duckdb(self.prepared_db_path)
        try:
            rows = con.execute("SELECT DISTINCT year FROM active_mines ORDER BY year").fetchall()
        except Exception:
            return []
        finally:
            con.close()
        years = [int(r[0]) for r in rows]
        if not years:
            return []
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.GRID, key="all",
                output_path=layout.grid_store_path(
                    self.ctx.data_root,
                    self.cfg.data_path,
                    self.output_filename,
                    grid_id=self.ctx.grid_id,
                    layout=self.ctx.layout,
                    v2_family="snl_mining",
                ),
                inputs=(self.prepared_db_path,), completion=Completion.PATH_EXISTS,
                meta={
                    "years": years,
                    **verify.verification_meta(
                        self.cfg.raw,
                        expected_vars=tuple(self.output_variables),
                        value_range=(0, 200),
                        # value_range=(0, 200) only makes sense for the uint16
                        # count family -- the float32 price-shock family is on a
                        # different physical scale (a sum of ln-prices, not a
                        # count). verify.py supports one value_range per target,
                        # so scope it by dtype here; the excluded (float32)
                        # variables still get the unconditional "sample isn't
                        # entirely NaN" check.
                        range_vars=tuple(v for v in self.output_variables if self.buffer_tables[v][3] == "uint16"),
                    ),
                },
            )
        ]

    def _output_root(self) -> str:
        # Unlike every other source, this used to hardcode "stage_2" and
        # ignore ctx.grid_id entirely -- so a run with pipeline.grid: ease6933
        # would still land in a legacy-named directory. Route through the
        # shared layout function like DataSource.output_root() does.
        return layout.output_root(
            self.ctx.data_root,
            self.cfg.data_path,
            PipelineStep.GRID,
            grid_id=self.ctx.grid_id,
            layout=self.ctx.layout,
        )

    def output_root(self, step: PipelineStep, *, namespace: str | None = None) -> str:
        if step is PipelineStep.GRID:
            return self._output_root()
        return super().output_root(step, namespace=namespace)

    def _create_empty_target_zarr(self, output_path: str, geobox, years: List[int]) -> bool:
        import dask.array as da
        import numpy as np
        import pandas as pd
        import xarray as xr
        from zarr.codecs import BloscCodec

        try:
            time_coords = pd.to_datetime([f"{year}-12-31" for year in sorted(years)])
            ny, nx = geobox.shape
            dim_y, dim_x = geobox.dimensions
            y_coords = geobox.coords[dim_y].values.round(5)
            x_coords = geobox.coords[dim_x].values.round(5)

            data_vars = {}
            for var in self.output_variables:
                dtype_name = self.buffer_tables[var][3]
                np_dtype = np.dtype(dtype_name)
                fill = DTYPE_FILL_VALUES[dtype_name]
                data_vars[var] = xr.DataArray(
                    da.full(
                        (len(time_coords), 1, ny, nx), fill, dtype=np_dtype,
                        chunks=(1, 1, self.tile_size, self.tile_size),
                    ),
                    dims=["time", "band", dim_y, dim_x],
                    coords={"time": time_coords, "band": [1], dim_y: y_coords, dim_x: x_coords},
                    attrs={"_FillValue": fill, "nodata": fill},
                )
            ds = xr.Dataset(
                data_vars,
                attrs={
                    "source_duckdb_path": self.duckdb_path,
                    "prepared_duckdb_path": self.prepared_db_path,
                    "metric_crs": self.metric_crs,
                    "radius_semantics": (
                        "mine_count_*: count of active mine buffers covering pixel center. "
                        "mine_priceshock_*: sum of share * ln(real price) over active, "
                        "price-matched mine buffers covering pixel center; NaN where no "
                        "price-matched mine buffer covers the pixel (see module docstring)."
                    ),
                },
            )

            compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", blocksize=0)
            base_encoding = {}
            for var in self.output_variables:
                dtype_name = self.buffer_tables[var][3]
                base_encoding[var] = {
                    "chunks": (1, 1, self.tile_size, self.tile_size),
                    "compressors": (compressor,),
                    "dtype": dtype_name,
                    "fill_value": DTYPE_FILL_VALUES[dtype_name],
                }
            ds, encoding = write_crs_and_grid_mapping_encoding(ds, geobox, base_encoding)
            ds.to_zarr(output_path, mode="w", compute=False, encoding=encoding, zarr_format=3, consolidated=False)
            return True
        except Exception:
            logger.exception("Error creating SNL mining zarr skeleton")
            return False

    def _fetch_features(self, con, table_name: str, value_column: str, year: int, tile_wkt: str):
        # `{value_column} IS NOT NULL` is what makes an all-unmatched-commodity
        # mine-year (mine_priceshock.value = SQL NULL, see
        # _create_mine_priceshock_table) simply not contribute a geometry row
        # when fetching mine_priceshock_* -- while mine_count_*'s `value`
        # column is never NULL, so this clause is a no-op for it.
        sql = f"""
            SELECT {value_column}, ST_AsWKB(geometry_raster) AS geom_wkb FROM {table_name}
            WHERE year = ? AND {value_column} IS NOT NULL
              AND ST_Intersects(geometry_raster, ST_GeomFromText(?))
        """
        return con.execute(sql, [int(year), tile_wkt]).fetchall()

    def _rasterize_tiles_to_zarr(self, output_path: str, geobox, years: List[int]) -> bool:
        import numpy as np
        import pandas as pd
        import shapely.wkb
        import xarray as xr
        from odc.geo import GeoboxTiles
        from odc.geo.geom import Geometry
        from odc.geo.xr import rasterize

        con = self._connect_duckdb(self.prepared_db_path)
        try:
            tiles = GeoboxTiles(geobox, (self.tile_size, self.tile_size))
            for year in years:
                for ix in range(tiles.shape[0]):
                    for iy in range(tiles.shape[1]):
                        tile_geobox = tiles[ix, iy]
                        bounds = tile_geobox.boundingbox
                        tile_wkt = (
                            f"POLYGON(({bounds.left} {bounds.bottom}, {bounds.right} {bounds.bottom}, "
                            f"{bounds.right} {bounds.top}, {bounds.left} {bounds.top}, {bounds.left} {bounds.bottom}))"
                        )
                        tile_arrays = {}
                        tile_touched = {}
                        for var in self.output_variables:
                            dtype_name = self.buffer_tables[var][3]
                            tile_arrays[var] = np.zeros(tile_geobox.shape, dtype=np.dtype(dtype_name))
                            if dtype_name == "float32":
                                tile_touched[var] = np.zeros(tile_geobox.shape, dtype=bool)
                        any_data = False

                        for var_name, (table_name, _radius_m, value_column, dtype_name) in self.buffer_tables.items():
                            rows = self._fetch_features(con, table_name, value_column, year, tile_wkt)
                            any_data = any_data or bool(rows)
                            for value, geom_wkb in rows:
                                geom = Geometry(shapely.wkb.loads(bytes(geom_wkb)), crs=str(tile_geobox.crs))
                                mask = rasterize(geom, tile_geobox).values.astype(bool)
                                if dtype_name == "float32":
                                    tile_arrays[var_name][mask] += np.float32(value)
                                    tile_touched[var_name] |= mask
                                else:
                                    tile_arrays[var_name] = tile_arrays[var_name] + (mask.astype(np.uint16) * int(value))

                        # An untouched pixel of a float32 (price-shock) variable
                        # must resolve to NaN, not the accumulator's additive
                        # identity 0 -- 0 is itself a legitimate price-shock
                        # value (see module docstring). uint16 count variables
                        # are unaffected: 0 has always been their correct
                        # "no mine nearby" value.
                        for var_name in self.output_variables:
                            dtype_name = self.buffer_tables[var_name][3]
                            if dtype_name == "float32":
                                tile_arrays[var_name] = np.where(
                                    tile_touched[var_name], tile_arrays[var_name], np.nan
                                ).astype(np.float32)

                        if any_data:
                            dim_y, dim_x = tile_geobox.dimensions
                            tile_ds = xr.Dataset(
                                {
                                    var: xr.DataArray(
                                        tile_arrays[var][None, None, :, :],
                                        dims=["time", "band", dim_y, dim_x],
                                        coords={
                                            "time": pd.to_datetime([f"{year}-12-31"]),
                                            "band": [1],
                                            dim_y: tile_geobox.coords[dim_y].values.round(5),
                                            dim_x: tile_geobox.coords[dim_x].values.round(5),
                                        },
                                    )
                                    for var in self.output_variables
                                }
                            )
                            tile_ds.to_zarr(output_path, mode="r+", region="auto", consolidated=False)
            return True
        except Exception:
            logger.exception("Error rasterizing SNL mining tiles")
            return False
        finally:
            con.close()

    def _execute_grid(self, target: StepTarget) -> bool:
        from src.data.sources.steps import is_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping existing output: %s", target.output_path)
            return True

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)
        try:
            geobox = self._get_or_create_geobox()
            years = target.meta["years"]
            if not self._create_empty_target_zarr(target.output_path, geobox, years):
                return False
            if not self._rasterize_tiles_to_zarr(target.output_path, geobox, years):
                return False
            return self._export_admin_count_tables(os.path.dirname(target.output_path))
        except Exception:
            logger.exception("Error processing SNL mining GRID target")
            return False

    def _export_admin_count_tables(self, output_dir: str) -> bool:
        """Write each admin-polygon mine-count table as a small `(GID_N, year)`-
        keyed parquet sidecar instead of rasterizing it -- the value is
        constant within a GID (module docstring). One file per admin level,
        e.g. `mine_count_adm1.parquet`, named after the variable."""
        import json

        from src.data.sources.misc.gadm import gid_mapping_path

        con = self._connect_duckdb(self.prepared_db_path)
        try:
            for variable, table_spec in self.admin_tables.items():
                gid_col = table_spec["code_column"]
                mapping_file = gid_mapping_path(self.ctx.data_root, self.ctx.grid_id, self.ctx.layout, gid_col)
                if not os.path.exists(mapping_file):
                    logger.error("GADM %s mapping file not found: %s", gid_col, mapping_file)
                    return False
                with open(mapping_file) as f:
                    code_to_id: Dict[str, int] = json.load(f)

                counts_df = con.execute(
                    f"SELECT year, adm_code, value FROM {table_spec['table_name']}"
                ).df()
                counts_df[gid_col] = counts_df["adm_code"].map(lambda c: code_to_id.get(c, 0))
                counts_df = counts_df[counts_df[gid_col] != 0]
                out_df = counts_df[[gid_col, "year", "value"]].rename(columns={"value": variable})

                out_path = os.path.join(output_dir, f"{variable}.parquet")
                out_df.to_parquet(out_path, index=False)
                logger.info(
                    "SNL mining %s table complete: %d (%s, year) rows -> %s",
                    variable, len(out_df), gid_col, out_path,
                )
            return True
        except Exception:
            logger.exception("Error exporting SNL mining admin-count tables")
            return False
        finally:
            con.close()


registry.register(SnlMiningSource.ID, __name__, SnlMiningSource.__name__, SnlMiningSource.STEPS, requires=SnlMiningSource.REQUIRES)
