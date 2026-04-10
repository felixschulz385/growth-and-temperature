import logging
import os
import re
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from gnt.data.common.geobox.geobox import get_or_create_geobox
from gnt.data.preprocess.sources.base import AbstractPreprocessor

logger = logging.getLogger(__name__)


class SnlMiningPreprocessor(AbstractPreprocessor):
    """
    Stage 2 preprocessor for SNL mining property tables.

    The workflow has two phases:
    1. Prepare year-specific vector features in DuckDB.
    2. Rasterize prepared features tile-by-tile onto the common geobox.

    Radius variables count mines whose metric-radius buffer covers the pixel
    center. ADM variables paint the year-specific mine counts of the containing
    polygon onto each pixel.
    """

    OUTPUT_VARIABLES = [
        "mine_count_10km",
        "mine_count_20km",
        "mine_count_50km",
        "mine_count_adm1",
        "mine_count_adm2",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.stage = kwargs.get("stage", "spatial")
        if self.stage != "spatial":
            raise ValueError(f"Unsupported stage: {self.stage}. Use 'spatial'.")

        self.year_range = kwargs.get("year_range")
        self.override = kwargs.get("override", False)

        hpc_target = kwargs.get("hpc_target")
        self.hpc_root = self._strip_remote_prefix(hpc_target)
        if not self.hpc_root:
            raise ValueError("hpc_target is required for HPC mode")

        self.source_name = kwargs.get("name", kwargs.get("preprocessor", "snl_mining"))
        self.data_path = kwargs.get("data_path", "snl_mining")
        self.index_dir = os.path.join(self.hpc_root, "hpc_data_index")
        os.makedirs(self.index_dir, exist_ok=True)

        aggregation = kwargs.get("aggregation", {}) or {}

        self.duckdb_path = self._resolve_duckdb_path(kwargs.get("duckdb_path"))
        self.prepared_db_path = self._resolve_prepared_db_path(
            kwargs.get("prepared_db_path", aggregation.get("prepared_db_path"))
        )

        self.properties_table = kwargs.get("properties_table", "properties")
        self.llm_years_table = kwargs.get("llm_years_table", "property_llm_years")
        self.work_history_table = kwargs.get("work_history_table", "property_work_history_events")

        self.latitude_column = kwargs.get("latitude_column", "latitude")
        self.longitude_column = kwargs.get("longitude_column", "longitude")
        self.opening_year_column = kwargs.get("opening_year_column", "actual_start_up_year")
        self.closing_year_column = kwargs.get("closing_year_column", "actual_closure_year")
        self.llm_opening_year_column = kwargs.get("llm_opening_year_column", "llm_opening_year")
        self.llm_closing_year_column = kwargs.get("llm_closing_year_column", "llm_closing_year")

        self.metric_crs = kwargs.get(
            "metric_crs", aggregation.get("metric_crs", "ESRI:54009")
        )
        self.tile_size = int(kwargs.get("tile_size", aggregation.get("tile_size", 2048)))

        self.output_filename = kwargs.get(
            "output_filename",
            aggregation.get("output_filename", "snl_mining_timeseries_reprojected.zarr"),
        )

        radius_variables = aggregation.get("radius_variables") or {
            "mine_count_10km": {"radius_km": 10, "table_name": "mine_buffers_10km"},
            "mine_count_20km": {"radius_km": 20, "table_name": "mine_buffers_20km"},
            "mine_count_50km": {"radius_km": 50, "table_name": "mine_buffers_50km"},
        }
        admin_variables = aggregation.get("admin_variables") or {
            "mine_count_adm1": {
                "table_name": "adm1_year_counts",
                "geometry_path": "misc/processed/stage_1/gadm/gadm_levelADM_1_simplified.gpkg",
                "code_column": "GID_1",
            },
            "mine_count_adm2": {
                "table_name": "adm2_year_counts",
                "geometry_path": "misc/processed/stage_1/gadm/gadm_levelADM_2_simplified.gpkg",
                "code_column": "GID_2",
            },
        }

        self.buffer_tables = {
            variable: (
                spec.get("table_name", f"{variable}_buffer"),
                int(spec["radius_km"]) * 1000,
            )
            for variable, spec in radius_variables.items()
        }
        self.admin_tables = {
            variable: {
                "table_name": spec["table_name"],
                "geometry_path": self._resolve_path_with_root(spec["geometry_path"]),
                "code_column": spec["code_column"],
            }
            for variable, spec in admin_variables.items()
        }

        self.output_variables = kwargs.get(
            "output_variables",
            aggregation.get(
                "output_variables",
                list(self.buffer_tables.keys()) + list(self.admin_tables.keys()),
            ),
        )
        self.output_variables = list(self.output_variables)

        self.temp_dir = kwargs.get("temp_dir") or tempfile.mkdtemp(
            prefix="snl_mining_processor_"
        )
        os.makedirs(self.temp_dir, exist_ok=True)

        self._init_parquet_index_path()

        logger.info("Initialized SnlMiningPreprocessor")
        logger.info("HPC root: %s", self.hpc_root)
        logger.info("DuckDB path: %s", self.duckdb_path)
        logger.info("Prepared DB path: %s", self.prepared_db_path)

    def _resolve_duckdb_path(self, configured_path: Optional[str]) -> str:
        path = configured_path or os.path.join(
            self.data_path,
            "processed",
            "stage_0",
            "manual_xls",
            "snl_mining_manual_export.duckdb",
        )
        return self._resolve_path_with_root(path)

    def _resolve_prepared_db_path(self, configured_path: Optional[str]) -> str:
        path = configured_path or os.path.join(
            self.data_path,
            "processed",
            "stage_1",
            "snl_mining_prepared.duckdb",
        )
        return self._resolve_path_with_root(path)

    def _resolve_path_with_root(self, path: str) -> str:
        clean_path = self._strip_remote_prefix(path)
        if os.path.isabs(clean_path):
            return clean_path
        if clean_path == self.hpc_root or clean_path.startswith(f"{self.hpc_root}{os.sep}"):
            return clean_path
        return os.path.join(self.hpc_root, clean_path)

    def _init_parquet_index_path(self) -> None:
        safe_data_path = self.data_path.replace("/", "_").replace("\\", "_")
        os.makedirs(self.index_dir, exist_ok=True)
        self.parquet_index_path = os.path.join(
            self.index_dir,
            f"parquet_{safe_data_path}.parquet",
        )

    def _strip_remote_prefix(self, path):
        if isinstance(path, str):
            return re.sub(r"^[^@]+@[^:]+:", "", path)
        return path

    def get_preprocessing_targets(
        self, stage: str, year_range: Tuple[int, int] = None
    ) -> List[Dict[str, Any]]:
        if stage != "spatial":
            raise ValueError(f"Unknown stage: {stage}")

        effective_year_range = year_range or self.year_range
        return [
            {
                "stage": "spatial",
                "output_path": os.path.join(
                    self.get_hpc_output_path("spatial"),
                    self.output_filename,
                ),
                "dependencies": [self.duckdb_path],
                "metadata": {
                    "data_type": "snl_mining_spatial",
                    "source_name": self.source_name,
                    "year_range": effective_year_range,
                    "metric_crs": self.metric_crs,
                    "input_tables": {
                        "properties": self.properties_table,
                        "property_llm_years": self.llm_years_table,
                        "property_work_history_events": self.work_history_table,
                    },
                    "output_variables": self.output_variables,
                },
            }
        ]

    def get_hpc_output_path(self, stage: str) -> str:
        if stage != "spatial":
            raise ValueError(f"Unknown stage: {stage}")
        return self._strip_remote_prefix(
            os.path.join(self.hpc_root, self.data_path, "processed", "stage_2")
        )

    def process_target(self, target: Dict[str, Any]) -> bool:
        stage = target.get("stage")
        if stage != "spatial":
            raise ValueError(f"Unknown stage: {stage}")

        output_path = self._strip_remote_prefix(target["output_path"])
        if not self.override and os.path.exists(output_path):
            logger.info("Skipping existing output: %s", output_path)
            return True

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.prepared_db_path), exist_ok=True)

        try:
            geobox = self._get_or_create_geobox()
            raster_crs = str(geobox.crs)

            years = self._prepare_duckdb_features(raster_crs)
            if not years:
                raise ValueError("No years found for SNL mining preprocessing")

            if not self._create_empty_target_zarr(output_path, geobox, years):
                return False

            return self._rasterize_tiles_to_zarr(output_path, geobox, years)
        except Exception as e:
            logger.exception("Error processing SNL mining target: %s", e)
            return False

    def _connect_duckdb(self, path: str):
        import duckdb

        con = duckdb.connect(path)
        try:
            con.execute("LOAD spatial;")
        except Exception:
            try:
                extension_dir = os.path.join(self.temp_dir, "duckdb_extensions")
                os.makedirs(extension_dir, exist_ok=True)
            except Exception:
                logger.debug("DuckDB extension_directory setting unavailable; continuing")
            con.execute("INSTALL spatial;")
            con.execute("LOAD spatial;")                
            try:
                con.execute("SET geometry_always_xy = true;")
            except Exception:
                logger.debug("DuckDB geometry_always_xy setting unavailable; continuing")
        return con

    def _prepare_duckdb_features(self, raster_crs: str) -> List[int]:
        if not os.path.exists(self.duckdb_path):
            raise FileNotFoundError(f"SNL mining DuckDB not found: {self.duckdb_path}")

        for variable, table_spec in self.admin_tables.items():
            geometry_path = self._resolve_relative_to_hpc(table_spec["geometry_path"])
            if not os.path.exists(geometry_path):
                raise FileNotFoundError(
                    f"Admin geometry for {variable} not found: {geometry_path}"
                )

        logger.info("Preparing DuckDB spatial features in %s", self.prepared_db_path)
        con = self._connect_duckdb(self.prepared_db_path)
        try:
            con.execute(f"ATTACH '{self.duckdb_path}' AS raw_db (READ_ONLY)")
            llm_years_available = self._raw_table_exists(con, self.llm_years_table)
            if not llm_years_available:
                logger.warning(
                    "LLM years table %s not found in stage 0 DuckDB; falling back to observed years only",
                    self.llm_years_table,
                )
            start_year, end_year = self._determine_year_bounds(con)
            logger.info("Using year range %s-%s for mining panel", start_year, end_year)

            self._create_active_mines_table(
                con, start_year, end_year, raster_crs, llm_years_available
            )
            for table_name, radius_m in self.buffer_tables.values():
                self._create_buffer_table(con, table_name, radius_m, raster_crs)
            for table_spec in self.admin_tables.values():
                self._create_admin_count_table(
                    con,
                    table_spec["table_name"],
                    self._resolve_relative_to_hpc(table_spec["geometry_path"]),
                    table_spec["code_column"],
                    raster_crs,
                )
            self._create_rtree_indexes(con)
            self._verify_rtree_queries(con, raster_crs)

            year_rows = con.execute(
                "SELECT DISTINCT year FROM active_mines ORDER BY year"
            ).fetchall()
            years = [int(row[0]) for row in year_rows]
            con.execute("DETACH raw_db")
            return years
        finally:
            con.close()

    def _determine_year_bounds(self, con) -> Tuple[int, int]:
        llm_years_available = self._raw_table_exists(con, self.llm_years_table)
        llm_open_expr = (
            f"y.{self.llm_opening_year_column}" if llm_years_available else "NULL"
        )
        llm_close_expr = (
            f"y.{self.llm_closing_year_column}" if llm_years_available else "NULL"
        )
        llm_join = (
            f"LEFT JOIN raw_db.main.{self.llm_years_table} AS y USING (property_id)"
            if llm_years_available
            else ""
        )
        if self.year_range:
            return int(self.year_range[0]), int(self.year_range[1])

        query = f"""
            SELECT
                CAST(MIN(COALESCE(p.{self.opening_year_column}, {llm_open_expr})) AS INTEGER) AS start_year,
                CAST(MAX(COALESCE(p.{self.closing_year_column}, {llm_close_expr},
                    COALESCE(p.{self.opening_year_column}, {llm_open_expr}))) AS INTEGER) AS end_year
            FROM raw_db.main.{self.properties_table} AS p
            {llm_join}
            WHERE COALESCE(p.{self.opening_year_column}, {llm_open_expr}) IS NOT NULL
              AND p.{self.latitude_column} IS NOT NULL
              AND p.{self.longitude_column} IS NOT NULL
        """
        start_year, end_year = con.execute(query).fetchone()
        if start_year is None or end_year is None:
            raise ValueError("Unable to infer mining year range from stage 0 tables")
        end_year = max(int(end_year), int(start_year))
        return int(start_year), int(end_year)

    def _create_active_mines_table(
        self,
        con,
        start_year: int,
        end_year: int,
        raster_crs: str,
        llm_years_available: bool,
    ) -> None:
        llm_open_expr = (
            f"y.{self.llm_opening_year_column}" if llm_years_available else "NULL"
        )
        llm_close_expr = (
            f"y.{self.llm_closing_year_column}" if llm_years_available else "NULL"
        )
        llm_join = (
            f"LEFT JOIN raw_db.main.{self.llm_years_table} AS y USING (property_id)"
            if llm_years_available
            else ""
        )
        query = f"""
            CREATE OR REPLACE TABLE active_mines AS
            WITH canonical_mines AS (
                SELECT
                    CAST(p.property_id AS VARCHAR) AS property_id,
                    CAST(p.{self.longitude_column} AS DOUBLE) AS longitude,
                    CAST(p.{self.latitude_column} AS DOUBLE) AS latitude,
                    CAST(COALESCE(p.{self.opening_year_column}, {llm_open_expr}) AS INTEGER) AS opening_year,
                    CAST(COALESCE(p.{self.closing_year_column}, {llm_close_expr}) AS INTEGER) AS closing_year,
                    ST_Point(CAST(p.{self.longitude_column} AS DOUBLE), CAST(p.{self.latitude_column} AS DOUBLE)) AS point_wgs84
                FROM raw_db.main.{self.properties_table} AS p
                {llm_join}
                WHERE COALESCE(p.{self.opening_year_column}, {llm_open_expr}) IS NOT NULL
                  AND p.{self.latitude_column} IS NOT NULL
                  AND p.{self.longitude_column} IS NOT NULL
            ),
            bounded_mines AS (
                SELECT
                    property_id,
                    longitude,
                    latitude,
                    opening_year,
                    closing_year,
                    CASE
                        WHEN closing_year IS NULL THEN {end_year}
                        ELSE closing_year
                    END AS closing_year_effective,
                    point_wgs84
                FROM canonical_mines
                WHERE opening_year <= {end_year}
                  AND COALESCE(closing_year, {end_year}) >= {start_year}
            ),
            expanded AS (
                SELECT
                    m.property_id,
                    yr.range::INTEGER AS year,
                    m.longitude,
                    m.latitude,
                    m.opening_year,
                    m.closing_year,
                    m.point_wgs84
                FROM bounded_mines AS m,
                LATERAL range(GREATEST(m.opening_year, {start_year}), LEAST(m.closing_year_effective, {end_year}) + 1) AS yr(range)
            )
            SELECT
                property_id,
                year,
                longitude,
                latitude,
                opening_year,
                closing_year,
                point_wgs84,
                ST_Transform(point_wgs84, 'EPSG:4326', '{self.metric_crs}', true) AS point_metric,
                ST_Transform(point_wgs84, 'EPSG:4326', '{raster_crs}', true) AS point_raster
            FROM expanded
        """
        logger.info("Creating active_mines table")
        con.execute(query)

    def _raw_table_exists(self, con, table_name: str) -> bool:
        query = """
            SELECT 1
            FROM information_schema.tables
            WHERE table_catalog = 'raw_db'
              AND table_schema = 'main'
              AND table_name = ?
            LIMIT 1
        """
        row = con.execute(query, [table_name]).fetchone()
        return row is not None

    def _create_buffer_table(self, con, table_name: str, radius_m: int, raster_crs: str) -> None:
        query = f"""
            CREATE OR REPLACE TABLE {table_name} AS
            WITH buffered AS (
                SELECT
                    property_id,
                    year,
                    1::INTEGER AS value,
                    ST_MakeValid(ST_Buffer(point_metric, {radius_m})) AS geometry_metric
                FROM active_mines
            )
            SELECT
                property_id,
                year,
                value,
                geometry_metric,
                ST_Transform(geometry_metric, '{self.metric_crs}', '{raster_crs}', true) AS geometry_raster
            FROM buffered
            WHERE geometry_metric IS NOT NULL
        """
        logger.info("Creating %s buffer table", table_name)
        con.execute(query)

    def _create_admin_count_table(
        self, con, table_name: str, gpkg_path: str, code_column: str, raster_crs: str
    ) -> None:
        escaped_path = gpkg_path.replace("'", "''")
        geometry_column = self._detect_gpkg_geometry_column(con, gpkg_path, code_column)
        query = f"""
            CREATE OR REPLACE TABLE {table_name} AS
            WITH admin_polygons AS (
                SELECT
                    CAST({code_column} AS VARCHAR) AS adm_code,
                    ST_MakeValid(ST_Transform({geometry_column}, 'EPSG:4326', '{raster_crs}', true)) AS geometry_raster
                FROM ST_Read('{escaped_path}')
                WHERE {code_column} IS NOT NULL
                  AND {geometry_column} IS NOT NULL
            ),
            assignments AS (
                SELECT
                    m.year,
                    a.adm_code
                FROM active_mines AS m
                JOIN admin_polygons AS a
                  ON ST_Intersects(a.geometry_raster, m.point_raster)
            ),
            counts AS (
                SELECT
                    year,
                    adm_code,
                    COUNT(*)::INTEGER AS value
                FROM assignments
                GROUP BY 1, 2
            )
            SELECT
                c.year,
                c.adm_code,
                c.value,
                a.geometry_raster
            FROM counts AS c
            JOIN admin_polygons AS a
            USING (adm_code)
        """
        logger.info("Creating %s table from %s", table_name, os.path.basename(gpkg_path))
        con.execute(query)

    def _detect_gpkg_geometry_column(self, con, gpkg_path: str, code_column: str) -> str:
        escaped_path = gpkg_path.replace("'", "''")
        describe_rows = con.execute(
            f"DESCRIBE SELECT * FROM ST_Read('{escaped_path}')"
        ).fetchall()
        column_names = {str(row[0]) for row in describe_rows}
        for candidate in ("geom", "geometry", "wkb_geometry"):
            if candidate in column_names:
                return candidate
        raise ValueError(
            f"Could not find a geometry column in {gpkg_path}; available columns: {sorted(column_names)} "
            f"(needed admin code column {code_column})"
        )

    def _create_rtree_indexes(self, con) -> None:
        index_specs = []
        for table_name, _ in self.buffer_tables.values():
            index_specs.append((f"idx_{table_name}_rtree", table_name))
        for table_spec in self.admin_tables.values():
            table_name = table_spec["table_name"]
            index_specs.append((f"idx_{table_name}_rtree", table_name))
        for index_name, table_name in index_specs:
            logger.info("Creating R-tree index %s on %s", index_name, table_name)
            con.execute(f"DROP INDEX IF EXISTS {index_name}")
            con.execute(f"CREATE INDEX {index_name} ON {table_name} USING RTREE (geometry_raster)")

    def _verify_rtree_queries(self, con, raster_crs: str) -> None:
        try:
            geobox = self._get_or_create_geobox()
            bounds = geobox.boundingbox
            sample_sql = f"""
                EXPLAIN
                SELECT value
                FROM mine_buffers_10km
                WHERE year = (SELECT MIN(year) FROM active_mines)
                  AND ST_Intersects(
                      geometry_raster,
                      ST_GeomFromText('POLYGON(({bounds.left} {bounds.bottom}, {bounds.right} {bounds.bottom}, {bounds.right} {bounds.top}, {bounds.left} {bounds.top}, {bounds.left} {bounds.bottom}))')
                  )
            """
            plan_rows = con.execute(sample_sql).fetchall()
            plan_text = "\n".join(str(row) for row in plan_rows)
            if "RTREE_INDEX_SCAN" not in plan_text:
                logger.info(
                    "DuckDB EXPLAIN did not report RTREE_INDEX_SCAN for SNL mining tile fetch. "
                    "Queries will still run, but spatial tile fetches may be slower."
                )
        except Exception:
            logger.debug(
                "Skipping DuckDB R-tree plan verification because EXPLAIN is not stable on this DuckDB build."
            )

    def _resolve_relative_to_hpc(self, path: str) -> str:
        return self._resolve_path_with_root(path)

    def _get_or_create_geobox(self):
        misc_level1_dir = self._resolve_path_with_root("misc/processed/stage_1/misc")
        os.makedirs(misc_level1_dir, exist_ok=True)
        return get_or_create_geobox(self.hpc_root, misc_level1_dir)

    def _create_empty_target_zarr(self, output_path: str, geobox, years: List[int]) -> bool:
        try:
            import dask.array as da
            import numpy as np
            import pandas as pd
            import rioxarray  # noqa: F401
            import xarray as xr
            from zarr.codecs import BloscCodec

            time_coords = pd.to_datetime([f"{year}-12-31" for year in sorted(years)])
            ny, nx = geobox.shape
            lat_coords = geobox.coords["latitude"].values.round(5)
            lon_coords = geobox.coords["longitude"].values.round(5)

            data_vars = {}
            for var in self.output_variables:
                data_vars[var] = xr.DataArray(
                    da.zeros((len(time_coords), 1, ny, nx), dtype=np.uint16, chunks=(1, 1, self.tile_size, self.tile_size)),
                    dims=["time", "band", "latitude", "longitude"],
                    coords={
                        "time": time_coords,
                        "band": [1],
                        "latitude": lat_coords,
                        "longitude": lon_coords,
                    },
                    attrs={"_FillValue": 0, "nodata": 0},
                )

            ds = xr.Dataset(
                data_vars,
                attrs={
                    "source_duckdb_path": self.duckdb_path,
                    "prepared_duckdb_path": self.prepared_db_path,
                    "metric_crs": self.metric_crs,
                    "radius_semantics": "count of active mine buffers covering pixel center",
                    "admin_semantics": "count of active mines in containing ADM polygon",
                    "duckdb_tile_fetch_pattern": "WHERE ST_Intersects(geometry_raster, <constant tile geometry>)",
                },
            ).rio.write_crs(geobox.crs)

            compressor = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", blocksize=0)
            encoding = {
                var: {
                    "chunks": (1, 1, self.tile_size, self.tile_size),
                    "compressors": (compressor,),
                    "dtype": "uint16",
                    "fill_value": 0,
                }
                for var in self.output_variables
            }

            logger.info("Creating empty target zarr at %s", output_path)
            ds.to_zarr(
                output_path,
                mode="w",
                compute=False,
                encoding=encoding,
                zarr_format=3,
                consolidated=False,
            )
            return True
        except Exception as e:
            logger.exception("Error creating SNL mining zarr skeleton: %s", e)
            return False

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
            total_tiles = tiles.shape[0] * tiles.shape[1] * len(years)
            processed_tiles = 0

            for year in years:
                logger.info("Rasterizing SNL mining year %s", year)
                for ix in range(tiles.shape[0]):
                    for iy in range(tiles.shape[1]):
                        logger.info(
                            "Processing SNL mining tile year=%s tile=(%s,%s) progress=%s/%s",
                            year,
                            ix,
                            iy,
                            processed_tiles + 1,
                            total_tiles,
                        )
                        tile_geobox = tiles[ix, iy]
                        tile_bounds = tile_geobox.boundingbox
                        tile_wkt = (
                            f"POLYGON(({tile_bounds.left} {tile_bounds.bottom}, "
                            f"{tile_bounds.right} {tile_bounds.bottom}, "
                            f"{tile_bounds.right} {tile_bounds.top}, "
                            f"{tile_bounds.left} {tile_bounds.top}, "
                            f"{tile_bounds.left} {tile_bounds.bottom}))"
                        )

                        tile_arrays = {
                            var: np.zeros(tile_geobox.shape, dtype=np.uint16)
                            for var in self.output_variables
                        }

                        any_data = False
                        for var_name, (table_name, _) in self.buffer_tables.items():
                            rows = self._fetch_features(con, table_name, year, tile_wkt)
                            if rows:
                                any_data = True
                            for value, geom_wkb in rows:
                                geom = Geometry(shapely.wkb.loads(bytes(geom_wkb)), crs=str(tile_geobox.crs))
                                mask = rasterize(geom, tile_geobox).values
                                tile_arrays[var_name] = tile_arrays[var_name] + (mask.astype(np.uint16) * int(value))

                        for var_name, table_spec in self.admin_tables.items():
                            rows = self._fetch_features(
                                con, table_spec["table_name"], year, tile_wkt
                            )
                            if rows:
                                any_data = True
                            for value, geom_wkb in rows:
                                geom = Geometry(shapely.wkb.loads(bytes(geom_wkb)), crs=str(tile_geobox.crs))
                                mask = rasterize(geom, tile_geobox).values
                                tile_arrays[var_name] = np.where(mask, np.uint16(value), tile_arrays[var_name])

                        if any_data:
                            tile_ds = xr.Dataset(
                                {
                                    var: xr.DataArray(
                                        tile_arrays[var][None, None, :, :],
                                        dims=["time", "band", "latitude", "longitude"],
                                        coords={
                                            "time": pd.to_datetime([f"{year}-12-31"]),
                                            "band": [1],
                                            "latitude": tile_geobox.coords["latitude"].values.round(5),
                                            "longitude": tile_geobox.coords["longitude"].values.round(5),
                                        },
                                    )
                                    for var in self.output_variables
                                }
                            )
                            tile_ds.to_zarr(
                                output_path,
                                mode="r+",
                                region="auto",
                                consolidated=False,
                            )

                        processed_tiles += 1
                        if processed_tiles % 100 == 0:
                            logger.info("Processed %s/%s year-tiles", processed_tiles, total_tiles)

            return True
        except Exception as e:
            logger.exception("Error rasterizing SNL mining tiles: %s", e)
            return False
        finally:
            con.close()

    def _fetch_features(self, con, table_name: str, year: int, tile_wkt: str):
        sql = f"""
            SELECT
                value,
                ST_AsWKB(geometry_raster) AS geom_wkb
            FROM {table_name}
            WHERE year = ?
              AND ST_Intersects(geometry_raster, ST_GeomFromText(?))
        """
        return con.execute(sql, [int(year), tile_wkt]).fetchall()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SnlMiningPreprocessor":
        return cls(**config)
