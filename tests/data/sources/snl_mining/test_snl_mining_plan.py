"""SnlMiningSource.plan() must reproduce the relevant slice of the old
SnlMiningPreprocessor's behaviour, plus the new PREPARE/GRID split.
Oracle: tests/data/preprocess/sources/test_characterization_snl_mining.py.
"""

import os

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.snl_mining.source import SnlMiningSource
from src.data.sources.steps import PipelineStep, TargetSelection


def _make_source(tmp_path, *, grid_id="legacy_4326", layout="legacy", **raw):
    ctx = PipelineContext(
        data_root=str(tmp_path / "data_root"),
        local_index_dir=str(tmp_path / "index"),
        grid_id=grid_id,
        layout=layout,
    )
    cfg = SourceConfig.from_dict("snl_mining", dict(raw))
    return SnlMiningSource(ctx, cfg), ctx


def test_no_fetch_step():
    assert SnlMiningSource.STEPS == (PipelineStep.PREPARE, PipelineStep.GRID)


def test_requires_gadm_prepare_and_grid():
    # PREPARE for GADM's polygon geometries (admin-count spatial join), GRID
    # for GID_N_code_mapping.json (translating admin-count tables into
    # gadm's integer ids for the join_on merge -- see module docstring).
    # PREPARE for commodity_prices' normalized (commodity, year) price table,
    # joined against the user-owned commodity_shares table to build
    # mine_priceshock_*.
    from src.data.sources import registry

    assert registry.resolve("snl_mining").requires == (
        ("gadm", PipelineStep.PREPARE), ("gadm", PipelineStep.GRID),
        ("commodity_prices", PipelineStep.PREPARE),
    )


def test_default_output_variables_is_radius_only(tmp_path):
    # Admin-polygon counts (mine_count_adm1/2) are no longer rasterized --
    # they're exported as per-GID parquet sidecars instead (see
    # _export_admin_count_tables), so they must not appear in the zarr's
    # output_variables.
    source, _ = _make_source(tmp_path)
    assert source.output_variables == [
        "mine_count_10km", "mine_count_20km", "mine_count_50km",
        "mine_priceshock_10km", "mine_priceshock_20km", "mine_priceshock_50km",
    ]


def test_default_duckdb_and_prepared_db_paths(tmp_path):
    source, ctx = _make_source(tmp_path)
    assert source.duckdb_path == os.path.join(
        ctx.data_root, "snl_mining", "raw", "manual_xls", "snl_mining_manual_export.duckdb"
    )
    assert source.prepared_db_path == os.path.join(ctx.data_root, "snl_mining", "processed", "stage_1", "snl_mining_prepared.duckdb")


def test_duckdb_path_honors_layout_v2(tmp_path):
    # Stage-0's manual export is this source's raw input -- routed through
    # output_root(FETCH) (-> layout.raw_root()) so it moves under
    # layout="v2" too, instead of hardcoding the legacy processed/stage_0
    # shape.
    v2_source, v2_ctx = _make_source(tmp_path, layout="v2")
    assert v2_source.duckdb_path == os.path.join(
        v2_ctx.data_root, "raw", "snl_mining", "manual_xls", "snl_mining_manual_export.duckdb"
    )


def test_prepared_db_path_honors_layout_v2(tmp_path):
    # PREPARE is snl_mining's own artefact, like every other source's --
    # routed through output_root() so it moves under layout="v2" too,
    # instead of hardcoding the legacy processed/stage_1 shape.
    v2_source, v2_ctx = _make_source(tmp_path, layout="v2")
    assert v2_source.prepared_db_path == os.path.join(
        v2_ctx.data_root, "prepared", "snl_mining", "snl_mining_prepared.duckdb"
    )


def test_prepared_db_path_config_override_still_wins(tmp_path):
    source, ctx = _make_source(tmp_path, layout="v2", aggregation={"prepared_db_path": "custom/prepared.duckdb"})
    assert source.prepared_db_path == os.path.join(ctx.data_root, "custom", "prepared.duckdb")


def test_default_radius_and_admin_variables(tmp_path):
    source, ctx = _make_source(tmp_path)
    assert source.buffer_tables == {
        "mine_count_10km": ("mine_buffers_10km", 10000, "value", "uint16"),
        "mine_count_20km": ("mine_buffers_20km", 20000, "value", "uint16"),
        "mine_count_50km": ("mine_buffers_50km", 50000, "value", "uint16"),
        "mine_priceshock_10km": ("mine_buffers_10km", 10000, "value_priceshock", "float32"),
        "mine_priceshock_20km": ("mine_buffers_20km", 20000, "value_priceshock", "float32"),
        "mine_priceshock_50km": ("mine_buffers_50km", 50000, "value_priceshock", "float32"),
    }
    assert source.admin_tables["mine_count_adm1"]["geometry_path"] == os.path.join(
        ctx.data_root, "misc", "processed", "stage_1", "gadm", "gadm_levelADM_1_simplified.gpkg"
    )


def test_commodity_prices_path_resolution_legacy(tmp_path):
    source, ctx = _make_source(tmp_path)
    assert source.commodity_prices_path == os.path.join(
        ctx.data_root, "commodity_prices", "processed", "stage_1", "commodity_prices.parquet"
    )


def test_commodity_prices_path_resolution_v2(tmp_path):
    source, ctx = _make_source(tmp_path, layout="v2")
    assert source.commodity_prices_path == os.path.join(
        ctx.data_root, "prepared", "commodity_prices", "commodity_prices.parquet"
    )


def test_commodity_prices_path_config_override_wins(tmp_path):
    source, ctx = _make_source(tmp_path, commodity_prices_path="custom/prices.parquet")
    assert source.commodity_prices_path == os.path.join(ctx.data_root, "custom", "prices.parquet")


def test_default_admin_variables_geometry_path_honors_layout_v2(tmp_path):
    # Cross-source reference to gadm's own PREPARE output -- must keep
    # finding it under layout="v2" too (mirrors
    # CountryClassificationsSource._plan_grid()'s equivalent gadm reference).
    source, ctx = _make_source(tmp_path, layout="v2")
    assert source.admin_tables["mine_count_adm1"]["geometry_path"] == os.path.join(
        ctx.data_root, "prepared", "misc", "gadm", "gadm_levelADM_1_simplified.gpkg"
    )
    assert source.admin_tables["mine_count_adm2"]["geometry_path"] == os.path.join(
        ctx.data_root, "prepared", "misc", "gadm", "gadm_levelADM_2_simplified.gpkg"
    )


def test_prepare_plan_empty_when_stage0_duckdb_missing(tmp_path):
    source, _ = _make_source(tmp_path)
    assert source.plan(PipelineStep.PREPARE, TargetSelection()) == []


def test_prepare_plan_empty_when_commodity_prices_missing(tmp_path):
    # duckdb_path present but commodity_prices' PREPARE output isn't --
    # REQUIRES on commodity_prices' PREPARE (see module docstring), plan()
    # stays defensively self-consistent with that gate.
    source, _ = _make_source(tmp_path)
    os.makedirs(os.path.dirname(source.duckdb_path), exist_ok=True)
    open(source.duckdb_path, "w").close()
    assert source.plan(PipelineStep.PREPARE, TargetSelection()) == []


def test_prepare_plan_target_when_stage0_duckdb_present(tmp_path):
    source, _ = _make_source(tmp_path)
    os.makedirs(os.path.dirname(source.duckdb_path), exist_ok=True)
    open(source.duckdb_path, "w").close()
    os.makedirs(os.path.dirname(source.commodity_prices_path), exist_ok=True)
    open(source.commodity_prices_path, "w").close()

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].output_path == source.prepared_db_path
    assert targets[0].inputs == (source.duckdb_path, source.commodity_prices_path)


def test_grid_plan_empty_when_prepared_db_missing(tmp_path):
    source, _ = _make_source(tmp_path)
    assert source.plan(PipelineStep.GRID, TargetSelection()) == []


def test_output_root_grid_matches_old_get_hpc_output_path(tmp_path):
    source, ctx = _make_source(tmp_path)
    assert source.output_root(PipelineStep.GRID) == os.path.join(ctx.data_root, "snl_mining", "processed", "stage_2")


def test_output_root_grid_honors_ease6933(tmp_path):
    # Regression test: _output_root() used to hardcode "stage_2" and ignore
    # ctx.grid_id entirely, unlike every other source's output_root().
    source, ctx = _make_source(tmp_path, grid_id="ease6933")
    assert source.output_root(PipelineStep.GRID) == os.path.join(
        ctx.data_root, "snl_mining", "processed", "stage_2_ease6933"
    )


def test_grid_target_uses_v2_family_path_under_layout_v2(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, layout="v2")
    os.makedirs(os.path.dirname(source.prepared_db_path), exist_ok=True)
    open(source.prepared_db_path, "w").close()

    class _FakeConnection:
        def execute(self, *args, **kwargs):
            return self

        def fetchall(self):
            return [(2020,)]

        def close(self):
            pass

    monkeypatch.setattr(source, "_connect_duckdb", lambda path: _FakeConnection())

    targets = source.plan(PipelineStep.GRID, TargetSelection())
    assert len(targets) == 1
    assert targets[0].output_path == os.path.join(ctx.data_root, "grid", "legacy_4326", "snl_mining.zarr")


def test_export_admin_count_tables_writes_gid_keyed_parquet(tmp_path):
    import json

    import duckdb
    import pandas as pd

    from src.data.sources.misc.gadm import gid_mapping_path

    source, ctx = _make_source(tmp_path)

    os.makedirs(os.path.dirname(source.prepared_db_path), exist_ok=True)
    con = duckdb.connect(source.prepared_db_path)
    con.execute("CREATE TABLE adm1_year_counts (year INTEGER, adm_code VARCHAR, value INTEGER)")
    con.execute(
        "INSERT INTO adm1_year_counts VALUES (2019, 'USA.1_1', 3), (2020, 'USA.1_1', 5), (2020, 'FRA.2_1', 1)"
    )
    con.execute("CREATE TABLE adm2_year_counts (year INTEGER, adm_code VARCHAR, value INTEGER)")
    con.execute("INSERT INTO adm2_year_counts VALUES (2020, 'USA.1.1_1', 2)")
    con.close()

    mapping_path_1 = gid_mapping_path(ctx.data_root, ctx.grid_id, ctx.layout, "GID_1")
    os.makedirs(os.path.dirname(mapping_path_1), exist_ok=True)
    with open(mapping_path_1, "w") as f:
        json.dump({"USA.1_1": 5}, f)  # FRA.2_1 deliberately absent

    mapping_path_2 = gid_mapping_path(ctx.data_root, ctx.grid_id, ctx.layout, "GID_2")
    with open(mapping_path_2, "w") as f:
        json.dump({"USA.1.1_1": 7}, f)

    output_dir = tmp_path / "grid_output"
    output_dir.mkdir()
    assert source._export_admin_count_tables(str(output_dir)) is True

    adm1_df = pd.read_parquet(output_dir / "mine_count_adm1.parquet")
    assert list(adm1_df.columns) == ["GID_1", "year", "mine_count_adm1"]
    # FRA.2_1 has no gadm mapping entry -> dropped, not zeroed.
    assert set(zip(adm1_df["GID_1"], adm1_df["year"], adm1_df["mine_count_adm1"])) == {
        (5, 2019, 3), (5, 2020, 5),
    }

    adm2_df = pd.read_parquet(output_dir / "mine_count_adm2.parquet")
    assert list(adm2_df.columns) == ["GID_2", "year", "mine_count_adm2"]
    assert set(zip(adm2_df["GID_2"], adm2_df["year"], adm2_df["mine_count_adm2"])) == {(7, 2020, 2)}


def test_export_admin_count_tables_fails_clearly_when_mapping_missing(tmp_path):
    import duckdb

    source, ctx = _make_source(tmp_path)

    os.makedirs(os.path.dirname(source.prepared_db_path), exist_ok=True)
    con = duckdb.connect(source.prepared_db_path)
    con.execute("CREATE TABLE adm1_year_counts (year INTEGER, adm_code VARCHAR, value INTEGER)")
    con.execute("CREATE TABLE adm2_year_counts (year INTEGER, adm_code VARCHAR, value INTEGER)")
    con.close()

    output_dir = tmp_path / "grid_output"
    output_dir.mkdir()
    assert source._export_admin_count_tables(str(output_dir)) is False


# --- _determine_year_bounds: plausible-range guard --------------------


def _attach_properties_db(source, tmp_path, rows, *, columns="property_id INTEGER, actual_start_up_year INTEGER, actual_closure_year INTEGER, latitude DOUBLE, longitude DOUBLE"):
    """A `raw_db`-attached connection with a `properties` table, matching the
    shape `_determine_year_bounds` expects to be called against (it queries
    `raw_db.main.{properties_table}`, so a caller must ATTACH first --
    mirrors `_execute_prepare`'s own setup)."""
    import duckdb

    raw_path = str(tmp_path / "raw_stage0.duckdb")
    raw_con = duckdb.connect(raw_path)
    raw_con.execute(f"CREATE TABLE properties ({columns})")
    raw_con.executemany(f"INSERT INTO properties VALUES ({','.join('?' * len(rows[0]))})", rows)
    raw_con.close()

    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{raw_path}' AS raw_db (READ_ONLY)")
    return con


def test_determine_year_bounds_excludes_implausible_year_from_min(tmp_path):
    # The exact bug found on real data: a garbled opening year (150, likely
    # a "1950" typo) must not drag the auto-detected start_year down to an
    # implausible value that later corrupts a zarr time coordinate.
    source, _ = _make_source(tmp_path)
    con = _attach_properties_db(
        source, tmp_path,
        rows=[
            (1, 150, None, 10.0, 20.0),  # implausible -- excluded from MIN/MAX
            (2, 1990, 2010, 11.0, 21.0),
            (3, 2000, None, 12.0, 22.0),
        ],
    )
    start_year, end_year = source._determine_year_bounds(con, llm_years_available=False)
    assert start_year == 1990
    assert end_year == 2010


def test_determine_year_bounds_all_plausible_unaffected(tmp_path):
    source, _ = _make_source(tmp_path)
    con = _attach_properties_db(
        source, tmp_path,
        rows=[(1, 1980, 1995, 10.0, 20.0), (2, 1990, 2010, 11.0, 21.0)],
    )
    start_year, end_year = source._determine_year_bounds(con, llm_years_available=False)
    assert (start_year, end_year) == (1980, 2010)


def test_determine_year_bounds_logs_warning_when_excluding(tmp_path, caplog):
    import logging

    source, _ = _make_source(tmp_path)
    con = _attach_properties_db(
        source, tmp_path,
        rows=[(1, 150, None, 10.0, 20.0), (2, 1990, 2010, 11.0, 21.0)],
    )
    with caplog.at_level(logging.WARNING):
        source._determine_year_bounds(con, llm_years_available=False)
    assert any("plausible range" in r.getMessage() for r in caplog.records)


def test_determine_year_bounds_config_year_range_bypasses_detection(tmp_path):
    # An explicit config year_range skips the DB query entirely -- no
    # plausibility filtering applies to a user-specified value.
    source, _ = _make_source(tmp_path, year_range=[100, 200])
    assert source._determine_year_bounds(con=None, llm_years_available=False) == (100, 200)


def test_get_or_create_geobox_delegates_to_shared_target_helper(tmp_path, monkeypatch):
    source, ctx = _make_source(tmp_path, grid_id="ease6933")

    calls = []

    def fake_get_target_geobox(passed_ctx):
        calls.append(passed_ctx)
        return "fake-canonical-geobox"

    import src.data.sources.snl_mining.source as snl_source_module

    monkeypatch.setattr(snl_source_module, "get_target_geobox", fake_get_target_geobox)

    assert source._get_or_create_geobox() == "fake-canonical-geobox"
    assert calls == [ctx]


# --- price-shock: _create_mine_priceshock_table / _create_buffer_table ------


def _write_prices_parquet(tmp_path, rows):
    """rows: list of (commodity, year, ln_price_real) tuples."""
    import pandas as pd

    path = str(tmp_path / "commodity_prices.parquet")
    pd.DataFrame(rows, columns=["commodity", "year", "ln_price_real"]).to_parquet(path, index=False)
    return path


def _attach_raw_db_with_shares(tmp_path, share_rows, *, filename="raw_stage0_shares.duckdb"):
    """A `raw_db`-attached in-memory connection with a `commodity_shares`
    table, matching the shape `_create_mine_priceshock_table` expects (it
    queries `raw_db.main.{commodity_shares_table}`) -- mirrors
    `_attach_properties_db`'s pattern for `properties`."""
    import duckdb

    raw_path = str(tmp_path / filename)
    raw_con = duckdb.connect(raw_path)
    raw_con.execute("CREATE TABLE commodity_shares (property_id VARCHAR, commodity VARCHAR, share DOUBLE)")
    if share_rows:
        raw_con.executemany("INSERT INTO commodity_shares VALUES (?, ?, ?)", share_rows)
    raw_con.close()

    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{raw_path}' AS raw_db (READ_ONLY)")
    return con


def test_create_mine_priceshock_table_fully_priced_mine(tmp_path):
    source, _ = _make_source(tmp_path)
    source.commodity_prices_path = _write_prices_parquet(tmp_path, [("gold", 2020, 7.5)])
    con = _attach_raw_db_with_shares(tmp_path, [("m1", "gold", 1.0)])
    con.execute("CREATE TABLE active_mines AS SELECT 'm1' AS property_id, 2020 AS year")

    source._create_mine_priceshock_table(con)

    row = con.execute("SELECT property_id, year, value FROM mine_priceshock").fetchone()
    assert row == ("m1", 2020, 7.5)


def test_create_mine_priceshock_table_partially_unpriced_ignores_unmatched_share(tmp_path):
    # m2 is 50% gold (priced) + 50% uranium (no WB series) -- the unmatched
    # half must be ignored by SUM(), not silently treated as a zero-price
    # contribution.
    source, _ = _make_source(tmp_path)
    source.commodity_prices_path = _write_prices_parquet(tmp_path, [("gold", 2020, 7.5)])
    con = _attach_raw_db_with_shares(tmp_path, [("m2", "gold", 0.5), ("m2", "uranium", 0.5)])
    con.execute("CREATE TABLE active_mines AS SELECT 'm2' AS property_id, 2020 AS year")

    source._create_mine_priceshock_table(con)

    row = con.execute("SELECT property_id, year, value FROM mine_priceshock").fetchone()
    assert row == ("m2", 2020, 0.5 * 7.5)


def test_create_mine_priceshock_table_fully_unpriced_mine_is_null_not_zero(tmp_path):
    # m3 produces only uranium (no WB series) -- value must be SQL NULL, not
    # 0, since 0 is itself a legitimate price-shock value (see module
    # docstring / _rasterize_tiles_to_zarr's NaN-fill handling).
    source, _ = _make_source(tmp_path)
    source.commodity_prices_path = _write_prices_parquet(tmp_path, [("gold", 2020, 7.5)])
    con = _attach_raw_db_with_shares(tmp_path, [("m3", "uranium", 1.0)])
    con.execute("CREATE TABLE active_mines AS SELECT 'm3' AS property_id, 2020 AS year")

    source._create_mine_priceshock_table(con)

    row = con.execute("SELECT property_id, year, value FROM mine_priceshock").fetchone()
    assert row[0:2] == ("m3", 2020)
    assert row[2] is None


def test_create_mine_priceshock_table_missing_shares_table_yields_empty_table(tmp_path, caplog):
    import logging

    import duckdb

    source, _ = _make_source(tmp_path)
    source.commodity_prices_path = _write_prices_parquet(tmp_path, [("gold", 2020, 7.5)])
    # raw_db with no commodity_shares table at all -- simulates the user's
    # ingestion not having run yet.
    raw_path = str(tmp_path / "raw_stage0_no_shares.duckdb")
    duckdb.connect(raw_path).close()
    con = duckdb.connect(":memory:")
    con.execute(f"ATTACH '{raw_path}' AS raw_db (READ_ONLY)")
    con.execute("CREATE TABLE active_mines AS SELECT 'm1' AS property_id, 2020 AS year")

    with caplog.at_level(logging.WARNING):
        source._create_mine_priceshock_table(con)

    assert con.execute("SELECT count(*) FROM mine_priceshock").fetchone() == (0,)
    assert any("Commodity shares table" in r.getMessage() for r in caplog.records)


def test_create_buffer_table_carries_value_and_value_priceshock(tmp_path):
    source, _ = _make_source(tmp_path)
    source.commodity_prices_path = _write_prices_parquet(tmp_path, [("gold", 2020, 7.5)])
    con = _attach_raw_db_with_shares(
        tmp_path, [("m1", "gold", 1.0), ("m2", "uranium", 1.0)],
    )
    con.execute("LOAD spatial;")
    # Two mines, one priced (m1 -> gold) and one fully unpriced (m2 -> uranium).
    con.execute(
        """
        CREATE TABLE active_mines AS
        SELECT * FROM (VALUES
            ('m1', 2020, ST_Point(0, 0)),
            ('m2', 2020, ST_Point(1, 1))
        ) AS t(property_id, year, point_metric)
        """
    )
    source._create_mine_priceshock_table(con)

    source._create_buffer_table(con, "mine_buffers_test", 10000, "EPSG:3857")

    rows = {
        r[0]: (r[1], r[2])
        for r in con.execute("SELECT property_id, value, value_priceshock FROM mine_buffers_test").fetchall()
    }
    assert rows["m1"] == (1, 7.5)
    assert rows["m2"][0] == 1
    assert rows["m2"][1] is None
