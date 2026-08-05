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
    from src.data.sources import registry

    assert registry.resolve("snl_mining").requires == (
        ("gadm", PipelineStep.PREPARE), ("gadm", PipelineStep.GRID),
    )


def test_default_output_variables_is_radius_only(tmp_path):
    # Admin-polygon counts (mine_count_adm1/2) are no longer rasterized --
    # they're exported as per-GID parquet sidecars instead (see
    # _export_admin_count_tables), so they must not appear in the zarr's
    # output_variables.
    source, _ = _make_source(tmp_path)
    assert source.output_variables == ["mine_count_10km", "mine_count_20km", "mine_count_50km"]


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
        "mine_count_10km": ("mine_buffers_10km", 10000),
        "mine_count_20km": ("mine_buffers_20km", 20000),
        "mine_count_50km": ("mine_buffers_50km", 50000),
    }
    assert source.admin_tables["mine_count_adm1"]["geometry_path"] == os.path.join(
        ctx.data_root, "misc", "processed", "stage_1", "gadm", "gadm_levelADM_1_simplified.gpkg"
    )


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


def test_prepare_plan_target_when_stage0_duckdb_present(tmp_path):
    source, _ = _make_source(tmp_path)
    os.makedirs(os.path.dirname(source.duckdb_path), exist_ok=True)
    open(source.duckdb_path, "w").close()

    targets = source.plan(PipelineStep.PREPARE, TargetSelection())
    assert len(targets) == 1
    assert targets[0].output_path == source.prepared_db_path
    assert targets[0].inputs == (source.duckdb_path,)


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
