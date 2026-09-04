"""End-to-end DuckDB assembly on a synthetic tiled-parquet fixture: block
aggregation per resampling method, static x annual merge, land mask, join_on
sidecar, fillna, derived pixel ids, column order, and grid-shake."""

import os

import numpy as np
import pytest

from tests.data.assemble.conftest import (
    grid,
    read_panel,
    run_create,
    run_update,
    write_land_mask,
    write_tiled_source,
)

W, H = 20, 16


@pytest.fixture
def fixture(tmp_path):
    root = str(tmp_path / "grid")

    # annual source: lst_mean (average), obs_count (sum)
    modis = write_tiled_source(
        root, "modis", W=W, H=H, years=[2000, 2001],
        value_fn=lambda r, c, y: {
            "lst_mean": float(r * 10 + c + (y - 2000)),
            "obs_count": 1.0,
        },
    )
    # static categorical source: dominant code (mode) + the GID key column
    gadm = write_tiled_source(
        root, "gadm", W=W, H=H, years=None,
        value_fn=lambda r, c, y: {"GID_0": f"C{(r * W + c) % 3}", "gadm_code": float((r + c) % 4)},
    )
    # sparse source with fillna default (like snl_mining -> 0)
    snl = write_tiled_source(
        root, "snl_mining", W=W, H=H, years=None,
        value_fn=lambda r, c, y: {"mine_count": 1.0},
        keep={(0, 0), (1, 1), (2, 2)},
    )
    # even rows only (partial blocks survive) AND nothing below row 12
    # (blocks 12-15 become fully ocean and drop out entirely).
    lm = write_land_mask(root, W=W, H=H, land=lambda r, c: (r % 2 == 0) and (r < 12))

    sidecar = str(tmp_path / "cc.parquet")
    import pandas as pd
    pd.DataFrame({"GID_0": ["C0", "C1", "C2"], "income_grp": [10, 20, 30]}).to_parquet(sidecar, index=False)

    datasets = {
        "modis": {"path": modis, "index_cols": ["pixel_id", "year"],
                  "resampling": {"default": "average", "*_count": "sum"}},
        "gadm": {"path": gadm, "index_cols": ["pixel_id"], "resampling": "mode"},
        "snl_mining": {"path": snl, "index_cols": ["pixel_id"], "resampling": "sum"},
        "country_classifications": {"path": sidecar, "join_on": "GID_0",
                                    "columns": ["GID_0", "income_grp"]},
    }
    return datasets, lm, str(tmp_path / "out")


def test_block_aggregation_average_and_sum(fixture):
    datasets, lm, out = fixture
    df = run_create(grid(F=2), datasets, out, land_mask_path=lm)

    # pixel_id 0, year 2000 = native rows {0,1} cols {0,1}; land keeps r=0 only.
    row = df[(df.pixel_id == 0) & (df.year == 2000)].iloc[0]
    assert row["lst_mean"] == pytest.approx((0 + 1) / 2)      # avg(0,1)
    assert row["obs_count"] == pytest.approx(2.0)             # sum of two 1s
    row01 = df[(df.pixel_id == 0) & (df.year == 2001)].iloc[0]
    assert row01["lst_mean"] == pytest.approx((1 + 2) / 2)    # +1 for year 2001


def test_static_source_broadcasts_across_years(fixture):
    datasets, lm, out = fixture
    df = run_create(grid(F=2), datasets, out, land_mask_path=lm)
    g = df[df.pixel_id == 0]
    assert set(g["year"]) == {2000, 2001}
    assert g["GID_0"].nunique() == 1  # same static value each year


def test_land_mask_drops_all_ocean_blocks(fixture):
    datasets, lm, out = fixture
    df_masked = run_create(grid(F=2), datasets, out + "_m", land_mask_path=lm)
    df_all = run_create(grid(F=2), datasets, out + "_a", land_mask_path=None)
    assert len(df_masked) < len(df_all)


def test_join_on_sidecar_and_fillna(fixture):
    datasets, lm, out = fixture
    df = run_create(grid(F=2), datasets, out, land_mask_path=lm)
    assert "income_grp" in df.columns
    # income_grp follows GID_0 (C0->10, C1->20, C2->30)
    for _, r in df.iterrows():
        assert r["income_grp"] == {"C0": 10, "C1": 20, "C2": 30}[r["GID_0"]]
    # snl_mining sparse -> mostly filled with 0, never NaN
    assert df["mine_count"].notna().all()
    assert (df["mine_count"] == 0).any()


def test_derived_pixel_id_column(fixture):
    datasets, lm, out = fixture
    df = run_create(grid(F=2), datasets, out, land_mask_path=lm, derived={"pixel_id_4km": 4000})
    assert "pixel_id_4km" in df.columns
    # coarser id: several distinct 2km pixel_ids map to one 4km id
    assert df["pixel_id_4km"].nunique() < df["pixel_id"].nunique()


def test_column_order_is_index_then_config_source_order(fixture):
    datasets, lm, out = fixture
    df = run_create(grid(F=2), datasets, out, land_mask_path=lm, derived={"pixel_id_4km": 4000})
    cols = list(df.columns)
    assert cols[:3] == ["pixel_id", "year", "pixel_id_4km"]
    assert cols.index("lst_mean") < cols.index("GID_0") < cols.index("mine_count") < cols.index("income_grp")
    assert cols[-2:] == ["ix", "iy"]


def test_native_grid_is_pure_join_no_aggregation(fixture):
    datasets, lm, out = fixture
    df = run_create(grid(F=1), datasets, out, land_mask_path=lm)
    # F=1: every kept land pixel is its own row; lst_mean is the raw value
    row = df[(df.pixel_id == df.pixel_id.min()) & (df.year == 2000)].iloc[0]
    assert row["obs_count"] == pytest.approx(1.0)


def test_grid_shake_changes_values(fixture):
    datasets, lm, out = fixture
    base = run_create(grid(F=4), datasets, out + "_b", land_mask_path=lm)
    shaken = run_create(grid(F=4, DR=2, DC=2), datasets, out + "_s", land_mask_path=lm,
                        shake_offset=(0.5, 0.5))
    merged = base.merge(shaken, on=["pixel_id", "year"], suffixes=("_b", "_s"), how="inner")
    assert not np.allclose(merged["lst_mean_b"], merged["lst_mean_s"], equal_nan=True)


def test_winsorize_clamps_outliers_before_aggregation(tmp_path):
    root = str(tmp_path / "g")
    # one extreme value per row; winsorize should pull it toward the p95
    src = write_tiled_source(
        root, "m", W=W, H=H, years=[2000],
        value_fn=lambda r, c, y: {"v": (1000.0 if (r, c) == (0, 0) else float(r + c))},
    )
    datasets = {"m": {"path": src, "index_cols": ["pixel_id", "year"],
                      "resampling": "average", "winsorize": 0.05}}
    df = run_create(grid(F=2), datasets, str(tmp_path / "out"))
    # block (0,0) held native cells (0,0)=1000 clamped and (0,1)=1, (1,0)=1, (1,1)=2
    assert df.sort_values("pixel_id").iloc[0]["v"] < 1000.0


def test_update_mode_refreshes_one_source_only(fixture):
    datasets, lm, out = fixture
    run_create(grid(F=2), datasets, out, land_mask_path=lm)
    before = read_panel(out)

    # bump modis values, re-run update for modis only
    root = datasets["modis"]["path"].rsplit("/modis", 1)[0]
    write_tiled_source(
        root, "modis", W=W, H=H, years=[2000, 2001],
        value_fn=lambda r, c, y: {"lst_mean": float(r * 10 + c + (y - 2000) + 100), "obs_count": 1.0},
    )
    after = run_update(grid(F=2), datasets, "modis", out, land_mask_path=lm)

    m = before.merge(after, on=["pixel_id", "year"], suffixes=("_0", "_1"))
    assert (m["lst_mean_1"] - m["lst_mean_0"]).round().eq(100).all()   # modis changed
    assert (m["GID_0_0"] == m["GID_0_1"]).all()                        # gadm untouched
    # column order is preserved across the update (not moved to the tail)
    assert list(before.columns) == list(after.columns)


def test_update_refuses_when_refreshed_source_is_empty(fixture):
    datasets, lm, out = fixture
    run_create(grid(F=2), datasets, out, land_mask_path=lm)

    root = datasets["modis"]["path"].rsplit("/modis", 1)[0]
    write_tiled_source(  # all-NaN -> every group dropped by HAVING
        root, "modis", W=W, H=H, years=[2000, 2001],
        value_fn=lambda r, c, y: {"lst_mean": float("nan"), "obs_count": float("nan")},
    )
    with pytest.raises(ValueError, match="refreshed aggregate|empty"):
        run_update(grid(F=2), datasets, "modis", out, land_mask_path=lm)


def test_spill_dir_lifecycle(tmp_path, monkeypatch):
    from src.data.assemble import sql_engine as se

    root = str(tmp_path / "g")
    src = write_tiled_source(root, "m", W=W, H=H, years=[2000],
                             value_fn=lambda r, c, y: {"v": float(r + c)})
    datasets = {"m": {"path": src, "index_cols": ["pixel_id", "year"], "resampling": "average"}}
    common = dict(
        datasets=datasets, resolution_m=10000.0, shake_offset=(0.0, 0.0),
        land_mask_path=None, compression="zstd", tile_size=2048, year_range=None,
        derived_pixel_ids=None, mode="create", datasource=None,
    )

    # default: a private assemble_* subdir under the scratch root, removed after;
    # a sibling belonging to another component is left untouched.
    scratch_root = str(tmp_path / "scratch_nobackup")
    monkeypatch.setattr(se, "DEFAULT_SPILL_ROOT", scratch_root)
    sibling = os.path.join(scratch_root, "snl_mining")
    os.makedirs(sibling)
    se.run_sql_assembly(output_path=str(tmp_path / "out1"), duckdb_cfg=se.DuckDBConfig(), **common)
    assert os.path.isdir(sibling)                                   # sibling untouched
    assert not any(n.startswith("assemble_") for n in os.listdir(scratch_root))  # our subdir gone

    # explicit temp_dir: used as-is, left in place
    explicit = str(tmp_path / "my_spill")
    se.run_sql_assembly(output_path=str(tmp_path / "out2"),
                        duckdb_cfg=se.DuckDBConfig(temp_dir=explicit), **common)
    assert os.path.isdir(explicit)


def test_colliding_output_column_names_are_rejected(tmp_path):
    root = str(tmp_path / "g")
    a = write_tiled_source(root, "a", W=W, H=H, years=[2000],
                           value_fn=lambda r, c, y: {"v": 1.0})
    b = write_tiled_source(root, "b", W=W, H=H, years=[2000],
                           value_fn=lambda r, c, y: {"v": 2.0})  # same column name, no prefix
    datasets = {
        "a": {"path": a, "index_cols": ["pixel_id", "year"], "resampling": "average"},
        "b": {"path": b, "index_cols": ["pixel_id", "year"], "resampling": "average"},
    }
    with pytest.raises(ValueError, match="collides"):
        run_create(grid(F=2), datasets, str(tmp_path / "out"))
