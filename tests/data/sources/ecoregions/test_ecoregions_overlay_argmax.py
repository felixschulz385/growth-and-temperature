"""Unit tests for `overlay.compute_dominant_classes` -- area-weighted
dominant-class-per-polygon via vector overlay, genuinely new logic with no
prior test coverage anywhere in this codebase (raster zonal-mode helpers in
`src/data/assemble/geometry.py` don't do polygon-polygon overlay at all).

All geometries use EPSG:3857 (a projected, metric CRS) for both the input
GeoDataFrames and the `crs=` passed to `compute_dominant_classes`, so
`to_crs()` is a no-op and areas are exactly the rectangles' hand-computed
areas -- no reprojection distortion to account for.
"""

import geopandas as gpd
import shapely.geometry as sg

from src.data.sources.ecoregions.overlay import compute_dominant_classes

CRS = "EPSG:3857"


def _rect(x0, y0, x1, y1):
    return sg.box(x0, y0, x1, y1)


def _class_gdf(rows):
    """rows: list of (REALM, BIOME_NUM, BIOME_NAME, ECO_ID, ECO_NAME, geometry)."""
    return gpd.GeoDataFrame(
        [{"REALM": r[0], "BIOME_NUM": r[1], "BIOME_NAME": r[2], "ECO_ID": r[3], "ECO_NAME": r[4]} for r in rows],
        geometry=[r[5] for r in rows],
        crs=CRS,
    )


def _gid_gdf(rows, gid_col="GID_3"):
    """rows: list of (gid_code, geometry)."""
    return gpd.GeoDataFrame({gid_col: [r[0] for r in rows]}, geometry=[r[1] for r in rows], crs=CRS)


def test_dominant_class_by_area_within_one_gid_unit():
    # 10x10 GID unit split 60/40 between two ecoregions sharing one REALM.
    gid = _gid_gdf([("A", _rect(0, 0, 10, 10))])
    classes = _class_gdf(
        [
            ("Nearctic", 1, "Biome One", 101, "Eco One", _rect(0, 0, 10, 6)),
            ("Nearctic", 2, "Biome Two", 102, "Eco Two", _rect(0, 6, 10, 10)),
        ]
    )

    result = compute_dominant_classes(gid, classes, gid_col="GID_3", crs=CRS)
    assert len(result) == 1
    row = result.iloc[0]

    assert row["GID_3"] == "A"
    assert row["dominant_realm"] == "Nearctic"
    assert row["realm_area_frac"] == 1.0  # both fragments share the same realm

    assert row["dominant_biome_num"] == 1
    assert row["dominant_biome_name"] == "Biome One"
    assert row["biome_area_frac"] == 0.6

    assert row["dominant_eco_id"] == 101
    assert row["dominant_eco_name"] == "Eco One"
    assert row["eco_area_frac"] == 0.6

    assert row["n_ecoregions_intersecting"] == 2


def test_gid_unit_fully_inside_one_ecoregion_has_frac_one_and_n_one():
    gid = _gid_gdf([("B", _rect(20, 0, 30, 10))])
    classes = _class_gdf(
        [("Palearctic", 3, "Biome Three", 103, "Eco Three", _rect(15, -5, 35, 15))]
    )

    result = compute_dominant_classes(gid, classes, gid_col="GID_3", crs=CRS)
    row = result.iloc[0]
    assert row["dominant_eco_id"] == 103
    assert row["eco_area_frac"] == 1.0
    assert row["n_ecoregions_intersecting"] == 1


def test_exact_area_tie_breaks_on_lowest_code():
    # Two ecoregions split a GID unit exactly 50/50 -- lower ECO_ID wins.
    gid = _gid_gdf([("C", _rect(0, 0, 10, 10))])
    classes = _class_gdf(
        [
            ("Nearctic", 1, "Biome One", 105, "Eco Five", _rect(0, 0, 5, 10)),
            ("Nearctic", 1, "Biome One", 104, "Eco Four", _rect(5, 0, 10, 10)),
        ]
    )

    result = compute_dominant_classes(gid, classes, gid_col="GID_3", crs=CRS)
    row = result.iloc[0]
    assert row["dominant_eco_id"] == 104
    assert row["eco_area_frac"] == 0.5


def test_multiple_gid_units_get_independent_dominant_classes():
    gid = _gid_gdf([("A", _rect(0, 0, 10, 10)), ("B", _rect(20, 0, 30, 10))])
    classes = _class_gdf(
        [
            ("Nearctic", 1, "Biome One", 101, "Eco One", _rect(0, 0, 10, 10)),
            ("Palearctic", 3, "Biome Three", 103, "Eco Three", _rect(20, 0, 30, 10)),
        ]
    )

    result = compute_dominant_classes(gid, classes, gid_col="GID_3", crs=CRS).set_index("GID_3")
    assert result.loc["A", "dominant_eco_id"] == 101
    assert result.loc["B", "dominant_eco_id"] == 103


def test_code_to_id_translates_and_drops_unmapped_rows():
    gid = _gid_gdf([("A", _rect(0, 0, 10, 10)), ("Z", _rect(20, 0, 30, 10))])
    classes = _class_gdf(
        [
            ("Nearctic", 1, "Biome One", 101, "Eco One", _rect(0, 0, 10, 10)),
            ("Palearctic", 3, "Biome Three", 103, "Eco Three", _rect(20, 0, 30, 10)),
        ]
    )

    result = compute_dominant_classes(gid, classes, gid_col="GID_3", crs=CRS, code_to_id={"A": 7})
    assert list(result["GID_3"]) == [7]
    assert list(result["GID_3_code"]) == ["A"]


def test_duplicate_code_with_two_label_spellings_does_not_raise():
    # RESOLVE has codes that appear with two spellings of the same name across
    # features; the label lookup must dedupe on the code, not the (code, label)
    # pair, or Series.map() raises InvalidIndexError.
    gid = _gid_gdf([("A", _rect(0, 0, 10, 10))])
    classes = _class_gdf(
        [
            ("Nearctic", 1, "Biome One", 101, "Eco One", _rect(0, 0, 10, 6)),
            ("Nearctic", 1, "Biome One ", 101, "Eco One ", _rect(0, 6, 10, 10)),
        ]
    )

    result = compute_dominant_classes(gid, classes, gid_col="GID_3", crs=CRS)
    row = result.iloc[0]
    assert row["dominant_biome_num"] == 1
    assert row["dominant_biome_name"] == "Biome One"  # sorted-first spelling wins
    assert row["dominant_eco_id"] == 101
    assert row["dominant_eco_name"] == "Eco One"


def test_no_intersection_returns_empty_frame_with_expected_columns():
    gid = _gid_gdf([("A", _rect(0, 0, 10, 10))])
    classes = _class_gdf([("Nearctic", 1, "Biome One", 101, "Eco One", _rect(100, 100, 110, 110))])

    result = compute_dominant_classes(gid, classes, gid_col="GID_3", crs=CRS)
    assert len(result) == 0
    assert "dominant_biome_num" in result.columns
    assert "n_ecoregions_intersecting" in result.columns
