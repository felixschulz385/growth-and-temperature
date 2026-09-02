"""Area-weighted dominant-class-per-polygon table.

Computes, for every polygon in one GeoDataFrame (e.g. GADM's `GID_3` admin
units), the area-weighted dominant class from another GeoDataFrame's
categorical attributes (RESOLVE's `REALM`/`BIOME_NUM`/`ECO_ID`) via a
polygon-polygon intersection (`geopandas.overlay`) and a per-group area
argmax -- deliberately not raster zonal-mode. `src/data/assemble/geometry.py`'s
`zonal_reduce_odc`/`assemble_geometry_weighted` only reduce raster cells
(mean/sum/count/min/max/first, no majority/mode), and would implicitly
area-weight via 1km pixel counts rather than exact polygon area; this module
computes exact geometric area instead, at a fraction of the pixel count.
"""

from __future__ import annotations

from typing import Dict, Optional

import geopandas as gpd
import pandas as pd

#: RESOLVE attribute -> (dominant-value column, area-fraction column, code->label column).
_LEVELS = {
    "REALM": ("dominant_realm", "realm_area_frac", None),
    "BIOME_NUM": ("dominant_biome_num", "biome_area_frac", "BIOME_NAME"),
    "ECO_ID": ("dominant_eco_id", "eco_area_frac", "ECO_NAME"),
}

_OUTPUT_COLUMNS = [
    "dominant_realm", "realm_area_frac",
    "dominant_biome_num", "dominant_biome_name", "biome_area_frac",
    "dominant_eco_id", "dominant_eco_name", "eco_area_frac",
    "n_ecoregions_intersecting",
]


def _dominant_per_group(fragments: pd.DataFrame, gid_col: str, level_col: str) -> pd.DataFrame:
    """Area-argmax of *level_col* per *gid_col* group. Ties (exact equal area)
    are broken by the lowest *level_col* value -- arbitrary but deterministic,
    so results are reproducible across runs/environments."""
    per_class = fragments.groupby([gid_col, level_col], as_index=False)["_area"].sum()
    per_class = per_class.sort_values([gid_col, "_area", level_col], ascending=[True, False, True])
    return per_class.groupby(gid_col).first()


def compute_dominant_classes(
    gid_gdf: gpd.GeoDataFrame,
    class_gdf: gpd.GeoDataFrame,
    *,
    gid_col: str,
    crs,
    code_to_id: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """One row per *gid_col* unit in *gid_gdf* (e.g. GADM's level-3 admin
    polygons): the area-weighted dominant `REALM`/`BIOME_NUM`/`ECO_ID` from
    *class_gdf* (RESOLVE ecoregions), each with the dominant class's share of
    that unit's total intersected area, plus `n_ecoregions_intersecting` as a
    heterogeneity/confidence flag.

    Both inputs are reprojected to *crs* (an equal-area CRS -- callers pass
    the pipeline's own canonical `geobox.crs`) before computing area, so area
    fractions reflect true ground area, not degrees^2.

    If *code_to_id* is given (e.g. GADM's own `{native_code: int_id}` mapping,
    `gadm.gid_mapping_path()`), the output carries both the raw string
    `{gid_col}_code` and an integer `gid_col` translated through it -- rows
    whose code has no entry in the mapping are dropped, mirroring
    `country_classifications.py`'s own `GID_0` translation.
    """
    gid = gid_gdf[[gid_col, "geometry"]].to_crs(crs)
    classes = class_gdf[["REALM", "BIOME_NUM", "BIOME_NAME", "ECO_ID", "ECO_NAME", "geometry"]].to_crs(crs)

    fragments = gpd.overlay(gid, classes, how="intersection", keep_geom_type=False)
    fragments["_area"] = fragments.geometry.area
    fragments = fragments[fragments["_area"] > 0].drop(columns="geometry")

    if fragments.empty:
        columns = [gid_col, f"{gid_col}_code"] if code_to_id is not None else [gid_col]
        return pd.DataFrame(columns=columns + _OUTPUT_COLUMNS)

    total_area = fragments.groupby(gid_col)["_area"].sum()
    result = pd.DataFrame(index=total_area.index)

    # Dedupe on the *code* only, not the (code, label) pair: RESOLVE carries a
    # handful of codes (e.g. BIOME_NUM 98/99, some ECO_IDs) with two spellings
    # of the same name across features. A full-row drop_duplicates() keeps both,
    # making `.set_index(level_col)` non-unique and `Series.map()` below raise
    # `InvalidIndexError`. Sort first so the retained label is deterministic
    # (lexicographically first), matching this module's tie-break philosophy.
    label_lookups = {
        level_col: (
            classes[[level_col, label_col]]
            .dropna()
            .sort_values([level_col, label_col])
            .drop_duplicates(subset=[level_col], keep="first")
            .set_index(level_col)[label_col]
        )
        for level_col, (_, _, label_col) in _LEVELS.items()
        if label_col is not None
    }

    for level_col, (dominant_col, frac_col, label_col) in _LEVELS.items():
        dominant = _dominant_per_group(fragments, gid_col, level_col)
        result[dominant_col] = dominant[level_col]
        result[frac_col] = dominant["_area"] / total_area
        if label_col is not None:
            label_out_col = "dominant_biome_name" if level_col == "BIOME_NUM" else "dominant_eco_name"
            result[label_out_col] = result[dominant_col].map(label_lookups[level_col])

    result["n_ecoregions_intersecting"] = fragments.groupby(gid_col)["ECO_ID"].nunique()
    result = result[_OUTPUT_COLUMNS]
    result.index.name = gid_col
    result = result.reset_index()

    if code_to_id is not None:
        result = result.rename(columns={gid_col: f"{gid_col}_code"})
        result[gid_col] = result[f"{gid_col}_code"].map(lambda c: code_to_id.get(c, 0))
        result = result[result[gid_col] != 0]
        result = result[[gid_col, f"{gid_col}_code"] + _OUTPUT_COLUMNS]

    return result.reset_index(drop=True)
