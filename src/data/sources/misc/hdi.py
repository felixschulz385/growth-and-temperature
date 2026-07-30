"""UNDP Human Development Index classification -- a pure parsing function,
independently unit-testable (docs/design/09-integrated-pipeline.md §7,
country_classifications' escape hatch to a future 4-way split).

Ported from `src/data/preprocess/sources/misc.py::MiscPreprocessor.
_process_country_classifications_target`'s HDI branch, unchanged in
behaviour: classifies each country-year into UNDP's four HDI bands, snapshots
at panel years 1991/1999/2011 (last observation at-or-before each year), and
returns one row per iso3 with boolean HDI_{LO,ME,HI,VH}_{year} columns.
"""

from __future__ import annotations

import pandas as pd

PANEL_YEARS = (1991, 1999, 2011)


def _assign_group(x) -> str | None:
    if pd.isna(x):
        return None
    if x < 0.550:
        return "Low"
    if x < 0.700:
        return "Medium"
    if x < 0.800:
        return "High"
    return "Very High"


def read_hdi(path: str) -> pd.DataFrame:
    """Read UNDP's HDR composite-indices CSV and return one row per `iso3`
    with boolean `HDI_LO_{year}`/`HDI_ME_{year}`/`HDI_HI_{year}`/`HDI_VH_{year}`
    columns for each of `PANEL_YEARS`."""
    hdi_cols = [f"hdi_{y}" for y in range(1990, 2024)]
    hdi = pd.read_csv(path, encoding="latin1", usecols=["iso3"] + hdi_cols)

    hdi = hdi.melt(id_vars=["iso3"], var_name="year", value_name="hdi")
    # Plain bracket assignment (not `.loc[:, "year"] =`) -- pandas >=3.0's
    # default `str` dtype for melt()-derived columns rejects an in-place
    # int-dtype cast via `.loc`; bracket assignment replaces the column
    # instead of casting in place, avoiding a `TypeError` the old code's
    # `.loc[:, "year"] = ...` pattern would hit under pandas >=3.0 (this
    # repo pins no pandas version -- docs/design/05-migration.md §3).
    hdi["year"] = hdi["year"].str[4:].astype(int)
    hdi["hdi_group"] = hdi["hdi"].apply(_assign_group)
    hdi.sort_values(["iso3", "year"], inplace=True)

    year_indicators = {}
    for panel_year in PANEL_YEARS:
        year_indicators[panel_year] = hdi.query(f"year <= {panel_year}").groupby("iso3").last()
    hdi = pd.concat(year_indicators).reset_index(names=["indicator_year", "iso3"]).drop(columns=["year"])

    hdi["HDI_LO"] = hdi.hdi_group.isin(["Low"])
    hdi["HDI_ME"] = hdi.hdi_group.isin(["Medium"])
    hdi["HDI_HI"] = hdi.hdi_group.isin(["High"])
    hdi["HDI_VH"] = hdi.hdi_group.isin(["Very High"])
    hdi = hdi[["iso3", "indicator_year", "HDI_LO", "HDI_ME", "HDI_HI", "HDI_VH"]]

    hdi_wide = hdi[["iso3"]].drop_duplicates().reset_index(drop=True)
    for year in PANEL_YEARS:
        year_data = hdi[hdi["indicator_year"] == year].set_index("iso3")
        for col in ("HDI_LO", "HDI_ME", "HDI_HI", "HDI_VH"):
            hdi_wide[f"{col}_{year}"] = (
                hdi_wide["iso3"].map(year_data[col].to_dict()).fillna(False).infer_objects(copy=False).astype(bool)
            )
    return hdi_wide
