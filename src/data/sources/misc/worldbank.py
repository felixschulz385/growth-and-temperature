"""World Bank income-group classification -- a pure parsing function,
independently unit-testable (docs/design/09-integrated-pipeline.md §7).

Ported from `src/data/preprocess/sources/misc.py::MiscPreprocessor.
_process_country_classifications_target`'s World Bank branch, unchanged in
behaviour: snapshots the "Country Analytical History" sheet at panel years
1991/1999/2011 (last observation at-or-before each year), and returns one row
per iso3 with boolean WB_{LO,LM,UM,HI}_{year} columns.
"""

from __future__ import annotations

import pandas as pd

PANEL_YEARS = (1991, 1999, 2011)


def read_worldbank(path: str) -> pd.DataFrame:
    """Read the World Bank "World By Income" xlsx and return one row per
    `iso3` with boolean `WB_LO_{year}`/`WB_LM_{year}`/`WB_UM_{year}`/
    `WB_HI_{year}` columns for each of `PANEL_YEARS`."""
    wb = pd.read_excel(path, sheet_name="Country Analytical History", header=4)
    wb.rename({wb.columns[0]: "iso3"}, axis=1, inplace=True)
    wb.drop(columns=[wb.columns[1]], inplace=True)
    wb = wb.iloc[6:-2, :]

    wb = wb.melt(id_vars=["iso3"], var_name="year", value_name="classification")
    # Two-digit year -> four-digit: ">50" is 19xx, else 20xx (the same rule
    # ported verbatim from the old code -- not re-derived here). Bracket
    # assignment, not `.loc[:, "year"] =` -- see hdi.py's identical fix.
    wb["year"] = wb.year.str[2:].apply(lambda x: int("19" + x) if int(x) > 50 else int("20" + x))
    wb = wb.query("classification!='..'")
    wb.sort_values(["iso3", "year"], inplace=True)

    year_indicators = {}
    for panel_year in PANEL_YEARS:
        year_indicators[panel_year] = wb.query(f"year <= {panel_year}").groupby("iso3").last()
    wb = pd.concat(year_indicators).reset_index(names=["indicator_year", "iso3"]).drop(columns=["year"])

    wb["WB_LO"] = wb.classification.isin(["L"])
    wb["WB_LM"] = wb.classification.isin(["LM", "LM*"])
    wb["WB_UM"] = wb.classification.isin(["UM"])
    wb["WB_HI"] = wb.classification.isin(["H"])
    wb = wb[["iso3", "indicator_year", "WB_LO", "WB_LM", "WB_UM", "WB_HI"]]

    wb_wide = wb[["iso3"]].drop_duplicates().reset_index(drop=True)
    for year in PANEL_YEARS:
        year_data = wb[wb["indicator_year"] == year].set_index("iso3")
        for col in ("WB_LO", "WB_LM", "WB_UM", "WB_HI"):
            wb_wide[f"{col}_{year}"] = (
                wb_wide["iso3"].map(year_data[col].to_dict()).fillna(False).infer_objects(copy=False).astype(bool)
            )
    return wb_wide
