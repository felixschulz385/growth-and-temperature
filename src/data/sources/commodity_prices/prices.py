"""World Bank Pink Sheet commodity prices -- a pure parsing function,
independently unit-testable (mirrors `src/data/sources/misc/worldbank.py`'s
`read_worldbank`/`src/data/sources/misc/hdi.py`'s `read_hdi` split).

Reads the "Annual Prices (Real)" sheet of the WB Commodity Markets ("Pink
Sheet") CMO Historical Data workbook -- already published in constant 2010
US dollars, so no separate CPI-deflation step is needed here (confirmed by
inspecting the sheet's own title cell: "annual prices, 1960 to present, real
2010 US dollars"). The 2010 base year differs from Berman et al.'s 2005, but
this is immaterial for a `share * ln(price)` term: a uniform rebasing only
shifts every commodity's series by a commodity-specific, time-invariant
constant, absorbed by any downstream fixed effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.sources.commodities import normalize_commodity

#: 0-indexed row of the column-header row in "Annual Prices (Real)"
#: (row 7 in a 1-indexed spreadsheet view) -- `pandas.read_excel(header=...)`
#: takes the 0-indexed row number of the header.
HEADER_ROW = 6

SHEET_NAME = "Annual Prices (Real)"


def read_and_normalize_prices(path: str) -> pd.DataFrame:
    """Read the WB Pink Sheet real-price annual sheet and return one row per
    (commodity, year), restricted to commodities with a canonical mapping
    (`commodities.normalize_commodity(..., source="worldbank")`).

    Columns: `commodity` (canonical key), `year` (int), `price_real` (float),
    `ln_price_real` (float, `log(price_real)`).
    """
    raw = pd.read_excel(path, sheet_name=SHEET_NAME, header=HEADER_ROW)
    raw = raw.rename(columns={raw.columns[0]: "year"})

    # Trailing rows after the last data year are blank across every column
    # (confirmed: the sheet has ~11 such rows after 2025) -- drop any row
    # whose "year" cell isn't a plain year value rather than relying on a
    # fixed row count, which would silently break on the sheet's next
    # monthly republish if the trailer grows/shrinks.
    raw["year"] = pd.to_numeric(raw["year"], errors="coerce")
    raw = raw.dropna(subset=["year"])
    raw["year"] = raw["year"].astype(int)

    long = raw.melt(id_vars=["year"], var_name="wb_column", value_name="price_real")
    long["commodity"] = long["wb_column"].map(lambda c: normalize_commodity(str(c), source="worldbank"))
    long = long.dropna(subset=["commodity"])

    # Missing/not-yet-started series are coded "…" in this workbook rather
    # than a blank cell -- coerce to numeric so those (and any other
    # non-numeric cell) become NaN and get dropped below, instead of
    # silently propagating a string into `np.log`.
    long["price_real"] = pd.to_numeric(long["price_real"], errors="coerce")
    long = long.dropna(subset=["price_real"])
    long = long[long["price_real"] > 0]

    long["ln_price_real"] = np.log(long["price_real"].to_numpy())

    result = long[["commodity", "year", "price_real", "ln_price_real"]].sort_values(["commodity", "year"])
    return result.reset_index(drop=True)
