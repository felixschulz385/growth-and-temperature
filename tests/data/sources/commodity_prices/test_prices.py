"""Unit tests for the pure WB Pink Sheet parsing function
(mirrors tests/data/sources/misc/test_hdi_worldbank.py's synthetic-fixture-xlsx
pattern)."""

import math

import pandas as pd
import pytest

from src.data.sources.commodity_prices.prices import HEADER_ROW, SHEET_NAME, read_and_normalize_prices


@pytest.fixture
def prices_xlsx(tmp_path):
    # Mirrors the real "Annual Prices (Real)" sheet shape: 6 padding rows
    # (title/description/updated/blank/blank), header row at row 6
    # (HEADER_ROW), then data rows, then trailing blank rows.
    rows = [[None] * 4 for _ in range(HEADER_ROW)]
    rows.append(["", "Gold", "Copper", "Cocoa"])  # Cocoa: no canonical mapping
    rows.append([2019, 1500.0, 6000.0, 2400.0])
    rows.append([2020, "…", 6200.0, 2500.0])  # WB's own missing-data sentinel
    rows.append([2021, 1750.0, 6400.0, 2600.0])
    rows.append([None, None, None, None])  # trailing blank row
    rows.append([None, None, None, None])

    path = tmp_path / "prices.xlsx"
    pd.DataFrame(rows).to_excel(path, sheet_name=SHEET_NAME, header=False, index=False)
    return str(path)


def test_read_and_normalize_prices_maps_known_columns(prices_xlsx):
    df = read_and_normalize_prices(prices_xlsx)
    gold_2019 = df[(df.commodity == "gold") & (df.year == 2019)].iloc[0]
    assert gold_2019["price_real"] == 1500.0
    assert math.isclose(gold_2019["ln_price_real"], math.log(1500.0))


def test_read_and_normalize_prices_drops_unmapped_columns(prices_xlsx):
    df = read_and_normalize_prices(prices_xlsx)
    assert "cocoa" not in set(df["commodity"])
    assert set(df["commodity"]) == {"gold", "copper"}


def test_read_and_normalize_prices_drops_missing_data_sentinel(prices_xlsx):
    # WB codes missing/not-yet-started series as "…", not a blank cell.
    df = read_and_normalize_prices(prices_xlsx)
    assert df[(df.commodity == "gold") & (df.year == 2020)].empty
    # Copper's 2020 value is a real number and must survive.
    assert not df[(df.commodity == "copper") & (df.year == 2020)].empty


def test_read_and_normalize_prices_drops_trailing_blank_rows(prices_xlsx):
    df = read_and_normalize_prices(prices_xlsx)
    assert set(df["year"]) == {2019, 2020, 2021}
