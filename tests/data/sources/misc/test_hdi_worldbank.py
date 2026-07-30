"""Unit tests for the pure HDI/World Bank parsing functions -- independently
testable now that they're split out of MiscPreprocessor's monolith
(docs/design/09-integrated-pipeline.md §7's escape-hatch motivation: neither
is testable in isolation today)."""

import pandas as pd
import pytest

from src.data.sources.misc.hdi import read_hdi
from src.data.sources.misc.worldbank import read_worldbank


@pytest.fixture
def hdi_csv(tmp_path):
    # HDI thresholds: Low <0.550, Medium <0.700, High <0.800, else Very High.
    # Country A: 0.500 (Low) throughout. Country B: 0.750 (High) throughout.
    cols = {"iso3": ["AAA", "BBB"]}
    for year in range(1990, 2024):
        cols[f"hdi_{year}"] = [0.500, 0.750]
    path = tmp_path / "hdr.csv"
    pd.DataFrame(cols).to_csv(path, index=False, encoding="latin1")
    return str(path)


def test_read_hdi_classifies_low_and_high(hdi_csv):
    wide = read_hdi(hdi_csv)
    row_a = wide[wide.iso3 == "AAA"].iloc[0]
    row_b = wide[wide.iso3 == "BBB"].iloc[0]
    for year in (1991, 1999, 2011):
        assert row_a[f"HDI_LO_{year}"] is True or row_a[f"HDI_LO_{year}"] == True  # noqa: E712
        assert row_a[f"HDI_ME_{year}"] == False  # noqa: E712
        assert row_b[f"HDI_HI_{year}"] == True  # noqa: E712
        assert row_b[f"HDI_LO_{year}"] == False  # noqa: E712


def test_read_hdi_threshold_boundaries(tmp_path):
    # Exactly at each boundary: 0.550 -> Medium (not Low, since Low is `< 0.550`).
    cols = {"iso3": ["X"]}
    for year in range(1990, 2024):
        cols[f"hdi_{year}"] = [0.550]
    path = tmp_path / "hdr.csv"
    pd.DataFrame(cols).to_csv(path, index=False, encoding="latin1")

    wide = read_hdi(str(path))
    row = wide.iloc[0]
    assert row["HDI_ME_1991"] == True  # noqa: E712
    assert row["HDI_LO_1991"] == False  # noqa: E712


@pytest.fixture
def worldbank_xlsx(tmp_path):
    # Mirrors the real sheet shape: 4 header rows, then a header row at row 5
    # (header=4), country code in col 0, country name in col 1, year columns
    # from col 2 onward as two-digit suffixes ("...91" style), 6 leading data
    # rows and 2 trailing rows sliced off by iloc[6:-2].
    header_row = ["Code", "Country", "FY91", "FY99", "FY11"]
    rows = [["pad"] * 5 for _ in range(4)]  # 4 blank header rows before header=4
    rows.append(header_row)
    # 6 leading rows sliced off (iloc[6:-2] on the melted frame's row axis --
    # applied to the *data* rows after the header, so pad with 6 dummy rows).
    for _ in range(6):
        rows.append(["ZZZ", "Padding", "..", "..", ".."])
    rows.append(["AAA", "Country A", "L", "L", "LM"])
    rows.append(["BBB", "Country B", "H", "H", "H"])
    rows.append(["trailer1", "", "", "", ""])
    rows.append(["trailer2", "", "", "", ""])

    path = tmp_path / "wb.xlsx"
    pd.DataFrame(rows).to_excel(path, sheet_name="Country Analytical History", header=False, index=False)
    return str(path)


def test_read_worldbank_classifies_income_groups(worldbank_xlsx):
    wide = read_worldbank(worldbank_xlsx)
    row_a = wide[wide.iso3 == "AAA"].iloc[0]
    row_b = wide[wide.iso3 == "BBB"].iloc[0]
    assert row_a["WB_LO_1991"] == True  # noqa: E712
    assert row_a["WB_LM_2011"] == True  # noqa: E712 (FY11 = "LM")
    assert row_b["WB_HI_1991"] == True  # noqa: E712
