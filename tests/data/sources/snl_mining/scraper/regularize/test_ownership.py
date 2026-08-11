from __future__ import annotations

import datetime

from src.data.sources.snl_mining.scraper.parsing.xls import StructuredBlock
from src.data.sources.snl_mining.scraper.regularize.subsections import ownership
from _builders import cell, table_block, workbook


def _former_owners_block() -> StructuredBlock:
    # 4-column mini-table encoded as 2 label+2 value cells per row (real
    # Ownership shape -- see plan doc §8): (Former Direct Owner, Type,
    # Equity Ownership (%), Date), header row included as real data.
    cells = [
        cell("a13", 0, 13, "Former Direct Owner", "label"),
        cell("b13", 3, 13, "Type", "value"),
        cell("c13", 6, 13, "Equity Ownership (%)", "label"),
        cell("d13", 9, 13, "Date", "value"),
        cell("a14", 0, 14, "Acme Mining Ltd.", "label"),
        cell("b14", 3, 14, "Optionor", "value"),
        cell("c14", 6, 14, "20", "label"),
        cell("d14", 9, 14, "39275", "value"),
    ]
    return StructuredBlock(
        sheet_name="Details", block_index=2, block_type="key_value", block_title="Former Direct Owner(s)",
        row_start=13, row_end=14, header_row_count=0, cells=tuple(cells),
    )


def test_ownership_current_and_former_owners():
    current_block = table_block(
        headers=["Owner", "Type", "Equity Ownership (%)", "Controlling Ownership (%)", "Corporate Headquarters"],
        rows=[["Beta Resources Corp.", "Venturer", "50", "50", "CO, USA"]],
        sheet_name="Details",
        block_title="Equity & Controlling Owner(s)/Operator",
    )
    wb = workbook(blocks=[current_block, _former_owners_block()])

    tables = ownership.regularize(wb, "m1")

    assert tables["detail_ownership__current"] == [
        {
            "owner": "Beta Resources Corp.",
            "owner_type": "Venturer",
            "equity_pct": 50.0,
            "controlling_pct": 50.0,
            "headquarters": "CO, USA",
        }
    ]
    assert tables["detail_ownership__former"] == [
        {
            "former_owner": "Acme Mining Ltd.",
            "owner_role": "Optionor",
            "equity_pct": 20.0,
            "transaction_date": datetime.date(2007, 7, 12),
        }
    ]


def test_ownership_historical_tables_melt_wide_year_columns():
    equity_block = table_block(
        headers=["Equity Ownership (%), Headquarters", "2016 Y", "2017 Y"],
        rows=[["Beta Resources Corp. , USA", "50", "NA"]],
        sheet_name="Historical",
    )
    control_block = table_block(
        headers=["Controlling Ownership (%), Headquarters", "2016 Y", "2017 Y"],
        rows=[["Beta Resources Corp. , USA", "100", "50"]],
        sheet_name="Historical",
        block_index=2,
    )
    wb = workbook(blocks=[equity_block, control_block])

    tables = ownership.regularize(wb, "m1")

    assert tables["detail_ownership__historical_equity"] == [
        {"owner": "Beta Resources Corp.", "headquarters": "USA", "year": 2016, "pct": 50.0}
    ]
    assert tables["detail_ownership__historical_control"] == [
        {"owner": "Beta Resources Corp.", "headquarters": "USA", "year": 2016, "pct": 100.0},
        {"owner": "Beta Resources Corp.", "headquarters": "USA", "year": 2017, "pct": 50.0},
    ]


def test_ownership_missing_sheets_yield_empty_tables():
    wb = workbook(blocks=[table_block(headers=["X"], rows=[["1"]], sheet_name="Details")])
    tables = ownership.regularize(wb, "m1")
    assert tables["detail_ownership__royalty"] == []
    assert tables["detail_ownership__historical_equity"] == []
