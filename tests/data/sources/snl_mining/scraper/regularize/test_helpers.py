from __future__ import annotations

import datetime

from src.data.sources.snl_mining.scraper.regularize import helpers
from _builders import key_value_block, table_block, text_block, workbook


def test_unescape_cell_text_replaces_x000d_artifact():
    assert helpers.unescape_cell_text("1976: began exploration._x000d_") == "1976: began exploration.\n"


def test_normalize_text_drops_na_placeholder():
    assert helpers.normalize_text("NA") is None
    assert helpers.normalize_text("  ") is None
    assert helpers.normalize_text(None) is None
    assert helpers.normalize_text(" Zinc ") == "Zinc"


def test_parse_number_handles_thousands_comma_and_accounting_parens():
    assert helpers.parse_number("106,186") == 106186.0
    assert helpers.parse_number("(1.05)") == -1.05
    assert helpers.parse_number("NA") is None
    assert helpers.parse_number("50%") == 50.0


def test_parse_number_does_not_misparse_us_thousands_as_european_decimal():
    # Regression guard: the unrelated manual-xls notebook's number parser
    # would corrupt "106,186" into "106.186" under a European decimal-comma
    # assumption -- this scraper's exports use plain US formatting (verified
    # against real data), so that heuristic must not be reused here.
    assert helpers.parse_number("1,234.56") == 1234.56


def test_parse_leading_number_strips_trailing_unit():
    assert helpers.parse_leading_number("0.630 ($/lb Zn)") == 0.630


def test_split_value_unit():
    assert helpers.split_value_unit("5,500 (tonnes/day)") == (5500.0, "tonnes/day")
    assert helpers.split_value_unit("garbage") == (None, None)


def test_parse_excel_serial_date_decodes_known_value():
    # 44229 -> 2021-02-02, verified against a real Financings export's
    # Announce Date column.
    assert helpers.parse_excel_serial_date("44229") == datetime.date(2021, 2, 2)


def test_parse_excel_serial_date_rejects_implausible_year():
    # A stray value that decodes far outside the shared plausible-year
    # range must not be accepted as a real date.
    assert helpers.parse_excel_serial_date("999999") is None


def test_parse_date_text_parses_us_slash_format():
    assert helpers.parse_date_text("8/27/2015") == datetime.date(2015, 8, 27)


def test_key_value_block_to_pairs_orders_by_row():
    block = key_value_block(pairs=[("Mining Method", "Open Pit"), ("Product Form", "Concentrate")])
    assert helpers.key_value_block_to_pairs(block) == [
        ("Mining Method", "Open Pit"),
        ("Product Form", "Concentrate"),
    ]


def test_table_block_to_rows_keys_by_header():
    block = table_block(headers=["Owner", "Equity (%)"], rows=[["Acme Corp.", "50"], ["Beta Inc.", "50"]])
    rows = helpers.table_block_to_rows(block)
    assert rows == [
        {"Owner": "Acme Corp.", "Equity (%)": "50"},
        {"Owner": "Beta Inc.", "Equity (%)": "50"},
    ]


def test_table_sections_from_first_header_row_splits_on_context_divider():
    # Mirrors Subcontractors' "Current"/"Past Subcontractors" shape: a
    # context-role single-cell row starts a new section, whose own first
    # row becomes that section's header.
    from src.data.sources.snl_mining.scraper.parsing.xls import StructuredBlock

    from _builders import cell

    cells = [
        cell("a1", 0, 1, "Company", "header"),
        cell("b1", 1, 1, "Role", "header"),
        cell("a2", 0, 2, "Acme Drilling", "data"),
        cell("b2", 1, 2, "Exploration", "data"),
        cell("a3", 0, 3, "Past Subcontractors", "context"),
        cell("a4", 0, 4, "Company", "data"),
        cell("b4", 1, 4, "Role", "data"),
        cell("a5", 0, 5, "Beta Surveys", "data"),
        cell("b5", 1, 5, "Geophysics", "data"),
    ]
    block = StructuredBlock(
        sheet_name="Sheet1", block_index=1, block_type="table", block_title=None,
        row_start=1, row_end=5, header_row_count=1, cells=tuple(cells),
    )
    sections = helpers.table_sections_from_first_header_row(block)
    assert sections == [
        (None, [{"Company": "Acme Drilling", "Role": "Exploration"}]),
        ("Past Subcontractors", [{"Company": "Beta Surveys", "Role": "Geophysics"}]),
    ]


def test_key_value_rows_groups_by_column_position():
    from _builders import cell
    from src.data.sources.snl_mining.scraper.parsing.xls import StructuredBlock

    cells = [
        cell("a1", 0, 1, "Former Direct Owner", "label"),
        cell("b1", 3, 1, "Type", "value"),
        cell("c1", 6, 1, "Equity Ownership (%)", "label"),
        cell("d1", 9, 1, "Date", "value"),
        cell("a2", 0, 2, "Acme Mining Ltd.", "label"),
        cell("b2", 3, 2, "Optionor", "value"),
        cell("c2", 6, 2, "20", "label"),
        cell("d2", 9, 2, "39275", "value"),
    ]
    block = StructuredBlock(
        sheet_name="Sheet1", block_index=1, block_type="key_value", block_title=None,
        row_start=1, row_end=2, header_row_count=0, cells=tuple(cells),
    )
    rows = helpers.key_value_rows(block)
    assert rows == [
        ["Former Direct Owner", "Type", "Equity Ownership (%)", "Date"],
        ["Acme Mining Ltd.", "Optionor", "20", "39275"],
    ]


def test_all_blocks_and_blocks_in_sheet():
    block_a = table_block(headers=["X"], rows=[["1"]], sheet_name="Details")
    block_b = text_block(lines=["hello"], sheet_name="Historical")
    wb = workbook(blocks=[block_a, block_b])
    assert helpers.all_blocks(wb) == [block_a, block_b]
    assert helpers.blocks_in_sheet(wb, "historical") == [block_b]
    assert helpers.blocks_in_sheet(wb, "nonexistent") == []
