"""`Cash Flow Analysis`, `Modeled Ore Costs`, `Modeled Product Costs`,
`Modeled Production`, `Modeled ROM Costs` -- five subsections sharing one
real shape: a "wide financial statement" table (year columns, e.g. `2024Y*,
2025Y, ..., 2028Y`), with a `Period Ended` row giving each year column's
Excel-serial period-end date, section-header rows (`context`-role single
cells like `"Revenue, Costs and Cash Flow ($M)"`) grouping the line items
below them, and the actual line-item values. Melted to one row per
`(section, line_item, year_label)`.

`Cost Curve` shares this module only nominally: its export is deliberately
disabled scraper-side (see `registry.py`), so `regularize_cost_curve` never
actually runs against real content -- the content-validation gate always
routes it to `content_mismatch` first.
"""

from __future__ import annotations

from ...parsing.xls import ParsedWorkbook
from ..helpers import all_blocks, find_block, normalize_text, parse_excel_serial_date, parse_number

CASH_FLOW_ANALYSIS_TABLE = "detail_cash_flow_analysis"
MODELED_ORE_COSTS_TABLE = "detail_modeled_ore_costs"
MODELED_PRODUCT_COSTS_TABLE = "detail_modeled_product_costs"
MODELED_PRODUCTION_TABLE = "detail_modeled_production"
MODELED_ROM_COSTS_TABLE = "detail_modeled_rom_costs"

_PERIOD_ENDED_LABEL = "Period Ended"


def _melt_financial_table(block) -> list[dict]:
    header_by_column = {cell.column_index: cell.value for cell in block.cells if cell.role == "header"}

    rows_by_number: dict[int, list] = {}
    for cell in block.cells:
        rows_by_number.setdefault(cell.row_number, []).append(cell)

    period_end_by_column: dict[int, object] = {}
    section: str | None = None
    melted: list[dict] = []

    for row_number in sorted(rows_by_number):
        row_cells = rows_by_number[row_number]
        if len(row_cells) == 1 and row_cells[0].role == "context":
            section = normalize_text(row_cells[0].value)
            continue

        data_cells = [c for c in row_cells if c.role == "data"]
        label_cell = next((c for c in data_cells if c.column_index == 0), None)
        if label_cell is None:
            continue
        label = normalize_text(label_cell.value)
        value_cells = [c for c in data_cells if c.column_index != 0]

        if label == _PERIOD_ENDED_LABEL:
            for cell in value_cells:
                period_end_by_column[cell.column_index] = parse_excel_serial_date(cell.value)
            continue

        for cell in value_cells:
            melted.append(
                {
                    "section": section,
                    "line_item": label,
                    "year_label": normalize_text(header_by_column.get(cell.column_index)),
                    "period_end_date": period_end_by_column.get(cell.column_index),
                    "value": parse_number(cell.value),
                }
            )
    return melted


def _regularize_financial_table(workbook: ParsedWorkbook, table_name: str) -> dict[str, list[dict]]:
    block = find_block(all_blocks(workbook), block_type="table")
    if block is None:
        return {table_name: []}
    return {table_name: _melt_financial_table(block)}


def regularize_cash_flow_analysis(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    return _regularize_financial_table(workbook, CASH_FLOW_ANALYSIS_TABLE)


def regularize_modeled_ore_costs(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    return _regularize_financial_table(workbook, MODELED_ORE_COSTS_TABLE)


def regularize_modeled_product_costs(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    return _regularize_financial_table(workbook, MODELED_PRODUCT_COSTS_TABLE)


def regularize_modeled_production(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    return _regularize_financial_table(workbook, MODELED_PRODUCTION_TABLE)


def regularize_modeled_rom_costs(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    return _regularize_financial_table(workbook, MODELED_ROM_COSTS_TABLE)


def regularize_cost_curve(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    return {}
