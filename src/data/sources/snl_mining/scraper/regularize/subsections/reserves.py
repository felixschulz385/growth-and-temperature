"""`Reserves & Resources` and `Reserves / Resources & Production Chart`.

`Reserves & Resources` has the hardest header shape found: a genuine 2-row
*spanning* header (a top row naming a commodity across several columns,
e.g. `Zinc` at column 2, plus a sub-header row giving each of those columns'
metric, e.g. `Grade (%)` / `Contained (tonnes)`), repeated across up to 4
blocks (Total / Measured & Indicated excl. reserves / M&I incl. reserves /
Inferred), each block itself covering one or more as-of-date snapshots
(a single-cell `context` row per snapshot). Melted to
`(category, as_of_date, commodity, metric, value)`.

`Reserves / Resources & Production Chart` is much simpler: a `Data` sheet
with one flat `Period, Reserves (tonnes), Resources (tonnes)` table -- the
commodity it's *for* lives in the `Chart` sheet's preamble text
("Commodities Zinc"), not in the table itself.
"""

from __future__ import annotations

from ...parsing.xls import ParsedWorkbook
from ..helpers import all_blocks, blocks_in_sheet, find_block, find_blocks, normalize_text, parse_date_text, parse_number, table_block_to_rows

RESERVES_RESOURCES_TABLE = "detail_reserves_resources"
PRODUCTION_CHART_TABLE = "detail_reserves_resources_production_chart"


def _top_header_for_column(top_headers: dict[int, str], column: int) -> str | None:
    candidates = [col for col in top_headers if col <= column]
    if not candidates:
        return None
    return top_headers[max(candidates)]


def _melt_reserves_block(block) -> list[dict]:
    rows_by_number: dict[int, list] = {}
    for cell in block.cells:
        rows_by_number.setdefault(cell.row_number, []).append(cell)
    row_numbers = sorted(rows_by_number)
    if len(row_numbers) < 2:
        return []

    top_headers = {c.column_index: c.value for c in rows_by_number[row_numbers[0]]}
    sub_headers = {c.column_index: c.value for c in rows_by_number[row_numbers[1]]}

    as_of_date = None
    melted: list[dict] = []
    for row_number in row_numbers[2:]:
        cells = rows_by_number[row_number]
        if len(cells) == 1 and cells[0].role == "context":
            as_of_date = parse_date_text(cells[0].value)
            continue
        label_cell = next((c for c in cells if c.column_index == 0), None)
        if label_cell is None:
            continue
        category = normalize_text(label_cell.value)
        for cell in cells:
            if cell.column_index == 0:
                continue
            top_label = _top_header_for_column(top_headers, cell.column_index)
            sub_label = sub_headers.get(cell.column_index)
            if sub_label:
                commodity, metric = top_label, sub_label
            else:
                commodity, metric = None, top_label
            value = parse_number(cell.value)
            if value is None:
                continue
            melted.append(
                {
                    "category": category,
                    "as_of_date": as_of_date,
                    "commodity": normalize_text(commodity),
                    "metric": normalize_text(metric),
                    "value": value,
                }
            )
    return melted


def regularize_reserves_resources(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    rows: list[dict] = []
    for block in find_blocks(all_blocks(workbook), block_type="table"):
        rows.extend(_melt_reserves_block(block))
    return {RESERVES_RESOURCES_TABLE: rows}


def regularize_reserves_resources_production_chart(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    commodity = None
    for block in blocks_in_sheet(workbook, "Chart"):
        for cell in block.cells:
            if cell.value and cell.value.startswith("Commodities "):
                commodity = normalize_text(cell.value.removeprefix("Commodities "))

    block = find_block(blocks_in_sheet(workbook, "Data"), block_type="table")
    rows = []
    if block is not None:
        for raw in table_block_to_rows(block):
            rows.append(
                {
                    "commodity": commodity,
                    "period": normalize_text(raw.get("Period")),
                    "reserves_tonnes": parse_number(raw.get("Reserves (tonnes)")),
                    "resources_tonnes": parse_number(raw.get("Resources (tonnes)")),
                }
            )
    return {PRODUCTION_CHART_TABLE: rows}
