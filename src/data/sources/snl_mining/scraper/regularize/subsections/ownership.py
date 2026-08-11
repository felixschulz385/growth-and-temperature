"""`Ownership` / `Ownership Structure` -- both sidebar links land on the same
underlying 3-sheet report (Details / Historical / Royalty), confirmed both
by real title inspection and by the aggregate content-label crosstab in the
plan doc §1. Produces four regularized tables from one workbook.

Real shapes (see plan doc §3 and §8):
- `Details` sheet: a `table` block ("Equity & Controlling Owner(s)/Operator")
  and a `key_value` block ("Former Direct Owner(s)") that's actually a
  4-column mini-table (Owner, Type, Equity %, Date) encoded as 2 label+2
  value cells per row -- `key_value_rows` unpacks it positionally.
- `Historical` sheet: two wide year-columned tables (equity then control
  ownership), row label combines "Owner , Headquarters" -- melted to long
  `(owner, headquarters, year, pct)`.
- `Royalty` sheet: another 4-column mini-table (Holder, Type, Royalty %, HQ).
"""

from __future__ import annotations

import re

from ...parsing.xls import ParsedWorkbook
from ..helpers import blocks_in_sheet, find_block, find_blocks, key_value_rows, normalize_text, parse_excel_serial_date, parse_number, table_block_to_rows

CURRENT_TABLE = "detail_ownership__current"
FORMER_TABLE = "detail_ownership__former"
HISTORICAL_EQUITY_TABLE = "detail_ownership__historical_equity"
HISTORICAL_CONTROL_TABLE = "detail_ownership__historical_control"
ROYALTY_TABLE = "detail_ownership__royalty"

_YEAR_COLUMN_RE = re.compile(r"^(\d{4})\s*Y$")


def _split_owner_headquarters(text: str | None) -> tuple[str | None, str | None]:
    normalized = normalize_text(text)
    if normalized is None:
        return None, None
    if "," not in normalized:
        return normalized, None
    owner, _, hq = normalized.rpartition(",")
    return owner.strip() or None, hq.strip() or None


def _melt_historical(block) -> list[dict]:
    rows = []
    for raw in table_block_to_rows(block):
        first_key = next(iter(raw), None)
        owner, headquarters = _split_owner_headquarters(raw.get(first_key) if first_key else None)
        for column, value in raw.items():
            match = _YEAR_COLUMN_RE.match(column.strip()) if column else None
            if not match:
                continue
            pct = parse_number(value)
            if pct is None:
                continue
            rows.append({"owner": owner, "headquarters": headquarters, "year": int(match.group(1)), "pct": pct})
    return rows


def regularize(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    details_blocks = blocks_in_sheet(workbook, "Details")
    historical_blocks = blocks_in_sheet(workbook, "Historical")
    royalty_blocks = blocks_in_sheet(workbook, "Royalty")

    current_rows = []
    current_block = find_block(details_blocks, block_type="table")
    if current_block is not None:
        for raw in table_block_to_rows(current_block):
            current_rows.append(
                {
                    "owner": raw.get("Owner"),
                    "owner_type": raw.get("Type"),
                    "equity_pct": parse_number(raw.get("Equity Ownership (%)")),
                    "controlling_pct": parse_number(raw.get("Controlling Ownership (%)")),
                    "headquarters": raw.get("Corporate Headquarters"),
                }
            )

    former_rows = []
    former_block = find_block(details_blocks, title_contains="Former Direct Owner")
    if former_block is not None:
        data_rows = key_value_rows(former_block)[1:]  # first row is the block's own column-header row
        for row in data_rows:
            if len(row) < 4:
                continue
            former_rows.append(
                {
                    "former_owner": normalize_text(row[0]),
                    "owner_role": normalize_text(row[1]),
                    "equity_pct": parse_number(row[2]),
                    "transaction_date": parse_excel_serial_date(row[3]),
                }
            )

    historical_equity_rows: list[dict] = []
    historical_control_rows: list[dict] = []
    historical_tables = find_blocks(historical_blocks, block_type="table")
    if len(historical_tables) >= 1:
        historical_equity_rows = _melt_historical(historical_tables[0])
    if len(historical_tables) >= 2:
        historical_control_rows = _melt_historical(historical_tables[1])

    royalty_rows = []
    royalty_block = find_block(royalty_blocks, title_contains="Royalty Holder")
    if royalty_block is not None:
        data_rows = key_value_rows(royalty_block)[1:]
        for row in data_rows:
            if len(row) < 4:
                continue
            royalty_rows.append(
                {
                    "holder": normalize_text(row[0]),
                    "royalty_type": normalize_text(row[1]),
                    "royalty_pct": parse_number(row[2]),
                    "headquarters": normalize_text(row[3]),
                }
            )

    return {
        CURRENT_TABLE: current_rows,
        FORMER_TABLE: former_rows,
        HISTORICAL_EQUITY_TABLE: historical_equity_rows,
        HISTORICAL_CONTROL_TABLE: historical_control_rows,
        ROYALTY_TABLE: royalty_rows,
    }
