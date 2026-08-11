"""`M&A History` -- one flat table of M&A transactions.

Real shape: text preamble + a single `table` block with headers
`Transaction ID, Announcement Date, Completion Date, Target, Buyer, Seller,
Percent Acquired (%), Earn-In?, Transaction Type, Announced Transaction
Value ($M)`.
"""

from __future__ import annotations

from ...parsing.xls import ParsedWorkbook
from ..helpers import all_blocks, find_block, parse_excel_serial_date, parse_number, table_block_to_rows

TABLE_NAME = "detail_m_a_history"


def regularize(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    block = find_block(all_blocks(workbook), block_type="table")
    if block is None:
        return {TABLE_NAME: []}

    rows = []
    for raw in table_block_to_rows(block):
        rows.append(
            {
                "transaction_id": raw.get("Transaction ID"),
                "announcement_date": parse_excel_serial_date(raw.get("Announcement Date")),
                "completion_date": parse_excel_serial_date(raw.get("Completion Date")),
                "target": raw.get("Target"),
                "buyer": raw.get("Buyer"),
                "seller": raw.get("Seller"),
                "percent_acquired": parse_number(raw.get("Percent Acquired (%)")),
                "earn_in": raw.get("Earn-In?"),
                "transaction_type": raw.get("Transaction Type"),
                "announced_transaction_value_usd_m": parse_number(raw.get("Announced Transaction Value ($M)")),
            }
        )
    return {TABLE_NAME: rows}
