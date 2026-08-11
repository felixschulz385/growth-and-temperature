"""`Financings` -- one flat table of security-offering transactions.

Real shape (see plan doc §8): text preamble + a single `table` block with
headers `Transaction ID, Issuer Name, Announce Date, Offering Type,
Transaction Status, Offering Price ($), Total Shares Offered (actual),
Offering Size ($000), Issue Currency`. `mine_id`/`xls_sha256`/
`regularized_at` are injected by the stage function, not here.
"""

from __future__ import annotations

from ...parsing.xls import ParsedWorkbook
from ..helpers import all_blocks, find_block, parse_excel_serial_date, parse_number, table_block_to_rows

TABLE_NAME = "detail_financings"


def regularize(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    block = find_block(all_blocks(workbook), block_type="table")
    if block is None:
        return {TABLE_NAME: []}

    rows = []
    for raw in table_block_to_rows(block):
        rows.append(
            {
                "transaction_id": raw.get("Transaction ID"),
                "issuer_name": raw.get("Issuer Name"),
                "announce_date": parse_excel_serial_date(raw.get("Announce Date")),
                "offering_type": raw.get("Offering Type"),
                "transaction_status": raw.get("Transaction Status"),
                "offering_price_usd": parse_number(raw.get("Offering Price ($)")),
                "total_shares_offered": parse_number(raw.get("Total Shares Offered (actual)")),
                "offering_size_usd000": parse_number(raw.get("Offering Size ($000)")),
                "issue_currency": raw.get("Issue Currency"),
            }
        )
    return {TABLE_NAME: rows}
