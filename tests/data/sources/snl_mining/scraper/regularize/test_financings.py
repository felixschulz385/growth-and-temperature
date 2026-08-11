from __future__ import annotations

import datetime

from src.data.sources.snl_mining.scraper.regularize.subsections import financings
from _builders import table_block, text_block, workbook


def test_financings_regularizes_flat_table():
    block = table_block(
        headers=[
            "Transaction ID", "Issuer Name", "Announce Date", "Offering Type",
            "Transaction Status", "Offering Price ($)", "Total Shares Offered (actual)",
            "Offering Size ($000)", "Issue Currency",
        ],
        rows=[["SPTRO123", "Acme Corp.", "44229", "Common Stock - ATM", "Priced", "1.50", "302,633", "454", "USD"]],
    )
    wb = workbook(blocks=[text_block(lines=["Testium Ridge | Financings"]), block])

    tables = financings.regularize(wb, "m1")
    rows = tables["detail_financings"]

    assert len(rows) == 1
    assert rows[0]["transaction_id"] == "SPTRO123"
    assert rows[0]["announce_date"] == datetime.date(2021, 2, 2)
    assert rows[0]["offering_price_usd"] == 1.5
    assert rows[0]["total_shares_offered"] == 302633.0


def test_financings_empty_when_no_table_block():
    wb = workbook(blocks=[text_block(lines=["No data"])])
    tables = financings.regularize(wb, "m1")
    assert tables["detail_financings"] == []
