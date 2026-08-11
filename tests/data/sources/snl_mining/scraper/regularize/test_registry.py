"""Tests for the content-validation gate (`classify_and_regularize`) --
the mechanism guarding against the real, load-bearing data-quality issue
found in the local scraper DB: a subsection export's actual content
frequently doesn't match the `subsection_label` it was filed under (e.g.
13% of real "Production" exports actually contain "Capacity & Costs"
content). See the plan doc for the full derivation.
"""

from __future__ import annotations

from src.data.sources.snl_mining.scraper.regularize.registry import (
    STATUS_COMPLETED,
    STATUS_CONTENT_MISMATCH,
    STATUS_UNKNOWN_TYPE,
    STATUS_UNVERIFIED,
    classify_and_regularize,
)
from _builders import table_block, workbook


def test_completed_when_title_matches_expected_fragment():
    wb = workbook(
        workbook_title="Testium Ridge | Financings",
        blocks=[
            table_block(
                headers=[
                    "Transaction ID", "Issuer Name", "Announce Date", "Offering Type",
                    "Transaction Status", "Offering Price ($)", "Total Shares Offered (actual)",
                    "Offering Size ($000)", "Issue Currency",
                ],
                rows=[["T1", "Acme Corp.", "44229", "Common Stock", "Priced", "1.5", "1000", "1500", "USD"]],
            )
        ],
    )
    status, tables = classify_and_regularize(wb, "m1", "Financings")
    assert status == STATUS_COMPLETED
    assert tables["detail_financings"][0]["transaction_id"] == "T1"


def test_content_mismatch_when_title_does_not_match():
    # Real, observed pattern: an export filed under "Production" whose
    # workbook title is actually "Capacity & Costs"-flavored stale content
    # from the previous page (see plan doc §1) -- must not be regularized
    # as if it were real Financings content.
    wb = workbook(workbook_title="Testium Ridge | Capacity & Costs", blocks=[])
    status, tables = classify_and_regularize(wb, "m1", "Financings")
    assert status == STATUS_CONTENT_MISMATCH
    assert tables == {}


def test_unverified_when_title_is_missing():
    wb = workbook(workbook_title=None, blocks=[table_block(headers=["Transaction ID"], rows=[["T1"]])])
    status, tables = classify_and_regularize(wb, "m1", "Financings")
    assert status == STATUS_UNVERIFIED
    # Still attempts regularization -- real data shows a large NaN-title
    # tail that shouldn't be silently dropped, just flagged distinctly.
    assert tables["detail_financings"][0]["transaction_id"] == "T1"


def test_unknown_type_for_unregistered_subsection_label():
    wb = workbook(workbook_title="Testium Ridge | Some New Page")
    status, tables = classify_and_regularize(wb, "m1", "Some New Page That Does Not Exist")
    assert status == STATUS_UNKNOWN_TYPE
    assert tables == {}


def test_registry_covers_exactly_the_27_fixed_subsection_types():
    from src.data.sources.snl_mining.scraper.regularize.registry import SUBSECTION_REGULARIZERS

    assert len(SUBSECTION_REGULARIZERS) == 27
