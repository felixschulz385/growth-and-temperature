"""Tests for the content-validation gate (`classify_and_regularize`) --
the mechanism guarding against the real, load-bearing data-quality issue
found in the local scraper DB: a subsection export's actual content
frequently doesn't match the `subsection_label` it was filed under (e.g.
13% of real "Production" exports actually contain "Capacity & Costs"
content). See the plan doc for the full derivation.

When the requested label doesn't validate, `classify_and_regularize` now
tries to reclassify purely from the workbook's own title before giving up
(`STATUS_RECLASSIFIED`) -- see the follow-up plan on reducing the ~20% skip
rate. It only does so when the title unambiguously matches exactly one
*function* among the 27 registered types; several types share the same
underlying regularizer (e.g. `Ownership`/`Ownership Structure`), so matching
either of those isn't ambiguous, but a title that could equally be two
functionally distinct types (e.g. `Modeled Ore Costs` vs `Modeled Product
Costs`) still falls back to `STATUS_CONTENT_MISMATCH` rather than guessing.
"""

from __future__ import annotations

from src.data.sources.snl_mining.scraper.regularize.registry import (
    STATUS_COMPLETED,
    STATUS_CONTENT_MISMATCH,
    STATUS_RECLASSIFIED,
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


def test_content_mismatch_when_title_matches_nothing_registered():
    # A title that doesn't validate the requested label and also doesn't
    # match any other registered type's fragments -- a genuine mismatch,
    # not a recoverable mislabel.
    wb = workbook(workbook_title="Testium Ridge | Totally Unrelated Page", blocks=[])
    status, tables = classify_and_regularize(wb, "m1", "Financings")
    assert status == STATUS_CONTENT_MISMATCH
    assert tables == {}


def test_reclassified_when_title_matches_a_different_registered_type():
    # Real, observed pattern: an export filed under one label whose workbook
    # title is actually a different, unambiguous type's stale content from
    # the previous page (see plan doc §1). "M&A History" is a good test
    # fragment precisely because it's *not* shared with any other entry
    # (unlike e.g. "Capacity & Costs", which is also listed as an alternate
    # fragment for "Reserves & Resources" and would be ambiguous). Previously
    # this was a dead STATUS_CONTENT_MISMATCH; it now recovers via the
    # actual content's own regularizer instead of being dropped.
    wb = workbook(workbook_title="Testium Ridge | M&A History", blocks=[])
    status, tables = classify_and_regularize(wb, "m1", "Financings")
    assert status == STATUS_RECLASSIFIED
    # Regularized under m_a_history's own table, not detail_financings.
    assert set(tables) == {"detail_m_a_history"}


def test_reclassified_when_requested_label_is_unregistered_but_title_matches():
    # A scraped subsection_label outside the fixed 27 (previously always
    # STATUS_UNKNOWN_TYPE) whose title nonetheless matches a registered type
    # unambiguously -- also recoverable now.
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
    status, tables = classify_and_regularize(wb, "m1", "Some New Page That Does Not Exist")
    assert status == STATUS_RECLASSIFIED
    assert tables["detail_financings"][0]["transaction_id"] == "T1"


def test_stays_content_mismatch_when_title_matches_two_different_functions():
    # "Mine Economics Modeled Ore Costs" is Modeled Ore Costs' own fragment
    # *and* one of Modeled Product Costs' alternate fragments -- those two
    # types have genuinely different regularizers, so this must not guess.
    wb = workbook(workbook_title="Testium Ridge | Mine Economics Modeled Ore Costs", blocks=[])
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
