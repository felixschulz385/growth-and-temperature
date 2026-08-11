"""`Property Profile` -- the richest single-workbook type: one sheet with
~7 meaningful blocks (general info, current owners, a claims-count summary,
comments, work-history preview, recent news, contained reserves/resources,
document highlights) plus a static risk-score legend table that's identical
across every mine and is therefore not regularized (it carries no per-mine
information). See plan doc §3/§8.
"""

from __future__ import annotations

import re

from ...parsing.xls import ParsedWorkbook
from ..helpers import (
    all_blocks,
    find_block,
    key_value_block_to_pairs,
    normalize_text,
    parse_date_text,
    parse_excel_serial_date,
    parse_number,
    table_block_to_rows,
)

GENERAL_TABLE = "detail_property_profile__general"
OWNERS_TABLE = "detail_property_profile__owners"
CLAIMS_SUMMARY_TABLE = "detail_property_profile__claims_summary"
RECENT_NEWS_TABLE = "detail_property_profile__recent_news"
CONTAINED_RESERVES_TABLE = "detail_property_profile__contained_reserves"
FILINGS_TABLE = "detail_property_profile__filings"
NARRATIVE_TABLE = "detail_property_profile__narrative"

_GENERAL_FIELD_MAP = {
    "Property ID": "property_id",
    "Also Known As": "also_known_as",
    "Property Type": "property_type",
    "Commodity(s)": "commodities",
    "Development Stage": "development_stage",
    "Activity Status": "activity_status",
    "Mine Type": "mine_type",
    "Country/Region": "country_region",
    "State or Province": "state_or_province",
}
_COUNT_RE = re.compile(r"\(([\d,]+)\)")


def _extract_count(text: str | None) -> int | None:
    normalized = normalize_text(text)
    if normalized is None:
        return None
    match = _COUNT_RE.search(normalized)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def regularize(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    blocks = all_blocks(workbook)

    general_row: dict = {}
    general_block = find_block(blocks, block_type="key_value")
    if general_block is not None:
        pairs = key_value_block_to_pairs(general_block)[1:]  # skip "General Information"/"Map" section header pair
        raw = dict(pairs)
        for source_label, field in _GENERAL_FIELD_MAP.items():
            general_row[field] = normalize_text(raw.get(source_label))
        general_row["total_insitu_value_usd_m"] = next(
            (parse_number(v) for k, v in raw.items() if k.startswith("Total In-Situ Value")), None
        )
        general_row["country_risk_score_outlook"] = next(
            (normalize_text(v) for k, v in raw.items() if k.startswith("Country/Region Risk Score")), None
        )

    owners_rows = []
    owners_block = None
    for block in blocks:
        if block.block_type == "table" and {"Owner", "Type", "Equity Ownership (%)"}.issubset(
            {c.value for c in block.cells if c.role == "header"}
        ):
            owners_block = block
            break
    if owners_block is not None:
        for raw in table_block_to_rows(owners_block):
            owners_rows.append(
                {
                    "owner": raw.get("Owner"),
                    "owner_type": raw.get("Type"),
                    "equity_pct": parse_number(raw.get("Equity Ownership (%)")),
                }
            )

    claims_summary_rows = []
    claims_block = None
    for block in blocks:
        if block.block_type != "key_value":
            continue
        pairs = key_value_block_to_pairs(block)
        labels = [label for label, _ in pairs]
        if any(label.startswith("Property (") for label in labels):
            claims_block = block
            property_count = next((_extract_count(l) for l in labels if l.startswith("Property (")), None)
            owner_claims_count = next((_extract_count(l) for l in labels if l.startswith("Property Owner(s) Claims")), None)
            linked_claims_count = next((_extract_count(v) for l, v in pairs if l.startswith("Property (")), None)
            claims_summary_rows.append(
                {
                    "property_count": property_count,
                    "linked_claims_count": linked_claims_count,
                    "owner_claims_count": owner_claims_count,
                }
            )
            break

    recent_news_rows = []
    news_block = find_block(blocks, title_contains="Recent News")
    if news_block is not None:
        for label, value in key_value_block_to_pairs(news_block):
            recent_news_rows.append({"news_date": parse_date_text(label), "headline": normalize_text(value)})

    contained_reserves_rows = []
    reserves_block = None
    for block in blocks:
        if block.block_type == "table" and {"Commodity", "Contained", "Unit"}.issubset(
            {c.value for c in block.cells if c.role == "header"}
        ):
            reserves_block = block
            break
    if reserves_block is not None:
        for raw in table_block_to_rows(reserves_block):
            contained_reserves_rows.append(
                {
                    "commodity": raw.get("Commodity"),
                    "contained": parse_number(raw.get("Contained")),
                    "unit": raw.get("Unit"),
                    "as_of_date": parse_excel_serial_date(raw.get("As Of Date")),
                }
            )

    filings_rows = []
    for block in blocks:
        if block.block_type != "key_value" or block in (general_block, news_block, claims_block):
            continue
        pairs = key_value_block_to_pairs(block)
        # "Document Highlights" is the one remaining key_value block whose
        # every value is a plain filing date (e.g. "3/5/2026") -- distinct
        # from general info (text values) and claims-summary ("Property (N)"
        # labels, already excluded by identity above).
        if pairs and all(parse_date_text(value) is not None for _, value in pairs):
            for label, value in pairs:
                filings_rows.append({"filing_type": normalize_text(label), "filing_date": parse_date_text(value)})
            break

    narrative_rows = []
    text_blocks = [b for b in blocks if b.block_type == "text"]
    for index, block in enumerate(text_blocks):
        if not block.cells:
            continue
        heading = normalize_text(block.cells[0].value)
        if heading not in ("Comments", "Work History"):
            continue
        if len(block.cells) > 1:
            # Heading + body land in the same block (Work History, when row
            # spacing keeps them contiguous).
            body = normalize_text(" ".join(cell.value for cell in block.cells[1:]))
        elif index + 1 < len(text_blocks):
            # Heading is its own 1-cell block (Comments, split by a row gap
            # from its body) -- the body is the next text block.
            body_block = text_blocks[index + 1]
            body = normalize_text(" ".join(cell.value for cell in body_block.cells))
        else:
            body = None
        narrative_rows.append({"section": heading, "text": body})

    return {
        GENERAL_TABLE: [general_row] if general_row else [],
        OWNERS_TABLE: owners_rows,
        CLAIMS_SUMMARY_TABLE: claims_summary_rows,
        RECENT_NEWS_TABLE: recent_news_rows,
        CONTAINED_RESERVES_TABLE: contained_reserves_rows,
        FILINGS_TABLE: filings_rows,
        NARRATIVE_TABLE: narrative_rows,
    }
