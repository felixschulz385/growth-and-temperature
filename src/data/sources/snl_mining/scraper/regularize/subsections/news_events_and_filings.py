"""`Documents`, `News`, `Events Calendar` -- all under the same sidebar
section, structurally distinct from each other.

- `Documents`: flat table, headers `Company, Document, Event Date, Filing
  Date, Abstract` (both date columns are Excel serial numbers; Filing Date
  carries a fractional time-of-day component that's dropped -- date-level
  granularity only, matching every other date field in this package).
- `News`: a `key_value` block where each `label` cell is `"headline\nbody"`
  and the matching `value` cell is a display timestamp like
  `"6/29/2017 6:09 PM ET"` (narrow no-break space + a trailing tz
  abbreviation) -- not a serial number, needs its own text-timestamp parser.
- `Events Calendar`: flat table, headers `Type, Event, Eastern Time, As
  Reported Date*, Phone Number, Access Code`; often empty ("No upcoming
  events scheduled").
"""

from __future__ import annotations

import re
from datetime import datetime

from ...parsing.xls import ParsedWorkbook
from ..helpers import all_blocks, find_block, key_value_block_to_pairs, normalize_text, parse_excel_serial_date, table_block_to_rows

DOCUMENTS_TABLE = "detail_documents"
NEWS_TABLE = "detail_news"
EVENTS_CALENDAR_TABLE = "detail_events_calendar"

_NEWS_TIMESTAMP_RE = re.compile(r"\s+(?:ET|CT|MT|PT)$", re.IGNORECASE)


def _parse_news_timestamp(value: str | None) -> datetime | None:
    text = normalize_text(value)
    if text is None:
        return None
    text = text.replace(" ", " ")
    text = _NEWS_TIMESTAMP_RE.sub("", text).strip()
    try:
        return datetime.strptime(text, "%m/%d/%Y %I:%M %p")
    except ValueError:
        return None


def regularize_documents(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    block = find_block(all_blocks(workbook), block_type="table")
    if block is None:
        return {DOCUMENTS_TABLE: []}
    rows = []
    for raw in table_block_to_rows(block):
        rows.append(
            {
                "company": raw.get("Company"),
                "document_type": raw.get("Document"),
                "event_date": parse_excel_serial_date(raw.get("Event Date")),
                "filing_date": parse_excel_serial_date(raw.get("Filing Date")),
                "abstract": normalize_text(raw.get("Abstract")),
            }
        )
    return {DOCUMENTS_TABLE: rows}


def regularize_news(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    block = find_block(all_blocks(workbook), block_type="key_value")
    if block is None:
        return {NEWS_TABLE: []}
    rows = []
    for label, value in key_value_block_to_pairs(block):
        text = normalize_text(label) or ""
        headline, _, body = text.partition("\n")
        rows.append(
            {
                "headline": headline or None,
                "body": normalize_text(body) if body else None,
                "published_at": _parse_news_timestamp(value),
            }
        )
    return {NEWS_TABLE: rows}


def regularize_events_calendar(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    block = find_block(all_blocks(workbook), block_type="table")
    if block is None:
        return {EVENTS_CALENDAR_TABLE: []}
    rows = []
    for raw in table_block_to_rows(block):
        rows.append(
            {
                "event_type": raw.get("Type"),
                "event": raw.get("Event"),
                "eastern_time": normalize_text(raw.get("Eastern Time")),
                "as_reported_date": normalize_text(raw.get("As Reported Date*")),
                "phone_number": normalize_text(raw.get("Phone Number")),
                "access_code": normalize_text(raw.get("Access Code")),
            }
        )
    return {EVENTS_CALENDAR_TABLE: rows}
