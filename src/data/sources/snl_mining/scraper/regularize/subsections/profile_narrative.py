"""`Work History` and `Comments` -- free-text narrative subsections, each a
single content block whose heading is the block's first cell and whose
remaining cells are one narrative segment per row (not one big blob to
regex-split, unlike the *manual* screener export's `FULL_WORK_HISTORY`
column that `notebooks/snl_mining_manual_xls_to_duckdb.ipynb` parses with
`WORK_HISTORY_PATTERN` -- the scraper's exports already arrive one segment
per cell).

`Work History` segments are frequently (not always) prefixed with a
`"YYYY[-YY]:"` date range -- when a segment lacks one, it continues the most
recently seen year (a real, observed pattern: a dated segment is often
followed by 1-2 undated continuation segments for the same year). `Comments`
segments have no consistent date-prefix convention, so they're kept as
plain ordered text; its separate "Bibliography" block is regularized as its
own citation-list table.
"""

from __future__ import annotations

import re

from ...parsing.xls import ParsedWorkbook
from ..helpers import all_blocks, normalize_text

WORK_HISTORY_EVENTS_TABLE = "detail_work_history_events"
COMMENTS_TABLE = "detail_comments__general"
BIBLIOGRAPHY_TABLE = "detail_comments__bibliography"

_YEAR_PREFIX_RE = re.compile(r"^(\d{4})(?:-(\d{2,4}))?:\s*(.*)$", re.DOTALL)


def _resolve_year_end(year_start: int, raw_end: str | None) -> int | None:
    if raw_end is None:
        return None
    if len(raw_end) == 4:
        return int(raw_end)
    # 2-digit continuation year (e.g. "1977-83") shares the start year's century.
    century = str(year_start)[:2]
    return int(century + raw_end)


def _heading_and_segments(workbook: ParsedWorkbook, heading_text: str) -> list[str]:
    for block in all_blocks(workbook):
        if block.block_type != "text" or not block.cells:
            continue
        heading = normalize_text(block.cells[0].value)
        if heading == heading_text:
            return [normalize_text(cell.value) for cell in block.cells[1:]]
    return []


def regularize_work_history(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    rows = []
    year_start: int | None = None
    year_end: int | None = None
    for sequence, segment in enumerate(_heading_and_segments(workbook, "Work History"), start=1):
        if segment is None:
            continue
        match = _YEAR_PREFIX_RE.match(segment)
        if match:
            year_start = int(match.group(1))
            year_end = _resolve_year_end(year_start, match.group(2))
            text = normalize_text(match.group(3))
        else:
            text = segment
        rows.append(
            {
                "event_sequence": sequence,
                "year_start": year_start,
                "year_end": year_end,
                "event_text": text,
            }
        )
    return {WORK_HISTORY_EVENTS_TABLE: rows}


def regularize_comments(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    comment_rows = [
        {"comment_sequence": sequence, "text": text}
        for sequence, text in enumerate(_heading_and_segments(workbook, "General Comments"), start=1)
        if text is not None
    ]
    bibliography_rows = [
        {"citation_sequence": sequence, "citation_text": text}
        for sequence, text in enumerate(_heading_and_segments(workbook, "Bibliography"), start=1)
        if text is not None
    ]
    return {COMMENTS_TABLE: comment_rows, BIBLIOGRAPHY_TABLE: bibliography_rows}
