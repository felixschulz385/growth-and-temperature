from __future__ import annotations

from src.data.sources.snl_mining.scraper.regularize.subsections import profile_narrative
from _builders import text_block, workbook


def test_work_history_parses_dated_segments_and_carries_forward_undated_ones():
    block = text_block(
        lines=[
            "Work History",
            "1976: GCO began exploration in the area._x000d_",
            "1977-83: GCO drilled more than 96 core holes.  (E&MJ 1/85)_x000d_",
            "GCO and Houston Oil did more drilling in the area.  (ARMR 1984)_x000d_",
        ]
    )
    wb = workbook(blocks=[block])

    tables = profile_narrative.regularize_work_history(wb, "m1")
    events = tables["detail_work_history_events"]

    assert len(events) == 3
    assert events[0] == {
        "event_sequence": 1,
        "year_start": 1976,
        "year_end": None,
        "event_text": "GCO began exploration in the area.",
    }
    assert events[1]["year_start"] == 1977
    assert events[1]["year_end"] == 1983
    # normalize_text collapses the export's double space before "(E&MJ...".
    assert events[1]["event_text"] == "GCO drilled more than 96 core holes. (E&MJ 1/85)"
    # No "YYYY:" prefix on this segment -- carries forward the previous
    # segment's year range rather than leaving it null.
    assert events[2]["year_start"] == 1977
    assert events[2]["year_end"] == 1983
    assert events[2]["event_text"] == "GCO and Houston Oil did more drilling in the area. (ARMR 1984)"


def test_work_history_empty_when_heading_not_found():
    wb = workbook(blocks=[text_block(lines=["Something Else"])])
    tables = profile_narrative.regularize_work_history(wb, "m1")
    assert tables["detail_work_history_events"] == []


def test_comments_splits_general_and_bibliography_sections():
    general_block = text_block(lines=["General Comments", "First comment.", "Second comment."], block_index=1)
    biblio_block = text_block(
        lines=["Bibliography", "MILS SEQUENCE #0020180001", "CALIFORNIA MINING JOURNAL 9/81"],
        block_index=2,
        start_row=10,
    )
    wb = workbook(blocks=[general_block, biblio_block])

    tables = profile_narrative.regularize_comments(wb, "m1")

    assert tables["detail_comments__general"] == [
        {"comment_sequence": 1, "text": "First comment."},
        {"comment_sequence": 2, "text": "Second comment."},
    ]
    assert tables["detail_comments__bibliography"] == [
        {"citation_sequence": 1, "citation_text": "MILS SEQUENCE #0020180001"},
        {"citation_sequence": 2, "citation_text": "CALIFORNIA MINING JOURNAL 9/81"},
    ]
