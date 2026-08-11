"""Small builders for synthetic ParsedWorkbook fixtures, used across the
regularize/ test modules. Not a test module itself (no `test_` prefix, so
pytest won't collect it) -- built directly as dataclass instances rather
than real .xlsx bytes, per the plan: exercising the parsing layer itself is
already covered by `parsing/xls.py`'s own real-file behavior; these tests
target the regularize/ layer's logic in isolation.
"""

from __future__ import annotations

from pathlib import Path

from src.data.sources.snl_mining.scraper.parsing.xls import (
    ParsedWorkbook,
    StructuredBlock,
    StructuredCell,
    StructuredSheet,
)


def _column_letter(index: int) -> str:
    letters = ""
    index += 1
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def cell(cell_ref: str, column_index: int, row_number: int, value: str, role: str) -> StructuredCell:
    return StructuredCell(
        cell_ref=cell_ref,
        column_index=column_index,
        column_name=_column_letter(column_index),
        row_number=row_number,
        value=value,
        role=role,
    )


def header_row(row_number: int, labels: list[str]) -> list[StructuredCell]:
    return [cell(f"col{i}{row_number}", i, row_number, label, "header") for i, label in enumerate(labels)]


def data_row(row_number: int, values: list[str]) -> list[StructuredCell]:
    return [cell(f"col{i}{row_number}", i, row_number, value, "data") for i, value in enumerate(values)]


def table_block(
    *,
    headers: list[str],
    rows: list[list[str]],
    block_index: int = 1,
    block_title: str | None = None,
    sheet_name: str = "Sheet1",
    header_row_number: int = 1,
) -> StructuredBlock:
    cells = list(header_row(header_row_number, headers))
    for offset, row_values in enumerate(rows, start=1):
        cells.extend(data_row(header_row_number + offset, row_values))
    return StructuredBlock(
        sheet_name=sheet_name,
        block_index=block_index,
        block_type="table",
        block_title=block_title,
        row_start=header_row_number,
        row_end=header_row_number + len(rows),
        header_row_count=1,
        cells=tuple(cells),
    )


def key_value_block(
    *,
    pairs: list[tuple[str, str]],
    block_index: int = 1,
    block_title: str | None = None,
    sheet_name: str = "Sheet1",
    start_row: int = 1,
) -> StructuredBlock:
    cells = []
    for offset, (label, value) in enumerate(pairs):
        row_number = start_row + offset
        cells.append(cell(f"a{row_number}", 0, row_number, label, "label"))
        cells.append(cell(f"b{row_number}", 1, row_number, value, "value"))
    return StructuredBlock(
        sheet_name=sheet_name,
        block_index=block_index,
        block_type="key_value",
        block_title=block_title,
        row_start=start_row,
        row_end=start_row + len(pairs) - 1 if pairs else start_row,
        header_row_count=0,
        cells=tuple(cells),
    )


def text_block(
    *,
    lines: list[str],
    block_index: int = 1,
    sheet_name: str = "Sheet1",
    start_row: int = 1,
) -> StructuredBlock:
    cells = [cell(f"a{start_row + i}", 0, start_row + i, line, "text") for i, line in enumerate(lines)]
    return StructuredBlock(
        sheet_name=sheet_name,
        block_index=block_index,
        block_type="text",
        block_title=None,
        row_start=start_row,
        row_end=start_row + len(lines) - 1,
        header_row_count=0,
        cells=tuple(cells),
    )


def workbook(
    *,
    blocks: list[StructuredBlock] | None = None,
    workbook_title: str | None = "Test Mine | Test Subsection",
    workbook_subtitle: str | None = "(Primary: Testium)",
    xls_sha256: str = "deadbeef",
) -> ParsedWorkbook:
    by_sheet: dict[str, list[StructuredBlock]] = {}
    for block in blocks or []:
        by_sheet.setdefault(block.sheet_name, []).append(block)
    sheets = tuple(
        StructuredSheet(sheet_name=name, sheet_index=index, blocks=tuple(sheet_blocks))
        for index, (name, sheet_blocks) in enumerate(by_sheet.items(), start=1)
    )
    return ParsedWorkbook(
        xls_path=Path("synthetic.xlsx"),
        xls_sha256=xls_sha256,
        workbook_title=workbook_title,
        workbook_subtitle=workbook_subtitle,
        primary_sheet_name=sheets[0].sheet_name if sheets else None,
        content_subsection_label=workbook_title.split("|", 1)[1].strip() if workbook_title and "|" in workbook_title else workbook_title,
        sheets=sheets,
        flat_cells=(),
    )
