"""Shared block-extraction and value-cleaning primitives for subsection
regularizers.

Value-cleaning conventions here are derived from actually inspecting real
scraped exports (see the plan under
`.claude/plans/design-a-modular-parser-elegant-stardust.md`), not ported
blindly from the unrelated manual-xls notebook's `parse_spglobal_number`:
this scraper's numeric cells use plain US formatting (`,` thousands
separator, `.` decimal point -- no locale-comma decimals were observed), and
date-typed table columns are Excel serial-day integers (the raw `<v>` cell
value survives even though the workbook displays it as a date), not date
strings.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from src.data.common.years import MAX_PLAUSIBLE_YEAR, MIN_PLAUSIBLE_YEAR

from ..parsing.xls import ParsedWorkbook, StructuredBlock, StructuredCell

_EXCEL_EPOCH = datetime(1899, 12, 30)
_LEADING_NUMBER_RE = re.compile(r"^\s*(-?[\d,]+(?:\.\d+)?)")
_NA_VALUES = {"NA", "N/A", "-", ""}


def unescape_cell_text(value: str | None) -> str | None:
    """Fix the literal `_x000d_`/`_x000a_` OOXML carriage-return/line-feed
    escape artifacts that show up unresolved inside narrative text cells."""
    if value is None:
        return None
    return (
        value.replace("_x000D_", "\n")
        .replace("_x000d_", "\n")
        .replace("_x000A_", "\n")
        .replace("_x000a_", "\n")
    )


def normalize_text(value: object, *, collapse_whitespace: bool = True) -> str | None:
    """Unescape, trim, and drop `NA`-style placeholders. Returns `None` for
    blank/placeholder values so callers can treat "missing" uniformly."""
    if value is None:
        return None
    text = unescape_cell_text(str(value)).strip()
    if not text or text.upper() in _NA_VALUES:
        return None
    if collapse_whitespace:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def parse_number(value: object) -> float | None:
    """Parse a numeric cell: strips `$`/`%`/thousands commas, honors
    accounting parentheses as negative. Returns `None` for blank/`NA`/
    unparseable values."""
    text = normalize_text(value)
    if text is None:
        return None
    text = text.replace("$", "").replace("%", "").strip()
    is_negative = text.startswith("(") and text.endswith(")")
    if is_negative:
        text = text[1:-1]
    text = text.replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return None
    return -number if is_negative else number


_VALUE_UNIT_RE = re.compile(r"^\s*(-?[\d,]+(?:\.\d+)?)\s*(?:\(([^)]*)\))?\s*$")


def split_value_unit(value: object) -> tuple[float | None, str | None]:
    """Split a `"5,500 (tonnes/day)"`-shaped cell into `(5500.0,
    "tonnes/day")`. Falls back to `(None, None)` when the text doesn't
    match that number(-unit) shape at all."""
    text = normalize_text(value)
    if text is None:
        return None, None
    match = _VALUE_UNIT_RE.match(text)
    if not match:
        return None, None
    return parse_number(match.group(1)), match.group(2)


def parse_leading_number(value: object) -> float | None:
    """Extract a leading numeric value from a string that also carries a
    trailing unit/label, e.g. `"0.630 ($/lb Zn)"` -> `0.630`."""
    text = normalize_text(value)
    if text is None:
        return None
    match = _LEADING_NUMBER_RE.match(text)
    if not match:
        return None
    return parse_number(match.group(1))


def parse_year(value: object) -> int | None:
    """Keep the first 4-digit year found in a value (mirrors the manual-xls
    notebook's `parse_year`, reused here since the same "sometimes textual,
    sometimes numeric" year quirk shows up in this scraper's exports too)."""
    text = normalize_text(value)
    if text is None:
        return None
    match = re.search(r"\d{4}", text)
    if not match:
        return None
    year = int(match.group(0))
    if not (MIN_PLAUSIBLE_YEAR <= year <= MAX_PLAUSIBLE_YEAR):
        return None
    return year


def parse_excel_serial_date(value: object) -> date | None:
    """Decode an Excel serial-day integer (the raw underlying value of a
    date-formatted cell) into a `date`. Only accepts values that decode into
    the shared plausible-year range, guarding against a stray non-date
    integer (e.g. a percentage) in a date-labeled column being silently
    misread as a date."""
    text = normalize_text(value)
    if text is None:
        return None
    try:
        serial = int(float(text))
    except ValueError:
        return None
    try:
        decoded = _EXCEL_EPOCH + timedelta(days=serial)
    except OverflowError:
        return None
    if not (MIN_PLAUSIBLE_YEAR <= decoded.year <= MAX_PLAUSIBLE_YEAR):
        return None
    return decoded.date()


def parse_date_text(value: object) -> date | None:
    """Parse a literal date string (e.g. `"8/27/2015"`), as seen in
    key_value/narrative blocks that render dates as text rather than through
    a numeric-date-formatted cell."""
    text = normalize_text(value)
    if text is None:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def key_value_block_to_pairs(block: StructuredBlock) -> list[tuple[str, str]]:
    """Flatten a `key_value`-type block's `label`/`value`-role cells into
    ordered `(label, value)` pairs. Callers decide how to group pairs: some
    blocks are one label->value dict, others are N repeated (name, amount)
    pairs (see `helpers.pair_up` for that case)."""
    pairs: list[tuple[str, str]] = []
    pending_label: str | None = None
    for cell in block.cells:
        if cell.role == "label":
            pending_label = cell.value
        elif cell.role == "value" and pending_label is not None:
            pairs.append((pending_label, cell.value))
            pending_label = None
    return pairs


def key_value_rows(block: StructuredBlock) -> list[list[str]]:
    """Group a `key_value` block's label/value cells into per-row value
    lists, ordered by column position -- for blocks that are really an
    N-column mini-table encoded as repeated label/value pairs per row (e.g.
    Ownership's "Former Direct Owner(s)": 4 columns -> 2 label + 2 value
    cells per row). The first returned row is typically the block's own
    in-band column-header row; callers skip it explicitly."""
    rows_by_number: dict[int, list[StructuredCell]] = {}
    for cell in block.cells:
        if cell.role in ("label", "value"):
            rows_by_number.setdefault(cell.row_number, []).append(cell)
    rows = []
    for row_number in sorted(rows_by_number):
        cells = sorted(rows_by_number[row_number], key=lambda c: c.column_index)
        rows.append([c.value for c in cells])
    return rows


def pair_up(values: list[str]) -> list[tuple[str, str]]:
    """Group a flat, repeated (name, amount, name, amount, ...) sequence
    into `(name, amount)` tuples -- the shape `key_value_block_to_pairs`
    returns for blocks like "Former Direct Owner(s)" or "Royalty Holder(s)"
    once the header pair itself is stripped off."""
    return [(values[i], values[i + 1]) for i in range(0, len(values) - 1, 2)]


def table_block_to_rows(block: StructuredBlock) -> list[dict[str, str]]:
    """Convert a `table`-type block's single-header-row layout into row
    dicts keyed by header text. Rows with a header cell missing a matching
    data cell (ragged rows) simply omit that key rather than erroring --
    real exports are not always rectangular."""
    header_by_column: dict[int, str] = {
        cell.column_index: cell.value for cell in block.cells if cell.role == "header"
    }
    if not header_by_column:
        return []

    rows_by_number: dict[int, dict[int, StructuredCell]] = {}
    for cell in block.cells:
        if cell.role != "data":
            continue
        rows_by_number.setdefault(cell.row_number, {})[cell.column_index] = cell

    rows: list[dict[str, str]] = []
    for row_number in sorted(rows_by_number):
        row_cells = rows_by_number[row_number]
        row = {
            header_by_column[col_idx]: cell.value
            for col_idx, cell in row_cells.items()
            if col_idx in header_by_column
        }
        if row:
            rows.append(row)
    return rows


def table_sections_from_first_header_row(block: StructuredBlock) -> list[tuple[str | None, list[dict[str, str]]]]:
    """Split a `table` block into `(section_title, rows)` groups at each
    single-cell `context`-role divider row (e.g. Subcontractors'
    "Past Subcontractors" line), treating only each group's own first row as
    its column header -- by row/column position, not the generic `role`
    field.

    Exists because the generic `header_row_count` heuristic misfires on
    tables where a real data row happens to look header-like (mostly text,
    few/no numbers -- e.g. Discoveries & Milestones' single discovery-detail
    row, or Subcontractors' lone "current subcontractor" row): those get
    misclassified as extra header rows and silently dropped by
    `table_block_to_rows`. Position-only header/data reconstruction sidesteps
    that misclassification. Not a fit for genuinely multi-row headers (see
    `capacity_costs.py`/Drill Results, handled separately)."""
    rows_by_number: dict[int, list[StructuredCell]] = {}
    for cell in block.cells:
        rows_by_number.setdefault(cell.row_number, []).append(cell)

    sections: list[tuple[str | None, list[dict[str, str]]]] = []
    section_title: str | None = None
    section_row_numbers: list[int] = []

    def flush() -> None:
        if not section_row_numbers:
            return
        header_row = rows_by_number[section_row_numbers[0]]
        header_by_column = {cell.column_index: cell.value for cell in header_row}
        data_rows = []
        for row_number in section_row_numbers[1:]:
            row = {
                header_by_column[cell.column_index]: cell.value
                for cell in rows_by_number[row_number]
                if cell.column_index in header_by_column
            }
            if row:
                data_rows.append(row)
        sections.append((section_title, data_rows))

    for row_number in sorted(rows_by_number):
        cells = rows_by_number[row_number]
        if len(cells) == 1 and cells[0].role == "context":
            flush()
            section_title = normalize_text(cells[0].value)
            section_row_numbers = []
        else:
            section_row_numbers.append(row_number)
    flush()
    return sections


def cells_by_position(block: StructuredBlock) -> dict[tuple[int, int], str]:
    """Index every cell in a block by `(row_number, column_index)` --
    the escape hatch for cross-tab/merged layouts (see `capacity_costs.py`)
    where the generic `header`/`data` role split doesn't line up with the
    block's real shape."""
    return {(cell.row_number, cell.column_index): cell.value for cell in block.cells}


def find_block(blocks: list[StructuredBlock], *, title_contains: str | None = None, block_type: str | None = None) -> StructuredBlock | None:
    """Find the first block matching an optional (case-insensitive) title
    substring and/or block_type -- used by per-type regularizers to pick out
    a specific sub-table/key_value block from a multi-block workbook."""
    needle = title_contains.casefold() if title_contains else None
    for block in blocks:
        if block_type is not None and block.block_type != block_type:
            continue
        if needle is not None and (block.block_title is None or needle not in block.block_title.casefold()):
            continue
        return block
    return None


def all_blocks(workbook: ParsedWorkbook) -> list[StructuredBlock]:
    """Flatten every block across every sheet in a parsed workbook, in
    sheet/block order."""
    return [block for sheet in workbook.sheets for block in sheet.blocks]


def blocks_in_sheet(workbook: ParsedWorkbook, sheet_name: str) -> list[StructuredBlock]:
    """Blocks belonging to one named sheet (case-insensitive) -- multi-sheet
    workbooks like `Ownership` (Details/Historical/Royalty) need to scope
    lookups to a specific sheet rather than searching the whole workbook."""
    needle = sheet_name.casefold()
    for sheet in workbook.sheets:
        if sheet.sheet_name.casefold() == needle:
            return list(sheet.blocks)
    return []


def find_blocks(blocks: list[StructuredBlock], *, title_contains: str | None = None, block_type: str | None = None) -> list[StructuredBlock]:
    """Like `find_block`, but returns every match (e.g. every `table` block
    in a workbook)."""
    needle = title_contains.casefold() if title_contains else None
    matches = []
    for block in blocks:
        if block_type is not None and block.block_type != block_type:
            continue
        if needle is not None and (block.block_title is None or needle not in block.block_title.casefold()):
            continue
        matches.append(block)
    return matches
