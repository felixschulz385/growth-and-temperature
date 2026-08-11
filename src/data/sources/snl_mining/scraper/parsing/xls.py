"""XLS/XLSX parsing helpers used by subsection export blocks."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from zipfile import ZipFile
import xml.etree.ElementTree as ET

PROPERTY_PROFILE_LABEL = "property profile"

_XML_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
_CELL_REF_RE = re.compile(r"([A-Z]+)(\d+)")
_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


@dataclass(frozen=True, slots=True)
class ParsedCell:
    sheet_name: str
    row_index: int
    column_name: str
    value: str


@dataclass(frozen=True, slots=True)
class StructuredCell:
    cell_ref: str
    column_index: int
    column_name: str
    row_number: int
    value: str
    role: str


@dataclass(frozen=True, slots=True)
class StructuredBlock:
    sheet_name: str
    block_index: int
    block_type: str
    block_title: str | None
    row_start: int
    row_end: int
    header_row_count: int
    cells: tuple[StructuredCell, ...]


@dataclass(frozen=True, slots=True)
class StructuredSheet:
    sheet_name: str
    sheet_index: int
    blocks: tuple[StructuredBlock, ...]


@dataclass(frozen=True, slots=True)
class ParsedWorkbook:
    xls_path: Path
    xls_sha256: str
    workbook_title: str | None
    workbook_subtitle: str | None
    primary_sheet_name: str | None
    content_subsection_label: str | None
    sheets: tuple[StructuredSheet, ...]
    flat_cells: tuple[ParsedCell, ...]


@dataclass(frozen=True, slots=True)
class _RowCell:
    cell_ref: str
    column_index: int
    column_name: str
    row_number: int
    value: str


@dataclass(frozen=True, slots=True)
class _SheetRow:
    row_number: int
    cells: tuple[_RowCell, ...]


def parse_subsection_xls(
    xls_path: str | Path,
    subsection_label: str,
) -> ParsedWorkbook:
    path = Path(xls_path)
    if path.suffix.casefold() != ".xlsx":
        raise ValueError(
            f"Unsupported export format for structural parsing: {path.suffix or '<none>'}"
        )
    return _parse_xlsx_workbook(path, subsection_label=subsection_label)


def parse_property_profile_special(xls_path: str | Path) -> ParsedWorkbook:
    return parse_subsection_xls(xls_path, subsection_label=PROPERTY_PROFILE_LABEL)


def normalize_subsection_label(value: str) -> str:
    return value.strip().casefold()


def _parse_xlsx_workbook(path: Path, subsection_label: str) -> ParsedWorkbook:
    workbook_hash = sha256(path.read_bytes()).hexdigest()
    sheets = _load_structured_sheets(path)
    flat_cells = tuple(
        ParsedCell(
            sheet_name=cell.sheet_name,
            row_index=cell.row_index,
            column_name=cell.column_name,
            value=cell.value,
        )
        for sheet in sheets
        for block in sheet.blocks
        for cell in _iter_flat_cells(block)
    )
    workbook_title, workbook_subtitle = _extract_workbook_headers(sheets)
    content_subsection_label = _infer_content_subsection_label(
        workbook_title=workbook_title,
        primary_sheet_name=sheets[0].sheet_name if sheets else None,
        requested_subsection_label=subsection_label,
    )
    return ParsedWorkbook(
        xls_path=path,
        xls_sha256=workbook_hash,
        workbook_title=workbook_title,
        workbook_subtitle=workbook_subtitle,
        primary_sheet_name=sheets[0].sheet_name if sheets else None,
        content_subsection_label=content_subsection_label,
        sheets=tuple(sheets),
        flat_cells=flat_cells,
    )


def _iter_flat_cells(block: StructuredBlock) -> tuple[ParsedCell, ...]:
    return tuple(
        ParsedCell(
            sheet_name=block.sheet_name,
            row_index=cell.row_number - 1,
            column_name=cell.column_name,
            value=cell.value,
        )
        for cell in block.cells
    )


def _load_structured_sheets(path: Path) -> list[StructuredSheet]:
    with ZipFile(path) as archive:
        shared_strings = _read_shared_strings(archive)
        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels_root.findall("pkgrel:Relationship", _XML_NS)
        }

        sheets: list[StructuredSheet] = []
        for sheet_index, sheet in enumerate(
            workbook_root.findall("main:sheets/main:sheet", _XML_NS), start=1
        ):
            relation_id = sheet.attrib[
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            ]
            target = rel_map[relation_id]
            sheet_xml = ET.fromstring(archive.read(f"xl/{target}"))
            rows = _read_sheet_rows(sheet_xml, shared_strings)
            blocks = _infer_sheet_blocks(str(sheet.attrib["name"]), rows)
            sheets.append(
                StructuredSheet(
                    sheet_name=str(sheet.attrib["name"]),
                    sheet_index=sheet_index,
                    blocks=tuple(blocks),
                )
            )
    return sheets


def _read_shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(text.text or "" for text in item.findall(".//main:t", _XML_NS))
        for item in root.findall("main:si", _XML_NS)
    ]


def _read_sheet_rows(sheet_xml: ET.Element, shared_strings: list[str]) -> list[_SheetRow]:
    rows: list[_SheetRow] = []
    for row in sheet_xml.findall("main:sheetData/main:row", _XML_NS):
        row_number = int(row.attrib["r"])
        cells: list[_RowCell] = []
        for cell in row.findall("main:c", _XML_NS):
            cell_ref = cell.attrib.get("r", "")
            match = _CELL_REF_RE.match(cell_ref)
            if match is None:
                continue
            value = _extract_cell_value(cell, shared_strings).strip()
            if value == "":
                continue
            column_name = match.group(1)
            cells.append(
                _RowCell(
                    cell_ref=cell_ref,
                    column_index=_column_name_to_index(column_name),
                    column_name=column_name,
                    row_number=row_number,
                    value=value,
                )
            )
        if cells:
            rows.append(_SheetRow(row_number=row_number, cells=tuple(cells)))
    return rows


def _extract_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find("main:is/main:t", _XML_NS)
        return "" if inline is None or inline.text is None else inline.text
    value_node = cell.find("main:v", _XML_NS)
    if value_node is None or value_node.text is None:
        return ""
    value = value_node.text
    if cell_type == "s":
        index = int(value)
        if 0 <= index < len(shared_strings):
            return shared_strings[index]
    return value


def _infer_sheet_blocks(sheet_name: str, rows: list[_SheetRow]) -> list[StructuredBlock]:
    row_groups = _group_rows_by_spacing(rows)
    blocks: list[StructuredBlock] = []
    for block_index, row_group in enumerate(row_groups, start=1):
        block_rows = _strip_leading_singletons_for_metadata(row_group)
        if not block_rows:
            continue
        block_title, content_rows = _split_block_title(block_rows)
        if not content_rows:
            content_rows = block_rows
        block_type = _infer_block_type(content_rows)
        header_row_count = _infer_header_row_count(block_type, content_rows)
        structured_cells: list[StructuredCell] = []
        for row_offset, row in enumerate(content_rows, start=1):
            for cell_position, cell in enumerate(row.cells):
                role = _infer_cell_role(
                    block_type=block_type,
                    header_row_count=header_row_count,
                    row_offset=row_offset,
                    cell_position=cell_position,
                    row=row,
                )
                structured_cells.append(
                    StructuredCell(
                        cell_ref=cell.cell_ref,
                        column_index=cell.column_index,
                        column_name=cell.column_name,
                        row_number=cell.row_number,
                        value=cell.value,
                        role=role,
                    )
                )
        blocks.append(
            StructuredBlock(
                sheet_name=sheet_name,
                block_index=block_index,
                block_type=block_type,
                block_title=block_title,
                row_start=content_rows[0].row_number,
                row_end=content_rows[-1].row_number,
                header_row_count=header_row_count,
                cells=tuple(structured_cells),
            )
        )
    return blocks


def _group_rows_by_spacing(rows: list[_SheetRow]) -> list[list[_SheetRow]]:
    if not rows:
        return []
    groups: list[list[_SheetRow]] = []
    current: list[_SheetRow] = [rows[0]]
    previous_row_number = rows[0].row_number
    for row in rows[1:]:
        if row.row_number - previous_row_number > 1:
            groups.append(current)
            current = [row]
        else:
            current.append(row)
        previous_row_number = row.row_number
    groups.append(current)
    return groups


def _strip_leading_singletons_for_metadata(rows: list[_SheetRow]) -> list[_SheetRow]:
    if not rows:
        return []
    first_content_index = 0
    while first_content_index < len(rows) and len(rows[first_content_index].cells) == 1:
        if first_content_index + 1 < len(rows) and len(rows[first_content_index + 1].cells) > 1:
            break
        first_content_index += 1
    return rows[first_content_index:] or rows


def _split_block_title(rows: list[_SheetRow]) -> tuple[str | None, list[_SheetRow]]:
    if len(rows) >= 2 and len(rows[0].cells) == 1 and len(rows[1].cells) > 1:
        return rows[0].cells[0].value, rows[1:]
    return None, rows


def _infer_block_type(rows: list[_SheetRow]) -> str:
    widths = [len(row.cells) for row in rows]
    if widths and all(width == 1 for width in widths):
        return "text"
    if _looks_like_key_value(rows):
        return "key_value"
    return "table"


def _looks_like_key_value(rows: list[_SheetRow]) -> bool:
    if any(len(row.cells) >= 5 and _is_consecutive_header_row(row) for row in rows):
        return False
    if rows and (sum(len(row.cells) for row in rows) / len(rows)) > 4:
        return False
    pair_like_rows = 0
    for row in rows:
        width = len(row.cells)
        if width in {2, 4, 6, 8}:
            pair_like_rows += 1
    if not rows:
        return False
    if pair_like_rows / len(rows) < 0.6:
        return False
    dense_rows = [
        row
        for row in rows
        if len(row.cells) >= 4
        and _has_repeated_label_value_spacing(row)
    ]
    if dense_rows:
        return True
    return all(len(row.cells) <= 2 for row in rows)


def _has_repeated_label_value_spacing(row: _SheetRow) -> bool:
    if len(row.cells) < 4 or len(row.cells) % 2 != 0:
        return False
    indices = [cell.column_index for cell in row.cells]
    gaps = [right - left for left, right in zip(indices, indices[1:])]
    return all(gap >= 1 for gap in gaps) and not _is_consecutive_header_row(row)


def _is_consecutive_header_row(row: _SheetRow) -> bool:
    indices = [cell.column_index for cell in row.cells]
    if len(indices) < 3:
        return False
    return indices == list(range(indices[0], indices[0] + len(indices)))


def _infer_header_row_count(block_type: str, rows: list[_SheetRow]) -> int:
    if block_type != "table":
        return 0
    header_rows = 0
    for row in rows:
        if len(row.cells) == 1:
            if header_rows == 0:
                header_rows += 1
                continue
            break
        if _row_is_header_like(row):
            header_rows += 1
            continue
        break
    return header_rows


def _row_is_header_like(row: _SheetRow) -> bool:
    if len(row.cells) <= 1:
        return True
    numeric_cells = sum(1 for cell in row.cells if _looks_numeric(cell.value))
    return numeric_cells == 0 or (
        row.cells[0].column_index > 0 and numeric_cells < len(row.cells) / 2
    )


def _infer_cell_role(
    block_type: str,
    header_row_count: int,
    row_offset: int,
    cell_position: int,
    row: _SheetRow,
) -> str:
    if block_type == "text":
        return "text"
    if block_type == "key_value":
        if len(row.cells) == 1:
            return "note"
        return "label" if cell_position % 2 == 0 else "value"
    if row_offset <= header_row_count:
        return "header"
    if len(row.cells) == 1:
        return "context"
    return "data"


def _extract_workbook_headers(sheets: list[StructuredSheet]) -> tuple[str | None, str | None]:
    first_cells = [
        cell.value
        for sheet in sheets[:1]
        for block in sheet.blocks[:2]
        for cell in block.cells[:2]
    ]
    title = first_cells[0] if first_cells else None
    subtitle = first_cells[1] if len(first_cells) > 1 else None
    return title, subtitle


def _infer_content_subsection_label(
    workbook_title: str | None,
    primary_sheet_name: str | None,
    requested_subsection_label: str,
) -> str | None:
    title_source = workbook_title or primary_sheet_name or requested_subsection_label
    normalized = title_source.strip() if title_source else ""
    if "|" in normalized:
        normalized = normalized.split("|", 1)[1].strip()
    return normalized or None


def _column_name_to_index(column_name: str) -> int:
    index = 0
    for char in column_name:
        index = (index * 26) + (ord(char.upper()) - 64)
    return index - 1


def _looks_numeric(value: str) -> bool:
    return bool(_NUMERIC_RE.match(value))
