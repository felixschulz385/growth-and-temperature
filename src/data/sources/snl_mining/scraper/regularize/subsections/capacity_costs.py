"""`Capacity & Costs` / `Production` -- both sidebar links frequently land on
the same underlying report (confirmed both by real title inspection and the
plan doc §1 content-label crosstab), and it's the hardest real shape found:
a 2-row header that's actually two side-by-side mini-tables ("Production
Capacity" at columns 0-5, "Production Costs" at columns 7-9) sharing no
common row key, with a numeric value embedded inside a header-role cell's
text (`"0.630 ($/lb Zn)"`). The generic `header`/`data` role split
misclassifies the top cost-summary row as `header` (dropping it entirely
from `table_block_to_rows`), so this extracts directly from
`cells_by_position` instead.
"""

from __future__ import annotations

from ...parsing.xls import ParsedWorkbook
from ..helpers import all_blocks, cells_by_position, find_block, key_value_block_to_pairs, normalize_text, parse_number, split_value_unit

PRODUCTION_TABLE = "detail_capacity_costs__production"
COST_TABLE = "detail_capacity_costs__cost_breakdown"
PROCESSING_TABLE = "detail_capacity_costs__processing_details"


def _row_numbers(block) -> list[int]:
    return sorted({cell.row_number for cell in block.cells})


def regularize(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    table_block = find_block(all_blocks(workbook), block_type="table")

    production_rows: list[dict] = []
    cost_rows: list[dict] = []
    if table_block is not None:
        positions = cells_by_position(table_block)
        for row_number in _row_numbers(table_block):
            commodity = positions.get((row_number, 0))
            if commodity and commodity not in ("Production Capacity", "Commodity"):
                amount, amount_unit = parse_number(positions.get((row_number, 1))), positions.get((row_number, 2))
                grade, grade_unit = parse_number(positions.get((row_number, 3))), positions.get((row_number, 4))
                recovery_rate = parse_number(positions.get((row_number, 5)))
                production_rows.append(
                    {
                        "commodity": normalize_text(commodity),
                        "production_amount": amount,
                        "production_unit": normalize_text(amount_unit),
                        "millhead_grade": grade,
                        "millhead_grade_unit": normalize_text(grade_unit),
                        "recovery_rate_pct": recovery_rate,
                    }
                )

            cost_label = positions.get((row_number, 7))
            if cost_label and cost_label not in ("Production Costs",):
                value, unit = split_value_unit(positions.get((row_number, 9)))
                cost_rows.append(
                    {
                        "cost_type": normalize_text(cost_label),
                        "cost_value": value,
                        "cost_unit": unit,
                    }
                )

    processing_rows: list[dict] = []
    processing_block = find_block(all_blocks(workbook), block_type="key_value", title_contains="Mining & Processing Details")
    if processing_block is not None:
        raw = dict(key_value_block_to_pairs(processing_block))
        daily_value, daily_unit = split_value_unit(raw.get("Daily Processing"))
        annual_value, annual_unit = split_value_unit(raw.get("Annual Processing"))
        processing_rows.append(
            {
                "mining_method": normalize_text(raw.get("Mining Method")),
                "processing_method": normalize_text(raw.get("Processing Method")),
                "product_form": normalize_text(raw.get("Product Form")),
                "daily_processing": daily_value,
                "daily_processing_unit": daily_unit,
                "annual_processing": annual_value,
                "annual_processing_unit": annual_unit,
                "stripping_ratio": parse_number((raw.get("Stripping Ratio") or "").split(":")[0]) if raw.get("Stripping Ratio") else None,
            }
        )

    return {
        PRODUCTION_TABLE: production_rows,
        COST_TABLE: cost_rows,
        PROCESSING_TABLE: processing_rows,
    }
