"""The remaining `profile`-section table subsections: `Location, Map &
Claims`, `Geology`, `Discoveries & Milestones`, `Drill Results`,
`Development Studies`, `Capital Costs`, `Subcontractors`. Each gets its own
`regularize_*` function (grouped in one module since none needs its own
shared helper beyond what's already in `helpers.py`), registered
individually in `registry.py`.
"""

from __future__ import annotations

from ...parsing.xls import ParsedWorkbook
from ..helpers import (
    all_blocks,
    cells_by_position,
    find_block,
    key_value_block_to_pairs,
    normalize_text,
    parse_excel_serial_date,
    parse_number,
    table_block_to_rows,
    table_sections_from_first_header_row,
)

LOCATION_TABLE = "detail_location_map_claims__location"
CLAIMS_TABLE = "detail_location_map_claims__claims"
GEOLOGY_TABLE = "detail_geology"
DISCOVERIES_TABLE = "detail_discoveries_milestones__discoveries"
MILESTONES_TABLE = "detail_discoveries_milestones__milestones"
DRILL_RESULTS_TABLE = "detail_drill_results"
DEVELOPMENT_STUDIES_TABLE = "detail_development_studies"
CAPITAL_COSTS_TABLE = "detail_capital_costs"
SUBCONTRACTORS_TABLE = "detail_subcontractors"


def regularize_location_map_claims(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    blocks = all_blocks(workbook)

    location_rows = []
    location_block = find_block(blocks, block_type="key_value")
    latlong_block = find_block(blocks, title_contains="Latitude/Longitude")
    if location_block is not None:
        raw = dict(key_value_block_to_pairs(location_block)[1:])  # skip "Location"/"Map" section-header pair
        latlong = dict(key_value_block_to_pairs(latlong_block)) if latlong_block is not None else {}
        location_rows.append(
            {
                "country_region": normalize_text(raw.get("Country/Region")),
                "state_or_province": normalize_text(raw.get("State or Province")),
                "district": normalize_text(raw.get("District")),
                "distance_from": normalize_text(raw.get("Distance From")),
                "decimal_degrees": normalize_text(latlong.get("Decimal Degrees")),
                "coordinate_accuracy": normalize_text(latlong.get("Coordinate Accuracy")),
            }
        )

    claims_rows = []
    claims_block = find_block(blocks, block_type="table", title_contains="Claims linked to Property")
    if claims_block is not None:
        for raw in table_block_to_rows(claims_block):
            claims_rows.append(
                {
                    "claim_name": raw.get("Claim Name"),
                    "claim_id": raw.get("Claim ID"),
                    "owners": raw.get("Owner(s)"),
                    "claim_type": raw.get("Type"),
                    "as_reported_type": raw.get("As Reported Type"),
                    "status": raw.get("Status"),
                    "date_granted": parse_excel_serial_date(raw.get("Date Granted")),
                    "expiry_date": parse_excel_serial_date(raw.get("Expiry Date")),
                    "source": raw.get("Source"),
                    "source_as_of_date": parse_excel_serial_date(raw.get("Source As Of Date")),
                }
            )

    return {LOCATION_TABLE: location_rows, CLAIMS_TABLE: claims_rows}


def regularize_geology(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    block = find_block(all_blocks(workbook), block_type="table", title_contains="Geology")
    rows = []
    if block is not None:
        for raw in table_block_to_rows(block):
            rows.append(
                {
                    "zone_name": raw.get("Zone Name"),
                    "orebody_type": raw.get("Orebody Type"),
                    "orebody_class": raw.get("Orebody Class"),
                    "ore_minerals": raw.get("Ore Minerals"),
                    "avg_depth_feet": parse_number(raw.get("Avg Depth (feet)")),
                    "comments": normalize_text(raw.get("Comments")),
                }
            )
    return {GEOLOGY_TABLE: rows}


def regularize_discoveries_milestones(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    blocks = all_blocks(workbook)

    discovery_rows = []
    discovery_block = find_block(blocks, block_type="table", title_contains="Discovery Details")
    if discovery_block is not None:
        # The single discovery-detail row often looks header-like (all text,
        # no numbers) and gets misclassified as a 2nd header row by the
        # generic role heuristic -- position-only reconstruction avoids that.
        for _, rows in table_sections_from_first_header_row(discovery_block):
            for raw in rows:
                discovery_rows.append(
                    {
                        "period": normalize_text(raw.get("Period")),
                        "event": normalize_text(raw.get("Event")),
                        "discovery_type": normalize_text(raw.get("Discovery Type")),
                        "owner_class_percent_ownership": normalize_text(raw.get("Owner, Class and Percent Ownership✱")),
                        "comment": normalize_text(raw.get("Mining Discovery Comment")),
                    }
                )

    milestones_rows = []
    milestones_block = find_block(blocks, block_type="key_value", title_contains="Milestones")
    if milestones_block is not None:
        for period, event_type in key_value_block_to_pairs(milestones_block)[1:]:  # skip "Period"/"Mining Event Type" header pair
            milestones_rows.append({"period": normalize_text(period), "event_type": normalize_text(event_type)})

    return {DISCOVERIES_TABLE: discovery_rows, MILESTONES_TABLE: milestones_rows}


def regularize_drill_results(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    block = find_block(all_blocks(workbook), block_type="table")
    rows: list[dict] = []
    if block is not None:
        # Genuinely 2-row header (row 2 extends/overrides columns 11-12 with
        # a per-report commodity-specific grade label) -- not a
        # misclassification, so read by fixed column position rather than
        # via table_sections_from_first_header_row's "first row only" rule.
        positions = cells_by_position(block)
        row_numbers = sorted({cell.row_number for cell in block.cells})
        header_rows = row_numbers[: block.header_row_count or 2]
        data_row_numbers = row_numbers[len(header_rows):]
        for row_number in data_row_numbers:
            if (row_number, 0) not in positions and (row_number, 1) not in positions:
                continue
            rows.append(
                {
                    "drill_date": parse_excel_serial_date(positions.get((row_number, 0))),
                    "hole_or_sampling_id": normalize_text(positions.get((row_number, 1))),
                    "reporting_company": normalize_text(positions.get((row_number, 2))),
                    "drill_or_sampling_type": normalize_text(positions.get((row_number, 3))),
                    "purpose": normalize_text(positions.get((row_number, 4))),
                    "from_m": parse_number(positions.get((row_number, 5))),
                    "to_m": parse_number(positions.get((row_number, 6))),
                    "interval_m": parse_number(positions.get((row_number, 7))),
                    "depth_m": parse_number(positions.get((row_number, 8))),
                    "primary_commodity": normalize_text(positions.get((row_number, 9))),
                    "primary_grade_equivalent": normalize_text(positions.get((row_number, 10))),
                    "interval_grade": parse_number(positions.get((row_number, 11))),
                    "grade_x_interval": parse_number(positions.get((row_number, 12))),
                    "source_document": normalize_text(positions.get((row_number, 13))),
                }
            )
    return {DRILL_RESULTS_TABLE: rows}


def regularize_development_studies(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    block = find_block(all_blocks(workbook), block_type="table")
    rows = []
    if block is not None:
        for raw in table_block_to_rows(block):
            rows.append(
                {
                    "study_date": parse_excel_serial_date(raw.get("Date")),
                    "title": normalize_text(raw.get("Title")),
                    "reporting_company": raw.get("Reporting Company"),
                    "study_type": raw.get("Type"),
                    "npv_usd000": parse_number(next((v for k, v in raw.items() if k.startswith("NPV") and "Discount" not in k), None)),
                    "npv_discount_pct": parse_number(next((v for k, v in raw.items() if k.startswith("NPV") and "Discount" in k), None)),
                    "capital_cost_usd000": parse_number(raw.get("Capital Cost ($000)")),
                    "commodities": normalize_text(raw.get("Commodity(s)")),
                    "annual_production_life_of_mine": normalize_text(raw.get("Annual Production Life of Mine")),
                    "irr_pct": parse_number(next((v for k, v in raw.items() if k.startswith("IRR")), None)),
                    "payback_period_years": parse_number(raw.get("Payback Period (years)")),
                }
            )
    return {DEVELOPMENT_STUDIES_TABLE: rows}


def regularize_capital_costs(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    block = find_block(all_blocks(workbook), block_type="table")
    rows = []
    if block is not None:
        for raw in table_block_to_rows(block):
            rows.append(
                {
                    "initial_announcement_date": parse_excel_serial_date(raw.get("Initial Announcement Date")),
                    "last_update_date": parse_excel_serial_date(raw.get("Last Update Date")),
                    "capital_cost_oid": normalize_text(raw.get("Mining Capital Cost OID")),
                    "capital_cost_type": raw.get("Mining Capital Cost Type"),
                    "status": raw.get("Status"),
                    "initial_cost_estimate_usd000": parse_number(raw.get("Initial Cost Estimate ($000)")),
                    "initial_expected_completion": normalize_text(raw.get("Initial Expected Completion")),
                    "most_recent_cost_estimate_usd000": parse_number(raw.get("Most Recent Cost Estimate or Completed Cost ($000)")),
                    "most_recent_expected_or_actual_completion": normalize_text(raw.get("Most Recent Expected or Actual Completion")),
                    "development_type": raw.get("Development Type"),
                    "development_stage": raw.get("Development Stage of the Capital Project"),
                    "comments": normalize_text(raw.get("Comments")),
                }
            )
    return {CAPITAL_COSTS_TABLE: rows}


def regularize_subcontractors(workbook: ParsedWorkbook, mine_id: str) -> dict[str, list[dict]]:
    block = find_block(all_blocks(workbook), block_type="table")
    rows: list[dict] = []
    if block is not None:
        for section_title, section_rows in table_sections_from_first_header_row(block):
            is_current = section_title is None
            for raw in section_rows:
                rows.append(
                    {
                        "status": "current" if is_current else "past",
                        "company": raw.get("Company"),
                        "role": raw.get("Role"),
                        "service": raw.get("Service"),
                        "start_date": normalize_text(raw.get("Start Date")),
                        "end_date": normalize_text(raw.get("Projected End Date") or raw.get("End Date")),
                    }
                )
    return {SUBCONTRACTORS_TABLE: rows}
