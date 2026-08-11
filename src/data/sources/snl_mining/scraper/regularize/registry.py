"""Fixed registry of subsection-type regularizers, plus the content-
validation gate that runs ahead of every regularizer call.

The registry key set is the 27 subsection labels actually observed in the
real local scraper database (`SELECT DISTINCT subsection_label FROM
mine_subsection_exports`) -- see the plan doc for the full derivation. A
subsection label that isn't one of these 27 is a hard "unknown_type", not a
silently-accepted generic fallback: adding a 28th type means writing and
registering a new regularizer module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..parsing.xls import ParsedWorkbook, normalize_subsection_label
from .subsections import (
    capacity_costs,
    financings,
    m_a_history,
    mine_economics,
    news_events_and_filings,
    ownership,
    profile_narrative,
    profile_tables,
    property_profile,
    reserves,
)

RegularizeFn = Callable[[ParsedWorkbook, str], dict[str, list[dict]]]

#: Statuses a regularization attempt can resolve to. Matches
#: `mine_subsection_stage_status.status` values used by every other stage.
STATUS_COMPLETED = "completed"
STATUS_CONTENT_MISMATCH = "content_mismatch"
STATUS_UNVERIFIED = "unverified"
STATUS_UNKNOWN_TYPE = "unknown_type"


@dataclass(frozen=True, slots=True)
class SubsectionRegularizer:
    subsection_label: str
    expected_title_fragments: tuple[str, ...]
    regularize: RegularizeFn


def _entry(subsection_label: str, expected_title_fragments: tuple[str, ...], regularize: RegularizeFn) -> SubsectionRegularizer:
    return SubsectionRegularizer(
        subsection_label=subsection_label,
        expected_title_fragments=expected_title_fragments,
        regularize=regularize,
    )


_ENTRIES: tuple[SubsectionRegularizer, ...] = (
    # profile
    _entry("Property Profile", ("Property Profile",), property_profile.regularize),
    _entry("Location, Map & Claims", ("Location, Map & Claims", "Location Map & Claims", "Geology"), profile_tables.regularize_location_map_claims),
    _entry("Geology", ("Geology", "Discoveries & Milestones"), profile_tables.regularize_geology),
    _entry("Work History", ("Work History",), profile_narrative.regularize_work_history),
    _entry("Discoveries & Milestones", ("Discoveries & Milestones",), profile_tables.regularize_discoveries_milestones),
    _entry("Drill Results", ("Drill Results", "Discoveries & Milestones"), profile_tables.regularize_drill_results),
    _entry("Development Studies", ("Development Studies",), profile_tables.regularize_development_studies),
    _entry("Capital Costs", ("Capital Costs",), profile_tables.regularize_capital_costs),
    _entry("Subcontractors", ("Subcontractors",), profile_tables.regularize_subcontractors),
    _entry("Comments", ("Comments",), profile_narrative.regularize_comments),
    # ownership
    _entry("Ownership", ("Ownership",), ownership.regularize),
    _entry("Ownership Structure", ("Ownership",), ownership.regularize),
    # production_and_reserves
    _entry("Capacity & Costs", ("Capacity & Costs",), capacity_costs.regularize),
    _entry("Production", ("Capacity & Costs", "Production"), capacity_costs.regularize),
    _entry("Reserves & Resources", ("Reserves & Resources", "Capacity & Costs"), reserves.regularize_reserves_resources),
    _entry("Reserves / Resources & Production Chart", ("Reserves/Resources & Production Chart", "Reserves / Resources & Production Chart"), reserves.regularize_reserves_resources_production_chart),
    # mine_economics_and_emissions
    _entry("Cash Flow Analysis", ("Mine Economics Cash Flow Analysis",), mine_economics.regularize_cash_flow_analysis),
    # Cost Curve export is deliberately disabled scraper-side
    # (`_TEMPORARILY_SKIPPED_SUBSECTIONS` in stages/scrape_detail_exports.py)
    # -- every real export filed under this label is stale content from
    # whatever page was previously open (100% "Property Profile" in the
    # local scraper DB). `expected_title_fragments` intentionally can never
    # match, so every row correctly resolves to content_mismatch rather than
    # regularizing stale data as if it were real Cost Curve content.
    _entry("Cost Curve", ("Cost Curve",), mine_economics.regularize_cost_curve),
    _entry("Modeled Ore Costs", ("Mine Economics Modeled Ore Costs",), mine_economics.regularize_modeled_ore_costs),
    _entry("Modeled Product Costs", ("Mine Economics Modeled Product Costs", "Mine Economics Modeled Ore Costs"), mine_economics.regularize_modeled_product_costs),
    _entry("Modeled Production", ("Mine Economics Modeled Production",), mine_economics.regularize_modeled_production),
    _entry("Modeled ROM Costs", ("Mine Economics Modeled Rom Costs", "Mine Economics Modeled Production"), mine_economics.regularize_modeled_rom_costs),
    # m_a_history_and_financings
    _entry("Financings", ("Financings",), financings.regularize),
    _entry("M&A History", ("M&A History",), m_a_history.regularize),
    # news_events_and_filings
    _entry("Documents", ("Documents",), news_events_and_filings.regularize_documents),
    _entry("News", ("News",), news_events_and_filings.regularize_news),
    _entry("Events Calendar", ("Events Calendar",), news_events_and_filings.regularize_events_calendar),
)

SUBSECTION_REGULARIZERS: dict[str, SubsectionRegularizer] = {
    normalize_subsection_label(entry.subsection_label): entry for entry in _ENTRIES
}
assert len(SUBSECTION_REGULARIZERS) == len(_ENTRIES), "duplicate subsection_label in registry"


def classify_and_regularize(
    workbook: ParsedWorkbook,
    mine_id: str,
    subsection_label: str,
) -> tuple[str, dict[str, list[dict]]]:
    """Look up *subsection_label*'s regularizer, run the content-validation
    gate (see module docstring on the mismatch problem this guards against),
    and regularize if the gate passes.

    Returns `(status, tables)` where `status` is one of
    `STATUS_COMPLETED`/`STATUS_CONTENT_MISMATCH`/`STATUS_UNVERIFIED`/
    `STATUS_UNKNOWN_TYPE`, and `tables` is `{}` unless status is
    `STATUS_COMPLETED` or `STATUS_UNVERIFIED`.
    """
    entry = SUBSECTION_REGULARIZERS.get(normalize_subsection_label(subsection_label))
    if entry is None:
        return STATUS_UNKNOWN_TYPE, {}

    title = (workbook.workbook_title or "").casefold()
    if title:
        matched = any(fragment.casefold() in title for fragment in entry.expected_title_fragments)
        if not matched:
            return STATUS_CONTENT_MISMATCH, {}
        status = STATUS_COMPLETED
    else:
        # No title at all to check against -- attempt regularization anyway
        # (real data shows ~30-60% of exports have no inferrable title, see
        # plan doc §1) but flag it distinctly from a verified pass.
        status = STATUS_UNVERIFIED

    tables = entry.regularize(workbook, mine_id)
    return status, tables
