"""Unit tests for the shared commodity vocabulary (src/data/sources/commodities.py)."""

from src.data.sources.commodities import (
    CANONICAL_COMMODITIES,
    SNL_COMMODITY_ALIASES,
    WORLD_BANK_COLUMNS,
    normalize_commodity,
)


def test_normalize_commodity_snl_aliases():
    assert normalize_commodity("Gold", source="snl") == "gold"
    assert normalize_commodity("Iron Ore", source="snl") == "iron_ore"
    assert normalize_commodity("U3O8", source="snl") == "uranium"


def test_normalize_commodity_snl_lanthanides_and_rare_earth_elements_share_canonical_key():
    assert normalize_commodity("Lanthanides", source="snl") == "rare_earths"
    assert normalize_commodity("Rare Earth Elements", source="snl") == "rare_earths"


def test_normalize_commodity_worldbank_aliases():
    assert normalize_commodity("Gold", source="worldbank") == "gold"
    assert normalize_commodity("Iron ore, cfr spot", source="worldbank") == "iron_ore"
    assert normalize_commodity("Coal, Australian", source="worldbank") == "coal"


def test_normalize_commodity_worldbank_only_matches_priced_columns():
    # "Coal, South African" is a real WB column but not the one chosen as the
    # canonical "coal" mapping -- it must not resolve to anything.
    assert normalize_commodity("Coal, South African", source="worldbank") is None
    assert normalize_commodity("Cocoa", source="worldbank") is None


def test_normalize_commodity_unrecognized_returns_none():
    assert normalize_commodity("Not A Real Commodity", source="snl") is None
    assert normalize_commodity("Not A Real Commodity", source="worldbank") is None


def test_normalize_commodity_is_whitespace_and_case_insensitive():
    assert normalize_commodity("  gold  ", source="snl") == "gold"
    assert normalize_commodity("GOLD", source="snl") == "gold"


def test_every_canonical_commodity_has_a_world_bank_columns_entry():
    # Completeness guard: a newly-added CANONICAL_COMMODITIES key must not be
    # forgotten from WORLD_BANK_COLUMNS (even if the value is None).
    assert set(CANONICAL_COMMODITIES) == set(WORLD_BANK_COLUMNS)


def test_every_snl_alias_maps_to_a_canonical_commodity():
    assert set(SNL_COMMODITY_ALIASES.values()) <= set(CANONICAL_COMMODITIES)


def test_world_bank_columns_values_are_unique_where_present():
    # Two canonical commodities must not both claim the same WB column --
    # that would make the worldbank-side reverse lookup ambiguous.
    priced = [v for v in WORLD_BANK_COLUMNS.values() if v is not None]
    assert len(priced) == len(set(priced))
