"""Shared commodity-name vocabulary, joining `commodity_prices` (World Bank
Pink Sheet columns) to `snl_mining` (SNL `primary_commodity` labels and the
user-owned commodity-share table it consumes during PREPARE).

A plain constants/pure-function module, not a `DataSource` subclass -- doesn't
violate docs/design/09-integrated-pipeline.md's "cross-source coupling is on
artefact paths, never a class import" rule, since nothing here imports another
source's *class*. Lives at the `sources/` package root alongside `layout.py`/
`verify.py`/`steps.py`, which are already commonly cross-imported by source
packages.

`CANONICAL_COMMODITIES` was originally seeded from the full distinct
`primary_commodity` list in `data/raw/snl_mining/database.duckdb::properties`
(42 raw labels, 41 distinct canonical keys -- "Lanthanides" and "Rare Earth
Elements" both normalize to "rare_earths"), then extended with 19 more keys
(`arsenic`, `barite`, `bismuth`, `borates`, `cadmium`, `gallium`, `garnet`,
`hafnium`, `indium`, `iridium`, `leucoxene`, `magnesium`, `mercury`,
`rhenium`, `rhodium`, `rubidium`, `ruthenium`, `strontium`, `thorium`) found
via `SELECT DISTINCT commodity FROM detail_reserves_resources` (the scraper's
richer `Contained(...)` reserves/resources table, source="snl" also covers
this vocabulary -- see `snl_mining.source._create_commodity_shares_table`)
that weren't present in the manual `properties.primary_commodity` vocabulary
at all. `WORLD_BANK_COLUMNS` intentionally leaves most of them mapped to
`None`: the World Bank Pink Sheet ("Annual Prices (Real)" sheet) has no
genuine world-market series for the majority of these -- mirroring how Berman
et al. (2017) themselves restrict their analysis to the 14-of-25 minerals
with usable world prices rather than substituting a proxy series. A commodity
mapped to `None` here contributes nothing to a mine's price-shock term (see
`snl_mining.source._create_mine_priceshock_table`), not zero and not an
error.

Bauxite/Alumina are deliberately left unmapped rather than proxied via the WB
"Aluminum" (refined-metal) series -- a defensible but real judgment call,
flagged in the implementation plan rather than decided silently.

Individual lanthanide/rare-earth element labels (`Cerium`, `Dysprosium`,
`Erbium`, `Europium`, `Gadolinium`, `Holmium`, `Lanthanum`, `Lutetium`,
`Neodymium`, `Praseodymium`, `Samarium`, `Terbium`, `Thulium`, `Ytterbium` --
found in `detail_reserves_resources`, reported individually rather than
aggregated) all alias to the existing `"rare_earths"` key, same as
`Lanthanides`/`Rare Earth Elements`; `Yttrium`/`Scandium` keep their own
canonical keys as before, since those already existed independently.
`Lithium Oxide` (a grade/assay reporting convention for the same commodity)
aliases to the existing `"lithium"` key rather than getting a separate one.
"""

from __future__ import annotations

from typing import Literal

#: Canonical, snake_case commodity keys -- the join key between
#: `commodity_prices`'s prepared price table and the (user-owned)
#: `commodity_shares` table `snl_mining`'s PREPARE step consumes.
CANONICAL_COMMODITIES: tuple[str, ...] = (
    "gold", "coal", "copper", "iron_ore", "uranium", "diamonds", "nickel",
    "silver", "zinc", "lithium", "rare_earths", "lead", "platinum",
    "graphite", "molybdenum", "phosphate", "tin", "bauxite", "manganese",
    "potash", "tungsten", "chromite", "cobalt", "ilmenite", "vanadium",
    "antimony", "tantalum", "niobium", "heavy_mineral_sands", "palladium",
    "titanium", "rutile", "zircon", "scandium", "alumina", "yttrium",
    "germanium", "chromium", "ferronickel", "ferrochrome", "caesium",
    # Added for detail_reserves_resources coverage (module docstring) --
    # not present in the manual properties.primary_commodity vocabulary.
    "arsenic", "barite", "bismuth", "borates", "cadmium", "gallium",
    "garnet", "hafnium", "indium", "iridium", "leucoxene", "magnesium",
    "mercury", "rhenium", "rhodium", "rubidium", "ruthenium", "strontium",
    "thorium",
)

#: canonical key -> exact World Bank "Annual Prices (Real)" sheet column
#: header string (`data/raw/commodity_prices/auxiliary/
#: CMO-Historical-Data-Annual.xlsx`, header row 7 / `header=6` in
#: `pd.read_excel`), or `None` if the Pink Sheet has no matching series.
WORLD_BANK_COLUMNS: dict[str, str | None] = {
    "gold": "Gold",
    "coal": "Coal, Australian",  # WB also publishes "Coal, South African";
                                  # Australian chosen as the single benchmark.
    "copper": "Copper",
    "iron_ore": "Iron ore, cfr spot",
    "nickel": "Nickel",
    "silver": "Silver",
    "zinc": "Zinc",
    "lead": "Lead",
    "platinum": "Platinum",
    "tin": "Tin",
    "phosphate": "Phosphate rock",
    "potash": "Potassium chloride",
    "uranium": None,
    "diamonds": None,
    "lithium": None,
    "rare_earths": None,
    "graphite": None,
    "molybdenum": None,
    "bauxite": None,
    "manganese": None,
    "tungsten": None,
    "chromite": None,
    "cobalt": None,
    "ilmenite": None,
    "vanadium": None,
    "antimony": None,
    "tantalum": None,
    "niobium": None,
    "heavy_mineral_sands": None,
    "palladium": None,
    "titanium": None,
    "rutile": None,
    "zircon": None,
    "scandium": None,
    "alumina": None,
    "yttrium": None,
    "germanium": None,
    "chromium": None,
    "ferronickel": None,
    "ferrochrome": None,
    "caesium": None,
    "arsenic": None,
    "barite": None,
    "bismuth": None,
    "borates": None,
    "cadmium": None,
    "gallium": None,
    "garnet": None,
    "hafnium": None,
    "indium": None,
    "iridium": None,
    "leucoxene": None,
    "magnesium": None,
    "mercury": None,
    "rhenium": None,
    "rhodium": None,
    "rubidium": None,
    "ruthenium": None,
    "strontium": None,
    "thorium": None,
}

#: SNL raw label -> canonical key, covering both `properties.primary_commodity`
#: (42 distinct values via `SELECT DISTINCT primary_commodity FROM properties`)
#: and `detail_reserves_resources.commodity` (63 distinct values via
#: `SELECT DISTINCT commodity FROM detail_reserves_resources` restricted to
#: `Contained(...)` rows -- the scraper's richer per-commodity vocabulary,
#: which includes ~28 individual REE/minor-element labels the manual side
#: never reports separately). Both use `source="snl"`.
SNL_COMMODITY_ALIASES: dict[str, str] = {
    "Gold": "gold",
    "Coal": "coal",
    "Copper": "copper",
    "Iron Ore": "iron_ore",
    "U3O8": "uranium",
    "Diamonds": "diamonds",
    "Nickel": "nickel",
    "Silver": "silver",
    "Zinc": "zinc",
    "Lithium": "lithium",
    "Lanthanides": "rare_earths",
    "Rare Earth Elements": "rare_earths",
    "Lead": "lead",
    "Platinum": "platinum",
    "Graphite": "graphite",
    "Molybdenum": "molybdenum",
    "Phosphate": "phosphate",
    "Tin": "tin",
    "Bauxite": "bauxite",
    "Manganese": "manganese",
    "Potash": "potash",
    "Tungsten": "tungsten",
    "Chromite": "chromite",
    "Cobalt": "cobalt",
    "Ilmenite": "ilmenite",
    "Vanadium": "vanadium",
    "Antimony": "antimony",
    "Tantalum": "tantalum",
    "Niobium": "niobium",
    "Heavy Mineral Sands": "heavy_mineral_sands",
    "Palladium": "palladium",
    "Titanium": "titanium",
    "Rutile": "rutile",
    "Zircon": "zircon",
    "Scandium": "scandium",
    "Alumina": "alumina",
    "Yttrium": "yttrium",
    "Germanium": "germanium",
    "Chromium": "chromium",
    "Ferronickel": "ferronickel",
    "Ferrochrome": "ferrochrome",
    "Caesium": "caesium",
    # detail_reserves_resources-only labels below (module docstring).
    "Arsenic": "arsenic",
    "Barite": "barite",
    "Bismuth": "bismuth",
    "Borates": "borates",
    "Cadmium": "cadmium",
    "Gallium": "gallium",
    "Garnet": "garnet",
    "Hafnium": "hafnium",
    "Indium": "indium",
    "Iridium": "iridium",
    "Leucoxene": "leucoxene",
    "Magnesium": "magnesium",
    "Mercury": "mercury",
    "Rhenium": "rhenium",
    "Rhodium": "rhodium",
    "Rubidium": "rubidium",
    "Ruthenium": "ruthenium",
    "Strontium": "strontium",
    "Thorium": "thorium",
    # Individual lanthanide/REE labels -> the existing rare_earths bucket.
    "Cerium": "rare_earths",
    "Dysprosium": "rare_earths",
    "Erbium": "rare_earths",
    "Europium": "rare_earths",
    "Gadolinium": "rare_earths",
    "Holmium": "rare_earths",
    "Lanthanum": "rare_earths",
    "Lutetium": "rare_earths",
    "Neodymium": "rare_earths",
    "Praseodymium": "rare_earths",
    "Samarium": "rare_earths",
    "Terbium": "rare_earths",
    "Thulium": "rare_earths",
    "Ytterbium": "rare_earths",
    # Grade/assay reporting convention for the same commodity as "Lithium".
    "Lithium Oxide": "lithium",
}

#: World Bank "Annual Prices (Real)" column header -> canonical key, built as
#: the inverse of `WORLD_BANK_COLUMNS` (only the mapped, non-`None` entries).
_WORLD_BANK_COLUMN_TO_CANONICAL: dict[str, str] = {
    column: canonical for canonical, column in WORLD_BANK_COLUMNS.items() if column is not None
}

_ALIAS_TABLES: dict[str, dict[str, str]] = {
    "snl": SNL_COMMODITY_ALIASES,
    "worldbank": _WORLD_BANK_COLUMN_TO_CANONICAL,
}


def normalize_commodity(label: str, *, source: Literal["snl", "worldbank"]) -> str | None:
    """Map a raw commodity label from `source` to its canonical key.

    Case/whitespace-normalized (exact-match lookup after `.strip()`, tried
    verbatim first since most labels are already exact). Returns `None` for
    an unrecognized label -- callers should log/skip, never raise, since a
    new commodity showing up in either upstream file is an expected event,
    not a corruption.
    """
    table = _ALIAS_TABLES[source]
    stripped = label.strip()
    if stripped in table:
        return table[stripped]
    for raw_label, canonical in table.items():
        if raw_label.strip().casefold() == stripped.casefold():
            return canonical
    return None
