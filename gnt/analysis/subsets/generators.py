"""
Country subset generation.

Free functions that build subset JSON files (continents, custom groupings,
research subsets, USA/World-ex-USA, and HDI/WB partitioned buckets) against a
shared :class:`~gnt.analysis.subsets.registry.CountryRegistry` and the
canonical schema in :mod:`~gnt.analysis.subsets.schema`.

Filenames are part of an implicit contract with ``SUBSET_ALIASES``
(:mod:`gnt.analysis.subsets.resolve`) and with any spatial-extent values
already typed into an ``analysis.xlsx`` workbook — do not change them without
updating both.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .registry import CountryRegistry, default_gadm_path, default_mapping_path, load_country_registry
from .schema import SubsetKind, build_subset_record, write_subset_record

logger = logging.getLogger(__name__)

DEFAULT_CONTINENTS_PATH = "data_nobackup/misc/raw/continents/continents.csv"
DEFAULT_OUTPUT_DIR = "data_nobackup/subsets"

# Hodler & Raschky (2014) countries. Regional favoritism.
# The Quarterly Journal of Economics, 129(2), 995-1033.
HODLER_RASCHKY_2014_COUNTRIES = [
    "Afghanistan", "Albania", "Algeria", "Angola", "Argentina", "Australia",
    "Austria", "Bangladesh", "Belarus", "Belgium", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina",
    "Botswana", "Brazil", "Bulgaria", "Burkina Faso", "Burundi", "Cambodia", "Cameroon", "Canada",
    "Central African Republic", "Chad", "Chile", "China", "Colombia", "Costa Rica", "Czechia",
    "Côte d'Ivoire", "Democratic Republic of the Congo", "Denmark", "Timor-Leste", "Ecuador", "El Salvador",
    "Eritrea", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany",
    "Ghana", "Greece", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "India", "Indonesia",
    "Iran", "Iraq", "Italy", "Japan", "Jordan", "Kazakhstan", "Kenya", "Laos", "Latvia", "Lebanon",
    "Liberia", "Lithuania", "North Macedonia", "Madagascar", "Malawi", "Malaysia", "Mali", "Mauritania", "México",
    "Mongolia", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nepal", "Netherlands", "New Zealand",
    "Nicaragua", "Niger", "Nigeria", "North Korea", "Norway", "Oman", "Pakistan", "Panama",
    "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Republic of the Congo",
    "Russia", "Rwanda", "Senegal", "Serbia", "Sierra Leone", "Slovakia", "Slovenia", "Somalia", "South Africa",
    "South Korea", "Spain", "Sri Lanka", "Sudan", "Sweden", "Taiwan", "Tajikistan", "Tanzania", "Thailand",
    "Togo", "Tunisia", "Uganda", "Ukraine", "United Kingdom", "United States", "Uruguay", "Venezuela",
    "Vietnam", "Yemen", "Zambia", "Zimbabwe",
]


def generate_continent_subsets(
    registry: CountryRegistry,
    continents_df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """Generate one subset file per continent code."""
    output_dir = Path(output_dir)
    output_files: Dict[str, Path] = {}

    valid_continents = continents_df.dropna(subset=["Continent_Code"])

    for continent_code in valid_continents["Continent_Code"].unique():
        if not isinstance(continent_code, str):
            logger.warning(f"Skipping invalid continent code: {continent_code}")
            continue

        rows = valid_continents.query("Continent_Code == @continent_code")
        country_codes = rows["Three_Letter_Country_Code"].tolist()
        country_ids = [
            registry.country_to_id[code]
            for code in country_codes
            if code in registry.country_to_id
        ]
        if not country_ids:
            logger.warning(f"No countries mapped for continent {continent_code}, skipping")
            continue

        continent_name = rows["Continent_Name"].iloc[0] if not rows.empty else continent_code

        record = build_subset_record(
            kind=SubsetKind.CONTINENT,
            name=continent_name,
            country_ids=country_ids,
            code=continent_code,
        )
        output_file = output_dir / f"continent_{continent_code.lower()}.json"
        write_subset_record(output_file, record)
        output_files[continent_code] = output_file
        logger.info(
            f"Generated subset for {continent_name} ({continent_code}): "
            f"{record['n_countries']} countries"
        )

    return output_files


def generate_custom_subset(
    registry: CountryRegistry,
    name: str,
    *,
    country_ids: Optional[List[int]] = None,
    country_codes: Optional[List[str]] = None,
    description: Optional[str] = None,
    output_dir: Path,
) -> Path:
    """Generate a custom subset file from explicit ids or ISO3 codes."""
    if country_ids is None and country_codes is None:
        raise ValueError("Must provide either country_ids or country_codes")

    if country_codes is not None:
        country_ids = [
            registry.country_to_id[code]
            for code in country_codes
            if code in registry.country_to_id
        ]

    record = build_subset_record(
        kind=SubsetKind.CUSTOM,
        name=name,
        country_ids=country_ids,
        description=description or f"Custom subset: {name}",
    )
    output_file = Path(output_dir) / f"custom_{name.lower().replace(' ', '_')}.json"
    write_subset_record(output_file, record)
    logger.info(f"Generated custom subset '{name}': {record['n_countries']} countries")
    return output_file


def generate_hodler_raschky_2014_subset(registry: CountryRegistry, output_dir: Path) -> Path:
    """Generate the Hodler & Raschky (2014) research subset."""
    if registry.gadm_name_to_iso3 is None:
        raise ValueError("GADM name mapping not loaded. Cannot generate subset.")

    country_ids: List[int] = []
    missing_countries: List[str] = []

    for country_name in HODLER_RASCHKY_2014_COUNTRIES:
        iso3_code = registry.gadm_name_to_iso3.get(country_name)
        if iso3_code is None:
            missing_countries.append(country_name)
            continue

        country_id = registry.country_to_id.get(iso3_code)
        if country_id is None:
            missing_countries.append(f"{country_name} (ISO3: {iso3_code})")
            continue

        country_ids.append(country_id)

    if missing_countries:
        logger.warning(f"Could not map {len(missing_countries)} countries: {missing_countries[:5]}...")

    record = build_subset_record(
        kind=SubsetKind.RESEARCH,
        name="Hodler & Raschky (2014)",
        country_ids=country_ids,
        description="Countries used in Hodler & Raschky (2014) Regional favoritism study",
        reference="Hodler, R., & Raschky, P. A. (2014). Regional favoritism. The Quarterly Journal of Economics, 129(2), 995-1033.",
        n_original=len(HODLER_RASCHKY_2014_COUNTRIES),
        n_missing=len(missing_countries),
        missing_names=missing_countries,
    )
    output_file = Path(output_dir) / "research_hodler_raschky_2014.json"
    write_subset_record(output_file, record)
    logger.info(
        f"Generated Hodler & Raschky (2014) subset: {record['n_countries']}"
        f"/{len(HODLER_RASCHKY_2014_COUNTRIES)} countries mapped"
    )
    return output_file


def generate_usa_subsets(registry: CountryRegistry, output_dir: Path) -> Dict[str, Path]:
    """Generate the USA subset and its complement, World ex/ USA."""
    usa_id = registry.country_to_id.get("USA")
    if usa_id is None:
        raise ValueError("USA not present in country mapping. Cannot generate subset.")

    world_ex_usa_ids = [
        country_id for code, country_id in registry.country_to_id.items()
        if code != "USA"
    ]

    usa_file = generate_custom_subset(
        registry,
        "USA",
        country_codes=["USA"],
        description="United States of America",
        output_dir=output_dir,
    )
    world_ex_usa_file = generate_custom_subset(
        registry,
        "World ex USA",
        country_ids=world_ex_usa_ids,
        description="All countries excluding the United States",
        output_dir=output_dir,
    )
    return {"usa": usa_file, "world_ex_usa": world_ex_usa_file}


def generate_partitioned_subset_ids(
    family: str,
    bucket_tokens: List[str],
    year: str,
    classifications_df: pd.DataFrame,
    registry: CountryRegistry,
) -> List[int]:
    """Compute country ids for an HDI/WB partitioned bucket from classification data."""
    required_columns = [f"{family}_{token}_{year}" for token in bucket_tokens]
    missing_columns = [col for col in required_columns if col not in classifications_df.columns]
    if missing_columns:
        raise FileNotFoundError(
            f"Subset '{family}_{'_'.join(bucket_tokens)}_{year}' requires missing "
            f"classification columns: {missing_columns}"
        )

    mask = classifications_df[required_columns].any(axis=1)
    iso3_codes = classifications_df.loc[mask, "iso3"].dropna().astype(str)
    return [
        registry.country_to_id[iso3]
        for iso3 in iso3_codes
        if iso3 in registry.country_to_id
    ]


def generate_all_default_subsets(
    project_root: Path,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """Generate continent, custom (USA/World-ex-USA), and research subset files."""
    project_root = Path(project_root)
    output_dir = Path(output_dir) if output_dir else project_root / DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files: Dict[str, Path] = {}

    mapping_path = default_mapping_path(project_root)
    gadm_path = default_gadm_path(project_root)
    continents_path = project_root / DEFAULT_CONTINENTS_PATH

    if not mapping_path.exists():
        logger.warning(f"Country mapping file not found: {mapping_path}. Cannot generate subsets.")
        return output_files

    registry = load_country_registry(mapping_path, gadm_path=gadm_path)

    if continents_path.exists():
        continents_df = pd.read_csv(continents_path)
        try:
            continent_files = generate_continent_subsets(registry, continents_df, output_dir)
            output_files.update(continent_files)
        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to generate continent subsets: {e}")
    else:
        logger.warning(f"Continents file not found: {continents_path}. Skipping continent subsets.")

    try:
        usa_files = generate_usa_subsets(registry, output_dir)
        output_files.update(usa_files)
    except ValueError as e:
        logger.warning(f"Failed to generate USA / World ex USA subsets: {e}")

    try:
        hr2014_file = generate_hodler_raschky_2014_subset(registry, output_dir)
        output_files["hodler_raschky_2014"] = hr2014_file
    except ValueError as e:
        logger.warning(f"Failed to generate Hodler & Raschky (2014) subset: {e}")

    logger.info(f"Generated {len(output_files)} default subset files")
    return output_files
