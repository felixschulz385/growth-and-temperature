"""
Country subset generation and resolution.

Public API:

* :class:`CountryRegistry` / :func:`load_country_registry` — shared
  ISO3<->country_id and country-name->ISO3 lookups.
* :func:`build_subset_record` / :func:`read_country_ids` — the canonical
  subset JSON schema.
* :data:`SUBSET_ALIASES` / :data:`PARTITIONED_SUBSET_RE` / :func:`resolve_subset`
  / :func:`list_available_subsets` — turning a subset name into country ids.
* :func:`generate_all_default_subsets` and the individual ``generate_*``
  functions — building subset files.
"""

from .generators import (
    generate_all_default_subsets,
    generate_continent_subsets,
    generate_custom_subset,
    generate_hodler_raschky_2014_subset,
    generate_partitioned_subset_ids,
    generate_usa_subsets,
)
from .registry import CountryRegistry, default_gadm_path, default_mapping_path, load_country_registry
from .resolve import PARTITIONED_SUBSET_RE, SUBSET_ALIASES, list_available_subsets, resolve_subset
from .schema import SubsetKind, SUBSET_SCHEMA_VERSION, build_subset_record, read_country_ids

__all__ = [
    "CountryRegistry",
    "PARTITIONED_SUBSET_RE",
    "SUBSET_ALIASES",
    "SUBSET_SCHEMA_VERSION",
    "SubsetKind",
    "build_subset_record",
    "default_gadm_path",
    "default_mapping_path",
    "generate_all_default_subsets",
    "generate_continent_subsets",
    "generate_custom_subset",
    "generate_hodler_raschky_2014_subset",
    "generate_partitioned_subset_ids",
    "generate_usa_subsets",
    "list_available_subsets",
    "load_country_registry",
    "read_country_ids",
    "resolve_subset",
]
