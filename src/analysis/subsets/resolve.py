"""
Subset name resolution.

Turns a subset name (a continent code, an alias like ``"USA"``, a bare
filename stem, or a canonical partitioned name like ``"HDI_LO_1999"``) into a
list of country ids, generating and caching HDI/WB partitioned subsets on
first use.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .generators import generate_partitioned_subset_ids
from .registry import default_classifications_path, default_mapping_path, load_country_registry
from .schema import SubsetKind, build_subset_record, read_country_ids, write_subset_record

logger = logging.getLogger(__name__)

# Single source of truth for the canonical partitioned-subset pattern
# (e.g. "HDI_LO_ME_HI_1999"). Shared with src.analysis.core.config, which
# imports this exact object rather than defining its own copy.
PARTITIONED_SUBSET_RE = re.compile(r"^(HDI|WB)_([A-Z_]+)_(\d{4})$")

SUBSET_ALIASES: Dict[str, str] = {
    "H&R": "research_hodler_raschky_2014",
    "USA": "custom_usa",
    "World ex/ USA": "custom_world_ex_usa",
    "World ex USA": "custom_world_ex_usa",
}

# Construction-time invariant: alias keys must never collide with the
# 2-letter-uppercase continent-code pattern, so the two resolution branches
# below stay mutually exclusive by construction.
_TWO_LETTER_ALIAS_KEYS = [key for key in SUBSET_ALIASES if len(key) == 2 and key.isupper()]
assert not _TWO_LETTER_ALIAS_KEYS, (
    f"SUBSET_ALIASES keys must not be 2-letter uppercase codes (collides with "
    f"continent-code resolution): {_TWO_LETTER_ALIAS_KEYS}"
)


def _resolve_and_cache_partitioned_subset(
    subset_name: str,
    *,
    subsets_dir: Path,
    project_root: Path,
    layout: str = "legacy",
) -> Optional[List[int]]:
    """Lazily generate and cache an HDI/WB partitioned subset, if requested."""
    match = PARTITIONED_SUBSET_RE.fullmatch(subset_name)
    if not match:
        return None

    family, bucket, year = match.groups()
    bucket_tokens = bucket.split("_")

    classifications_path = default_classifications_path(project_root, layout=layout)
    mapping_path = default_mapping_path(project_root, layout=layout)

    if not classifications_path.exists() or not mapping_path.exists():
        return None

    classifications_df = pd.read_parquet(classifications_path)
    registry = load_country_registry(mapping_path)
    country_ids = generate_partitioned_subset_ids(
        family, bucket_tokens, year, classifications_df, registry
    )

    record = build_subset_record(
        kind=SubsetKind.PARTITIONED,
        name=subset_name,
        country_ids=country_ids,
        generated_from="country_classifications",
    )
    subset_file = Path(subsets_dir) / f"{subset_name}.json"
    write_subset_record(subset_file, record)
    logger.info(
        f"Generated subset '{subset_name}' from country classifications: "
        f"{record['n_countries']} countries"
    )
    return record["country_ids"]


def resolve_subset(
    subset_name: str,
    *,
    subsets_dir: Path,
    project_root: Path,
    layout: str = "legacy",
) -> List[int]:
    """Resolve a subset name to a list of country ids.

    Resolution is evaluated as mutually exclusive branches on the *original*
    input name (never on an already-substituted value), in this order:

    1. Alias lookup (``SUBSET_ALIASES``) -> target filename stem.
    2. Two-letter uppercase code -> ``continent_<code>.json``.
    3. Name ending in ``.json`` -> used verbatim.
    4. Canonical partitioned name (``HDI_*``/``WB_*``) -> lazily generated
       and cached from country classification data.
    5. Otherwise -> ``<name>.json``.

    Raises
    ------
    FileNotFoundError
        When no matching subset file exists and the name isn't a partitioned
        subset that can be generated on the fly.
    """
    subsets_dir = Path(subsets_dir)

    if subset_name in SUBSET_ALIASES:
        subset_file = subsets_dir / f"{SUBSET_ALIASES[subset_name]}.json"
    elif len(subset_name) == 2 and subset_name.isupper():
        subset_file = subsets_dir / f"continent_{subset_name.lower()}.json"
    elif subset_name.endswith(".json"):
        subset_file = subsets_dir / subset_name
    else:
        subset_file = subsets_dir / f"{subset_name}.json"

    if not subset_file.exists():
        generated_country_ids = _resolve_and_cache_partitioned_subset(
            subset_name,
            subsets_dir=subsets_dir,
            project_root=project_root,
            layout=layout,
        )
        if generated_country_ids is not None:
            return generated_country_ids

        available = sorted(f.stem for f in subsets_dir.glob("*.json")) if subsets_dir.exists() else []
        raise FileNotFoundError(f"Subset '{subset_name}' not found. Available: {available}")

    country_ids = read_country_ids(subset_file)
    logger.info(f"Loaded subset '{subset_name}': {len(country_ids)} countries")
    return country_ids


def list_available_subsets(subsets_dir: Path) -> Dict[str, str]:
    """Enumerate generated subset files and known aliases."""
    subsets_dir = Path(subsets_dir)
    info: Dict[str, str] = {}
    if subsets_dir.exists():
        for path in sorted(subsets_dir.glob("*.json")):
            info[path.stem] = str(path)
    for alias, target in SUBSET_ALIASES.items():
        info[alias] = f"alias -> {target}"
    return info
