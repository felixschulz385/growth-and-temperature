"""
Canonical JSON schema for country subset files.

Every subset generator writes records through :func:`build_subset_record` so
that all subset files — continents, custom groupings, research subsets, and
partitioned HDI/WB buckets — share one consistent set of required keys, with
kind-specific fields layered on top.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

SUBSET_SCHEMA_VERSION = 1


class SubsetKind(str, Enum):
    CONTINENT = "continent"
    CUSTOM = "custom"
    RESEARCH = "research"
    PARTITIONED = "partitioned"


def build_subset_record(
    *,
    kind: SubsetKind,
    name: str,
    country_ids: List[int],
    description: Optional[str] = None,
    code: Optional[str] = None,
    reference: Optional[str] = None,
    n_original: Optional[int] = None,
    n_missing: Optional[int] = None,
    missing_names: Optional[List[str]] = None,
    generated_from: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a canonical subset record.

    Required keys on every record: ``schema_version, kind, name, country_ids,
    n_countries``. All other parameters are optional and additive per *kind*.
    """
    deduped_ids = sorted(set(country_ids))

    record: Dict[str, Any] = {
        "schema_version": SUBSET_SCHEMA_VERSION,
        "kind": kind.value if isinstance(kind, SubsetKind) else kind,
        "name": name,
        "country_ids": deduped_ids,
        "n_countries": len(deduped_ids),
    }

    if description is not None:
        record["description"] = description
    if code is not None:
        record["code"] = code
    if reference is not None:
        record["reference"] = reference
    if n_original is not None:
        record["n_original"] = n_original
    if n_missing is not None:
        record["n_missing"] = n_missing
    if missing_names is not None:
        record["missing_names"] = missing_names
    if generated_from is not None:
        record["generated_from"] = generated_from

    return record


def write_subset_record(path: Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)


def load_subset_record(path: Path) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def read_country_ids(path: Path) -> List[int]:
    """Read the ``country_ids`` list from a subset JSON file."""
    return load_subset_record(path)["country_ids"]
