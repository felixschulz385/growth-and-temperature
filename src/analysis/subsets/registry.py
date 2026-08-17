"""
Shared country-identity registry.

Loads the ISO3-to-country-id mapping (and, optionally, a GADM country-name
table) exactly once per call site. This is the single source of truth for
country identity used by subset generation, subset resolution, and the
country-classification rasterization step in the preprocessing pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

#: Where GADM's country-id (GID_0) sidecar lands (src/data/sources/layout.py's
#: grid_store_path(), family="country_id") -- same sidecar filename,
#: alongside country_id.zarr under grid/<grid_id>/, which is why
#: default_mapping_path() takes a grid_id too.
COUNTRY_MAPPING_PATH_TEMPLATE = "data_nobackup/grid/{grid_id}/GID_0_code_mapping.json"
#: PREPARE-stage artefact -- gadm's own simplified level-0 vector.
DEFAULT_GADM_PATH = "data_nobackup/prepared/misc/gadm/gadm_levelADM_0_simplified.gpkg"
#: country_classifications' PREPARE-stage output (a genuine PREPARE-stage
#: artefact, unlike COUNTRY_MAPPING_PATH_TEMPLATE above which is GRID-stage
#: despite being read alongside PREPARE data by resolve.py) -- under the
#: top-level `prepared/` tree (src/data/sources/layout.py's
#: output_root(..., PipelineStep.PREPARE)).
CLASSIFICATIONS_PATH = "data_nobackup/prepared/misc/country_classifications/classifications.parquet"


@dataclass(frozen=True)
class CountryRegistry:
    """ISO3 <-> country_id lookups, plus optional GADM country-name lookup."""

    country_to_id: Dict[str, int]
    id_to_country: Dict[int, str] = field(default_factory=dict)
    gadm_name_to_iso3: Optional[Dict[str, str]] = None

    def iso3_to_id(self, iso3: str) -> Optional[int]:
        return self.country_to_id.get(iso3)

    def name_to_id(self, country_name: str) -> Optional[int]:
        if self.gadm_name_to_iso3 is None:
            raise ValueError(
                "GADM name mapping not loaded; pass gadm_path to load_country_registry()."
            )
        iso3 = self.gadm_name_to_iso3.get(country_name)
        if iso3 is None:
            return None
        return self.country_to_id.get(iso3)


def default_mapping_path(project_root: Path, *, grid_id: str = "legacy_4326") -> Path:
    """GADM's `grid/<grid_id>/GID_0_code_mapping.json` sidecar."""
    rel_path = COUNTRY_MAPPING_PATH_TEMPLATE.format(grid_id=grid_id)
    return Path(project_root) / rel_path


def default_gadm_path(project_root: Path) -> Path:
    return Path(project_root) / DEFAULT_GADM_PATH


def default_classifications_path(project_root: Path) -> Path:
    return Path(project_root) / CLASSIFICATIONS_PATH


def load_country_registry(
    mapping_path: Path,
    gadm_path: Optional[Path] = None,
) -> CountryRegistry:
    """Load a :class:`CountryRegistry` from a country-code-mapping JSON file.

    Parameters
    ----------
    mapping_path:
        Path to a JSON file mapping ISO3 codes to integer country ids.
    gadm_path:
        Optional path to a GADM ADM0 geopackage. When given, builds a
        country-name -> ISO3 lookup for name-based subset generation
        (e.g. research subsets keyed by country name). Requires
        ``geopandas``, which is only imported when this path is provided.
    """
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Country mapping file not found: {mapping_path}")

    with open(mapping_path) as fh:
        country_to_id: Dict[str, int] = json.load(fh)

    id_to_country = {country_id: iso3 for iso3, country_id in country_to_id.items()}

    gadm_name_to_iso3: Optional[Dict[str, str]] = None
    if gadm_path is not None:
        gadm_path = Path(gadm_path)
        if gadm_path.exists():
            import geopandas as gpd

            gadm = gpd.read_file(gadm_path).drop(columns=["geometry"])
            gadm_name_to_iso3 = gadm.set_index("COUNTRY")["GID_0"].to_dict()

    return CountryRegistry(
        country_to_id=country_to_id,
        id_to_country=id_to_country,
        gadm_name_to_iso3=gadm_name_to_iso3,
    )
