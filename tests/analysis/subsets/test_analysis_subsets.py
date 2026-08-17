import json

import pandas as pd
import pytest

from src.analysis.core.config import CANONICAL_PARTITIONED_SPATIAL_EXTENT_RE
from src.analysis.subsets import (
    PARTITIONED_SUBSET_RE,
    SUBSET_ALIASES,
    SubsetKind,
    build_subset_record,
    generate_all_default_subsets,
    generate_continent_subsets,
    generate_custom_subset,
    generate_hodler_raschky_2014_subset,
    generate_usa_subsets,
    list_available_subsets,
    load_country_registry,
    read_country_ids,
    resolve_subset,
)
from src.analysis.subsets.registry import CountryRegistry
from src.analysis.subsets.schema import load_subset_record, write_subset_record


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _write_mapping(path, mapping):
    path.write_text(json.dumps(mapping))


def test_load_country_registry_without_gadm(tmp_path):
    mapping_path = tmp_path / "country_code_mapping.json"
    _write_mapping(mapping_path, {"AAA": 1, "BBB": 2})

    registry = load_country_registry(mapping_path)

    assert registry.country_to_id == {"AAA": 1, "BBB": 2}
    assert registry.id_to_country == {1: "AAA", 2: "BBB"}
    assert registry.gadm_name_to_iso3 is None


def test_load_country_registry_with_gadm(tmp_path, monkeypatch):
    mapping_path = tmp_path / "country_code_mapping.json"
    _write_mapping(mapping_path, {"AAA": 1, "BBB": 2})
    gadm_path = tmp_path / "gadm.gpkg"
    gadm_path.write_text("not a real geopackage")

    fake_gadm_df = pd.DataFrame(
        [
            {"COUNTRY": "Wonderland", "GID_0": "AAA", "geometry": None},
            {"COUNTRY": "Narnia", "GID_0": "BBB", "geometry": None},
        ]
    )
    monkeypatch.setattr("geopandas.read_file", lambda path: fake_gadm_df)

    registry = load_country_registry(mapping_path, gadm_path=gadm_path)

    assert registry.gadm_name_to_iso3 == {"Wonderland": "AAA", "Narnia": "BBB"}
    assert registry.name_to_id("Wonderland") == 1


def test_iso3_to_id_missing_code_returns_none(tmp_path):
    mapping_path = tmp_path / "country_code_mapping.json"
    _write_mapping(mapping_path, {"AAA": 1})

    registry = load_country_registry(mapping_path)

    assert registry.iso3_to_id("ZZZ") is None


def test_name_to_id_without_gadm_raises():
    registry = CountryRegistry(country_to_id={"AAA": 1})

    with pytest.raises(ValueError):
        registry.name_to_id("Wonderland")


def test_load_country_registry_missing_mapping_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_country_registry(tmp_path / "does_not_exist.json")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", list(SubsetKind))
def test_build_subset_record_required_keys_present_for_every_kind(kind):
    record = build_subset_record(kind=kind, name="X", country_ids=[3, 1, 2])

    for key in ("schema_version", "kind", "name", "country_ids", "n_countries"):
        assert key in record
    assert record["n_countries"] == len(record["country_ids"])


def test_build_subset_record_sorts_and_dedupes_country_ids():
    record = build_subset_record(
        kind=SubsetKind.CUSTOM, name="X", country_ids=[5, 1, 5, 3]
    )

    assert record["country_ids"] == [1, 3, 5]
    assert record["n_countries"] == 3


def test_read_country_ids_roundtrip(tmp_path):
    record = build_subset_record(kind=SubsetKind.CUSTOM, name="X", country_ids=[2, 1])
    path = tmp_path / "x.json"
    write_subset_record(path, record)

    assert read_country_ids(path) == [1, 2]
    assert load_subset_record(path)["name"] == "X"


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

def test_resolve_subset_alias_keys_never_collide_with_continent_codes():
    two_letter_uppercase_keys = [k for k in SUBSET_ALIASES if len(k) == 2 and k.isupper()]
    assert two_letter_uppercase_keys == []


def test_resolve_subset_continent_two_letter_code(tmp_path):
    subsets_dir = tmp_path / "subsets"
    subsets_dir.mkdir()
    record = build_subset_record(
        kind=SubsetKind.CONTINENT, name="Africa", country_ids=[1, 2], code="AF"
    )
    write_subset_record(subsets_dir / "continent_af.json", record)

    country_ids = resolve_subset("AF", subsets_dir=subsets_dir, project_root=tmp_path)

    assert country_ids == [1, 2]


@pytest.mark.parametrize("alias,target", list(SUBSET_ALIASES.items()))
def test_resolve_subset_every_alias_entry(tmp_path, alias, target):
    subsets_dir = tmp_path / "subsets"
    subsets_dir.mkdir()
    record = build_subset_record(kind=SubsetKind.CUSTOM, name=alias, country_ids=[42])
    write_subset_record(subsets_dir / f"{target}.json", record)

    country_ids = resolve_subset(alias, subsets_dir=subsets_dir, project_root=tmp_path)

    assert country_ids == [42]


def test_resolve_subset_json_suffix_passthrough(tmp_path):
    subsets_dir = tmp_path / "subsets"
    subsets_dir.mkdir()
    record = build_subset_record(kind=SubsetKind.CUSTOM, name="X", country_ids=[7])
    write_subset_record(subsets_dir / "my_custom.json", record)

    country_ids = resolve_subset(
        "my_custom.json", subsets_dir=subsets_dir, project_root=tmp_path
    )

    assert country_ids == [7]


def test_resolve_subset_bare_name_appends_json(tmp_path):
    subsets_dir = tmp_path / "subsets"
    subsets_dir.mkdir()
    record = build_subset_record(kind=SubsetKind.CUSTOM, name="X", country_ids=[9])
    write_subset_record(subsets_dir / "my_custom.json", record)

    country_ids = resolve_subset("my_custom", subsets_dir=subsets_dir, project_root=tmp_path)

    assert country_ids == [9]


def test_resolve_subset_missing_file_raises_with_available_list(tmp_path):
    subsets_dir = tmp_path / "subsets"
    subsets_dir.mkdir()
    record = build_subset_record(kind=SubsetKind.CUSTOM, name="X", country_ids=[1])
    write_subset_record(subsets_dir / "existing.json", record)

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_subset("nonexistent", subsets_dir=subsets_dir, project_root=tmp_path)

    assert "existing" in str(excinfo.value)


def _write_partitioned_fixture(project_root, rows, mapping):
    classifications_dir = project_root / "data_nobackup" / "prepared" / "misc" / "adm" / "country_classifications"
    mapping_dir = project_root / "data_nobackup" / "prepared" / "misc" / "adm" / "gadm"
    classifications_dir.mkdir(parents=True)
    mapping_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(classifications_dir / "classifications.parquet", index=False)
    mapping_dir.joinpath("GID_0_code_mapping.json").write_text(json.dumps(mapping))


def test_resolve_subset_generates_partitioned_hdi_subset_from_classifications(tmp_path):
    project_root = tmp_path
    subsets_dir = project_root / "data_nobackup" / "subsets"
    _write_partitioned_fixture(
        project_root,
        [
            {"iso3": "AAA", "HDI_LO_1999": True, "HDI_ME_1999": False, "HDI_HI_1999": False},
            {"iso3": "BBB", "HDI_LO_1999": False, "HDI_ME_1999": True, "HDI_HI_1999": False},
            {"iso3": "CCC", "HDI_LO_1999": False, "HDI_ME_1999": False, "HDI_HI_1999": False},
        ],
        {"AAA": 10, "BBB": 20, "CCC": 30},
    )

    country_ids = resolve_subset(
        "HDI_LO_ME_HI_1999", subsets_dir=subsets_dir, project_root=project_root
    )

    assert country_ids == [10, 20]
    assert (subsets_dir / "HDI_LO_ME_HI_1999.json").exists()


def test_resolve_subset_generates_partitioned_wb_subset_from_classifications(tmp_path):
    project_root = tmp_path
    subsets_dir = project_root / "data_nobackup" / "subsets"
    _write_partitioned_fixture(
        project_root,
        [
            {"iso3": "AAA", "WB_LO_1999": True, "WB_LM_1999": False, "WB_UM_1999": False},
            {"iso3": "BBB", "WB_LO_1999": False, "WB_LM_1999": False, "WB_UM_1999": False},
            {"iso3": "CCC", "WB_LO_1999": False, "WB_LM_1999": True, "WB_UM_1999": True},
        ],
        {"AAA": 10, "BBB": 20, "CCC": 30},
    )

    country_ids = resolve_subset(
        "WB_LO_LM_UM_1999", subsets_dir=subsets_dir, project_root=project_root
    )

    assert country_ids == [10, 30]
    assert (subsets_dir / "WB_LO_LM_UM_1999.json").exists()


def test_list_available_subsets_merges_files_and_aliases(tmp_path):
    subsets_dir = tmp_path / "subsets"
    subsets_dir.mkdir()
    record = build_subset_record(kind=SubsetKind.CUSTOM, name="X", country_ids=[1])
    write_subset_record(subsets_dir / "custom_usa.json", record)

    info = list_available_subsets(subsets_dir)

    assert "custom_usa" in info
    assert info["USA"] == "alias -> custom_usa"


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def test_generate_continent_subsets_schema_and_partition(tmp_path):
    registry = CountryRegistry(country_to_id={"AAA": 1, "BBB": 2, "CCC": 3})
    continents_df = pd.DataFrame(
        [
            {"Three_Letter_Country_Code": "AAA", "Continent_Code": "AF", "Continent_Name": "Africa"},
            {"Three_Letter_Country_Code": "BBB", "Continent_Code": "AF", "Continent_Name": "Africa"},
            {"Three_Letter_Country_Code": "CCC", "Continent_Code": "EU", "Continent_Name": "Europe"},
        ]
    )

    output_files = generate_continent_subsets(registry, continents_df, tmp_path)

    assert set(output_files) == {"AF", "EU"}
    af_record = load_subset_record(output_files["AF"])
    assert af_record["code"] == "AF"
    assert af_record["country_ids"] == [1, 2]
    assert af_record["kind"] == "continent"


def test_generate_custom_subset_by_codes_and_by_ids(tmp_path):
    registry = CountryRegistry(country_to_id={"AAA": 1, "BBB": 2})

    by_codes = generate_custom_subset(
        registry, "By Codes", country_codes=["AAA", "BBB"], output_dir=tmp_path
    )
    by_ids = generate_custom_subset(
        registry, "By Ids", country_ids=[1, 2], output_dir=tmp_path
    )

    assert read_country_ids(by_codes) == [1, 2]
    assert read_country_ids(by_ids) == [1, 2]


def test_generate_custom_subset_requires_ids_or_codes(tmp_path):
    registry = CountryRegistry(country_to_id={"AAA": 1})

    with pytest.raises(ValueError):
        generate_custom_subset(registry, "Nothing", output_dir=tmp_path)


def test_generate_usa_subsets_partition_is_complete(tmp_path):
    registry = CountryRegistry(country_to_id={"USA": 1, "BBB": 2, "CCC": 3})

    files = generate_usa_subsets(registry, tmp_path)

    usa_ids = set(read_country_ids(files["usa"]))
    world_ex_usa_ids = set(read_country_ids(files["world_ex_usa"]))

    assert usa_ids == {1}
    assert world_ex_usa_ids == {2, 3}
    assert usa_ids.isdisjoint(world_ex_usa_ids)
    assert usa_ids | world_ex_usa_ids == set(registry.country_to_id.values())


def test_generate_hodler_raschky_2014_reports_missing_countries_in_json(tmp_path, monkeypatch):
    from src.analysis.subsets.generators import HODLER_RASCHKY_2014_COUNTRIES

    present = HODLER_RASCHKY_2014_COUNTRIES[:5]
    gadm_name_to_iso3 = {name: f"I{i:02d}" for i, name in enumerate(present)}
    country_to_id = {iso3: idx for idx, iso3 in enumerate(gadm_name_to_iso3.values())}
    registry = CountryRegistry(
        country_to_id=country_to_id,
        gadm_name_to_iso3=gadm_name_to_iso3,
    )

    output_file = generate_hodler_raschky_2014_subset(registry, tmp_path)
    record = load_subset_record(output_file)

    assert record["n_original"] == len(HODLER_RASCHKY_2014_COUNTRIES)
    assert record["n_countries"] == len(present)
    assert record["n_missing"] == len(HODLER_RASCHKY_2014_COUNTRIES) - len(present)
    assert record["n_countries"] + record["n_missing"] == record["n_original"]
    assert len(record["missing_names"]) == record["n_missing"]


def test_generate_all_default_subsets_continues_after_one_category_fails(tmp_path):
    project_root = tmp_path
    mapping_dir = project_root / "data_nobackup" / "prepared" / "misc" / "adm" / "gadm"
    mapping_dir.mkdir(parents=True)
    mapping_dir.joinpath("GID_0_code_mapping.json").write_text(
        json.dumps({"USA": 1, "BBB": 2})
    )
    # No continents.csv and no GADM geopackage present, so continent
    # subsets and the HR2014 research subset should both be skipped, but
    # USA / World-ex-USA (which only need the mapping file) should succeed.

    output_files = generate_all_default_subsets(project_root)

    assert "usa" in output_files
    assert "world_ex_usa" in output_files
    assert "hodler_raschky_2014" not in output_files


def test_generate_all_default_subsets_returns_empty_when_mapping_missing(tmp_path):
    output_files = generate_all_default_subsets(tmp_path)

    assert output_files == {}


# ---------------------------------------------------------------------------
# Shared regex identity (regression guard against re-duplication)
# ---------------------------------------------------------------------------

def test_canonical_partitioned_regex_is_shared_object():
    assert CANONICAL_PARTITIONED_SPATIAL_EXTENT_RE is PARTITIONED_SUBSET_RE
