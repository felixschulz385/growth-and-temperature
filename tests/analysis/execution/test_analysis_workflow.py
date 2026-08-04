import sys
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.core.config import AnalysisConfig, resolve_table_spatial_extent_label
from src.analysis.execution import runner
from src.cli.config import load_config_with_env_vars


def test_excel_configs_are_loaded_through_analysis_config(monkeypatch, tmp_path):
    class DummyAnalysisConfig:
        def __init__(self, config_path):
            self.config_path = Path(config_path)

        def as_workflow_config(self):
            return {
                "analyses": {"duckreg": {"specifications": {}, "defaults": {}}},
                "output": {"base_path": "out"},
                "config_path": str(self.config_path),
            }

    monkeypatch.setattr("src.analysis.core.config.AnalysisConfig", DummyAnalysisConfig)

    config_path = tmp_path / "analysis.xlsx"
    config_path.write_text("")

    config = load_config_with_env_vars(config_path)

    assert config["config_path"] == str(config_path)
    assert config["output"]["base_path"] == "out"


def test_build_geographic_query_uses_spatial_extent(monkeypatch):
    monkeypatch.setattr(runner, "load_subset", lambda subset_name: [1, 2])

    query = runner.build_geographic_query({"spatial_extent": "AF"})

    assert query == "(country.isin([1, 2]))"


def test_build_geographic_query_combines_subset_and_country_filters(monkeypatch):
    monkeypatch.setattr(runner, "load_subset", lambda subset_name: [1, 2])

    query = runner.build_geographic_query(
        {
            "spatial_extent": "AF",
            "countries": [10, 20],
            "country_col": "country_id",
        }
    )

    assert query == "(country_id.isin([1, 2])) & (country_id.isin([10, 20]))"


def test_load_subset_generates_partitioned_hdi_subset_from_classifications(tmp_path):
    project_root = tmp_path
    subsets_dir = project_root / "data_nobackup" / "subsets"
    classifications_dir = (
        project_root
        / "data_nobackup"
        / "misc"
        / "processed"
        / "stage_1"
        / "country_classifications"
    )
    mapping_dir = (
        project_root
        / "data_nobackup"
        / "misc"
        / "processed"
        / "stage_2"
        / "gadm"
    )
    classifications_dir.mkdir(parents=True)
    mapping_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"iso3": "AAA", "HDI_LO_1999": True, "HDI_ME_1999": False, "HDI_HI_1999": False},
            {"iso3": "BBB", "HDI_LO_1999": False, "HDI_ME_1999": True, "HDI_HI_1999": False},
            {"iso3": "CCC", "HDI_LO_1999": False, "HDI_ME_1999": False, "HDI_HI_1999": False},
        ]
    ).to_parquet(classifications_dir / "classifications.parquet", index=False)
    mapping_dir.joinpath("GID_0_code_mapping.json").write_text(
        '{"AAA": 10, "BBB": 20, "CCC": 30}'
    )

    country_ids = runner.load_subset(
        "HDI_LO_ME_HI_1999",
        subsets_dir=subsets_dir,
        project_root=project_root,
    )

    assert country_ids == [10, 20]
    assert (subsets_dir / "HDI_LO_ME_HI_1999.json").exists()


def test_load_subset_generates_partitioned_wb_subset_from_classifications(tmp_path):
    project_root = tmp_path
    subsets_dir = project_root / "data_nobackup" / "subsets"
    classifications_dir = (
        project_root
        / "data_nobackup"
        / "misc"
        / "processed"
        / "stage_1"
        / "country_classifications"
    )
    mapping_dir = (
        project_root
        / "data_nobackup"
        / "misc"
        / "processed"
        / "stage_2"
        / "gadm"
    )
    classifications_dir.mkdir(parents=True)
    mapping_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"iso3": "AAA", "WB_LO_1999": True, "WB_LM_1999": False, "WB_UM_1999": False},
            {"iso3": "BBB", "WB_LO_1999": False, "WB_LM_1999": False, "WB_UM_1999": False},
            {"iso3": "CCC", "WB_LO_1999": False, "WB_LM_1999": True, "WB_UM_1999": True},
        ]
    ).to_parquet(classifications_dir / "classifications.parquet", index=False)
    mapping_dir.joinpath("GID_0_code_mapping.json").write_text(
        '{"AAA": 10, "BBB": 20, "CCC": 30}'
    )

    country_ids = runner.load_subset(
        "WB_LO_LM_UM_1999",
        subsets_dir=subsets_dir,
        project_root=project_root,
    )

    assert country_ids == [10, 30]
    assert (subsets_dir / "WB_LO_LM_UM_1999.json").exists()


@pytest.mark.parametrize(
    ("spatial_extent", "expected"),
    [
        ("HDI < V. High", "HDI_LO_ME_HI_1999"),
        ("HDI excl V High", "HDI_LO_ME_HI_1999"),
        ("WB < High", "WB_LO_LM_UM_1999"),
        ("WB excl High", "WB_LO_LM_UM_1999"),
        ("HDI_LO_ME_HI_1999", "HDI_LO_ME_HI_1999"),
        ("WB_LO_LM_UM_1999", "WB_LO_LM_UM_1999"),
    ],
)
def test_resolve_table_spatial_extent_label_supports_excluding_top_bucket(
    spatial_extent, expected
):
    assert (
        resolve_table_spatial_extent_label(spatial_extent, "2000-2020", "1km")
        == expected
    )


def test_analysis_config_reads_runtime_settings_sheet(tmp_path):
    config_path = tmp_path / "analysis.xlsx"

    with pd.ExcelWriter(config_path) as writer:
        pd.DataFrame(
            [
                {"key": "max_iterations", "value": 321},
                {"key": "tolerance", "value": 1e-4},
                {"key": "check_interval_growth", "value": False},
                {"key": "singleton_pruning", "value": "one_pass"},
                {"key": "residual_type", "value": "FLOAT"},
            ]
        ).to_excel(writer, sheet_name="Settings", index=False)
        pd.DataFrame(
            [
                {
                    "model_name": "m1",
                    "dependent": "y",
                    "independent": "x",
                    "data_source": "1km",
                    "fixed_effects": "0",
                    "clustering": "ADM2",
                    "instruments": "0",
                    "section": "OLS",
                    "subsection": "Base",
                }
            ]
        ).to_excel(writer, sheet_name="Models", index=False)
        pd.DataFrame(
            [{"table_name": "t1", "model_name": "m1", "order": 1}]
        ).to_excel(writer, sheet_name="Models in Tables", index=False)

    cfg = AnalysisConfig(config_path)
    settings = cfg.get_runtime_settings()

    assert settings["max_iterations"] == 321
    assert settings["tolerance"] == 1e-4
    assert settings["check_interval_growth"] is False
    assert settings["singleton_pruning"] == "one_pass"
    assert settings["residual_type"] == "FLOAT"
    assert settings["threads"] == 4


def test_run_duckreg_passes_settings_sheet_fixef_tuning(monkeypatch, tmp_path):
    captured = {}

    class DummyModel:
        def as_dict(self):
            return {}

        def summary(self):
            return "summary"

        def tidy(self):
            return pd.DataFrame()

    def fake_duckreg(**kwargs):
        captured.update(kwargs)
        return DummyModel()

    monkeypatch.setitem(sys.modules, "duckreg", SimpleNamespace(duckreg=fake_duckreg))
    monkeypatch.setitem(
        sys.modules,
        "duckreg.utils.summary",
        SimpleNamespace(format_model_summary=lambda *args, **kwargs: "summary"),
    )

    class DummyConfig:
        def get_model_spec(self, *args, **kwargs):
            return {
                "description": "OLS - BASE",
                "data_source": "data_nobackup/assembled/1km.parquet",
                "formula": "y ~ x | pixel_id_5km + country*year",
                "fixed_effects_label": "PX5K+CY",
                "resolution": "1km",
                "temporal_extent": "2000-2020",
                "spatial_extent": "full_sample",
                "clustering": "ADM2",
                "cluster1_col": "subdivision",
                "variant_path": ["m1", "PX5K+CY", "1km", "2000-2020", "full_sample", "ADM2"],
            }

        def get_runtime_settings(self, overrides=None):
            settings = {
                "se_method": "CRV1",
                "fitter": "duckdb",
                "fe_method": "demean",
                "compression": 5,
                "seed": 42,
                "n_bootstraps": 0,
                "threads": 4,
                "memory_limit": "112GB",
                "max_temp_directory_size": "768GB",
                "max_iterations": 321,
                "tolerance": 1e-4,
                "check_interval": 50,
                "convergence_sample": 0.05,
                "min_iterations_before_check": 10,
                "check_interval_growth": True,
                "max_check_interval": 100,
                "singleton_pruning": "one_pass",
                "fe_order": "ascending_groups",
                "drop_constant_variables": True,
                "residual_type": "FLOAT",
            }
            if overrides:
                settings.update({k: v for k, v in overrides.items() if v is not None})
            return settings

    runner.run_duckreg(DummyConfig(), "m1", output_dir=None, verbose=False)

    assert captured["max_iterations"] == 321
    assert captured["tolerance"] == 1e-4
    assert captured["check_interval"] == 50
    assert captured["convergence_sample"] == 0.05
    assert captured["min_iterations_before_check"] == 10
    assert captured["check_interval_growth"] is True
    assert captured["max_check_interval"] == 100
    assert captured["singleton_pruning"] == "one_pass"
    assert captured["fe_order"] == "ascending_groups"
    assert captured["drop_constant_variables"] is True
    assert captured["residual_type"] == "FLOAT"
    assert captured["compression"] == 5
