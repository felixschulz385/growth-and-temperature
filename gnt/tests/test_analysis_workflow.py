from pathlib import Path

from gnt.analysis.execution import runner
from gnt.cli.config import load_config_with_env_vars


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

    monkeypatch.setattr("gnt.analysis.core.config.AnalysisConfig", DummyAnalysisConfig)

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
