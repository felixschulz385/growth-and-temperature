from pathlib import Path

import pytest

from gnt.analysis import runner
from gnt.analysis import workflow


def test_load_config_delegates_to_analysis_config(monkeypatch):
    class DummyAnalysisConfig:
        def __init__(self, config_path):
            self.config_path = config_path

        def as_workflow_config(self):
            return {
                "analyses": {"duckreg": {"specifications": {}, "defaults": {}}},
                "output": {"base_path": "out"},
                "config_path": self.config_path,
            }

    monkeypatch.setattr(workflow, "AnalysisConfig", DummyAnalysisConfig)

    config = workflow.load_config("analysis.xlsx")

    assert config["config_path"] == "analysis.xlsx"
    assert config["output"]["base_path"] == "out"


def test_run_duckreg_accepts_legacy_dict_config(monkeypatch, tmp_path):
    calls = []

    def fake_run_duckreg(config, spec_name, **kwargs):
        calls.append((config, spec_name, kwargs))
        return config.get_model_spec(spec_name)

    legacy_config = {
        "analyses": {
            "duckreg": {
                "specifications": {
                    "model_a": {
                        "model_name": "model_a",
                        "formula": "y ~ x",
                        "data_source": "data.parquet",
                    }
                },
                "defaults": {},
            }
        },
        "output": {"base_path": str(tmp_path)},
    }
    monkeypatch.setattr(workflow, "_run_duckreg", fake_run_duckreg)

    spec = workflow.run_duckreg(
        legacy_config,
        "model_a",
        output_dir="results",
        verbose=False,
        dataset_override="override.parquet",
    )

    assert spec["formula"] == "y ~ x"
    assert calls[0][1] == "model_a"
    assert calls[0][2] == {
        "output_dir": "results",
        "verbose": False,
        "dataset_override": "override.parquet",
    }
    assert calls[0][0].base_path == Path(tmp_path)


def test_legacy_dict_adapter_rejects_variant_overrides():
    config = {
        "analyses": {
            "duckreg": {
                "specifications": {"model_a": {"formula": "y ~ x"}},
                "defaults": {},
            }
        }
    }
    adapter = workflow._WorkflowConfigAdapter(config)

    with pytest.raises(ValueError, match="Variant overrides require AnalysisConfig"):
        adapter.get_model_spec("model_a", fixed_effects="PX")


def test_build_geographic_query_accepts_legacy_subset_key(monkeypatch):
    monkeypatch.setattr(runner, "load_subset", lambda subset_name: [1, 2])

    query = workflow.build_geographic_query({"subset": "AF"})

    assert query == "(country.isin([1, 2]))"
