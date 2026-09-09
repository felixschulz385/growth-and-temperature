from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.viz.models import FittedModel, as_fitted_model, from_duckreg, from_results_json

_RESULTS_JSON = {
    "analysis_metadata": {
        "analysis_type": "duckreg",
        "formula": "y ~ x + treat | unit + year",
        "fixed_effects": "unit+year",
        "spec_name": "demo",
        "clustering": "country",
    },
    "coefficients": {
        "names": ["x", "treat"],
        "estimates": [2.01, 0.48],
        "std_errors": [0.05, 0.12],
        "p_values": [0.0, 0.001],
    },
    "model_statistics": {"n_obs": 800},
}


def test_from_results_json_dict():
    fm = from_results_json(_RESULTS_JSON)
    assert fm.backend == "results_json"
    assert list(fm.coef_table["term"]) == ["x", "treat"]
    assert fm.coef_table.loc[0, "estimate"] == pytest.approx(2.01)
    # CI filled from se when absent
    assert fm.coef_table.loc[0, "ci_low"] == pytest.approx(2.01 - 1.959963984540054 * 0.05)
    assert fm.fe_dims == ["unit", "year"]
    assert fm.nobs == 800
    assert fm.label == "demo"


def test_from_results_json_path(tmp_path):
    import json

    p = tmp_path / "demo" / "results_20240101_000000.json"
    p.parent.mkdir()
    p.write_text(json.dumps(_RESULTS_JSON))
    fm = from_results_json(str(p))
    assert list(fm.coef_table["term"]) == ["x", "treat"]


def test_results_json_model_has_no_residuals():
    fm = from_results_json(_RESULTS_JSON)
    with pytest.raises(ValueError):
        _ = fm.residuals
    assert fm.has_residuals is False


def test_as_fitted_model_passthrough_and_dict():
    fm = from_results_json(_RESULTS_JSON)
    assert as_fitted_model(fm) is fm
    assert isinstance(as_fitted_model(_RESULTS_JSON), FittedModel)


def test_from_duckreg_adapts_tidy():
    class _FakeDuckreg:  # mimics duckreg's .tidy() / .summary() surface
        def tidy(self):
            return pd.DataFrame(
                {
                    "variable": ["Intercept", "log_ntl"],
                    "estimate": [300.0, -0.18],
                    "std_error": [1.0, 0.27],
                    "p_value": [0.0, 0.5],
                    "ci_lower": [298.0, -0.71],
                    "ci_upper": [302.0, 0.35],
                }
            )

        def summary(self):
            return {
                "model_spec": {"formula": "y ~ 1 | fe | (x ~ z)", "fe_cols": ["pixel_id", "GID_0^year"],
                               "cluster_col": "GID_0"},
                "sample_info": {"n_obs": 25_000_000},
            }

    fm = from_duckreg(_FakeDuckreg(), label="2SLS")
    assert fm.backend == "duckreg"
    assert list(fm.coef_table["term"]) == ["Intercept", "log_ntl"]
    assert fm.coef_table.loc[fm.coef_table["term"] == "log_ntl", "ci_low"].iloc[0] == pytest.approx(-0.71)
    assert fm.fe_dims == ["pixel_id", "GID_0^year"]
    assert fm.nobs == 25_000_000
    assert fm.label == "2SLS"
    with pytest.raises(ValueError):  # no row-level residuals from a compressed fit
        _ = fm.residuals


def test_from_pyfixest_roundtrip(panel_frame):
    pf = pytest.importorskip("pyfixest")
    fit = pf.feols("y ~ x + treat | unit + year", data=panel_frame)
    fm = as_fitted_model(fit)
    assert fm.backend == "pyfixest"
    assert set(fm.coef_table["term"]) >= {"x", "treat"}
    assert fm.coef_table.loc[fm.coef_table["term"] == "x", "estimate"].iloc[0] == pytest.approx(2.0, abs=0.1)
    assert fm.has_residuals
    assert fm.residuals.shape[0] == len(panel_frame)
    assert fm.fitted.shape[0] == len(panel_frame)
