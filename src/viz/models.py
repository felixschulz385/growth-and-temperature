"""A backend-neutral view of a fitted regression, for the model-consuming plots.

``FittedModel`` is the stable interface the renderers in ``src.viz.regplot``
depend on. Adapters turn a concrete fit into one:

* :func:`from_pyfixest` -- a ``pyfixest.feols`` / ``fepois`` result (coefficients
  *and* row-level residuals / fitted values);
* :func:`from_results_json` -- a persisted duckreg result dict or a path to one
  (coefficients only);
* :func:`from_duckreg` -- a live ``duckreg`` model: a documented stub for now.

:func:`as_fitted_model` dispatches on type. All third-party imports are lazy, so
``import src.viz`` works without pyfixest or duckreg installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_COEF_COLUMNS = ["term", "estimate", "se", "ci_low", "ci_high", "p"]


@dataclass
class FittedModel:
    """Normalised fit. ``coef_table`` always has columns ``term, estimate, se,
    ci_low, ci_high, p``; the rest are best-effort and may be ``None``.
    """

    coef_table: pd.DataFrame
    formula: str | None = None
    fe_dims: list[str] = field(default_factory=list)
    nobs: int | None = None
    vcov_type: str | None = None
    backend: str = "unknown"
    label: str | None = None
    _resid: np.ndarray | None = None
    _fitted: np.ndarray | None = None
    _frame: pd.DataFrame | None = None

    @property
    def residuals(self) -> np.ndarray:
        if self._resid is None:
            raise ValueError(
                f"this {self.backend!r} model carries no row-level residuals; "
                "residual diagnostics need a pyfixest fit"
            )
        return self._resid

    @property
    def fitted(self) -> np.ndarray:
        if self._fitted is None:
            raise ValueError(
                f"this {self.backend!r} model carries no fitted values; "
                "residual diagnostics need a pyfixest fit"
            )
        return self._fitted

    @property
    def has_residuals(self) -> bool:
        return self._resid is not None and self._fitted is not None


def _coef_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in _COEF_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[_COEF_COLUMNS]


def _fill_ci(df: pd.DataFrame) -> pd.DataFrame:
    missing = df["ci_low"].isna() | df["ci_high"].isna()
    if missing.any():
        half = 1.959963984540054 * df["se"]
        df.loc[missing, "ci_low"] = df["estimate"] - half
        df.loc[missing, "ci_high"] = df["estimate"] + half
    return df


# ---------------------------------------------------------------------------
# pyfixest
# ---------------------------------------------------------------------------


def from_pyfixest(fit, *, data: pd.DataFrame | None = None, label: str | None = None) -> FittedModel:
    """Adapt a ``pyfixest`` estimation result."""
    tidy = fit.tidy().reset_index()
    col = {c.lower(): c for c in tidy.columns}

    def pick(*candidates: str):
        for cand in candidates:
            if cand in tidy.columns:
                return tidy[cand]
            if cand.lower() in col:
                return tidy[col[cand.lower()]]
        return pd.Series([np.nan] * len(tidy))

    rows = pd.DataFrame(
        {
            "term": pick("Coefficient", "index"),
            "estimate": pick("Estimate"),
            "se": pick("Std. Error"),
            "ci_low": pick("2.5%"),
            "ci_high": pick("97.5%"),
            "p": pick("Pr(>|t|)"),
        }
    ).to_dict("records")
    coef_table = _fill_ci(_coef_frame(rows))

    fe_raw = getattr(fit, "_fixef", None)
    fe_dims = [p for p in str(fe_raw).split("+") if p and fe_raw] if fe_raw else []

    resid = fitted = None
    for getter in (lambda: fit.resid(), lambda: fit._u_hat):
        try:
            resid = np.asarray(getter(), dtype=float).ravel()
            if resid.size:
                break
        except Exception:
            resid = None
    try:
        fitted = np.asarray(fit.predict(), dtype=float).ravel()
    except Exception:
        fitted = None
    if fitted is None or not np.isfinite(fitted).any():
        # IV (`Feiv`) has no `.predict()`; reconstruct within-fitted values from
        # the (FE-partialled) outcome pyfixest keeps on the model.
        y = getattr(fit, "_Y", None)
        if y is not None and resid is not None:
            y = np.asarray(y, dtype=float).ravel()
            if y.shape == resid.shape:
                fitted = y - resid

    frame = data if data is not None else getattr(fit, "_data", None)

    return FittedModel(
        coef_table=coef_table,
        formula=getattr(fit, "_fml", None) or getattr(fit, "formula", None),
        fe_dims=fe_dims,
        nobs=int(getattr(fit, "_N", 0)) or None,
        vcov_type=str(getattr(fit, "_vcov_type", None) or "") or None,
        backend="pyfixest",
        label=label,
        _resid=resid,
        _fitted=fitted,
        _frame=frame,
    )


# ---------------------------------------------------------------------------
# duckreg results JSON
# ---------------------------------------------------------------------------


def from_results_json(source, *, label: str | None = None) -> FittedModel:
    """Adapt a persisted duckreg result: a dict, or a path to a ``results_*.json``."""
    from src.analysis.io.results import get_coefficient_data, get_model_metadata

    if isinstance(source, (str, Path)):
        with open(source) as fh:
            model = json.load(fh)
        default_label = Path(source).parent.name
    elif isinstance(source, dict):
        model = source
        default_label = None
    else:
        raise TypeError(f"from_results_json: expected a dict or path, got {type(source)!r}")

    cd = get_coefficient_data(model) or {}
    names = cd.get("names", cd.get("coef_names", []))
    est = cd.get("estimates", cd.get("coefficients", []))
    se = cd.get("std_errors", [])
    p = cd.get("p_values", [])
    lo = cd.get("conf_int_lower", [])
    hi = cd.get("conf_int_upper", [])

    def at(seq, i):
        return seq[i] if i < len(seq) else np.nan

    rows = [
        {
            "term": name,
            "estimate": at(est, i),
            "se": at(se, i),
            "ci_low": at(lo, i),
            "ci_high": at(hi, i),
            "p": at(p, i),
        }
        for i, name in enumerate(names)
    ]
    coef_table = _fill_ci(_coef_frame(rows))

    meta = get_model_metadata(model) or {}
    return FittedModel(
        coef_table=coef_table,
        formula=meta.get("formula"),
        fe_dims=[p for p in str(meta.get("fixed_effects", "")).split("+") if p],
        nobs=meta.get("n_obs") or model.get("model_statistics", {}).get("n_obs"),
        vcov_type=meta.get("clustering"),
        backend="results_json",
        label=label or meta.get("spec_name") or default_label,
    )


# ---------------------------------------------------------------------------
# duckreg (live model) -- extension seam
# ---------------------------------------------------------------------------


def from_duckreg(model, *, label: str | None = None) -> FittedModel:
    """Adapt a live ``duckreg`` model (``DuckRegression`` / ``Duck2SLS``).

    Coefficients come from ``model.tidy()`` (columns ``variable, estimate,
    std_error, p_value, ci_lower, ci_upper``); FE / nobs / clustering from
    ``model.summary()`` when it returns the standardised dict. duckreg compresses
    the design matrix, so there are no row-level residuals -- ``.residuals`` /
    ``.fitted`` stay unavailable.
    """
    tidy = model.tidy()
    rename = {"variable": "term", "estimate": "estimate", "std_error": "se",
              "p_value": "p", "ci_lower": "ci_low", "ci_high": "ci_high",
              "ci_upper": "ci_high"}
    df = tidy.rename(columns=rename)
    coef_table = _fill_ci(_coef_frame(df.to_dict("records")))

    formula = fe_dims = nobs = vcov = None
    try:
        summ = model.summary()
        if isinstance(summ, dict):
            spec = summ.get("model_spec", {})
            formula = spec.get("formula")
            fe_dims = spec.get("fe_cols")
            vcov = spec.get("cluster_col")
            nobs = summ.get("sample_info", {}).get("n_obs")
    except Exception:
        pass

    return FittedModel(
        coef_table=coef_table,
        formula=formula,
        fe_dims=list(fe_dims) if fe_dims else [],
        nobs=nobs,
        vcov_type=str(vcov) if vcov else None,
        backend="duckreg",
        label=label,
    )


# ---------------------------------------------------------------------------


def as_fitted_model(obj, *, label: str | None = None) -> FittedModel:
    """Coerce ``obj`` to a :class:`FittedModel` (pass-through if it already is one)."""
    if isinstance(obj, FittedModel):
        return obj
    if isinstance(obj, (str, Path, dict)):
        return from_results_json(obj, label=label)

    module = (type(obj).__module__ or "").split(".")[0]
    if module == "pyfixest":
        return from_pyfixest(obj, label=label)
    if module == "duckreg":
        return from_duckreg(obj, label=label)
    if hasattr(obj, "tidy") and hasattr(obj, "resid"):
        return from_pyfixest(obj, label=label)

    raise TypeError(f"as_fitted_model: don't know how to adapt {type(obj)!r}")
