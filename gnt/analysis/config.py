"""
Unified configuration reader for ``orchestration/configs/analysis.xlsx``.

This is the single source of truth for all three concerns that previously
read the workbook independently:

* **Model execution** (workflow.py / run.py) — formula, data source,
  per-model settings overrides.
* **SLURM submission** (submit_table_analysis.py) — table membership,
  wall-clock time budgets.
* **Table rendering** (generate_tables.py) — table display configuration,
  output formats.

Sheets consumed
---------------
Settings         key / value  → global defaults (threads, memory_limit, …)
Models           one row per model specification
Models in Tables table_name / model_name / order
Tables           (optional) per-table display overrides for generate_tables
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ANALYSIS_DIR = Path(__file__).parent           # gnt/analysis/
PROJECT_ROOT = _ANALYSIS_DIR.parent.parent       # project root
RESULTS_DIR = PROJECT_ROOT / "output" / "analysis"
DEFAULT_EXCEL = PROJECT_ROOT / "orchestration" / "configs" / "analysis.xlsx"

# Column names in the Models sheet that map to per-model DuckReg settings
_SETTING_KEYS = (
    'se_method', 'fitter', 'fe_method', 'round_strata', 'seed',
    'n_bootstraps', 'threads', 'memory_limit', 'max_temp_directory_size',
)


# ---------------------------------------------------------------------------
# Runtime helpers (used by both config.py and slurm.py)
# ---------------------------------------------------------------------------

def parse_runtime_to_seconds(runtime_str: str) -> int:
    """Parse ``HH:MM:SS`` or ``D-HH:MM:SS`` to integer seconds."""
    runtime_str = str(runtime_str).strip()
    days = 0
    if '-' in runtime_str:
        day_part, time_part = runtime_str.split('-', 1)
        days = int(day_part)
    else:
        time_part = runtime_str

    parts = time_part.split(':')
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        h, m, s = 0, int(parts[0]), int(parts[1])
    else:
        raise ValueError(f"Cannot parse runtime: {runtime_str!r}")

    return days * 86400 + h * 3600 + m * 60 + s


def seconds_to_slurm_time(seconds: int) -> str:
    """Convert integer seconds to ``D-HH:MM:SS`` SLURM format."""
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{days}-{hours:02d}:{minutes:02d}:{secs:02d}"


# ---------------------------------------------------------------------------
# AnalysisConfig
# ---------------------------------------------------------------------------

class AnalysisConfig:
    """Single interface for reading ``analysis.xlsx``.

    Parameters
    ----------
    excel_path:
        Path to the Excel workbook.  Defaults to
        ``orchestration/configs/analysis.xlsx``.
    results_dir:
        Base directory where analysis results live.  Defaults to
        ``output/analysis`` inside the project root.
    """

    def __init__(
        self,
        excel_path: Optional[str | Path] = None,
        results_dir: Optional[str | Path] = None,
    ):
        self.excel_path = Path(excel_path) if excel_path else DEFAULT_EXCEL
        if not self.excel_path.exists():
            raise FileNotFoundError(
                f"Analysis Excel workbook not found: {self.excel_path}"
            )
        self.base_path = Path(results_dir) if results_dir else RESULTS_DIR
        self._xl: Optional[pd.ExcelFile] = None
        self._cache: Dict[str, Any] = {}

    # ── lazy sheet accessors ────────────────────────────────────────────────

    def _xl_file(self) -> pd.ExcelFile:
        if self._xl is None:
            self._xl = pd.ExcelFile(self.excel_path)
        return self._xl

    def _sheet(self, name: str) -> Optional[pd.DataFrame]:
        key = f"_sheet_{name}"
        if key not in self._cache:
            xl = self._xl_file()
            self._cache[key] = (
                pd.read_excel(xl, sheet_name=name)
                if name in xl.sheet_names
                else None
            )
        return self._cache[key]

    # ── convenience properties ──────────────────────────────────────────────

    @property
    def df_settings(self) -> pd.DataFrame:
        """``Settings`` sheet as a raw DataFrame."""
        df = self._sheet('Settings')
        if df is None:
            raise ValueError("'Settings' sheet not found in workbook")
        return df

    @property
    def df_models(self) -> pd.DataFrame:
        """``Models`` sheet as a raw DataFrame."""
        df = self._sheet('Models')
        if df is None:
            raise ValueError("'Models' sheet not found in workbook")
        return df

    @property
    def df_models_in_tables(self) -> pd.DataFrame:
        """``Models in Tables`` sheet as a raw DataFrame."""
        df = self._sheet('Models in Tables')
        if df is None:
            raise ValueError("'Models in Tables' sheet not found in workbook")
        return df

    @property
    def df_tables(self) -> Optional[pd.DataFrame]:
        """``Tables`` sheet, or *None* if the sheet does not exist."""
        return self._sheet('Tables')

    # ── Settings / defaults ─────────────────────────────────────────────────

    def get_defaults(self) -> Dict[str, Any]:
        """Return global setting defaults from the ``Settings`` sheet."""
        if '_defaults' in self._cache:
            return self._cache['_defaults']

        defaults: Dict[str, Any] = {}
        for _, row in self.df_settings.iterrows():
            key = str(row['key']).strip()
            val = row['value']
            if pd.isna(val):
                continue
            expanded = os.path.expandvars(str(val).strip())
            try:
                defaults[key] = int(expanded)
            except ValueError:
                try:
                    defaults[key] = float(expanded)
                except ValueError:
                    defaults[key] = expanded

        self._cache['_defaults'] = defaults
        return defaults

    # ── Model specs ─────────────────────────────────────────────────────────

    def get_model_names(self) -> List[str]:
        """Return all model names defined in the ``Models`` sheet."""
        return self.df_models['model_name'].dropna().str.strip().tolist()

    def get_model_spec(self, model_name: str) -> Dict[str, Any]:
        """Return the full specification dict for *model_name*.

        The returned dict mirrors the structure expected by
        :func:`~gnt.analysis.runner.run_duckreg`:

        .. code-block:: python

            {
                'description': str,
                'data_source': str,           # expanded path
                'formula': str,               # patsy-style
                'settings': dict,             # per-model overrides
                'cluster1_col': str,          # optional
                'query': str,                 # optional SQL WHERE
                'subset': str,               # optional subset name
            }
        """
        df = self.df_models
        rows = df[df['model_name'].str.strip() == model_name]
        if rows.empty:
            available = df['model_name'].str.strip().tolist()
            raise KeyError(
                f"Model '{model_name}' not found in 'Models' sheet. "
                f"Available: {available}"
            )
        row = rows.iloc[0]

        # ── formula ─────────────────────────────────────────────────────────
        dependent = str(row['dependent']).strip()
        independent = str(row['independent']).strip()

        fe_raw = str(row.get('fixed_effects', '0')).strip()
        fixed_effects: List[str] = (
            [fe.strip() for fe in fe_raw.split(',')]
            if fe_raw not in ('0', 'nan', '')
            else []
        )

        instruments_raw = str(row.get('instruments', '0')).strip()
        has_iv = instruments_raw not in ('0', 'nan', '')

        formula = f"{dependent} ~ {independent}"
        if fixed_effects:
            formula += " | " + " + ".join(fixed_effects)
        if has_iv:
            formula += f" | {instruments_raw}"

        # ── data source ──────────────────────────────────────────────────────
        data_source = str(row['data_source']).strip()
        if not data_source.startswith('/') and not data_source.startswith('$'):
            data_source = os.path.expandvars(
                f"data_nobackup/assembled/{data_source}.parquet"
            )
        else:
            data_source = os.path.expandvars(data_source)

        # ── per-model settings overrides ─────────────────────────────────────
        model_settings: Dict[str, Any] = {}
        for key in _SETTING_KEYS:
            val = row.get(key)
            if val is not None and pd.notna(val) and str(val).strip() not in ('', 'nan'):
                if isinstance(val, float) and val == int(val):
                    val = int(val)
                model_settings[key] = val

        spec: Dict[str, Any] = {
            'description': (
                f"{row.get('section', 'Analysis')} - {row.get('subsection', model_name)}"
            ),
            'data_source': data_source,
            'formula': formula,
            'settings': model_settings,
        }

        cluster = str(row.get('clustering', '0')).strip()
        if cluster not in ('0', 'nan', ''):
            spec['cluster1_col'] = cluster

        query = str(row.get('query', '')).strip()
        if query and query != 'nan':
            spec['query'] = query

        subset = str(row.get('subset', '')).strip()
        if subset and subset != 'nan':
            spec['subset'] = subset

        return spec

    def get_all_model_specs(self) -> Dict[str, Dict[str, Any]]:
        """Return a ``{model_name: spec_dict}`` mapping for all models."""
        return {name: self.get_model_spec(name) for name in self.get_model_names()}

    # ── Table membership ────────────────────────────────────────────────────

    def get_all_table_names(self) -> List[str]:
        """Return unique table names in ``Models in Tables`` sheet order."""
        return (
            self.df_models_in_tables['table_name']
            .dropna()
            .unique()
            .tolist()
        )

    def get_models_for_table(self, table_name: str) -> List[str]:
        """Return model names for *table_name* sorted by the ``order`` column."""
        rows = self.df_models_in_tables[
            self.df_models_in_tables['table_name'] == table_name
        ].sort_values('order')
        if rows.empty:
            available = self.get_all_table_names()
            raise KeyError(
                f"Table '{table_name}' not found in 'Models in Tables'. "
                f"Available: {available}"
            )
        return rows['model_name'].tolist()

    # ── Runtime budgets (SLURM) ─────────────────────────────────────────────

    def get_model_runtime_seconds(self, model_name: str) -> int:
        """Return the ``max_runtime`` for *model_name* converted to seconds."""
        df = self.df_models
        rows = df[df['model_name'].str.strip() == model_name]
        if rows.empty:
            raise KeyError(f"Model '{model_name}' not found in 'Models' sheet")
        runtime_str = rows.iloc[0]['max_runtime']
        return parse_runtime_to_seconds(runtime_str)

    def get_table_runtime_seconds(self, table_name: str) -> int:
        """Return the sum of ``max_runtime`` for all models in *table_name*."""
        models = self.get_models_for_table(table_name)
        return sum(self.get_model_runtime_seconds(m) for m in models)

    def get_table_runtime_slurm(self, table_name: str) -> str:
        """Return SLURM-formatted combined runtime for *table_name*."""
        return seconds_to_slurm_time(self.get_table_runtime_seconds(table_name))

    # ── Table display config (generate_tables) ──────────────────────────────

    def get_models_for_table_with_labels(self, table_name: str) -> List[Tuple[str, Optional[str]]]:
        """Return ``[(model_name, model_label), …]`` for *table_name*, sorted by ``order``.

        ``model_label`` is *None* when the column is absent or the cell is empty.
        """
        rows = self.df_models_in_tables[
            self.df_models_in_tables['table_name'] == table_name
        ].sort_values('order')
        if rows.empty:
            available = self.get_all_table_names()
            raise KeyError(
                f"Table '{table_name}' not found in 'Models in Tables'. "
                f"Available: {available}"
            )
        result: List[Tuple[str, Optional[str]]] = []
        for _, row in rows.iterrows():
            name = str(row['model_name']).strip()
            label_raw = row.get('model_label')
            label: Optional[str] = (
                str(label_raw).strip()
                if label_raw is not None and pd.notna(label_raw) and str(label_raw).strip()
                else None
            )
            result.append((name, label))
        return result

    def get_table_display_config(self, table_name: str) -> Dict[str, Any]:
        """Return a ``create_regression_table``-compatible kwarg dict.

        Reads per-table display overrides from the optional ``Tables`` sheet.
        The ``model_paths`` key always contains the ordered list of model names
        for *table_name*.

        ``Tables`` sheet columns recognised:
        ``caption``, ``notes`` (``|``-separated list), ``show_first_stage``,
        ``table_environment``, ``table_size`` (alias: ``fontsize``), ``decimals``, ``stars``,
        ``include_ci``, ``model_display_names`` (comma/semicolon-separated),
        ``output_formats``, ``show_stats``,
        ``select_coefs_keys`` / ``select_coefs_labels`` (semicolon-separated),
        ``select_instruments_keys`` / ``select_instruments_labels``

        ``Models in Tables`` sheet columns recognised:
        ``model_label`` — used as column display names when
        ``model_display_names`` is not set in the ``Tables`` sheet.
        Any additional columns beyond ``order``, ``table_name``,
        ``model_name``, ``model_label`` are collected into a
        ``custom_rows`` dict (``{column_name: [value_per_model]}``) and
        appended to the bottom block of the rendered table.
        """
        model_label_pairs = self.get_models_for_table_with_labels(table_name)
        models = [name for name, _ in model_label_pairs]
        labels = [label for _, label in model_label_pairs]
        config: Dict[str, Any] = {'model_paths': models}

        # Use model_label column from Models in Tables as default display names
        if any(lbl is not None for lbl in labels):
            config['model_names'] = [
                lbl if lbl is not None else name
                for name, lbl in model_label_pairs
            ]

        if self.df_tables is None:
            return config

        trows = self.df_tables[self.df_tables['table_name'] == table_name]
        if trows.empty:
            return config

        r = trows.iloc[0]

        # scalar string overrides
        for col in ('caption', 'show_first_stage', 'table_environment', 'table_size'):
            val = r.get(col)
            if val is not None and pd.notna(val):
                config[col] = str(val).strip()

        # 'fontsize' is a user-friendly alias for 'table_size' (takes precedence)
        fontsize_val = r.get('fontsize')
        if fontsize_val is not None and pd.notna(fontsize_val):
            config['table_size'] = str(fontsize_val).strip()

        # notes: pipe-separated → list of strings
        notes_val = r.get('notes')
        if notes_val is not None and pd.notna(notes_val):
            notes_parts = [s.strip() for s in str(notes_val).split('|') if s.strip()]
            config['notes'] = notes_parts if len(notes_parts) > 1 else notes_parts[0]

        # scalar int override
        for col in ('decimals',):
            val = r.get(col)
            if val is not None and pd.notna(val):
                config[col] = int(val)

        # scalar bool overrides (handle both Python bool and Excel string "True"/"False")
        for col in ('stars', 'include_ci'):
            val = r.get(col)
            if val is not None and pd.notna(val):
                if isinstance(val, bool):
                    config[col] = val
                elif isinstance(val, str):
                    config[col] = val.strip().lower() not in ('false', '0', 'no', '')
                else:
                    config[col] = bool(val)

        # comma-or-semicolon separated list helper
        def _split(col_name: str, sep: str = None) -> Optional[List[str]]:
            val = r.get(col_name)
            if val is None or pd.isna(val):
                return None
            raw = str(val).strip()
            if not raw:
                return None
            # auto-detect separator: prefer ';' when present, else ','
            delimiter = sep or (';' if ';' in raw else ',')
            return [s.strip() for s in raw.split(delimiter) if s.strip()]

        # model display names from Tables sheet override model_label
        disp = _split('model_display_names')
        if disp:
            config['model_names'] = disp
        fmts = _split('output_formats')
        if fmts:
            config['output_formats'] = fmts
        stats = _split('show_stats')
        if stats:
            config['show_stats'] = stats

        # select_coefs: zip keys and labels into a dict
        coef_keys = _split('select_coefs_keys', sep=';')
        coef_labels = _split('select_coefs_labels', sep=';')
        if coef_keys:
            if coef_labels and len(coef_labels) == len(coef_keys):
                config['select_coefs'] = dict(zip(coef_keys, coef_labels))
            else:
                config['select_coefs'] = {k: k for k in coef_keys}

        # select_instruments: zip keys and labels into a dict
        inst_keys = _split('select_instruments_keys', sep=';')
        inst_labels = _split('select_instruments_labels', sep=';')
        if inst_keys:
            if inst_labels and len(inst_labels) == len(inst_keys):
                config['select_instruments'] = dict(zip(inst_keys, inst_labels))
            else:
                config['select_instruments'] = {k: k for k in inst_keys}

        # Extra columns in Models in Tables → custom_rows at bottom of table
        _mit_standard_cols = {'order', 'table_name', 'model_name', 'model_label'}
        extra_cols = [
            c for c in self.df_models_in_tables.columns
            if c not in _mit_standard_cols
        ]
        if extra_cols:
            # Reload rows in order so values align with models
            mit_rows = self.df_models_in_tables[
                self.df_models_in_tables['table_name'] == table_name
            ].sort_values('order')
            extra_custom: Dict[str, List] = {}
            for col in extra_cols:
                vals = []
                for _, row in mit_rows.iterrows():
                    v = row.get(col)
                    vals.append(
                        str(v).strip() if v is not None and pd.notna(v) else ''
                    )
                # Only include the column when at least one model has a value
                if any(v != '' for v in vals):
                    extra_custom[col] = vals
            if extra_custom:
                # Merge with any custom_rows already in config
                existing = config.get('custom_rows')
                if existing is None:
                    config['custom_rows'] = extra_custom
                elif isinstance(existing, dict):
                    config['custom_rows'] = [existing, extra_custom]
                else:
                    config['custom_rows'] = list(existing) + [extra_custom]

        return config

    # ── Convenience / diagnostics ───────────────────────────────────────────

    def get_missing_models(self, analysis_type: str = 'duckreg') -> Dict[str, List[str]]:
        """Return ``{table_name: [missing_model_names]}`` for models without results.

        A model is considered *missing* when its result directory does not
        exist under ``self.base_path``.
        """
        missing: Dict[str, List[str]] = {}
        for table in self.get_all_table_names():
            absent = [
                m for m in self.get_models_for_table(table)
                if not (self.base_path / analysis_type / m).exists()
            ]
            if absent:
                missing[table] = absent
        return missing

    def table_runtime_summary(self) -> pd.DataFrame:
        """Return a DataFrame with runtime info for every table.

        Columns: ``table_name``, ``n_models``, ``total_seconds``,
        ``slurm_time``.
        """
        rows = []
        for table in self.get_all_table_names():
            models = self.get_models_for_table(table)
            secs = self.get_table_runtime_seconds(table)
            rows.append({
                'table_name': table,
                'n_models': len(models),
                'total_seconds': secs,
                'slurm_time': seconds_to_slurm_time(secs),
            })
        return pd.DataFrame(rows)

    def as_workflow_config(self) -> Dict[str, Any]:
        """Return a dict compatible with the legacy ``workflow.py`` ``load_config`` format."""
        return {
            'analyses': {
                'duckreg': {
                    'specifications': self.get_all_model_specs(),
                    'defaults': self.get_defaults(),
                }
            },
            'output': {'base_path': str(self.base_path)},
        }
